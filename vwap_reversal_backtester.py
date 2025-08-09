"""
VWAP Reversal Strategy Backtester
Implements a daily strategy that:
1. Selects top 5 stocks based on VWAP reversal signals
2. Buys stocks below VWAP, sells stocks above VWAP
3. Exits positions when price converges to VWAP
4. Downloads 1 year of data for comprehensive backtesting
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
import time
from tqdm import tqdm

# Import existing modules
from core.data_handler import DataHandler
from utils.logger import get_logger
from cache_manager import add_cache_management_to_sidebar

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

class VWAPReversalBacktester:
    """
    VWAP Reversal Strategy Backtester
    
    Strategy Logic:
    - Daily: Select top 5 stocks with highest VWAP reversal signals
    - Buy stocks below VWAP (negative VWAP distance)
    - Sell stocks above VWAP (positive VWAP distance)
    - Exit when price converges to VWAP (within threshold)
    """
    
    def __init__(self, data_handler: DataHandler = None):
        """Initialize the VWAP reversal backtester."""
        self.data_handler = data_handler or DataHandler()
        self.results = {}
        self.trades = []
        self.daily_positions = []
        self.equity_curve = []
        
    def calculate_vwap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators for a stock."""
        data = df.copy()
        
        # Calculate VWAP
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # VWAP distance and position
        data['vwap_distance'] = ((data['close'] - data['vwap']) / data['vwap']) * 100
        data['above_vwap'] = data['close'] > data['vwap']
        data['below_vwap'] = data['close'] < data['vwap']
        
        # VWAP reversal signals (change in position relative to VWAP)
        data['vwap_cross_above'] = (data['close'] > data['vwap']) & (data['close'].shift(1) <= data['vwap'].shift(1))
        data['vwap_cross_below'] = (data['close'] < data['vwap']) & (data['close'].shift(1) >= data['vwap'].shift(1))
        
        # VWAP reversal strength (magnitude of distance change)
        data['vwap_distance_change'] = data['vwap_distance'].diff()
        data['vwap_reversal_strength'] = abs(data['vwap_distance_change'])
        
        # Convergence indicators
        data['vwap_convergence'] = abs(data['vwap_distance']) < 0.5  # Within 0.5% of VWAP
        data['vwap_convergence_strong'] = abs(data['vwap_distance']) < 0.2  # Within 0.2% of VWAP
        
        return data
    
    def get_vwap_reversal_signal(self, df: pd.DataFrame) -> float:
        """Calculate VWAP reversal signal strength for a stock."""
        if len(df) < 2:
            return 0.0
        
        # Get latest data
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate reversal signal based on:
        # 1. Current VWAP distance magnitude
        # 2. Recent VWAP cross
        # 3. Change in VWAP distance
        
        signal = 0.0
        
        # Base signal from current VWAP distance
        signal += abs(latest['vwap_distance'])
        
        # Bonus for recent VWAP cross
        if latest['vwap_cross_above'] or latest['vwap_cross_below']:
            signal *= 1.5
        
        # Bonus for strong VWAP distance change
        if abs(latest['vwap_distance_change']) > 1.0:  # More than 1% change
            signal *= 1.3
        
        return signal
    
    def download_universe_data(self, universe: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download 1 year of data for the entire universe."""
        st.info(f"📥 Downloading {len(universe)} stocks data from {start_date} to {end_date}...")
        
        all_data = {}
        
        with st.spinner("Downloading data..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(self._download_single_stock, symbol, start_date, end_date): symbol 
                    for symbol in universe
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(universe)):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            all_data[symbol] = data
                            st.success(f"✅ Downloaded {symbol}")
                        else:
                            st.warning(f"⚠️ No data for {symbol}")
                    except Exception as e:
                        st.error(f"❌ Error downloading {symbol}: {str(e)}")
                        logger.error(f"Error downloading {symbol}: {e}")
        
        st.success(f"📊 Downloaded data for {len(all_data)} stocks")
        return all_data
    
    def _download_single_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download data for a single stock."""
        try:
            data = self.data_handler.get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                # Calculate VWAP indicators
                data = self.calculate_vwap_indicators(data)
                return data
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
        
        return None
    
    def select_daily_stocks(self, all_data: Dict[str, pd.DataFrame], date: str, top_k: int = 5) -> List[Dict]:
        """Select top K stocks based on VWAP reversal signals for a given date."""
        daily_signals = []
        
        for symbol, data in all_data.items():
            # Filter data for the specific date
            date_data = data[data.index.date == pd.to_datetime(date).date()]
            
            if len(date_data) < 10:  # Need minimum data points
                continue
            
            # Calculate VWAP reversal signal
            signal_strength = self.get_vwap_reversal_signal(date_data)
            
            if signal_strength > 0:
                latest = date_data.iloc[-1]
                daily_signals.append({
                    'symbol': symbol,
                    'signal_strength': signal_strength,
                    'vwap_distance': latest['vwap_distance'],
                    'above_vwap': latest['above_vwap'],
                    'close': latest['close'],
                    'vwap': latest['vwap'],
                    'volume': latest['volume']
                })
        
        # Sort by signal strength and select top K
        daily_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        return daily_signals[:top_k]
    
    def simulate_trade(self, symbol: str, position: str, entry_price: float, 
                      entry_vwap: float, entry_date: str, data: pd.DataFrame) -> Dict:
        """Simulate a trade from entry to exit."""
        # Find entry index in data
        entry_idx = data[data.index.date == pd.to_datetime(entry_date).date()].index[-1]
        entry_loc = data.index.get_loc(entry_idx)
        
        # Look for exit conditions in subsequent data
        exit_price = entry_price
        exit_date = entry_date
        exit_reason = "End of data"
        pnl = 0.0
        
        for i in range(entry_loc + 1, len(data)):
            current_data = data.iloc[i]
            current_price = current_data['close']
            current_vwap = current_data['vwap']
            current_date = current_data.name.strftime('%Y-%m-%d')
            
            # Check exit conditions
            if position == 'LONG':
                # Exit long when price converges to VWAP (within 0.5%)
                if abs((current_price - current_vwap) / current_vwap) < 0.005:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Converged to VWAP"
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    break
                # Stop loss if price moves further away from VWAP
                elif (current_price - current_vwap) / current_vwap < -0.02:  # 2% below VWAP
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Stop loss - moved away from VWAP"
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    break
            else:  # SHORT
                # Exit short when price converges to VWAP (within 0.5%)
                if abs((current_price - current_vwap) / current_vwap) < 0.005:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Converged to VWAP"
                    pnl = ((entry_price - exit_price) / entry_price) * 100
                    break
                # Stop loss if price moves further away from VWAP
                elif (current_price - current_vwap) / current_vwap > 0.02:  # 2% above VWAP
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Stop loss - moved away from VWAP"
                    pnl = ((entry_price - exit_price) / entry_price) * 100
                    break
        
        return {
            'symbol': symbol,
            'position': position,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_vwap': entry_vwap,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl,
            'hold_days': (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
        }
    
    def run_backtest(self, universe: List[str], start_date: str, end_date: str, 
                    top_k: int = 5, initial_capital: float = 100000) -> Dict[str, Any]:
        """Run the VWAP reversal backtest."""
        st.info("🚀 Starting VWAP Reversal Backtest...")
        
        # Download all data
        all_data = self.download_universe_data(universe, start_date, end_date)
        if not all_data:
            st.error("❌ No data available for backtest")
            return {}
        
        # Generate trading dates
        trading_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = [d.strftime('%Y-%m-%d') for d in trading_dates 
                        if d.weekday() < 5]  # Weekdays only
        
        st.info(f"📅 Running backtest for {len(trading_dates)} trading days")
        
        # Initialize tracking variables
        current_capital = initial_capital
        equity_curve = []
        all_trades = []
        daily_positions = []
        
        with st.spinner("Running backtest..."):
            for date in tqdm(trading_dates):
                # Select top stocks for the day
                daily_stocks = self.select_daily_stocks(all_data, date, top_k)
                
                if not daily_stocks:
                    continue
                
                # Simulate trades for selected stocks
                daily_trades = []
                for stock in daily_stocks:
                    symbol = stock['symbol']
                    vwap_distance = stock['vwap_distance']
                    entry_price = stock['close']
                    entry_vwap = stock['vwap']
                    
                    # Determine position based on VWAP position
                    if vwap_distance < 0:  # Below VWAP -> LONG
                        position = 'LONG'
                    else:  # Above VWAP -> SHORT
                        position = 'SHORT'
                    
                    # Simulate trade
                    trade = self.simulate_trade(symbol, position, entry_price, 
                                              entry_vwap, date, all_data[symbol])
                    
                    if trade:
                        daily_trades.append(trade)
                        all_trades.append(trade)
                
                # Update capital and track daily positions
                daily_pnl = sum(trade['pnl_pct'] for trade in daily_trades)
                current_capital *= (1 + daily_pnl / 100)
                
                equity_curve.append({
                    'date': date,
                    'capital': current_capital,
                    'daily_pnl': daily_pnl,
                    'num_trades': len(daily_trades)
                })
                
                daily_positions.append({
                    'date': date,
                    'stocks': daily_stocks,
                    'trades': daily_trades,
                    'capital': current_capital
                })
        
        # Calculate performance metrics
        performance = self._calculate_performance(equity_curve, all_trades, initial_capital)
        
        # Store results
        self.results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_return': performance['total_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'win_rate': performance['win_rate'],
            'total_trades': len(all_trades),
            'equity_curve': equity_curve,
            'trades': all_trades,
            'daily_positions': daily_positions,
            'performance': performance
        }
        
        st.success("✅ Backtest completed!")
        return self.results
    
    def _calculate_performance(self, equity_curve: List[Dict], trades: List[Dict], 
                             initial_capital: float) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not equity_curve:
            return {}
        
        # Basic metrics
        final_capital = equity_curve[-1]['capital']
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i]['capital'] - equity_curve[i-1]['capital']) / equity_curve[i-1]['capital']
            daily_returns.append(daily_return)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        
        # Maximum drawdown
        peak = initial_capital
        max_drawdown = 0
        for point in equity_curve:
            if point['capital'] > peak:
                peak = point['capital']
            drawdown = (peak - point['capital']) / peak
            max_drawdown = max(max_drawdown, drawdown)
        max_drawdown *= 100
        
        # Win rate
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_pnl': np.mean([t['pnl_pct'] for t in trades]) if trades else 0,
            'num_trades': len(trades)
        } 