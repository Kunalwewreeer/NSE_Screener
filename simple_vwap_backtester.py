"""
Simple VWAP Reversal Strategy Backtester
Implements the core VWAP reversal strategy with basic visualization.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings

# Import existing modules
from core.data_handler import DataHandler
from utils.logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

class SimpleVWAPBacktester:
    """Simple VWAP reversal strategy backtester."""
    
    def __init__(self):
        self.data_handler = DataHandler()
        self.results = {}
    
    def calculate_vwap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators for intraday data."""
        data = df.copy()
        
        # For intraday data, calculate VWAP with daily resets
        # Group by date and calculate VWAP for each day
        data['date_only'] = data.index.date
        
        # Calculate VWAP for each day separately
        daily_vwap = data.groupby('date_only').apply(
            lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
        )
        
        # Flatten the result and assign back to the original dataframe
        data['vwap'] = daily_vwap.values
        
        # VWAP distance and position
        data['vwap_distance'] = ((data['close'] - data['vwap']) / data['vwap']) * 100
        data['above_vwap'] = data['close'] > data['vwap']
        data['below_vwap'] = data['close'] < data['vwap']
        
        # VWAP reversal signals
        data['vwap_cross_above'] = (data['close'] > data['vwap']) & (data['close'].shift(1) <= data['vwap'].shift(1))
        data['vwap_cross_below'] = (data['close'] < data['vwap']) & (data['close'].shift(1) >= data['vwap'].shift(1))
        
        # VWAP distance change
        data['vwap_distance_change'] = data['vwap_distance'].diff()
        
        # Remove the temporary date column
        data = data.drop('date_only', axis=1)
        
        return data
    
    def get_vwap_reversal_signal(self, df: pd.DataFrame) -> float:
        """Calculate VWAP reversal signal strength."""
        if len(df) < 2:
            return 0.0
        
        latest = df.iloc[-1]
        
        # Base signal from current VWAP distance
        signal = abs(latest['vwap_distance'])
        
        # Bonus for recent VWAP cross
        if latest['vwap_cross_above'] or latest['vwap_cross_below']:
            signal *= 1.5
        
        # Bonus for strong VWAP distance change
        if abs(latest['vwap_distance_change']) > 1.0:
            signal *= 1.3
        
        return signal
    
    def download_universe_data(self, universe: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download data for the universe."""
        st.info(f"📥 Downloading {len(universe)} stocks data...")
        
        all_data = {}
        
        try:
            # Download all data at once using the data handler
            data_dict = self.data_handler.get_historical_data(universe, start_date, end_date, interval="minute")
            
            if isinstance(data_dict, dict):
                # Data handler returned a dict of DataFrames
                for symbol, data in data_dict.items():
                    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                        data = self.calculate_vwap_indicators(data)
                        all_data[symbol] = data
                        st.success(f"✅ Downloaded {symbol}")
                    else:
                        st.warning(f"⚠️ No data for {symbol}")
            elif isinstance(data_dict, pd.DataFrame):
                # Data handler returned a single DataFrame (shouldn't happen with multiple symbols)
                st.warning("⚠️ Unexpected single DataFrame returned")
            else:
                st.warning("⚠️ Unexpected data format returned")
                
        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            # Fallback: try downloading symbols one by one
            st.info("🔄 Trying individual symbol download...")
            for symbol in universe:
                try:
                    data = self.data_handler.get_historical_data(symbol, start_date, end_date, interval="minute")
                    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                        data = self.calculate_vwap_indicators(data)
                        all_data[symbol] = data
                        st.success(f"✅ Downloaded {symbol}")
                    else:
                        st.warning(f"⚠️ No data for {symbol}")
                except Exception as e:
                    st.error(f"❌ Error downloading {symbol}: {str(e)}")
        
        st.success(f"📊 Downloaded data for {len(all_data)} stocks")
        return all_data
    
    def select_daily_stocks(self, all_data: Dict[str, pd.DataFrame], date: str, top_k: int = 5) -> List[Dict]:
        """Select top K stocks based on VWAP reversal signals for intraday data."""
        daily_signals = []
        
        for symbol, data in all_data.items():
            # Filter data for the specific date
            date_data = data[data.index.date == pd.to_datetime(date).date()]
            
            if len(date_data) < 5:  # Need at least 5 minutes of data
                continue
            
            # For intraday, use the last 30 minutes of data for signal calculation
            # This gives us enough data to calculate VWAP and signals
            recent_data = date_data.tail(min(30, len(date_data)))
            
            # Calculate VWAP reversal signal
            signal_strength = self.get_vwap_reversal_signal(recent_data)
            
            if signal_strength > 0:
                latest = recent_data.iloc[-1]
                daily_signals.append({
                    'symbol': symbol,
                    'signal_strength': signal_strength,
                    'vwap_distance': latest['vwap_distance'],
                    'above_vwap': latest['above_vwap'],
                    'close': latest['close'],
                    'vwap': latest['vwap']
                })
        
        # Sort by signal strength and select top K
        daily_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        return daily_signals[:top_k]
    
    def simulate_trade(self, symbol: str, position: str, entry_price: float, 
                      entry_vwap: float, entry_date: str, data: pd.DataFrame) -> Dict:
        """Simulate a trade from entry to exit."""
        # Find entry index
        entry_idx = data[data.index.date == pd.to_datetime(entry_date).date()].index[-1]
        entry_loc = data.index.get_loc(entry_idx)
        
        # Look for exit conditions
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
                # Exit when price converges to VWAP (within 0.5%)
                if abs((current_price - current_vwap) / current_vwap) < 0.005:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Converged to VWAP"
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    break
                # Stop loss if price moves further away
                elif (current_price - current_vwap) / current_vwap < -0.02:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Stop loss"
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    break
            else:  # SHORT
                # Exit when price converges to VWAP
                if abs((current_price - current_vwap) / current_vwap) < 0.005:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Converged to VWAP"
                    pnl = ((entry_price - exit_price) / entry_price) * 100
                    break
                # Stop loss if price moves further away
                elif (current_price - current_vwap) / current_vwap > 0.02:
                    exit_price = current_price
                    exit_date = current_date
                    exit_reason = "Stop loss"
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
            'pnl_pct': pnl
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
        
        with st.spinner("Running backtest..."):
            for date in trading_dates:
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
                
                # Update capital
                daily_pnl = sum(trade['pnl_pct'] for trade in daily_trades)
                current_capital *= (1 + daily_pnl / 100)
                
                equity_curve.append({
                    'date': date,
                    'capital': current_capital,
                    'daily_pnl': daily_pnl,
                    'num_trades': len(daily_trades)
                })
        
        # Calculate performance metrics
        performance = self._calculate_performance(equity_curve, all_trades, initial_capital)
        
        # Store results
        self.results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_return': performance.get('total_return', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown': performance.get('max_drawdown', 0),
            'win_rate': performance.get('win_rate', 0),
            'total_trades': len(all_trades),
            'equity_curve': equity_curve,
            'trades': all_trades,
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
        
        # Sharpe ratio
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
    
    def plot_results(self):
        """Plot basic backtest results."""
        if not self.results:
            st.warning("No backtest results to plot")
            return
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{self.results['total_return']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{self.results['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{self.results['max_drawdown']:.2f}%")
        with col4:
            st.metric("Win Rate", f"{self.results['win_rate']:.1f}%")
        
        # Equity curve
        if self.results.get('equity_curve') and len(self.results['equity_curve']) > 0:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            
            # Ensure date column exists and is properly formatted
            if 'date' in equity_df.columns:
                equity_df['date'] = pd.to_datetime(equity_df['date'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['capital'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (₹)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Equity curve data missing date column")
        else:
            st.warning("No equity curve data available")
        
        # Trade summary
        st.subheader("Trade Summary")
        if self.results.get('trades') and len(self.results['trades']) > 0:
            trades_df = pd.DataFrame(self.results['trades'])
            st.dataframe(trades_df.tail(10), use_container_width=True)
        else:
            st.warning("No trades available")


def create_simple_vwap_dashboard():
    """Create the simple VWAP reversal backtest dashboard."""
    st.set_page_config(page_title="Simple VWAP Backtester", layout="wide")
    
    st.title("🔍 Simple VWAP Reversal Strategy Backtester")
    st.markdown("""
    **Strategy Logic:**
    - Daily selection of top 5 stocks based on VWAP reversal signals
    - Buy stocks below VWAP, sell stocks above VWAP
    - Exit positions when price converges to VWAP (within 0.5%)
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Backtest Configuration")
    
    # Universe selection
    universe_type = st.sidebar.selectbox(
        "Stock Universe",
        ["nifty50", "nifty100", "nifty500"],
        help="Select the stock universe for backtesting"
    )
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days for minute data
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime(start_date),
        max_value=datetime.now()
    ).strftime('%Y-%m-%d')
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    ).strftime('%Y-%m-%d')
    
    # Strategy parameters
    top_k = st.sidebar.slider(
        "Top K Stocks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top stocks to select daily"
    )
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000,
        help="Initial capital for backtesting"
    )
    
    # Main content
    if st.button("🚀 Run VWAP Reversal Backtest", type="primary"):
        with st.spinner("Initializing backtester..."):
            # Initialize components
            backtester = SimpleVWAPBacktester()
            
            # Get universe
            universe = backtester.data_handler.get_stocks_by_universe(universe_type)
            st.info(f"📊 Universe: {len(universe)} stocks from {universe_type}")
            
            # Run backtest
            results = backtester.run_backtest(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                initial_capital=initial_capital
            )
            
            if results:
                st.success("✅ Backtest completed successfully!")
                
                # Display results
                backtester.plot_results()
            else:
                st.error("❌ Backtest failed. Please check the configuration and try again.")


if __name__ == "__main__":
    create_simple_vwap_dashboard() 