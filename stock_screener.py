"""
Advanced Stock Screener for Intraday Trading
Screens stocks based on technical indicators and price performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from typing import Dict, List, Tuple, Optional
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Import our existing modules
from core.data_handler import DataHandler
from utils.logger import get_logger
from cache_manager import add_cache_management_to_sidebar, add_data_completeness_settings, validate_screening_data

logger = get_logger(__name__)

class StockScreener:
    """Advanced stock screener for intraday trading opportunities."""
    
    def __init__(self):
        self.data_handler = DataHandler()
        
    def get_universe_stocks(self, universe_type: str = "nifty50") -> List[str]:
        """Get stock universe based on type."""
        return self.data_handler.get_stocks_by_universe(universe_type)
    
    def calculate_screening_indicators(self, df: pd.DataFrame, is_early_morning: bool = False) -> pd.DataFrame:
        """Calculate comprehensive indicators for screening."""
        data = df.copy()
        
        try:
            # Price action metrics
            data['day_open'] = data['open'].iloc[0]  # First open of the day
            data['current_price'] = data['close']
            data['pct_change_from_open'] = ((data['close'] - data['day_open']) / data['day_open']) * 100
            data['high_of_day'] = data['high'].expanding().max()
            data['low_of_day'] = data['low'].expanding().min()
            data['range_pct'] = ((data['high_of_day'] - data['low_of_day']) / data['day_open']) * 100
            
            # Volume metrics
            data['avg_volume'] = data['volume'].expanding().mean()
            data['volume_spike'] = data['volume'] / data['avg_volume']
            
            if not is_early_morning:
                # Full technical analysis (after 9:35 AM)
                data['volume_sma_20'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma_20']
                
                # Moving averages
                data['sma_5'] = data['close'].rolling(5).mean()
                data['sma_10'] = data['close'].rolling(10).mean()
                data['sma_20'] = data['close'].rolling(20).mean()
                data['ema_5'] = data['close'].ewm(span=5).mean()
                data['ema_10'] = data['close'].ewm(span=10).mean()
            else:
                # Early morning mode - simplified metrics
                data['volume_ratio'] = data['volume_spike']  # Use spike as ratio
                # Set minimal moving averages
                data['sma_5'] = data['close'].rolling(min(5, len(data))).mean()
                data['sma_10'] = data['close'].rolling(min(10, len(data))).mean() 
                data['sma_20'] = data['close'].rolling(min(20, len(data))).mean()
                data['ema_5'] = data['close'].ewm(span=min(5, len(data))).mean()
                data['ema_10'] = data['close'].ewm(span=min(10, len(data))).mean()
            
            # Price vs MA signals
            data['above_sma5'] = data['close'] > data['sma_5']
            data['above_sma10'] = data['close'] > data['sma_10']
            data['above_sma20'] = data['close'] > data['sma_20']
            data['ma_alignment'] = (data['sma_5'] > data['sma_10']).astype(int) + (data['sma_10'] > data['sma_20']).astype(int)
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            data['rsi_momentum'] = data['rsi'] > 50
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            data['macd_bullish'] = data['macd'] > data['macd_signal']
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # ATR
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift())
            low_close_prev = abs(data['low'] - data['close'].shift())
            
            # Use numpy maximum to avoid DataFrame ambiguity
            true_range = np.maximum.reduce([
                high_low.values,
                high_close_prev.values,
                low_close_prev.values
            ])
            
            true_range_series = pd.Series(true_range, index=data.index)
            data['atr'] = true_range_series.rolling(14).mean()
            data['atr_pct'] = (data['atr'] / data['close']) * 100
            
            # VWAP
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            data['above_vwap'] = data['close'] > data['vwap']
            data['vwap_distance'] = ((data['close'] - data['vwap']) / data['vwap']) * 100
            
            # Recent volume ratio calculation (for strategy filter)
            # Calculate recent volume activity compared to average volume per period
            # This gives us a ratio where >1 means recent volume is higher than average
            
            # Calculate average volume per period up to current point
            data['avg_volume_per_period'] = data['volume'].expanding().mean()
            
            # Calculate recent volume averages for different windows
            data['recent_avg_volume_5'] = data['volume'].rolling(5).mean()
            data['recent_avg_volume_10'] = data['volume'].rolling(10).mean()
            data['recent_avg_volume_15'] = data['volume'].rolling(15).mean()
            data['recent_avg_volume_20'] = data['volume'].rolling(20).mean()
            data['recent_avg_volume_30'] = data['volume'].rolling(30).mean()
            
            # Calculate ratios: recent_avg / overall_avg (>1 = more active, <1 = less active)
            data['recent_volume_ratio_5'] = data['recent_avg_volume_5'] / data['avg_volume_per_period']
            data['recent_volume_ratio_10'] = data['recent_avg_volume_10'] / data['avg_volume_per_period']
            data['recent_volume_ratio_15'] = data['recent_avg_volume_15'] / data['avg_volume_per_period']
            data['recent_volume_ratio_20'] = data['recent_avg_volume_20'] / data['avg_volume_per_period']
            data['recent_volume_ratio_30'] = data['recent_avg_volume_30'] / data['avg_volume_per_period']
            
            # Momentum indicators
            data['momentum_10'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100
            data['momentum_20'] = ((data['close'] - data['close'].shift(20)) / data['close'].shift(20)) * 100
            
            # Recent minute-level returns for ranking (very short-term momentum)
            data['return_1min'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * 100
            data['return_2min'] = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) * 100
            data['return_3min'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3)) * 100
            data['return_5min'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) * 100
            data['return_10min'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100
            data['return_15min'] = ((data['close'] - data['close'].shift(15)) / data['close'].shift(15)) * 100
            data['return_30min'] = ((data['close'] - data['close'].shift(30)) / data['close'].shift(30)) * 100
            
            # Breakout signals
            data['near_high'] = (data['close'] / data['high_of_day']) > 0.95
            data['near_low'] = (data['close'] / data['low_of_day']) < 1.05
            
            # Bearish signals
            data['below_sma5'] = data['close'] < data['sma_5']
            data['below_sma10'] = data['close'] < data['sma_10']
            data['below_sma20'] = data['close'] < data['sma_20']
            data['below_vwap'] = data['close'] < data['vwap']
            data['rsi_oversold'] = data['rsi'] < 30
            data['rsi_overbought'] = data['rsi'] > 70
            data['macd_bearish'] = data['macd'] < data['macd_signal']
            data['ma_bearish_alignment'] = (data['sma_5'] < data['sma_10']).astype(int) + (data['sma_10'] < data['sma_20']).astype(int)
            
            # Composite scoring
            data['bullish_score'] = (
                data['above_sma5'].astype(int) +
                data['above_sma10'].astype(int) +
                data['above_sma20'].astype(int) +
                data['above_vwap'].astype(int) +
                data['rsi_momentum'].astype(int) +
                data['macd_bullish'].astype(int) +
                (data['volume_ratio'] > 1.2).astype(int) +
                data['ma_alignment']
            )
            
            data['bearish_score'] = (
                data['below_sma5'].astype(int) +
                data['below_sma10'].astype(int) +
                data['below_sma20'].astype(int) +
                data['below_vwap'].astype(int) +
                data['rsi_oversold'].astype(int) +
                data['macd_bearish'].astype(int) +
                (data['volume_ratio'] > 1.2).astype(int) +  # High volume on bearish moves
                data['ma_bearish_alignment']
            )
            
            # Interest score - combines both bullish and bearish signals
            data['interest_score'] = data['bullish_score'] + data['bearish_score']
            data['signal_direction'] = np.where(
                data['bullish_score'] > data['bearish_score'], 'BULLISH',
                np.where(data['bearish_score'] > data['bullish_score'], 'BEARISH', 'NEUTRAL')
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating screening indicators: {e}")
            return data
    
    def fetch_and_analyze_stock(self, symbol: str, start_date: str, end_date: str, 
                               cutoff_time: Optional[str] = None, request_delay: float = 0.1) -> Optional[Dict]:
        """Fetch and analyze a single stock with proper cutoff time handling."""
        try:
            # Add configurable delay to avoid API rate limits
            time.sleep(request_delay)
            # Fetch data
            stock_data = self.data_handler.get_historical_data(
                symbols=[symbol],
                from_date=start_date,
                to_date=end_date,
                interval="minute",
                refresh_cache=False
            )
            
            if stock_data is None or len(stock_data) == 0 or symbol not in stock_data:
                return None
                
            df = stock_data[symbol]
            if df.empty or len(df) < 20:
                return None
            
            # Filter to only the analysis date if we have a date range
            if start_date != end_date:
                # If we fetched multiple days, filter to only the end_date (analysis date)
                analysis_date_str = end_date
                df = df[df.index.date == pd.to_datetime(analysis_date_str).date()]
                
                if df.empty or len(df) < 10:
                    logger.warning(f"No data for {symbol} on analysis date {analysis_date_str}")
                    return None
            
            # Check if it's early morning analysis
            is_early_morning = False
            if cutoff_time:
                cutoff_hour = int(cutoff_time.split(':')[0])
                cutoff_minute = int(cutoff_time.split(':')[1])
                is_early_morning = cutoff_hour == 9 and cutoff_minute < 35
            
            # Calculate indicators for full day data first
            df_full = self.calculate_screening_indicators(df.copy(), is_early_morning=False)
            
            # Apply cutoff time if specified
            if cutoff_time:
                try:
                    # Parse cutoff time more precisely - use end_date (analysis date) not start_date
                    cutoff_datetime = pd.to_datetime(f"{end_date} {cutoff_time}")
                    
                    # Handle timezone-aware data
                    if df.index.tz is not None:
                        # If data has timezone, make cutoff timezone-aware to match
                        cutoff_datetime = cutoff_datetime.tz_localize(df.index.tz)
                    
                    # Filter data up to cutoff time
                    df_cutoff = df[df.index <= cutoff_datetime].copy()
                    
                    if df_cutoff.empty:
                        # If no data before cutoff, take first 25% of day
                        cutoff_idx = max(1, len(df) // 4)
                        df_cutoff = df.iloc[:cutoff_idx].copy()
                    elif len(df_cutoff) < 10:
                        # If too little data, extend to at least 10 records
                        cutoff_idx = min(20, len(df))
                        df_cutoff = df.iloc[:cutoff_idx].copy()
                    
                    # Recalculate indicators with cutoff data only
                    df_cutoff = self.calculate_screening_indicators(df_cutoff, is_early_morning=is_early_morning)
                    
                except Exception as e:
                    logger.warning(f"Error parsing cutoff time for {symbol}: {e}")
                    # Fallback to first 30% of day
                    cutoff_idx = max(10, len(df) * 30 // 100)
                    df_cutoff = df.iloc[:cutoff_idx].copy()
                    df_cutoff = self.calculate_screening_indicators(df_cutoff)
            else:
                df_cutoff = df_full.copy()
            
            if df_cutoff.empty:
                return None
            
            # Get metrics at cutoff time
            latest_cutoff = df_cutoff.iloc[-1]
            cutoff_time_actual = df_cutoff.index[-1]
            
            # Get end-of-day metrics
            latest_eod = df_full.iloc[-1]
            eod_time_actual = df_full.index[-1]
            
            # Calculate screening metrics - CUTOFF TIME
            cutoff_metrics = {
                'symbol': symbol,
                'cutoff_time': cutoff_time_actual.strftime('%H:%M:%S'),
                'cutoff_price': latest_cutoff['close'],
                'day_open': latest_cutoff['day_open'],
                'cutoff_pct_change': latest_cutoff['pct_change_from_open'],
                'cutoff_range_pct': latest_cutoff['range_pct'],
                'cutoff_volume_ratio': latest_cutoff.get('volume_ratio', 1.0),
                'cutoff_volume_spike': latest_cutoff.get('volume_spike', 1.0),
                'cutoff_rsi': latest_cutoff.get('rsi', 50),
                'cutoff_macd': latest_cutoff.get('macd', 0),
                'cutoff_bb_position': latest_cutoff.get('bb_position', 0.5),
                'cutoff_vwap_distance': latest_cutoff.get('vwap_distance', 0),
                'cutoff_momentum_10': latest_cutoff.get('momentum_10', 0),
                'cutoff_bullish_score': latest_cutoff.get('bullish_score', 0),
                'cutoff_bearish_score': latest_cutoff.get('bearish_score', 0),
                'cutoff_interest_score': latest_cutoff.get('interest_score', 0),
                'cutoff_signal_direction': latest_cutoff.get('signal_direction', 'NEUTRAL'),
                'cutoff_above_vwap': latest_cutoff.get('above_vwap', False),
                'cutoff_ma_alignment': latest_cutoff.get('ma_alignment', 0),
                # Recent volume ratios for strategy filter
                'recent_volume_ratio_5': latest_cutoff.get('recent_volume_ratio_5', 1.0),
                'recent_volume_ratio_10': latest_cutoff.get('recent_volume_ratio_10', 1.0),
                'recent_volume_ratio_15': latest_cutoff.get('recent_volume_ratio_15', 1.0),
                'recent_volume_ratio_20': latest_cutoff.get('recent_volume_ratio_20', 1.0),
                'recent_volume_ratio_30': latest_cutoff.get('recent_volume_ratio_30', 1.0),
                # Recent minute-level returns for ranking
                'return_1min': latest_cutoff.get('return_1min', 0.0),
                'return_2min': latest_cutoff.get('return_2min', 0.0),
                'return_3min': latest_cutoff.get('return_3min', 0.0),
                'return_5min': latest_cutoff.get('return_5min', 0.0),
                'return_10min': latest_cutoff.get('return_10min', 0.0),
                'return_15min': latest_cutoff.get('return_15min', 0.0),
                'return_30min': latest_cutoff.get('return_30min', 0.0),
            }
            
            # Calculate screening metrics - END OF DAY
            eod_metrics = {
                'eod_time': eod_time_actual.strftime('%H:%M:%S'),
                'eod_price': latest_eod['close'],
                'eod_pct_change': latest_eod['pct_change_from_open'],
                'eod_range_pct': latest_eod['range_pct'],
                'eod_volume_ratio': latest_eod.get('volume_ratio', 1.0),
                'eod_volume_spike': latest_eod.get('volume_spike', 1.0),
                'eod_rsi': latest_eod.get('rsi', 50),
                'eod_macd': latest_eod.get('macd', 0),
                'eod_bb_position': latest_eod.get('bb_position', 0.5),
                'eod_vwap_distance': latest_eod.get('vwap_distance', 0),
                'eod_momentum_10': latest_eod.get('momentum_10', 0),
                'eod_bullish_score': latest_eod.get('bullish_score', 0),
                'eod_bearish_score': latest_eod.get('bearish_score', 0),
                'eod_interest_score': latest_eod.get('interest_score', 0),
                'eod_signal_direction': latest_eod.get('signal_direction', 'NEUTRAL'),
                'eod_above_vwap': latest_eod.get('above_vwap', False),
                'eod_ma_alignment': latest_eod.get('ma_alignment', 0),
            }
            
            # Performance from cutoff to EOD
            performance_metrics = {
                'cutoff_to_eod_change': ((latest_eod['close'] - latest_cutoff['close']) / latest_cutoff['close']) * 100,
                'cutoff_to_eod_high': ((df_full.loc[df_cutoff.index[-1]:, 'high'].max() - latest_cutoff['close']) / latest_cutoff['close']) * 100,
                'cutoff_to_eod_low': ((df_full.loc[df_cutoff.index[-1]:, 'low'].min() - latest_cutoff['close']) / latest_cutoff['close']) * 100,
                'total_records_cutoff': len(df_cutoff),
                'total_records_full': len(df_full),
            }
            
            # Combine all metrics
            all_metrics = {**cutoff_metrics, **eod_metrics, **performance_metrics}
            
            # Add primary screening fields for backward compatibility
            all_metrics.update({
                'current_price': latest_cutoff['close'],  # Use cutoff price for screening
                'pct_change_from_open': latest_cutoff['pct_change_from_open'],  # Use cutoff change
                'range_pct': latest_cutoff['range_pct'],
                'volume_ratio': latest_cutoff.get('volume_ratio', 1.0),
                'volume_spike': latest_cutoff.get('volume_spike', 1.0),
                'rsi': latest_cutoff.get('rsi', 50),
                'macd': latest_cutoff.get('macd', 0),
                'macd_histogram': latest_cutoff.get('macd_histogram', 0),
                'bb_position': latest_cutoff.get('bb_position', 0.5),
                'atr_pct': latest_cutoff.get('atr_pct', 0),
                'vwap_distance': latest_cutoff.get('vwap_distance', 0),
                'momentum_10': latest_cutoff.get('momentum_10', 0),
                'momentum_20': latest_cutoff.get('momentum_20', 0),
                'bullish_score': latest_cutoff.get('bullish_score', 0),
                'bearish_score': latest_cutoff.get('bearish_score', 0),
                'interest_score': latest_cutoff.get('interest_score', 0),
                'signal_direction': latest_cutoff.get('signal_direction', 'NEUTRAL'),
                'above_vwap': latest_cutoff.get('above_vwap', False),
                'above_sma5': latest_cutoff.get('above_sma5', False),
                'above_sma10': latest_cutoff.get('above_sma10', False),
                'above_sma20': latest_cutoff.get('above_sma20', False),
                'near_high': latest_cutoff.get('near_high', False),
                'near_low': latest_cutoff.get('near_low', False),
                'ma_alignment': latest_cutoff.get('ma_alignment', 0),
                'data': df_full  # Store full data for detailed analysis
            })
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def screen_stocks(self, universe: List[str], start_date: str, end_date: str,
                     cutoff_time: Optional[str] = None, max_workers: int = 3, 
                     request_delay: float = 0.1, refresh_cache: bool = False,
                     min_data_points: int = 30, allow_incomplete_data: bool = True) -> pd.DataFrame:
        """Screen multiple stocks in parallel."""
        
        st.info(f"📊 Screening {len(universe)} stocks...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_and_analyze_stock, symbol, start_date, end_date, cutoff_time, request_delay): symbol
                for symbol in universe
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    progress = completed / len(universe)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {completed}/{len(universe)} stocks - Last: {symbol}")
                except Exception as e:
                    if "Too many requests" in str(e) or "rate limit" in str(e).lower():
                        logger.warning(f"Rate limit hit for {symbol}, adding delay...")
                        time.sleep(1)  # Wait 1 second on rate limit
                        st.warning(f"⚠️ Rate limit reached. Slowing down requests...")
                    else:
                        logger.error(f"Error processing {symbol}: {e}")
                    completed += 1
        
        progress_bar.empty()
        status_text.empty()
        
        if not results:
            st.warning("No stocks found matching criteria")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Map cutoff metrics to main screening columns for filtering
        df['pct_change_from_open'] = df['cutoff_pct_change']
        df['bullish_score'] = df['cutoff_bullish_score']
        df['bearish_score'] = df['cutoff_bearish_score']
        df['interest_score'] = df['cutoff_interest_score']
        df['volume_ratio'] = df['cutoff_volume_ratio']
        df['range_pct'] = df['cutoff_range_pct']
        df['vwap_distance'] = df['cutoff_vwap_distance']
        df['rsi'] = df['cutoff_rsi']
        df['momentum_10'] = df['cutoff_momentum_10']
        df['atr_pct'] = df.get('cutoff_atr_pct', 0)  # In case this doesn't exist
        
        # Sort by multiple criteria (you can customize this)
        df = df.sort_values([
            'pct_change_from_open',
            'bullish_score',
            'volume_ratio',
            'range_pct'
        ], ascending=[False, False, False, False])
        
        st.success(f"✅ Successfully screened {len(results)} stocks")
        return df
    
    def get_top_stocks(self, screened_df: pd.DataFrame, criteria: str, top_k: int) -> pd.DataFrame:
        """Get top K stocks based on specified criteria."""
        
        criteria_mapping = {
            'pct_change': 'pct_change_from_open',
            'volume_ratio': 'volume_ratio',
            'bullish_score': 'bullish_score',
            'bearish_score': 'bearish_score',
            'interest_score': 'interest_score',
            'range_pct': 'range_pct',
            'rsi_momentum': 'rsi',
            'vwap_distance': 'vwap_distance',
            'momentum_10': 'momentum_10',
            'atr_pct': 'atr_pct',
            # Recent minute-level returns for ranking
            'return_1min': 'return_1min',
            'return_2min': 'return_2min',
            'return_3min': 'return_3min',
            'return_5min': 'return_5min',
            'return_10min': 'return_10min',
            'return_15min': 'return_15min',
            'return_30min': 'return_30min'
        }
        
        sort_column = criteria_mapping.get(criteria, 'pct_change_from_open')
        
        # Handle special cases
        if criteria == 'pct_change':
            # For percentage change, use absolute values to capture both bullish and bearish moves
            screened_df_copy = screened_df.copy()
            screened_df_copy['abs_pct_change'] = abs(screened_df_copy['pct_change_from_open'])
            top_stocks = screened_df_copy.nlargest(top_k, 'abs_pct_change')
        elif criteria == 'rsi_momentum':
            # For RSI, we want values between 50-80 (bullish momentum)
            filtered_df = screened_df[(screened_df['rsi'] >= 50) & (screened_df['rsi'] <= 80)]
            top_stocks = filtered_df.nlargest(top_k, 'rsi')
        elif criteria == 'vwap_distance':
            # For VWAP distance, we want absolute values to capture both above and below VWAP
            screened_df_copy = screened_df.copy()
            screened_df_copy['abs_vwap_distance'] = abs(screened_df_copy['vwap_distance'])
            top_stocks = screened_df_copy.nlargest(top_k, 'abs_vwap_distance')
        elif criteria.startswith('return_'):
            # For recent minute returns, rank by absolute values to capture strongest moves in either direction
            screened_df_copy = screened_df.copy()
            abs_column_name = f'abs_{sort_column}'
            screened_df_copy[abs_column_name] = abs(screened_df_copy[sort_column])
            top_stocks = screened_df_copy.nlargest(top_k, abs_column_name)
        else:
            top_stocks = screened_df.nlargest(top_k, sort_column)
        
        return top_stocks
    
    def run_vwap_mean_reversion_backtest(self, screened_df: pd.DataFrame, top_k: int = 5, 
                                       start_date: str = None, end_date: str = None, 
                                       cutoff_time: str = "10:15:00") -> Dict:
        """
        Run VWAP mean reversion backtest:
        - Select top K stocks by absolute VWAP distance
        - Long stocks below VWAP (negative distance)  
        - Short stocks above VWAP (positive distance)
        - Hold till EOD
        """
        try:
            st.info(f"Debug: Starting backtest with {len(screened_df)} stocks, top_k={top_k}")
            
            # Get top stocks by absolute VWAP distance
            top_stocks = self.get_top_stocks(screened_df, 'vwap_distance', top_k)
            
            st.info(f"Debug: Selected {len(top_stocks)} top stocks for backtest")
            
            if top_stocks.empty:
                return {"error": "No stocks found for backtesting"}
            
            backtest_results = []
            total_pnl = 0
            total_trades = 0
            winning_trades = 0
            
            st.info(f"🔄 Running VWAP Mean Reversion Backtest on {len(top_stocks)} stocks...")
            progress_bar = st.progress(0)
            
            for idx, (_, stock) in enumerate(top_stocks.iterrows()):
                symbol = stock['symbol']
                cutoff_vwap_distance = stock.get('cutoff_vwap_distance', 0)
                cutoff_price = stock.get('cutoff_price', 0)
                eod_price = stock.get('eod_price', 0)
                
                st.info(f"Debug: Processing {symbol} - VWAP dist: {cutoff_vwap_distance:.2f}%, cutoff: {cutoff_price}, eod: {eod_price}")
                
                if cutoff_price == 0 or eod_price == 0:
                    st.warning(f"Debug: Skipping {symbol} - missing price data")
                    continue
                
                # Determine position based on VWAP distance
                if cutoff_vwap_distance > 0:
                    # Stock above VWAP -> SHORT
                    position = "SHORT"
                    # For short: profit when price goes down
                    pnl_pct = ((cutoff_price - eod_price) / cutoff_price) * 100
                else:
                    # Stock below VWAP -> LONG  
                    position = "LONG"
                    # For long: profit when price goes up
                    pnl_pct = ((eod_price - cutoff_price) / cutoff_price) * 100
                
                # Calculate trade metrics
                is_winner = pnl_pct > 0
                if is_winner:
                    winning_trades += 1
                
                trade_result = {
                    'symbol': symbol,
                    'position': position,
                    'cutoff_time': stock.get('cutoff_time', cutoff_time),
                    'cutoff_price': cutoff_price,
                    'eod_price': eod_price,
                    'vwap_distance': cutoff_vwap_distance,
                    'pnl_pct': pnl_pct,
                    'pnl_absolute': (pnl_pct / 100) * cutoff_price,
                    'is_winner': is_winner,
                    'trade_reason': f"{'Above' if cutoff_vwap_distance > 0 else 'Below'} VWAP by {abs(cutoff_vwap_distance):.2f}%"
                }
                
                backtest_results.append(trade_result)
                total_pnl += pnl_pct
                total_trades += 1
                
                st.info(f"Debug: {symbol} {position} - P&L: {pnl_pct:.2f}%")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(top_stocks))
            
            progress_bar.empty()
            
            st.info(f"Debug: Completed {total_trades} trades, {winning_trades} winners")
            
            # Calculate summary statistics
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                avg_pnl = total_pnl / total_trades
                
                # Calculate additional metrics
                winning_pnl = sum([t['pnl_pct'] for t in backtest_results if t['is_winner']])
                losing_pnl = sum([t['pnl_pct'] for t in backtest_results if not t['is_winner']])
                
                avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
                avg_loss = losing_pnl / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
                
                profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
                
                summary = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': win_rate,
                    'total_pnl_pct': total_pnl,
                    'avg_pnl_pct': avg_pnl,
                    'avg_win_pct': avg_win,
                    'avg_loss_pct': avg_loss,
                    'profit_factor': profit_factor,
                    'trades': backtest_results
                }
                
                st.info(f"Debug: Backtest summary - Win rate: {win_rate:.1f}%, Total P&L: {total_pnl:.2f}%")
                
                return summary
            else:
                return {"error": "No valid trades executed"}
                
        except Exception as e:
            st.error(f"Debug: Error in VWAP mean reversion backtest: {e}")
            logger.error(f"Error in VWAP mean reversion backtest: {e}")
            return {"error": str(e)}

def create_screening_dashboard():
    """Create the main screening dashboard."""
    
    st.set_page_config(page_title="Stock Screener", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .bullish { color: #28a745; font-weight: bold; }
    .bearish { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
    .cache-refresh { background-color: #ffc107; color: #000; padding: 0.5rem; border-radius: 0.3rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔍 Advanced Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown("### Find the best trading opportunities based on technical indicators and price action")
    
    # Initialize screener
    screener = StockScreener()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Screening Controls")
    
    # Cache Management Section
    st.sidebar.subheader("🗄️ Cache Management")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        refresh_cache = st.button("🔄 Refresh Cache", type="secondary", help="Force refresh all cached data")
    with col2:
        clear_cache = st.button("🗑️ Clear Cache", type="secondary", help="Clear all cached data")
    
    if refresh_cache:
        st.sidebar.success("✅ Cache refresh initiated! Data will be re-downloaded from API.")
        st.session_state['refresh_cache'] = True
    elif clear_cache:
        st.sidebar.success("✅ Cache cleared! Next run will download fresh data.")
        st.session_state['clear_cache'] = True
    
    # Show cache status
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    if os.path.exists(cache_dir):
        cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])
        st.sidebar.info(f"📁 Cache Status: {cache_files} files cached")
    else:
        st.sidebar.info("📁 Cache Status: No cache directory found")
    
    # Universe selection
    universe_type = st.sidebar.selectbox(
        "Stock Universe",
        options=["nifty50", "nifty100", "nifty500"],
        index=0,  # Default to nifty50 to avoid rate limits
        help="Choose the universe of stocks to screen"
    )
    
    if universe_type == "nifty500":
        st.sidebar.warning("⚠️ Nifty 500 may hit API rate limits. Consider using lower concurrent requests and higher delays.")
    
    # Date selection - single date only
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    analysis_date = st.sidebar.date_input(
        "Analysis Date", 
        value=yesterday,
        max_value=today,
        help="Date for intraday analysis (yesterday recommended for complete data)"
    )
    
    if analysis_date == today:
        st.sidebar.info("📅 Today's data may be incomplete. Yesterday is recommended for full analysis.")
    elif analysis_date > today:
        st.sidebar.error("❌ Cannot analyze future dates")
        st.stop()
    
    # Time cutoff with minute precision
    st.sidebar.markdown("**⏰ Analysis Cutoff Time**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cutoff_hour = st.selectbox(
            "Hour", 
            options=list(range(9, 16)), 
            index=1,  # Default to 10 AM (index 1 in range 9-15)
            help="Hour (9 AM to 3 PM)"
        )
    with col2:
        cutoff_minute = st.selectbox(
            "Minute", 
            options=list(range(0, 60)), 
            index=47,  # Default to 47 minutes
            help="Any minute (0-59)"
        )
    
    cutoff_time = datetime.strptime(f"{cutoff_hour:02d}:{cutoff_minute:02d}", "%H:%M").time()
    
    # Display current cutoff time for debugging
    st.sidebar.info(f"🕐 Selected Cutoff Time: **{cutoff_time.strftime('%H:%M')}**")
    st.sidebar.info(f"🔧 Debug - Hour: {cutoff_hour}, Minute: {cutoff_minute}")
    
    # Check if it's early morning (before 9:35)
    is_early_morning = cutoff_hour == 9 and cutoff_minute < 35
    
    if is_early_morning:
        st.sidebar.warning(f"⚠️ Early Morning Mode - Using simplified metrics (% change, volume) as technical indicators need more data")
    else:
        st.sidebar.success(f"📅 Full Analysis Mode - All technical indicators available")
    
    # Data Completeness Settings
    st.sidebar.subheader("📊 Data Completeness")
    
    min_data_points = st.sidebar.slider(
        "Min Data Points Required",
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of data points required for analysis (higher = more complete data)"
    )
    
    allow_incomplete_data = st.sidebar.checkbox(
        "Allow Incomplete Data",
        value=True,
        help="Allow screening with incomplete data (useful for live trading)"
    )
    
    if allow_incomplete_data:
        st.sidebar.info("✅ Will proceed with available data even if incomplete")
    else:
        st.sidebar.warning("⚠️ Will skip stocks with incomplete data")
    
    # Screening criteria - adjust based on early morning mode
    if is_early_morning:
        screening_options = [
            "pct_change", "volume_spike", "range_pct",
            # Recent minute returns for early momentum
            "return_1min", "return_2min", "return_3min", "return_5min"
        ]
        help_text = "Early morning mode: Simple metrics + recent minute returns for momentum"
        default_idx = 0
    else:
        screening_options = [
            "interest_score",      # Both bullish and bearish signals
            "pct_change", "volume_ratio", "bullish_score",
            "bearish_score",       # Bearish signals only
            "range_pct", "rsi_momentum", "vwap_distance", 
            "momentum_10", "atr_pct",
            # Recent minute returns for momentum ranking
            "return_1min", "return_2min", "return_3min", "return_5min",
            "return_10min", "return_15min", "return_30min"
        ]
        help_text = "Full analysis mode: All technical indicators + recent minute returns"
        default_idx = 0
    
    screening_criteria = st.sidebar.selectbox(
        "Primary Screening Criteria",
        options=screening_options,
        index=default_idx,
        help=help_text
    )
    
    # Top K selection
    top_k = st.sidebar.slider(
        "Top K Stocks",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of top stocks to display"
    )
    
    # API Rate Limiting
    st.sidebar.markdown("**⚡ API Settings**")
    max_workers = st.sidebar.slider(
        "Concurrent Requests",
        min_value=1,
        max_value=5,
        value=2,
        help="Lower values reduce API rate limit issues"
    )
    
    request_delay = st.sidebar.slider(
        "Request Delay (ms)",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Delay between API requests"
    )
    
    # Advanced filters
    st.sidebar.subheader("📊 Advanced Filters")
    
    # Percentage change filter - using absolute values for both directions
    min_abs_pct_change = st.sidebar.slider(
        "Min Absolute % Change",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Minimum absolute percentage change from day's open (captures both bullish and bearish moves)"
    )
    
    min_volume_ratio = st.sidebar.slider(
        "Min Volume Ratio",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Minimum volume compared to average"
    )
    
    min_interest_score = st.sidebar.slider(
        "Min Interest Score",
        min_value=0,
        max_value=16,
        value=4,
        help="Minimum interest score (bullish + bearish signals - captures interesting stocks)"
    )
    
    min_abs_vwap_distance = st.sidebar.slider(
        "Min Absolute VWAP Distance (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Minimum absolute distance from VWAP (both above and below) - 0 means no filter"
    )
    
    max_recent_volume_ratio = st.sidebar.slider(
        "Max Recent Volume Ratio",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Maximum ratio of recent volume to total volume (1.0 means no filter, lower = less recent activity)"
    )
    
    recent_candles = st.sidebar.slider(
        "Recent Candles Count",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Number of recent candles to analyze for volume"
    )
    
    # Run screening
    if st.sidebar.button("🚀 Run Stock Screening", type="primary"):
        
        # Get universe
        universe = screener.get_universe_stocks(universe_type)
        
        # Convert dates to strings
        date_str = analysis_date.strftime('%Y-%m-%d')
        cutoff_str = cutoff_time.strftime('%H:%M:%S')
        
        # Debug: Show what cutoff string is being used
        st.info(f"🔧 Debug - Using cutoff_str: {cutoff_str} from cutoff_time: {cutoff_time}")
        
        # For intraday analysis, we need to ensure we have data
        # If analysis_date is today, use yesterday as start to ensure data availability
        if analysis_date == datetime.now().date():
            start_date_for_api = analysis_date - timedelta(days=1)
            st.info("📅 Using yesterday's data as start date to ensure data availability for today's analysis")
        else:
            start_date_for_api = analysis_date
        
        start_date_str = start_date_for_api.strftime('%Y-%m-%d')
        end_date_str = (start_date_for_api + timedelta(days=1)).strftime('%Y-%m-%d')
        #end_date_str = date_str
        
        # Determine if we should refresh cache
        refresh_cache_flag = st.session_state.get('refresh_cache', False)
        if refresh_cache_flag:
            st.info("🔄 Cache refresh mode: Will re-download all data from API")
            st.session_state['refresh_cache'] = False  # Reset flag
        
        # Run screening with enhanced data handling
        with st.spinner("🔍 Screening stocks..."):
            try:
                screened_df = screener.screen_stocks(
                    universe=universe,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cutoff_time=cutoff_str,
                    max_workers=max_workers,
                    request_delay=request_delay / 1000.0,  # Convert ms to seconds
                    refresh_cache=refresh_cache_flag,
                    min_data_points=min_data_points,
                    allow_incomplete_data=allow_incomplete_data
                )
                
                if screened_df.empty:
                    st.error("❌ No stocks found with sufficient data. Try:")
                    st.error("• Reducing 'Min Data Points Required'")
                    st.error("• Enabling 'Allow Incomplete Data'")
                    st.error("• Refreshing cache to get latest data")
                    st.error("• Using a different date (yesterday recommended)")
                    st.stop()
                
            except Exception as e:
                st.error(f"❌ Error during screening: {str(e)}")
                st.error("Try refreshing cache or using different settings")
                st.stop()
        
        if not screened_df.empty:
            # Apply filters using absolute percentage change, interest score, VWAP distance, and recent volume
            # Create the recent volume ratio column name based on selected candles
            recent_volume_column = f'recent_volume_ratio_{recent_candles}'
            
            # Build filter conditions
            filter_conditions = [
                (abs(screened_df['pct_change_from_open']) >= min_abs_pct_change),
                (screened_df['volume_ratio'] >= min_volume_ratio),
                (screened_df['interest_score'] >= min_interest_score)
            ]
            
            # Add VWAP distance filter if specified
            if min_abs_vwap_distance > 0:
                filter_conditions.append(abs(screened_df['vwap_distance']) >= min_abs_vwap_distance)
                # Debug: Show filter stats
                st.info(f"🔧 VWAP Distance Filter: |vwap_distance| >= {min_abs_vwap_distance}%")
                if not screened_df.empty:
                    vwap_stats = abs(screened_df['vwap_distance']).describe()
                    st.write(f"Absolute VWAP Distance Stats: Min={vwap_stats['min']:.3f}%, Max={vwap_stats['max']:.3f}%, Mean={vwap_stats['mean']:.3f}%")
            else:
                st.info(f"🔧 VWAP Distance Filter: SKIPPED (value = {min_abs_vwap_distance}%, set > 0 to activate)")
            
            # Add recent volume ratio filter if specified
            if max_recent_volume_ratio < 1.0 and recent_volume_column in screened_df.columns:
                filter_conditions.append(screened_df[recent_volume_column] <= max_recent_volume_ratio)
                # Debug: Show filter stats
                st.info(f"🔧 Recent Volume Filter: {recent_volume_column} <= {max_recent_volume_ratio}")
                if not screened_df.empty:
                    vol_stats = screened_df[recent_volume_column].describe()
                    st.write(f"Recent Volume Ratio Stats: Min={vol_stats['min']:.3f}, Max={vol_stats['max']:.3f}, Mean={vol_stats['mean']:.3f}")
            elif max_recent_volume_ratio >= 1.0:
                st.info(f"🔧 Recent Volume Filter: SKIPPED (value = {max_recent_volume_ratio}, set < 1.0 to activate)")
            elif recent_volume_column not in screened_df.columns:
                st.warning(f"🔧 Recent Volume Filter: MISSING COLUMN ({recent_volume_column})")
                st.info(f"Available columns: {[col for col in screened_df.columns if 'recent_volume' in col]}")
            
            # Apply all filters with debug info
            filtered_df = screened_df
            st.info(f"🔧 Debug: Starting with {len(filtered_df)} stocks")
            
            for i, condition in enumerate(filter_conditions):
                before_count = len(filtered_df)
                filtered_df = filtered_df[condition]
                after_count = len(filtered_df)
                st.write(f"Filter {i+1}: {before_count} → {after_count} stocks ({before_count - after_count} filtered out)")
            
            if filtered_df.empty:
                st.warning("No stocks match your filtering criteria. Try relaxing the filters.")
            else:
                # Get top stocks
                top_stocks = screener.get_top_stocks(filtered_df, screening_criteria, top_k)
                
                # Store in session state
                st.session_state['screened_stocks'] = top_stocks
                st.session_state['screening_date'] = start_date_str
                st.session_state['cutoff_time'] = cutoff_str
                
                # Display summary
                st.markdown("## 📊 Screening Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Screened", len(screened_df))
                with col2:
                    st.metric("After Filters", len(filtered_df))
                with col3:
                    st.metric("Top Stocks", len(top_stocks))
                with col4:
                    avg_change = top_stocks['pct_change_from_open'].mean()
                    st.metric("Avg % Change", f"{avg_change:.2f}%")
                
                # Display top stocks table with cutoff and EOD comparison
                st.markdown("### 🏆 Top Performing Stocks (Cutoff vs EOD Comparison)")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Cutoff Time Metrics", "🕐 EOD Metrics", "📈 Performance Comparison", "🎯 VWAP Backtest"])
                
                with tab1:
                    st.markdown(f"**Metrics at Cutoff Time ({cutoff_str})**")
                    # Add recent volume ratio column if it exists
                    recent_vol_col = f'cutoff_recent_volume_ratio_{recent_candles}'
                    
                    cutoff_columns = [
                        'symbol', 'cutoff_time', 'cutoff_price', 'cutoff_pct_change', 
                        'cutoff_range_pct', 'cutoff_volume_ratio', 'cutoff_interest_score',
                        'cutoff_bullish_score', 'cutoff_bearish_score', 'cutoff_signal_direction',
                        'cutoff_rsi', 'cutoff_vwap_distance', 'cutoff_above_vwap'
                    ]
                    
                    # Add recent volume column if it exists
                    if recent_vol_col in top_stocks.columns:
                        cutoff_columns.append(recent_vol_col)
                    
                    # Add recent minute returns columns if they exist (show key ones)
                    recent_return_cols = ['return_1min', 'return_3min', 'return_5min', 'return_10min']
                    for col in recent_return_cols:
                        if col in top_stocks.columns:
                            cutoff_columns.append(col)
                    
                    if all(col in top_stocks.columns for col in cutoff_columns):
                        cutoff_df = top_stocks[cutoff_columns].copy()
                        
                        # Create column names list
                        column_names = [
                            'Symbol', 'Cutoff Time', 'Price@Cutoff', '% Change@Cutoff', 
                            'Range%@Cutoff', 'Vol Ratio@Cutoff', 'Interest Score@Cutoff',
                            'Bull Score@Cutoff', 'Bear Score@Cutoff', 'Signal@Cutoff',
                            'RSI@Cutoff', 'VWAP Dist@Cutoff', 'Above VWAP@Cutoff'
                        ]
                        
                        # Add recent volume column name if it exists
                        if recent_vol_col in top_stocks.columns:
                            column_names.append(f'Recent Vol Ratio({recent_candles})')
                        
                        # Add recent returns column names if they exist
                        recent_return_names = {'return_1min': '1min Return%', 'return_3min': '3min Return%', 
                                             'return_5min': '5min Return%', 'return_10min': '10min Return%'}
                        for col in recent_return_cols:
                            if col in top_stocks.columns:
                                column_names.append(recent_return_names.get(col, col))
                        
                        # Rename columns for display
                        cutoff_df.columns = column_names
                        
                        # Format numeric columns
                        numeric_cols = cutoff_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if 'Return%' in col or 'Change%' in col or 'Dist%' in col:
                                cutoff_df[col] = cutoff_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                            elif 'Ratio' in col:
                                cutoff_df[col] = cutoff_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            elif 'Score' in col:
                                cutoff_df[col] = cutoff_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                            else:
                                cutoff_df[col] = cutoff_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                        
                        st.dataframe(cutoff_df, use_container_width=True)
                    else:
                        st.error("Missing required columns for cutoff time display")
                        st.write("Available columns:", list(top_stocks.columns))
                
                with tab2:
                    st.markdown("**End of Day Metrics**")
                    eod_columns = [
                        'symbol', 'eod_price', 'eod_pct_change', 'eod_range_pct', 
                        'eod_volume_ratio', 'eod_interest_score', 'eod_bullish_score', 
                        'eod_bearish_score', 'eod_signal_direction', 'eod_rsi', 
                        'eod_vwap_distance', 'eod_above_vwap'
                    ]
                    
                    if all(col in top_stocks.columns for col in eod_columns):
                        eod_df = top_stocks[eod_columns].copy()
                        eod_df.columns = [
                            'Symbol', 'EOD Price', 'EOD % Change', 'EOD Range %', 
                            'EOD Vol Ratio', 'EOD Interest Score', 'EOD Bull Score', 
                            'EOD Bear Score', 'EOD Signal', 'EOD RSI', 'EOD VWAP Dist', 'EOD Above VWAP'
                        ]
                        
                        # Format numeric columns
                        numeric_cols = eod_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if 'Change%' in col or 'Range%' in col or 'Dist%' in col:
                                eod_df[col] = eod_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                            elif 'Ratio' in col:
                                eod_df[col] = eod_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            elif 'Score' in col:
                                eod_df[col] = eod_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                            else:
                                eod_df[col] = eod_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                        
                        st.dataframe(eod_df, use_container_width=True)
                    else:
                        st.warning("EOD metrics not available - data may be incomplete")
                
                with tab3:
                    st.markdown("**Performance Comparison (Cutoff vs EOD)**")
                    comparison_columns = [
                        'symbol', 'cutoff_pct_change', 'eod_pct_change', 
                        'performance_diff', 'signal_accuracy'
                    ]
                    
                    if all(col in top_stocks.columns for col in comparison_columns):
                        comp_df = top_stocks[comparison_columns].copy()
                        comp_df.columns = [
                            'Symbol', 'Cutoff % Change', 'EOD % Change', 
                            'Performance Diff', 'Signal Accuracy'
                        ]
                        
                        # Format numeric columns
                        for col in comp_df.columns:
                            if col != 'Symbol':
                                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        
                        st.dataframe(comp_df, use_container_width=True)
                    else:
                        st.warning("Performance comparison not available - data may be incomplete")
                
                with tab4:
                    st.markdown("**VWAP Mean Reversion Backtest**")
                    if st.button("Run VWAP Backtest"):
                        with st.spinner("Running VWAP backtest..."):
                            try:
                                backtest_results = screener.run_vwap_mean_reversion_backtest(
                                    screened_df, top_k=5, 
                                    start_date=start_date_str, 
                                    end_date=end_date_str,
                                    cutoff_time=cutoff_str
                                )
                                
                                if backtest_results:
                                    st.success("Backtest completed successfully!")
                                    st.write("Results:", backtest_results)
                                else:
                                    st.warning("Backtest failed - insufficient data")
                            except Exception as e:
                                st.error(f"Backtest error: {str(e)}")
                    else:
                        st.info("Click 'Run VWAP Backtest' to execute the backtest")
                
                # Detailed analysis section
                st.markdown("## 🔍 Detailed Stock Analysis")
                
                if 'screened_stocks' in st.session_state:
                    selected_stock = st.selectbox(
                        "Select a stock for detailed analysis:",
                        options=top_stocks['symbol'].tolist(),
                        index=0
                    )
                    
                    if selected_stock:
                        stock_data = top_stocks[top_stocks['symbol'] == selected_stock].iloc[0]
                        create_detailed_stock_analysis(stock_data)
                        
                        # Show stock chart if data is available
                        try:
                            stock_df = screener.data_handler.get_historical_data(
                                selected_stock, start_date_str, end_date_str, "5minute"
                            )
                            if stock_df is not None and not stock_df.empty:
                                create_stock_chart(stock_df, selected_stock)
                        except Exception as e:
                            st.warning(f"Could not load chart data for {selected_stock}: {str(e)}")
                
                # Visualization section
                st.markdown("## 📈 Screening Visualization")
                create_screening_visualization(top_stocks)
    
    # Show cached data info
    if st.sidebar.checkbox("Show Cache Info", value=False):
        st.sidebar.subheader("📁 Cache Information")
        cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            st.sidebar.write(f"Cache files: {len(cache_files)}")
            for file in cache_files[:5]:  # Show first 5 files
                st.sidebar.write(f"• {file}")
            if len(cache_files) > 5:
                st.sidebar.write(f"... and {len(cache_files) - 5} more")
        else:
            st.sidebar.write("No cache directory found")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🔍 Advanced Stock Screener | Built for Live Trading | Data from Zerodha Kite API
    </div>
    """, unsafe_allow_html=True)

def create_screening_visualization(top_stocks: pd.DataFrame):
    """Create visualization for screening results."""
    
    st.markdown("### 📊 Screening Visualization")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "% Change vs Volume Ratio",
            "Bullish Score Distribution", 
            "RSI vs VWAP Distance",
            "Range % vs ATR %"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: % Change vs Volume Ratio
    fig.add_trace(
        go.Scatter(
            x=top_stocks['volume_ratio'],
            y=top_stocks['pct_change_from_open'],
            mode='markers+text',
            text=top_stocks['symbol'].str.replace('.NS', ''),
            textposition="top center",
            marker=dict(
                size=top_stocks['bullish_score'] * 3,
                color=top_stocks['pct_change_from_open'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="% Change")
            ),
            name="Stocks"
        ),
        row=1, col=1
    )
    
    # Plot 2: Bullish Score Distribution
    fig.add_trace(
        go.Histogram(
            x=top_stocks['bullish_score'],
            nbinsx=9,
            name="Bull Score",
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Plot 3: RSI vs VWAP Distance
    fig.add_trace(
        go.Scatter(
            x=top_stocks['rsi'],
            y=top_stocks['vwap_distance'],
            mode='markers+text',
            text=top_stocks['symbol'].str.replace('.NS', ''),
            textposition="top center",
            marker=dict(
                size=8,
                color=top_stocks['ma_alignment'],
                colorscale='Viridis',
                showscale=False
            ),
            name="RSI-VWAP"
        ),
        row=2, col=1
    )
    
    # Plot 4: Range % vs ATR %
    fig.add_trace(
        go.Scatter(
            x=top_stocks['atr_pct'],
            y=top_stocks['range_pct'],
            mode='markers+text',
            text=top_stocks['symbol'].str.replace('.NS', ''),
            textposition="top center",
            marker=dict(
                size=8,
                color='orange'
            ),
            name="Volatility"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Stock Screening Analysis Dashboard"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Volume Ratio", row=1, col=1)
    fig.update_yaxes(title_text="% Change from Open", row=1, col=1)
    fig.update_xaxes(title_text="Bullish Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="VWAP Distance %", row=2, col=1)
    fig.update_xaxes(title_text="ATR %", row=2, col=2)
    fig.update_yaxes(title_text="Range %", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def create_detailed_stock_analysis(stock_data: pd.Series):
    """Create detailed analysis for selected stock."""
    
    st.markdown(f"### 🔍 Detailed Analysis: {stock_data['symbol'].replace('.NS', '')}")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        change_color = "🟢" if stock_data['pct_change_from_open'] > 0 else "🔴"
        st.metric("Price Change", f"{stock_data['pct_change_from_open']:.2f}%", 
                 delta=f"{change_color} from open")
    
    with col2:
        vol_color = "🟢" if stock_data['volume_ratio'] > 1.5 else "🟡" if stock_data['volume_ratio'] > 1 else "🔴"
        st.metric("Volume Ratio", f"{stock_data['volume_ratio']:.2f}x",
                 delta=f"{vol_color} vs avg")
    
    with col3:
        bull_color = "🟢" if stock_data['bullish_score'] >= 5 else "🟡" if stock_data['bullish_score'] >= 3 else "🔴"
        st.metric("Bullish Score", f"{stock_data['bullish_score']}/8",
                 delta=f"{bull_color} signals")
    
    with col4:
        rsi_color = "🟢" if 40 <= stock_data['rsi'] <= 70 else "🟡" if 30 <= stock_data['rsi'] <= 80 else "🔴"
        st.metric("RSI", f"{stock_data['rsi']:.1f}",
                 delta=f"{rsi_color} momentum")
    
    with col5:
        vwap_color = "🟢" if stock_data['above_vwap'] else "🔴"
        st.metric("VWAP Distance", f"{stock_data['vwap_distance']:.2f}%",
                 delta=f"{vwap_color} vs VWAP")
    
    # Technical signals summary
    st.markdown("#### 📊 Technical Signals")
    
    signals_col1, signals_col2, signals_col3 = st.columns(3)
    
    with signals_col1:
        st.markdown("**Moving Averages**")
        ma_signals = []
        if stock_data['above_sma5']:
            ma_signals.append("✅ Above SMA 5")
        else:
            ma_signals.append("❌ Below SMA 5")
            
        if stock_data['above_sma10']:
            ma_signals.append("✅ Above SMA 10")
        else:
            ma_signals.append("❌ Below SMA 10")
            
        if stock_data['above_sma20']:
            ma_signals.append("✅ Above SMA 20")
        else:
            ma_signals.append("❌ Below SMA 20")
        
        for signal in ma_signals:
            st.markdown(signal)
    
    with signals_col2:
        st.markdown("**Position Signals**")
        position_signals = []
        
        if stock_data['above_vwap']:
            position_signals.append("✅ Above VWAP")
        else:
            position_signals.append("❌ Below VWAP")
            
        if stock_data['near_high']:
            position_signals.append("⚠️ Near Day High")
        else:
            position_signals.append("📊 Not at High")
            
        if stock_data['near_low']:
            position_signals.append("⚠️ Near Day Low")
        else:
            position_signals.append("📊 Not at Low")
        
        for signal in position_signals:
            st.markdown(signal)
    
    with signals_col3:
        st.markdown("**Momentum Signals**")
        momentum_signals = []
        
        if stock_data['momentum_10'] > 2:
            momentum_signals.append("✅ Strong 10-bar momentum")
        elif stock_data['momentum_10'] > 0:
            momentum_signals.append("🟡 Positive 10-bar momentum")
        else:
            momentum_signals.append("❌ Negative 10-bar momentum")
            
        if stock_data['momentum_20'] > 2:
            momentum_signals.append("✅ Strong 20-bar momentum")
        elif stock_data['momentum_20'] > 0:
            momentum_signals.append("🟡 Positive 20-bar momentum")
        else:
            momentum_signals.append("❌ Negative 20-bar momentum")
        
        alignment_desc = ["❌ Bearish", "🟡 Mixed", "✅ Bullish"][min(int(stock_data['ma_alignment']), 2)]
        momentum_signals.append(f"MA Alignment: {alignment_desc}")
        
        for signal in momentum_signals:
            st.markdown(signal)
    
    # Chart if data is available
    if 'data' in stock_data and stock_data['data'] is not None:
        st.markdown("#### 📈 Price Chart with Indicators")
        create_stock_chart(stock_data['data'], stock_data['symbol'])

def create_stock_chart(df: pd.DataFrame, symbol: str):
    """Create detailed stock chart with indicators."""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol.replace('.NS', '')} - Price & Indicators", "RSI", "Volume"),
        row_width=[0.7, 0.15, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_5' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_5'], name='SMA 5', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    if 'vwap' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['vwap'], name='VWAP', 
                      line=dict(color='purple', width=2)),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                      line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', 
               marker_color='lightblue'),
        row=3, col=1
    )
    
    if 'volume_sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['volume_sma_20'], name='Vol SMA', 
                      line=dict(color='blue')),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    create_screening_dashboard()