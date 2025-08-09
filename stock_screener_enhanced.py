"""
Enhanced Stock Screener for Live Deployment
Screens stocks based on technical indicators and price performance
With cache refresh and data completeness handling
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
import shutil
warnings.filterwarnings('ignore')

# Import our existing modules
from core.data_handler import DataHandler
from utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedStockScreener:
    """Enhanced stock screener for live deployment with cache management."""
    
    def __init__(self):
        self.data_handler = DataHandler()
        
    def get_universe_stocks(self, universe_type: str = "nifty50") -> List[str]:
        """Get stock universe based on type."""
        return self.data_handler.get_stocks_by_universe(universe_type)
    
    def clear_cache(self):
        """Clear all cached data."""
        cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                logger.info("Cache cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False
        return True
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
        info = {
            'cache_dir_exists': os.path.exists(cache_dir),
            'cache_files': [],
            'total_files': 0,
            'total_size_mb': 0
        }
        
        if info['cache_dir_exists']:
            files = os.listdir(cache_dir)
            info['cache_files'] = files
            info['total_files'] = len(files)
            
            total_size = 0
            for file in files:
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info
    
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
            
            # VWAP
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            data['vwap_distance'] = ((data['close'] - data['vwap']) / data['vwap']) * 100
            data['above_vwap'] = data['close'] > data['vwap']
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data['atr'] = true_range.rolling(14).mean()
            data['atr_pct'] = (data['atr'] / data['close']) * 100
            
            # Momentum indicators
            data['momentum_5'] = data['close'].pct_change(5) * 100
            data['momentum_10'] = data['close'].pct_change(10) * 100
            data['momentum_20'] = data['close'].pct_change(20) * 100
            
            # Recent minute returns (for early morning analysis)
            data['return_1min'] = data['close'].pct_change(1) * 100
            data['return_2min'] = data['close'].pct_change(2) * 100
            data['return_3min'] = data['close'].pct_change(3) * 100
            data['return_5min'] = data['close'].pct_change(5) * 100
            data['return_10min'] = data['close'].pct_change(10) * 100
            data['return_15min'] = data['close'].pct_change(15) * 100
            data['return_30min'] = data['close'].pct_change(30) * 100
            
            # Recent volume analysis
            for candles in [5, 10, 15, 20]:
                if len(data) >= candles:
                    recent_volume = data['volume'].tail(candles).sum()
                    total_volume = data['volume'].sum()
                    data[f'recent_volume_ratio_{candles}'] = recent_volume / total_volume if total_volume > 0 else 0
            
            # Scoring system
            data['bullish_score'] = (
                (data['close'] > data['sma_5']).astype(int) +
                (data['close'] > data['sma_10']).astype(int) +
                (data['close'] > data['sma_20']).astype(int) +
                (data['rsi'] > 50).astype(int) +
                (data['macd'] > data['macd_signal']).astype(int) +
                (data['close'] > data['vwap']).astype(int) +
                (data['volume_ratio'] > 1.2).astype(int) +
                (data['pct_change_from_open'] > 0).astype(int)
            )
            
            data['bearish_score'] = (
                (data['close'] < data['sma_5']).astype(int) +
                (data['close'] < data['sma_10']).astype(int) +
                (data['close'] < data['sma_20']).astype(int) +
                (data['rsi'] < 50).astype(int) +
                (data['macd'] < data['macd_signal']).astype(int) +
                (data['close'] < data['vwap']).astype(int) +
                (data['volume_ratio'] > 1.2).astype(int) +
                (data['pct_change_from_open'] < 0).astype(int)
            )
            
            data['interest_score'] = data['bullish_score'] + data['bearish_score']
            
            # Signal direction
            data['signal_direction'] = np.where(
                data['bullish_score'] > data['bearish_score'], 'BULLISH',
                np.where(data['bearish_score'] > data['bullish_score'], 'BEARISH', 'NEUTRAL')
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def fetch_and_analyze_stock(self, symbol: str, start_date: str, end_date: str, 
                               cutoff_time: Optional[str] = None, request_delay: float = 0.1,
                               min_data_points: int = 30, allow_incomplete_data: bool = True) -> Optional[Dict]:
        """Fetch and analyze a single stock with enhanced data validation."""
        try:
            # Fetch data
            df = self.data_handler.get_historical_data(
                symbol, start_date, end_date, "5minute", refresh_cache=False
            )
            
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Check data completeness
            if len(df) < min_data_points:
                if not allow_incomplete_data:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} points < {min_data_points} required")
                    return None
                else:
                    logger.info(f"Using incomplete data for {symbol}: {len(df)} points (minimum {min_data_points} recommended)")
            
            # Calculate indicators
            is_early_morning = cutoff_time and cutoff_time < "09:35:00"
            df = self.calculate_screening_indicators(df, is_early_morning)
            
            if df.empty:
                return None
            
            # Get cutoff time data
            if cutoff_time:
                try:
                    cutoff_dt = pd.to_datetime(f"{start_date} {cutoff_time}")
                    # Find the closest time to cutoff
                    df['datetime'] = pd.to_datetime(df.index)
                    time_diff = abs(df['datetime'] - cutoff_dt)
                    cutoff_idx = time_diff.idxmin()
                    
                    if cutoff_idx in df.index:
                        cutoff_data = df.loc[cutoff_idx]
                        
                        # Get EOD data (last row)
                        eod_data = df.iloc[-1]
                        
                        return {
                            'symbol': symbol,
                            'cutoff_time': cutoff_time,
                            'cutoff_price': cutoff_data['close'],
                            'cutoff_pct_change': cutoff_data['pct_change_from_open'],
                            'cutoff_range_pct': cutoff_data['range_pct'],
                            'cutoff_volume_ratio': cutoff_data['volume_ratio'],
                            'cutoff_interest_score': cutoff_data['interest_score'],
                            'cutoff_bullish_score': cutoff_data['bullish_score'],
                            'cutoff_bearish_score': cutoff_data['bearish_score'],
                            'cutoff_signal_direction': cutoff_data['signal_direction'],
                            'cutoff_rsi': cutoff_data['rsi'],
                            'cutoff_vwap_distance': cutoff_data['vwap_distance'],
                            'cutoff_above_vwap': cutoff_data['above_vwap'],
                            'eod_price': eod_data['close'],
                            'eod_pct_change': eod_data['pct_change_from_open'],
                            'eod_range_pct': eod_data['range_pct'],
                            'eod_volume_ratio': eod_data['volume_ratio'],
                            'eod_interest_score': eod_data['interest_score'],
                            'eod_bullish_score': eod_data['bullish_score'],
                            'eod_bearish_score': eod_data['bearish_score'],
                            'eod_signal_direction': eod_data['signal_direction'],
                            'eod_rsi': eod_data['rsi'],
                            'eod_vwap_distance': eod_data['vwap_distance'],
                            'eod_above_vwap': eod_data['above_vwap'],
                            'performance_diff': eod_data['pct_change_from_open'] - cutoff_data['pct_change_from_open'],
                            'signal_accuracy': 1 if (
                                (cutoff_data['signal_direction'] == 'BULLISH' and eod_data['pct_change_from_open'] > cutoff_data['pct_change_from_open']) or
                                (cutoff_data['signal_direction'] == 'BEARISH' and eod_data['pct_change_from_open'] < cutoff_data['pct_change_from_open'])
                            ) else 0,
                            'data_points': len(df),
                            'data_completeness': len(df) / min_data_points if min_data_points > 0 else 1.0
                        }
                    else:
                        logger.warning(f"Cutoff time {cutoff_time} not found in data for {symbol}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Error processing cutoff time for {symbol}: {e}")
                    return None
            else:
                # No cutoff time - use latest data
                latest_data = df.iloc[-1]
                return {
                    'symbol': symbol,
                    'current_price': latest_data['close'],
                    'pct_change_from_open': latest_data['pct_change_from_open'],
                    'range_pct': latest_data['range_pct'],
                    'volume_ratio': latest_data['volume_ratio'],
                    'interest_score': latest_data['interest_score'],
                    'bullish_score': latest_data['bullish_score'],
                    'bearish_score': latest_data['bearish_score'],
                    'signal_direction': latest_data['signal_direction'],
                    'rsi': latest_data['rsi'],
                    'vwap_distance': latest_data['vwap_distance'],
                    'above_vwap': latest_data['above_vwap'],
                    'data_points': len(df),
                    'data_completeness': len(df) / min_data_points if min_data_points > 0 else 1.0
                }
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def screen_stocks(self, universe: List[str], start_date: str, end_date: str,
                     cutoff_time: Optional[str] = None, max_workers: int = 3, 
                     request_delay: float = 0.1, refresh_cache: bool = False,
                     min_data_points: int = 30, allow_incomplete_data: bool = True) -> pd.DataFrame:
        """Screen stocks with enhanced data handling."""
        
        if refresh_cache:
            logger.info("Refreshing cache - will re-download all data")
            self.clear_cache()
        
        results = []
        
        def process_stock(symbol):
            try:
                time.sleep(request_delay)
                result = self.fetch_and_analyze_stock(
                    symbol, start_date, end_date, cutoff_time, request_delay,
                    min_data_points, allow_incomplete_data
                )
                return result
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(process_stock, symbol): symbol for symbol in universe}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        if not results:
            raise ValueError("No valid data found for any stock. Check API response and symbol list.")
        
        df = pd.DataFrame(results)
        
        # Add recent volume ratio columns if they exist
        for candles in [5, 10, 15, 20]:
            col_name = f'recent_volume_ratio_{candles}'
            if col_name in df.columns:
                df[f'cutoff_{col_name}'] = df[col_name]
        
        # Add recent minute returns if they exist
        for minutes in [1, 2, 3, 5, 10, 15, 30]:
            col_name = f'return_{minutes}min'
            if col_name in df.columns:
                df[col_name] = df[col_name]
        
        return df
    
    def get_top_stocks(self, screened_df: pd.DataFrame, criteria: str, top_k: int) -> pd.DataFrame:
        """Get top stocks based on criteria."""
        if screened_df.empty:
            return screened_df
        
        # Handle different criteria
        if criteria == "interest_score":
            sorted_df = screened_df.sort_values('interest_score', ascending=False)
        elif criteria == "pct_change":
            sorted_df = screened_df.sort_values('pct_change_from_open', ascending=False)
        elif criteria == "volume_ratio":
            sorted_df = screened_df.sort_values('volume_ratio', ascending=False)
        elif criteria == "bullish_score":
            sorted_df = screened_df.sort_values('bullish_score', ascending=False)
        elif criteria == "bearish_score":
            sorted_df = screened_df.sort_values('bearish_score', ascending=False)
        elif criteria == "range_pct":
            sorted_df = screened_df.sort_values('range_pct', ascending=False)
        elif criteria == "rsi_momentum":
            sorted_df = screened_df.sort_values('rsi', ascending=False)
        elif criteria == "vwap_distance":
            sorted_df = screened_df.sort_values('vwap_distance', ascending=False)
        elif criteria == "momentum_10":
            sorted_df = screened_df.sort_values('momentum_10', ascending=False)
        elif criteria == "atr_pct":
            sorted_df = screened_df.sort_values('atr_pct', ascending=False)
        elif criteria == "volume_spike":
            sorted_df = screened_df.sort_values('volume_spike', ascending=False)
        elif criteria.startswith("return_"):
            # Recent minute returns
            if criteria in screened_df.columns:
                sorted_df = screened_df.sort_values(criteria, ascending=False)
            else:
                sorted_df = screened_df.sort_values('pct_change_from_open', ascending=False)
        else:
            # Default to interest score
            sorted_df = screened_df.sort_values('interest_score', ascending=False)
        
        return sorted_df.head(top_k)

def create_enhanced_screening_dashboard():
    """Create the enhanced screening dashboard with cache management."""
    
    st.set_page_config(page_title="Enhanced Stock Screener", layout="wide")
    
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
    
    st.markdown('<h1 class="main-header">🔍 Enhanced Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown("### Live Deployment Ready - Find the best trading opportunities with cache management")
    
    # Initialize screener
    screener = EnhancedStockScreener()
    
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
        if screener.clear_cache():
            st.sidebar.success("✅ Cache cleared! Next run will download fresh data.")
        else:
            st.sidebar.error("❌ Failed to clear cache")
    
    # Show cache status
    cache_info = screener.get_cache_info()
    if cache_info['cache_dir_exists']:
        st.sidebar.info(f"📁 Cache Status: {cache_info['total_files']} files ({cache_info['total_size_mb']:.1f} MB)")
    else:
        st.sidebar.info("📁 Cache Status: No cache directory found")
    
    # Universe selection
    universe_type = st.sidebar.selectbox(
        "Stock Universe",
        options=["nifty50", "nifty100", "nifty500"],
        index=0,
        help="Choose the universe of stocks to screen"
    )
    
    if universe_type == "nifty500":
        st.sidebar.warning("⚠️ Nifty 500 may hit API rate limits. Consider using lower concurrent requests and higher delays.")
    
    # Date selection
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
    
    # Time cutoff
    st.sidebar.markdown("**⏰ Analysis Cutoff Time**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cutoff_hour = st.selectbox("Hour", options=list(range(9, 16)), index=1)
    with col2:
        cutoff_minute = st.selectbox("Minute", options=list(range(0, 60)), index=47)
    
    cutoff_time = datetime.strptime(f"{cutoff_hour:02d}:{cutoff_minute:02d}", "%H:%M").time()
    st.sidebar.info(f"🕐 Selected Cutoff Time: **{cutoff_time.strftime('%H:%M')}**")
    
    # Data Completeness Settings
    st.sidebar.subheader("📊 Data Completeness")
    
    min_data_points = st.sidebar.slider(
        "Min Data Points Required",
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of data points required for analysis"
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
    
    # Screening criteria
    screening_options = [
        "interest_score", "pct_change", "volume_ratio", "bullish_score",
        "bearish_score", "range_pct", "rsi_momentum", "vwap_distance", 
        "momentum_10", "atr_pct", "volume_spike",
        "return_1min", "return_3min", "return_5min", "return_10min"
    ]
    
    screening_criteria = st.sidebar.selectbox(
        "Primary Screening Criteria",
        options=screening_options,
        index=0,
        help="Choose the primary ranking criteria"
    )
    
    # Top K selection
    top_k = st.sidebar.slider(
        "Top K Stocks",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of top stocks to display"
    )
    
    # API Settings
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
    
    # Run screening
    if st.sidebar.button("🚀 Run Stock Screening", type="primary"):
        
        # Get universe
        universe = screener.get_universe_stocks(universe_type)
        
        # Convert dates to strings
        date_str = analysis_date.strftime('%Y-%m-%d')
        cutoff_str = cutoff_time.strftime('%H:%M:%S')
        
        # For intraday analysis, ensure we have data
        if analysis_date == datetime.now().date():
            start_date_for_api = analysis_date - timedelta(days=1)
            st.info("📅 Using yesterday's data as start date to ensure data availability for today's analysis")
        else:
            start_date_for_api = analysis_date
        
        start_date_str = start_date_for_api.strftime('%Y-%m-%d')
        end_date_str = (start_date_for_api + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Determine if we should refresh cache
        refresh_cache_flag = st.session_state.get('refresh_cache', False)
        if refresh_cache_flag:
            st.info("🔄 Cache refresh mode: Will re-download all data from API")
            st.session_state['refresh_cache'] = False
        
        # Run screening
        with st.spinner("🔍 Screening stocks..."):
            try:
                screened_df = screener.screen_stocks(
                    universe=universe,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cutoff_time=cutoff_str,
                    max_workers=max_workers,
                    request_delay=request_delay / 1000.0,
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
            # Get top stocks
            top_stocks = screener.get_top_stocks(screened_df, screening_criteria, top_k)
            
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
                st.metric("Top Stocks", len(top_stocks))
            with col3:
                avg_change = top_stocks['pct_change_from_open'].mean()
                st.metric("Avg % Change", f"{avg_change:.2f}%")
            with col4:
                avg_completeness = top_stocks['data_completeness'].mean()
                st.metric("Avg Data Completeness", f"{avg_completeness:.1%}")
            
            # Display results
            st.markdown("### 🏆 Top Performing Stocks")
            
            # Show key metrics
            display_columns = [
                'symbol', 'cutoff_pct_change', 'cutoff_volume_ratio', 
                'cutoff_interest_score', 'cutoff_signal_direction',
                'data_points', 'data_completeness'
            ]
            
            if all(col in top_stocks.columns for col in display_columns):
                display_df = top_stocks[display_columns].copy()
                display_df.columns = [
                    'Symbol', 'Cutoff % Change', 'Volume Ratio', 
                    'Interest Score', 'Signal', 'Data Points', 'Completeness'
                ]
                
                # Format numeric columns
                display_df['Cutoff % Change'] = display_df['Cutoff % Change'].apply(lambda x: f"{x:.2f}%")
                display_df['Volume Ratio'] = display_df['Volume Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Interest Score'] = display_df['Interest Score'].apply(lambda x: f"{x:.1f}")
                display_df['Completeness'] = display_df['Completeness'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Missing required columns for display")
                st.write("Available columns:", list(top_stocks.columns))
    
    # Show cache info
    if st.sidebar.checkbox("Show Cache Info", value=False):
        st.sidebar.subheader("📁 Cache Information")
        cache_info = screener.get_cache_info()
        if cache_info['cache_dir_exists']:
            st.sidebar.write(f"Cache files: {cache_info['total_files']}")
            st.sidebar.write(f"Cache size: {cache_info['total_size_mb']:.1f} MB")
            for file in cache_info['cache_files'][:5]:
                st.sidebar.write(f"• {file}")
            if len(cache_info['cache_files']) > 5:
                st.sidebar.write(f"... and {len(cache_info['cache_files']) - 5} more")
        else:
            st.sidebar.write("No cache directory found")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🔍 Enhanced Stock Screener | Live Deployment Ready | Data from Zerodha Kite API
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_enhanced_screening_dashboard() 