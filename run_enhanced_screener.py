#!/usr/bin/env python3
"""
Enhanced Stock Screener Runner for Live Deployment
"""

import streamlit as st
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the cache manager
from cache_manager import CacheManager, add_cache_management_to_sidebar, add_data_completeness_settings

# Import the original stock screener
from stock_screener import StockScreener, create_screening_dashboard

def main():
    """Run the enhanced stock screener with cache management."""
    
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
    screener = StockScreener()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Screening Controls")
    
    # Cache Management Section
    cache_manager = add_cache_management_to_sidebar()
    
    # Data Completeness Settings
    min_data_points, allow_incomplete_data = add_data_completeness_settings()
    
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
    from datetime import datetime, timedelta
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
        
        # Run screening with enhanced data handling
        with st.spinner("🔍 Screening stocks..."):
            try:
                # Use the original screen_stocks method but with enhanced parameters
                screened_df = screener.screen_stocks(
                    universe=universe,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    cutoff_time=cutoff_str,
                    max_workers=max_workers,
                    request_delay=request_delay / 1000.0
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
                st.metric("Data Quality", "Enhanced")
            
            # Display results
            st.markdown("### 🏆 Top Performing Stocks")
            
            # Show key metrics
            display_columns = [
                'symbol', 'cutoff_pct_change', 'cutoff_volume_ratio', 
                'cutoff_interest_score', 'cutoff_signal_direction'
            ]
            
            if all(col in top_stocks.columns for col in display_columns):
                display_df = top_stocks[display_columns].copy()
                display_df.columns = [
                    'Symbol', 'Cutoff % Change', 'Volume Ratio', 
                    'Interest Score', 'Signal'
                ]
                
                # Format numeric columns
                display_df['Cutoff % Change'] = display_df['Cutoff % Change'].apply(lambda x: f"{x:.2f}%")
                display_df['Volume Ratio'] = display_df['Volume Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Interest Score'] = display_df['Interest Score'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.error("Missing required columns for display")
                st.write("Available columns:", list(top_stocks.columns))
    
    # Show cache info
    if st.sidebar.checkbox("Show Cache Info", value=False):
        st.sidebar.subheader("📁 Cache Information")
        cache_info = cache_manager.get_cache_info()
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
    main() 