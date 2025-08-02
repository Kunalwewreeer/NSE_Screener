"""
Integrated Trading Dashboard with Stock Screening
Combines the existing dashboard with advanced stock screening capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import existing modules
from core.data_handler import DataHandler
from stock_screener import StockScreener

def main():
    """Main integrated dashboard."""
    
    st.set_page_config(page_title="Integrated Trading Dashboard", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .tab-header { color: #2c3e50; font-size: 1.2rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎯 Integrated Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Stock Screener", "📈 Timeline Analysis", "🔍 Market Overview"])
    
    with tab1:
        st.markdown('<div class="tab-header">🔍 Advanced Stock Screener</div>', unsafe_allow_html=True)
        create_screener_tab()
    
    with tab2:
        st.markdown('<div class="tab-header">📈 Individual Stock Analysis</div>', unsafe_allow_html=True)
        create_timeline_tab()
    
    with tab3:
        st.markdown('<div class="tab-header">🔍 Market Overview</div>', unsafe_allow_html=True)
        create_market_overview_tab()

def create_screener_tab():
    """Create the stock screener tab."""
    
    screener = StockScreener()
    
    # Sidebar controls for screener
    st.sidebar.header("🔍 Stock Screening")
    
    # Universe selection
    universe_type = st.sidebar.selectbox(
        "Stock Universe",
        options=["nifty50", "nifty100"],
        help="Choose the universe of stocks to screen"
    )
    
    # Date and time selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        screen_date = st.date_input(
            "Screening Date",
            value=datetime.now().date() - timedelta(days=1),
            help="Date to analyze"
        )
    
    with col2:
        cutoff_time = st.time_input(
            "Cutoff Time",
            value=datetime.strptime("12:00", "%H:%M").time(),
            help="Analyze up to this time"
        )
    
    # Screening criteria
    screening_criteria = st.sidebar.selectbox(
        "Primary Criteria",
        options=[
            "pct_change", "volume_ratio", "bullish_score", 
            "range_pct", "rsi_momentum", "vwap_distance"
        ],
        index=0
    )
    
    top_k = st.sidebar.slider("Top K Stocks", 5, 30, 10)
    
    # Advanced filters
    with st.sidebar.expander("🎛️ Advanced Filters"):
        min_pct_change = st.slider("Min % Change", -5.0, 10.0, 0.0, 0.5)
        min_volume_ratio = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1)
        min_bullish_score = st.slider("Min Bullish Score", 0, 8, 3)
    
    # Run screening
    if st.sidebar.button("🚀 Screen Stocks", type="primary"):
        
        universe = screener.get_universe_stocks(universe_type)
        
        with st.spinner("🔍 Screening stocks..."):
            screened_df = screener.screen_stocks(
                universe=universe,
                start_date=screen_date.strftime('%Y-%m-%d'),
                end_date=screen_date.strftime('%Y-%m-%d'),
                cutoff_time=cutoff_time.strftime('%H:%M:%S')
            )
        
        if not screened_df.empty:
            # Apply filters
            filtered_df = screened_df[
                (screened_df['pct_change_from_open'] >= min_pct_change) &
                (screened_df['volume_ratio'] >= min_volume_ratio) &
                (screened_df['bullish_score'] >= min_bullish_score)
            ]
            
            if not filtered_df.empty:
                top_stocks = screener.get_top_stocks(filtered_df, screening_criteria, top_k)
                st.session_state['screened_stocks'] = top_stocks
                
                # Display results
                display_screening_results(top_stocks)
            else:
                st.warning("No stocks match your criteria. Try relaxing the filters.")
    
    # Display existing results
    if 'screened_stocks' in st.session_state:
        st.markdown("---")
        display_screening_results(st.session_state['screened_stocks'])

def create_timeline_tab():
    """Create the timeline analysis tab (existing dashboard functionality)."""
    
    st.markdown("### Select stocks from screener results or choose manually")
    
    data_handler = DataHandler()
    
    # Check if we have screened stocks
    if 'screened_stocks' in st.session_state:
        screened_stocks = st.session_state['screened_stocks']
        
        st.markdown("#### 🎯 Use Screened Stocks")
        col1, col2 = st.columns(2)
        
        with col1:
            use_screened = st.checkbox("Use screened stocks", value=True)
        
        with col2:
            if use_screened:
                selected_symbols = st.multiselect(
                    "Select from screened stocks",
                    options=screened_stocks['symbol'].tolist(),
                    default=screened_stocks['symbol'].head(5).tolist(),
                    format_func=lambda x: f"{x.replace('.NS', '')} ({screened_stocks[screened_stocks['symbol']==x]['pct_change_from_open'].iloc[0]:.2f}%)"
                )
            else:
                selected_symbols = st.multiselect(
                    "Select stocks manually",
                    options=data_handler.get_nifty50_stocks(),
                    default=['RELIANCE.NS', 'TCS.NS'],
                    format_func=lambda x: x.replace('.NS', '')
                )
    else:
        st.info("💡 Run the stock screener first to get targeted stock recommendations")
        selected_symbols = st.multiselect(
            "Select stocks manually",
            options=data_handler.get_nifty50_stocks(),
            default=['RELIANCE.NS', 'TCS.NS'],
            format_func=lambda x: x.replace('.NS', '')
        )
    
    if selected_symbols:
        # Date selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now().date())
        
        if st.button("📊 Load Data for Timeline Analysis", type="primary"):
            with st.spinner("📈 Loading data..."):
                all_data = data_handler.get_historical_data(
                    symbols=selected_symbols,
                    from_date=start_date.strftime('%Y-%m-%d'),
                    to_date=end_date.strftime('%Y-%m-%d'),
                    interval="minute",
                    refresh_cache=True
                )
            
            if all_data:
                st.session_state['timeline_data'] = all_data
                st.success(f"✅ Loaded data for {len(all_data)} stocks")
    
    # Display timeline analysis if data is available
    if 'timeline_data' in st.session_state:
        create_timeline_analysis(st.session_state['timeline_data'])

def create_market_overview_tab():
    """Create market overview tab."""
    
    st.markdown("### 🔍 Market Overview & Heatmaps")
    
    if 'screened_stocks' in st.session_state:
        screened_stocks = st.session_state['screened_stocks']
        
        # Market heatmap
        st.markdown("#### 📊 Market Heatmap")
        create_market_heatmap(screened_stocks)
        
        # Sector analysis
        st.markdown("#### 🏭 Performance Analysis")
        create_performance_analysis(screened_stocks)
        
    else:
        st.info("Run the stock screener first to see market overview")

def display_screening_results(top_stocks: pd.DataFrame):
    """Display screening results."""
    
    st.markdown("### 🏆 Top Performing Stocks")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Found", len(top_stocks))
    with col2:
        avg_change = top_stocks['pct_change_from_open'].mean()
        st.metric("Avg % Change", f"{avg_change:.2f}%")
    with col3:
        avg_volume = top_stocks['volume_ratio'].mean()
        st.metric("Avg Volume Ratio", f"{avg_volume:.2f}x")
    with col4:
        avg_bullish = top_stocks['bullish_score'].mean()
        st.metric("Avg Bullish Score", f"{avg_bullish:.1f}/8")
    
    # Results table
    display_columns = [
        'symbol', 'current_price', 'pct_change_from_open', 'range_pct',
        'volume_ratio', 'bullish_score', 'rsi', 'above_vwap'
    ]
    
    display_df = top_stocks[display_columns].copy()
    display_df.columns = [
        'Symbol', 'Price', '% Change', 'Range %',
        'Vol Ratio', 'Bull Score', 'RSI', 'Above VWAP'
    ]
    
    # Format and style
    styled_df = display_df.style.format({
        'Price': '₹{:.2f}',
        '% Change': '{:.2f}%',
        'Range %': '{:.2f}%',
        'Vol Ratio': '{:.2f}',
        'RSI': '{:.1f}'
    }).apply(lambda x: [
        'background-color: #d4edda' if val > 2 else 
        'background-color: #f8d7da' if val < 0 else ''
        for val in x
    ], subset=['% Change'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Quick visualization
    create_quick_screening_viz(top_stocks)

def create_timeline_analysis(all_data: dict):
    """Create timeline analysis for selected stocks."""
    
    st.markdown("### ⏯️ Timeline Analysis")
    
    # Stock selector
    available_stocks = list(all_data.keys())
    selected_stock = st.selectbox(
        "Select Stock for Timeline Analysis",
        options=available_stocks,
        format_func=lambda x: x.replace('.NS', '')
    )
    
    if selected_stock and selected_stock in all_data:
        stock_data = all_data[selected_stock]
        
        if not stock_data.empty and len(stock_data) > 20:
            # Timeline controls
            max_index = len(stock_data) - 1
            current_index = st.slider(
                "Timeline Position",
                min_value=20,
                max_value=max_index,
                value=min(50, max_index),
                help="Drag to move through the trading day"
            )
            
            # Current time display
            current_time = stock_data.index[current_index]
            st.info(f"📅 Analyzing: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get current data
            current_data = stock_data.iloc[:current_index+1].copy()
            
            # Simple indicators (avoiding the complex ones that caused errors)
            current_data['sma_5'] = current_data['close'].rolling(5).mean()
            current_data['sma_20'] = current_data['close'].rolling(20).mean()
            current_data['volume_sma'] = current_data['volume'].rolling(20).mean()
            
            # Display current metrics
            latest = current_data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"₹{latest['close']:.2f}")
            with col2:
                price_change = ((latest['close'] - current_data['open'].iloc[0]) / current_data['open'].iloc[0]) * 100
                st.metric("% Change", f"{price_change:.2f}%")
            with col3:
                if not pd.isna(latest.get('volume_sma', np.nan)):
                    vol_ratio = latest['volume'] / latest['volume_sma']
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
                else:
                    st.metric("Volume", f"{latest['volume']:,.0f}")
            with col4:
                day_range = ((current_data['high'].max() - current_data['low'].min()) / current_data['open'].iloc[0]) * 100
                st.metric("Day Range", f"{day_range:.2f}%")
            
            # Simple chart
            create_simple_timeline_chart(current_data, selected_stock, current_time)

def create_simple_timeline_chart(data: pd.DataFrame, symbol: str, current_time):
    """Create a simple timeline chart without complex indicators."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol.replace('.NS', '')} - Price", "Volume"),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Moving averages if available
    if 'sma_5' in data.columns and not data['sma_5'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_5'], name='SMA 5', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    if 'sma_20' in data.columns and not data['sma_20'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['volume'], name='Volume', 
               marker_color='lightblue'),
        row=2, col=1
    )
    
    if 'volume_sma' in data.columns and not data['volume_sma'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['volume_sma'], name='Vol SMA', 
                      line=dict(color='orange')),
            row=2, col=1
        )
    
    # Current time marker
    fig.add_vline(x=current_time, line_dash="dash", line_color="red", 
                  annotation_text="Current Time")
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title=f"Timeline Analysis - {current_time.strftime('%H:%M:%S')}",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_quick_screening_viz(top_stocks: pd.DataFrame):
    """Create quick visualization for screening results."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("% Change vs Volume Ratio", "Bullish Score Distribution")
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=top_stocks['volume_ratio'],
            y=top_stocks['pct_change_from_open'],
            mode='markers+text',
            text=top_stocks['symbol'].str.replace('.NS', ''),
            textposition="top center",
            marker=dict(
                size=top_stocks['bullish_score'] * 2,
                color=top_stocks['pct_change_from_open'],
                colorscale='RdYlGn',
                showscale=True
            ),
            name="Stocks"
        ),
        row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=top_stocks['bullish_score'],
            nbinsx=9,
            name="Distribution",
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Volume Ratio", row=1, col=1)
    fig.update_yaxes(title_text="% Change", row=1, col=1)
    fig.update_xaxes(title_text="Bullish Score", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def create_market_heatmap(screened_stocks: pd.DataFrame):
    """Create market heatmap."""
    
    # Simple heatmap based on performance
    fig = go.Figure(data=go.Heatmap(
        z=[screened_stocks['pct_change_from_open'].values],
        x=screened_stocks['symbol'].str.replace('.NS', ''),
        y=['% Change'],
        colorscale='RdYlGn',
        zmid=0,
        text=screened_stocks['pct_change_from_open'].round(2),
        texttemplate="%{text}%",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Stock Performance Heatmap",
        height=200,
        xaxis_title="Stocks",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_analysis(screened_stocks: pd.DataFrame):
    """Create performance analysis charts."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top gainers
        st.markdown("**🚀 Top Gainers**")
        top_gainers = screened_stocks.nlargest(5, 'pct_change_from_open')
        for _, stock in top_gainers.iterrows():
            st.markdown(f"• {stock['symbol'].replace('.NS', '')}: {stock['pct_change_from_open']:.2f}%")
    
    with col2:
        # High volume stocks
        st.markdown("**📊 High Volume Activity**")
        high_volume = screened_stocks.nlargest(5, 'volume_ratio')
        for _, stock in high_volume.iterrows():
            st.markdown(f"• {stock['symbol'].replace('.NS', '')}: {stock['volume_ratio']:.2f}x")

if __name__ == "__main__":
    main()