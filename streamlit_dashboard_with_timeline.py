#!/usr/bin/env python3
"""
🚀 STREAMLIT COMPREHENSIVE INTRADAY TRADING DASHBOARD - WITH TIME PROGRESSION
============================================================================

Interactive web-based dashboard with time progression feature for step-by-step
analysis through the trading day with live indicator values and ranges.

Run with: streamlit run streamlit_dashboard_with_timeline.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta, time
import sys
import os
import warnings
import time as time_module
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from core.data_handler import fetch_data

# Page configuration
st.set_page_config(
    page_title="🎯 Comprehensive Intraday Trading Dashboard with Timeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .timeline-controls {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .indicator-good { background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .indicator-neutral { background-color: #fff3cd; color: #856404; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .indicator-bad { background-color: #f8d7da; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .metric-big { font-size: 1.5rem; font-weight: bold; }
    .time-marker { background-color: #007bff; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_nifty50_symbols():
    """Get Nifty 50 stock symbols."""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
        'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'M&M.NS',
        'NTPC.NS', 'HCLTECH.NS', 'POWERGRID.NS', 'TATAMOTORS.NS', 'BAJFINANCE.NS'
    ]

@st.cache_data(ttl=300)
def fetch_stock_data(symbols, start_date, end_date, interval='minute'):
    """Fetch stock data with caching."""
    try:
        data = fetch_data(symbols=symbols, start=start_date, end=end_date, interval=interval)
        return data if data else {}
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators."""
    data = df.copy()
    
    try:
        # Moving Averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Volume
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Price action
        data['price_change'] = data['close'].pct_change() * 100
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

def get_indicator_status(value, indicator_type):
    """Get indicator status and color coding."""
    if indicator_type == 'rsi':
        if value < 30:
            return 'OVERSOLD', 'good'  # Good for buying
        elif value > 70:
            return 'OVERBOUGHT', 'bad'  # Bad for buying
        else:
            return 'NEUTRAL', 'neutral'
    
    elif indicator_type == 'macd':
        if value > 0:
            return 'BULLISH', 'good'
        elif value < 0:
            return 'BEARISH', 'bad'
        else:
            return 'NEUTRAL', 'neutral'
    
    elif indicator_type == 'volume_ratio':
        if value > 2.0:
            return 'HIGH VOLUME', 'good'
        elif value > 1.5:
            return 'ABOVE AVERAGE', 'neutral'
        else:
            return 'NORMAL', 'neutral'
    
    elif indicator_type == 'bb_position':
        if value < 0.2:
            return 'NEAR LOWER BB', 'good'  # Near support
        elif value > 0.8:
            return 'NEAR UPPER BB', 'bad'   # Near resistance
        else:
            return 'MIDDLE RANGE', 'neutral'
    
    return 'UNKNOWN', 'neutral'

def create_timeline_chart(symbol, data, current_index):
    """Create chart showing data up to current timeline position."""
    
    # Get data up to current position
    current_data = data.iloc[:current_index+1].copy()
    current_data = calculate_technical_indicators(current_data)
    
    if current_data.empty or len(current_data) < 5:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'{symbol.replace(".NS", "")} - Timeline Progress ({len(current_data)} bars)',
            'Volume Analysis',
            'RSI & MACD',
            'Bollinger Bands Position'
        ],
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=current_data.index,
            open=current_data['open'],
            high=current_data['high'],
            low=current_data['low'],
            close=current_data['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_5' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['sma_5'], name='SMA 5', line=dict(color='red', width=1)), row=1, col=1)
    if 'sma_20' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['sma_20'], name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    if 'vwap' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['vwap'], name='VWAP', line=dict(color='purple', width=1)), row=1, col=1)
    
    # Current position marker
    current_price = current_data['close'].iloc[-1]
    current_time = current_data.index[-1]
    fig.add_trace(
        go.Scatter(
            x=[current_time],
            y=[current_price],
            mode='markers',
            marker=dict(size=15, color='orange', symbol='circle'),
            name='Current Position'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['green' if current_data['close'].iloc[i] >= current_data['open'].iloc[i] else 'red' for i in range(len(current_data))]
    fig.add_trace(
        go.Bar(x=current_data.index, y=current_data['volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    if 'volume_sma' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['volume_sma'], name='Volume SMA', line=dict(color='blue', width=1)), row=2, col=1)
    
    # RSI
    if 'rsi' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['rsi'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
    
    # MACD
    if 'macd' in current_data.columns:
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['macd'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['macd_signal'], name='MACD Signal', line=dict(color='red', width=1)), row=3, col=1)
    
    # Bollinger Bands Position
    if 'bb_upper' in current_data.columns and 'bb_lower' in current_data.columns:
        bb_position = (current_data['close'] - current_data['bb_lower']) / (current_data['bb_upper'] - current_data['bb_lower'])
        fig.add_trace(go.Scatter(x=current_data.index, y=bb_position, name='BB Position', line=dict(color='orange', width=2)), row=4, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title=f"Timeline Analysis - {symbol.replace('.NS', '')} - {current_time.strftime('%H:%M:%S')}",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    """Main Streamlit application with timeline feature."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Intraday Trading Dashboard with Timeline</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Step-by-Step Analysis Through the Trading Day with Live Indicator Values")
    
    # Sidebar controls
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Date selection
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )
    
    start_date = end_date
    
    # Stock selection
    all_symbols = get_nifty50_symbols()
    selected_symbols = st.sidebar.multiselect(
        "Select Stocks",
        options=all_symbols,
        default=all_symbols[:5],
        help="Choose stocks to analyze"
    )
    
    # Fetch data button
    if st.sidebar.button("🚀 Fetch Data", type="primary"):
        
        with st.spinner(f"📈 Fetching data for {len(selected_symbols)} stocks..."):
            
            all_data = fetch_stock_data(
                symbols=selected_symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='minute'
            )
            
            if not all_data or len(all_data) == 0:
                st.error("❌ No data fetched. Please check your API connection.")
                return
            
            st.success(f"✅ Data fetched for {len(all_data)} stocks")
            st.session_state['stock_data'] = all_data
            st.session_state['analysis_date'] = end_date
    
    # Main timeline analysis
    if 'stock_data' in st.session_state:
        
        all_data = st.session_state['stock_data']
        
        # Stock selector for timeline
        st.markdown("## ⏯️ Timeline Analysis")
        
        available_stocks = list(all_data.keys())
        selected_stock = st.selectbox(
            "Select Stock for Timeline Analysis",
            options=available_stocks,
            format_func=lambda x: x.replace('.NS', ''),
            help="Choose stock for step-by-step analysis"
        )
        
        if selected_stock and selected_stock in all_data:
            
            stock_data = all_data[selected_stock]
            
            if not stock_data.empty and len(stock_data) > 10:
                
                # Timeline controls
                st.markdown('<div class="timeline-controls">', unsafe_allow_html=True)
                st.markdown("### 🎮 Timeline Controls")
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    # Timeline slider
                    max_index = len(stock_data) - 1
                    current_index = st.slider(
                        "Timeline Position",
                        min_value=20,  # Need some data for indicators
                        max_value=max_index,
                        value=min(50, max_index),  # Start at 50 or max
                        step=1,
                        help="Drag to move through the trading day"
                    )
                
                with col2:
                    # Quick jump buttons
                    if st.button("📍 Market Open"):
                        current_index = 20
                        st.rerun()
                    
                    if st.button("🕐 Mid Day"):
                        current_index = min(max_index // 2, max_index)
                        st.rerun()
                
                with col3:
                    if st.button("📍 Market Close"):
                        current_index = max_index
                        st.rerun()
                    
                    # Auto-play toggle
                    auto_play = st.checkbox("▶️ Auto Play", help="Automatically advance timeline")
                
                with col4:
                    # Speed control
                    speed = st.selectbox("Speed", options=[1, 2, 5, 10], index=1, help="Auto-play speed")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Current time display
                current_time = stock_data.index[current_index]
                progress_pct = (current_index / max_index) * 100
                
                st.markdown(f"""
                <div class="time-marker">
                    🕐 Current Time: {current_time.strftime('%H:%M:%S')} | 
                    📊 Progress: {progress_pct:.1f}% | 
                    📈 Bar {current_index + 1} of {len(stock_data)}
                </div>
                """, unsafe_allow_html=True)
                
                # Get current data point
                current_data = stock_data.iloc[:current_index+1].copy()
                current_data = calculate_technical_indicators(current_data)
                
                if not current_data.empty:
                    latest = current_data.iloc[-1]
                    
                    # Live indicator values
                    st.markdown("## 📊 Live Indicator Values")
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    # Price
                    price_change = latest.get('price_change', 0)
                    col1.metric("Price", f"₹{latest['close']:.1f}", f"{price_change:+.1f}%")
                    
                    # RSI
                    rsi_val = latest.get('rsi', 50)
                    rsi_status, rsi_color = get_indicator_status(rsi_val, 'rsi')
                    col2.metric("RSI", f"{rsi_val:.1f}", rsi_status)
                    
                    # MACD
                    macd_val = latest.get('macd', 0)
                    macd_status, macd_color = get_indicator_status(macd_val, 'macd')
                    col3.metric("MACD", f"{macd_val:.3f}", macd_status)
                    
                    # Volume Ratio
                    vol_ratio = latest.get('volume_ratio', 1)
                    vol_status, vol_color = get_indicator_status(vol_ratio, 'volume_ratio')
                    col4.metric("Volume", f"{vol_ratio:.1f}x", vol_status)
                    
                    # Bollinger Band Position
                    if not pd.isna(latest.get('bb_upper', np.nan)) and not pd.isna(latest.get('bb_lower', np.nan)):
                        bb_pos = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                        bb_status, bb_color = get_indicator_status(bb_pos, 'bb_position')
                        col5.metric("BB Position", f"{bb_pos:.1%}", bb_status)
                    else:
                        col5.metric("BB Position", "N/A", "CALCULATING")
                    
                    # Moving Average Signal
                    sma5 = latest.get('sma_5', latest['close'])
                    sma20 = latest.get('sma_20', latest['close'])
                    if sma5 > sma20 and latest['close'] > sma5:
                        ma_signal = "BULLISH"
                        ma_color = "good"
                    elif sma5 < sma20 and latest['close'] < sma5:
                        ma_signal = "BEARISH"
                        ma_color = "bad"
                    else:
                        ma_signal = "NEUTRAL"
                        ma_color = "neutral"
                    col6.metric("MA Trend", ma_signal, f"5>{20}" if sma5 > sma20 else f"5<20")
                    
                    # Indicator ranges and meanings
                    st.markdown("## 📋 Indicator Ranges & Meanings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **📈 RSI (Relative Strength Index)**
                        - 🟢 **0-30**: Oversold (Good buying opportunity)
                        - 🟡 **30-70**: Neutral zone (Normal trading)
                        - 🔴 **70-100**: Overbought (Consider selling)
                        
                        **📊 MACD (Moving Average Convergence Divergence)**
                        - 🟢 **Positive**: Bullish momentum
                        - 🔴 **Negative**: Bearish momentum
                        - 🟡 **Near Zero**: Weak momentum
                        
                        **📈 Volume Ratio**
                        - 🟢 **>2.0x**: High interest/breakout
                        - 🟡 **1.5-2.0x**: Above average activity
                        - ⚪ **<1.5x**: Normal trading volume
                        """)
                    
                    with col2:
                        st.markdown("""
                        **📊 Bollinger Band Position**
                        - 🟢 **0-20%**: Near lower band (Support)
                        - 🟡 **20-80%**: Middle range (Normal)
                        - 🔴 **80-100%**: Near upper band (Resistance)
                        
                        **📈 Moving Average Trend**
                        - 🟢 **Bullish**: Price > SMA5 > SMA20
                        - 🔴 **Bearish**: Price < SMA5 < SMA20
                        - 🟡 **Neutral**: Mixed signals
                        
                        **⏰ Best Trading Times**
                        - 🟢 **9:15-10:30**: Opening volatility
                        - 🟡 **10:30-14:30**: Mid-day consolidation
                        - 🟢 **14:30-15:30**: Closing moves
                        """)
                    
                    # Create and display timeline chart
                    fig = create_timeline_chart(selected_stock, stock_data, current_index)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Auto-play functionality
                    if auto_play and current_index < max_index:
                        time_module.sleep(1.0 / speed)  # Delay based on speed
                        st.rerun()
                
            else:
                st.error("Insufficient data for timeline analysis")

if __name__ == "__main__":
    main() 