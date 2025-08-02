#!/usr/bin/env python3
"""
🚀 STREAMLIT COMPREHENSIVE INTRADAY TRADING DASHBOARD
====================================================

Interactive web-based dashboard for analyzing ALL Nifty 50 stocks
with complete technical indicators for intraday trading decisions.

Run with: streamlit run streamlit_trading_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from core.data_handler import fetch_data

# Page configuration
st.set_page_config(
    page_title="🎯 Comprehensive Intraday Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .signal-strong-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-buy {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
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
        'NTPC.NS', 'HCLTECH.NS', 'POWERGRID.NS', 'TATAMOTORS.NS', 'BAJFINANCE.NS',
        'HDFCLIFE.NS', 'TECHM.NS', 'SBILIFE.NS', 'ADANIPORTS.NS', 'ONGC.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'BAJAJFINSV.NS', 'DRREDDY.NS',
        'EICHERMOT.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'INDUSINDBK.NS',
        'APOLLOHOSP.NS', 'HEROMOTOCO.NS', 'UPL.NS', 'TATASTEEL.NS', 'BPCL.NS',
        'HINDALCO.NS', 'BAJAJ-AUTO.NS', 'TATACONSUM.NS', 'LTIM.NS', 'ADANIENT.NS'
    ]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbols, start_date, end_date, interval='minute'):
    """Fetch stock data with caching."""
    try:
        data = fetch_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            interval=interval
        )
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}

def filter_data_for_date(data, target_date):
    """Filter data to show only the specific target date."""
    if data.empty:
        return data
    
    # Convert target_date to datetime if it's a string or date
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif hasattr(target_date, 'date'):
        target_date = target_date.date()
    
    # Filter data for the specific date
    filtered_data = data[data.index.date == target_date].copy()
    return filtered_data

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators."""
    data = df.copy()
    
    try:
        # ===== MOVING AVERAGES =====
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        
        # VWAP
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # ===== MOMENTUM INDICATORS =====
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
        
        # Stochastic
        low_min = data['low'].rolling(window=14).min()
        high_max = data['high'].rolling(window=14).max()
        data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # ===== VOLATILITY INDICATORS =====
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # ATR
        data['tr1'] = data['high'] - data['low']
        data['tr2'] = abs(data['high'] - data['close'].shift())
        data['tr3'] = abs(data['low'] - data['close'].shift())
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # ===== VOLUME INDICATORS =====
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # OBV
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        
        # ===== PRICE ACTION =====
        data['price_change'] = data['close'].pct_change() * 100
        data['volatility'] = data['close'].pct_change().rolling(20).std() * 100
        
        # ===== SUPPORT/RESISTANCE =====
        data['pivot'] = (data['high'].shift() + data['low'].shift() + data['close'].shift()) / 3
        data['r1'] = 2 * data['pivot'] - data['low'].shift()
        data['s1'] = 2 * data['pivot'] - data['high'].shift()
        
        # ===== GENERATE SIGNALS =====
        data = generate_trading_signals(data)
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

def generate_trading_signals(df):
    """Generate comprehensive trading signals."""
    
    # Initialize signal columns
    df['buy_signals'] = 0
    df['sell_signals'] = 0
    df['signal_strength'] = 0
    df['signal_type'] = 'NEUTRAL'
    
    # ===== TREND SIGNALS =====
    ma_bullish = (df['sma_5'] > df['sma_20']) & (df['close'] > df['sma_5'])
    ma_bearish = (df['sma_5'] < df['sma_20']) & (df['close'] < df['sma_5'])
    
    # ===== MOMENTUM SIGNALS =====
    rsi_oversold = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
    rsi_overbought = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
    rsi_bullish = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    rsi_bearish = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    
    macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # ===== VOLATILITY SIGNALS =====
    bb_oversold = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
    bb_overbought = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
    
    # ===== VOLUME SIGNALS =====
    volume_breakout = (df['volume_ratio'] > 2.0) & (df['close'] > df['close'].shift(1))
    volume_selling = (df['volume_ratio'] > 2.0) & (df['close'] < df['close'].shift(1))
    
    # Count signals
    df['buy_signals'] = (
        ma_bullish.astype(int) * 3 +
        rsi_oversold.astype(int) * 2 +
        rsi_bullish.astype(int) * 1 +
        macd_bullish.astype(int) * 2 +
        bb_oversold.astype(int) * 1 +
        volume_breakout.astype(int) * 2
    )
    
    df['sell_signals'] = (
        ma_bearish.astype(int) * 3 +
        rsi_overbought.astype(int) * 2 +
        rsi_bearish.astype(int) * 1 +
        macd_bearish.astype(int) * 2 +
        bb_overbought.astype(int) * 1 +
        volume_selling.astype(int) * 2
    )
    
    df['signal_strength'] = df['buy_signals'] - df['sell_signals']
    
    # Classify signals
    df.loc[df['signal_strength'] >= 6, 'signal_type'] = 'STRONG BUY'
    df.loc[(df['signal_strength'] >= 3) & (df['signal_strength'] < 6), 'signal_type'] = 'BUY'
    df.loc[(df['signal_strength'] >= 1) & (df['signal_strength'] < 3), 'signal_type'] = 'WEAK BUY'
    df.loc[(df['signal_strength'] <= -1) & (df['signal_strength'] > -3), 'signal_type'] = 'WEAK SELL'
    df.loc[(df['signal_strength'] <= -3) & (df['signal_strength'] > -6), 'signal_type'] = 'SELL'
    df.loc[df['signal_strength'] <= -6, 'signal_type'] = 'STRONG SELL'
    
    return df

def create_comprehensive_chart(symbol, data, chart_date=None, chart_mode="Full Range"):
    """Create comprehensive interactive chart with all indicators."""
    
    # Filter data based on chart mode
    if chart_mode == "Single Day" and chart_date:
        df = filter_data_for_date(data, chart_date)
        chart_title_suffix = f" - {chart_date}"
    else:
        df = data.copy()
        chart_title_suffix = ""
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    if df.empty or len(df) < 10:  # Reduced minimum for single day
        return None, None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'{symbol.replace(".NS", "")}{chart_title_suffix} - Price Action & Indicators',
            'Volume Analysis',
            'RSI & Stochastic',
            'MACD'
        ],
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # ===== MAIN PRICE CHART =====
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_5'], name='SMA 5', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], name='EMA 9', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name='VWAP', line=dict(color='purple', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray', width=1), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray', width=1), fill='tonexty', opacity=0.2), row=1, col=1)
    
    # Support/Resistance
    if not df['pivot'].isna().all():
        fig.add_hline(y=df['pivot'].iloc[-1], line_dash="dash", line_color="orange", opacity=0.7, row=1, col=1)
    if not df['r1'].isna().all():
        fig.add_hline(y=df['r1'].iloc[-1], line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
    if not df['s1'].isna().all():
        fig.add_hline(y=df['s1'].iloc[-1], line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
    
    # Trading signals
    buy_signals = df[df['buy_signals'] >= 3]
    sell_signals = df[df['sell_signals'] >= 3]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
    
    # ===== VOLUME CHART =====
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['volume_sma'], name='Volume SMA', line=dict(color='blue', width=1)), row=2, col=1)
    
    # ===== RSI & STOCHASTIC =====
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], name='Stoch %K', line=dict(color='orange', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], name='Stoch %D', line=dict(color='red', width=1)), row=3, col=1)
    
    # ===== MACD =====
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='MACD Signal', line=dict(color='red', width=1)), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['macd_histogram'], name='MACD Histogram', opacity=0.6), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title=f"Comprehensive Technical Analysis - {symbol.replace('.NS', '')}{chart_title_suffix}",
        xaxis_rangeslider_visible=False
    )
    
    # Get latest metrics
    latest = df.iloc[-1]
    metrics = {
        'price': latest['close'],
        'change_pct': latest.get('price_change', 0),
        'volume_ratio': latest.get('volume_ratio', 1),
        'rsi': latest.get('rsi', 50),
        'macd': latest.get('macd', 0),
        'signal_type': latest.get('signal_type', 'NEUTRAL'),
        'signal_strength': latest.get('signal_strength', 0),
        'buy_signals': latest.get('buy_signals', 0),
        'sell_signals': latest.get('sell_signals', 0),
        'atr': latest.get('atr', 0),
        'bb_position': (latest['close'] - latest.get('bb_lower', latest['close'])) / (latest.get('bb_upper', latest['close']) - latest.get('bb_lower', latest['close'])) if latest.get('bb_upper', 0) != latest.get('bb_lower', 0) else 0.5
    }
    
    return fig, metrics

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Comprehensive Intraday Trading Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Interactive Analysis of All Nifty 50 Stocks with Complete Technical Indicators")
    
    # Sidebar controls
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Date selection
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )
    
    days_back = st.sidebar.selectbox(
        "Days of Data",
        options=[1, 2, 3, 5, 7],
        index=2,
        help="Number of days of historical data to fetch"
    )
    
    start_date = end_date - timedelta(days=days_back)
    
    # Stock selection
    all_symbols = get_nifty50_symbols()
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        options=["Top Performers", "All Stocks", "Custom Selection"],
        help="Choose how many stocks to analyze"
    )
    
    if analysis_mode == "Custom Selection":
        selected_symbols = st.sidebar.multiselect(
            "Select Stocks",
            options=all_symbols,
            default=all_symbols[:10],
            help="Choose specific stocks to analyze"
        )
    elif analysis_mode == "Top Performers":
        num_stocks = st.sidebar.slider("Number of Stocks", 5, 25, 15)
        selected_symbols = all_symbols[:num_stocks]
    else:
        selected_symbols = all_symbols
    
    # Fetch data button
    if st.sidebar.button("🚀 Fetch & Analyze Data", type="primary"):
        
        with st.spinner(f"📈 Fetching data for {len(selected_symbols)} stocks..."):
            
            # Fetch data
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
            
            # Store in session state
            st.session_state['stock_data'] = all_data
            st.session_state['analysis_date'] = end_date
            st.session_state['start_date'] = start_date
    
    # Main analysis
    if 'stock_data' in st.session_state:
        
        all_data = st.session_state['stock_data']
        analysis_date = st.session_state['analysis_date']
        
        st.markdown(f"### 📅 Analysis for {analysis_date}")
        
        # Calculate metrics for all stocks
        stock_metrics = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (symbol, data) in enumerate(all_data.items()):
            
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(all_data)})")
            progress_bar.progress((i + 1) / len(all_data))
            
            if data.empty or len(data) < 50:
                continue
            
            try:
                df_with_indicators = calculate_technical_indicators(data)
                latest = df_with_indicators.iloc[-1]
                
                metrics = {
                    'symbol': symbol,
                    'price': latest['close'],
                    'change_pct': latest.get('price_change', 0),
                    'volume_ratio': latest.get('volume_ratio', 1),
                    'rsi': latest.get('rsi', 50),
                    'macd': latest.get('macd', 0),
                    'signal_type': latest.get('signal_type', 'NEUTRAL'),
                    'signal_strength': latest.get('signal_strength', 0),
                    'buy_signals': latest.get('buy_signals', 0),
                    'sell_signals': latest.get('sell_signals', 0),
                    'atr': latest.get('atr', 0),
                    'bb_position': (latest['close'] - latest.get('bb_lower', latest['close'])) / (latest.get('bb_upper', latest['close']) - latest.get('bb_lower', latest['close'])) if latest.get('bb_upper', 0) != latest.get('bb_lower', 0) else 0.5
                }
                
                stock_metrics.append(metrics)
                
            except Exception as e:
                st.warning(f"Error analyzing {symbol}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        if not stock_metrics:
            st.error("No stocks could be analyzed")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(stock_metrics)
        summary_df = summary_df.sort_values('signal_strength', ascending=False)
        
        # ===== MARKET OVERVIEW =====
        st.markdown("## 📊 Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_stocks = len(summary_df)
        strong_buy = len(summary_df[summary_df['signal_type'] == 'STRONG BUY'])
        buy_signals = len(summary_df[summary_df['signal_type'].str.contains('BUY')])
        sell_signals = len(summary_df[summary_df['signal_type'].str.contains('SELL')])
        avg_rsi = summary_df['rsi'].mean()
        
        col1.metric("Total Stocks", total_stocks)
        col2.metric("Strong Buy", strong_buy, delta=f"{strong_buy/total_stocks:.1%}")
        col3.metric("Buy Signals", buy_signals, delta=f"{buy_signals/total_stocks:.1%}")
        col4.metric("Sell Signals", sell_signals, delta=f"{sell_signals/total_stocks:.1%}")
        col5.metric("Avg RSI", f"{avg_rsi:.1f}")
        
        # ===== TOP OPPORTUNITIES TABLE =====
        st.markdown("## 🏆 Top Trading Opportunities")
        
        # Format the summary table
        display_df = summary_df.copy()
        display_df['Symbol'] = display_df['symbol'].str.replace('.NS', '')
        display_df['Price'] = display_df['price'].apply(lambda x: f"₹{x:.1f}")
        display_df['Change %'] = display_df['change_pct'].apply(lambda x: f"{x:+.1f}%")
        display_df['Volume'] = display_df['volume_ratio'].apply(lambda x: f"{x:.1f}x")
        display_df['RSI'] = display_df['rsi'].apply(lambda x: f"{x:.0f}")
        display_df['Signal'] = display_df['signal_type']
        display_df['Strength'] = display_df['signal_strength'].apply(lambda x: f"{x:+d}")
        
        # Color code signals
        def color_signal(val):
            if 'STRONG BUY' in val:
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif 'BUY' in val:
                return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold'
            elif 'SELL' in val:
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            else:
                return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        
        styled_df = display_df[['Symbol', 'Price', 'Change %', 'Signal', 'Strength', 'RSI', 'Volume']].style.applymap(
            color_signal, subset=['Signal']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # ===== INDIVIDUAL STOCK ANALYSIS =====
        st.markdown("## 📈 Individual Stock Analysis")
        
        # Chart controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stock selector
            stock_symbols = [s.replace('.NS', '') for s in summary_df['symbol'].tolist()]
            selected_stock_display = st.selectbox(
                "Select Stock for Detailed Analysis",
                options=stock_symbols,
                index=0
            )
        
        with col2:
            # Chart date filter
            chart_mode = st.selectbox(
                "Chart Time Range",
                options=["Single Day", "Multi-Day", "Full Range"],
                index=0,
                help="Choose how much data to show in the chart"
            )
            
            if chart_mode == "Single Day":
                # Date selector for single day view
                available_dates = []
                selected_stock_full = selected_stock_display + '.NS'
                if selected_stock_full in all_data:
                    sample_data = all_data[selected_stock_full]
                    if not sample_data.empty:
                        available_dates = sorted(sample_data.index.date.unique(), reverse=True)
                
                if available_dates:
                    chart_date = st.selectbox(
                        "Select Date",
                        options=available_dates,
                        index=0,
                        help="Choose specific date to analyze"
                    )
                else:
                    chart_date = analysis_date
        
        selected_stock = selected_stock_display + '.NS'
        
        if selected_stock in all_data:
            
            # Create comprehensive chart
            fig, metrics = create_comprehensive_chart(selected_stock, all_data[selected_stock], chart_date, chart_mode)
            
            if fig and metrics:
                
                # Display metrics
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("Price", f"₹{metrics['price']:.1f}", f"{metrics['change_pct']:+.1f}%")
                col2.metric("Signal", metrics['signal_type'])
                col3.metric("Strength", f"{metrics['signal_strength']:+d}", f"B:{metrics['buy_signals']} S:{metrics['sell_signals']}")
                col4.metric("RSI", f"{metrics['rsi']:.0f}")
                col5.metric("Volume", f"{metrics['volume_ratio']:.1f}x")
                col6.metric("BB Position", f"{metrics['bb_position']:.1%}")
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.markdown("### 🔍 Key Insights")
                
                insights = []
                
                if metrics['signal_type'] in ['STRONG BUY', 'BUY']:
                    insights.append("🟢 **Bullish signals detected** - Consider long positions")
                elif metrics['signal_type'] in ['STRONG SELL', 'SELL']:
                    insights.append("🔴 **Bearish signals detected** - Consider short positions or exit longs")
                
                if metrics['rsi'] < 30:
                    insights.append("📉 **RSI oversold** - Potential bounce opportunity")
                elif metrics['rsi'] > 70:
                    insights.append("📈 **RSI overbought** - Potential correction ahead")
                
                if metrics['volume_ratio'] > 2:
                    insights.append("📊 **High volume activity** - Strong institutional interest")
                
                if metrics['bb_position'] < 0.2:
                    insights.append("🎯 **Near lower Bollinger Band** - Potential support level")
                elif metrics['bb_position'] > 0.8:
                    insights.append("⚠️ **Near upper Bollinger Band** - Potential resistance level")
                
                for insight in insights:
                    st.markdown(insight)
            
            else:
                st.error(f"Unable to analyze {selected_stock}")

if __name__ == "__main__":
    main() 