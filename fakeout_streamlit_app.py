"""
Streamlit App for Fakeout Detection System
Interactive platform for detecting and analyzing fakeout reversals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple

from fakeout_detector import FakeoutDetector, create_sample_data

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fakeout Detector",
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
    }
    .signal-card {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .long-signal { border-left: 4px solid #2ca02c; }
    .short-signal { border-left: 4px solid #d62728; }
</style>
""", unsafe_allow_html=True)

def create_sample_data_with_controls():
    """Create sample data with user controls."""
    st.sidebar.subheader("📊 Data Configuration")
    
    # Date range
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now().date() - timedelta(days=7),
        max_value=datetime.now().date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )
    
    # Time range
    start_time = st.sidebar.time_input("Start Time", value=datetime.strptime("09:15", "%H:%M").time())
    end_time = st.sidebar.time_input("End Time", value=datetime.strptime("15:30", "%H:%M").time())
    
    # Candle frequency
    candle_freq = st.sidebar.selectbox(
        "Candle Frequency",
        ["1min", "5min", "15min", "30min"],
        index=1
    )
    
    # Base price and volatility
    base_price = st.sidebar.number_input("Base Price", value=18500.0, step=100.0)
    volatility = st.sidebar.slider("Volatility", 1.0, 20.0, 8.0, 0.5)
    
    # Fakeout frequency
    fakeout_freq = st.sidebar.slider("Fakeout Frequency (every N candles)", 10, 50, 25)
    
    # Generate data
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)
    
    # Convert frequency to pandas format
    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min"}
    freq = freq_map[candle_freq]
    
    dates = pd.date_range(start_datetime, end_datetime, freq=freq)
    
    # Generate realistic price data
    np.random.seed(42)
    data = []
    
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            # Add trend and noise
            trend = np.sin(i / 100) * 10
            noise = np.random.normal(0, volatility)
            price = data[-1]['close'] + trend + noise
        
        # Create OHLCV
        open_price = price
        high_price = price + abs(np.random.normal(0, volatility * 1.2))
        low_price = price - abs(np.random.normal(0, volatility * 1.2))
        close_price = price + np.random.normal(0, volatility * 0.8)
        volume = np.random.randint(50000, 200000)
        
        # Add fakeout patterns
        if i % fakeout_freq == fakeout_freq // 2:
            if np.random.choice([True, False]):
                # Resistance fakeout
                high_price += volatility * 2
                close_price = price - volatility * 1.5
            else:
                # Support fakeout
                low_price -= volatility * 2
                close_price = price + volatility * 1.5
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate VWAP
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df, vwap

def create_detector_config():
    """Create detector configuration with sidebar controls."""
    st.sidebar.subheader("⚙️ Detection Parameters")
    
    config = {}
    
    # Level detection parameters
    st.sidebar.markdown("**Level Detection**")
    config['wick_threshold_pct'] = st.sidebar.slider(
        "Wick Threshold (%)", 0.1, 2.0, 0.3, 0.1,
        help="Minimum wick percentage for breakout candle"
    )
    
    config['confirmation_threshold_pct'] = st.sidebar.slider(
        "Confirmation Threshold (%)", 0.1, 2.0, 0.5, 0.1,
        help="Minimum reversal percentage for confirmation"
    )
    
    config['level_tolerance_pct'] = st.sidebar.slider(
        "Level Tolerance (%)", 0.05, 1.0, 0.1, 0.05,
        help="Tolerance around level for breakout detection"
    )
    
    # Signal parameters
    st.sidebar.markdown("**Signal Parameters**")
    config['lookback_window'] = st.sidebar.slider(
        "Lookback Window", 5, 50, 20,
        help="Candles to look back for level calculation"
    )
    
    config['min_candles_between_signals'] = st.sidebar.slider(
        "Min Candles Between Signals", 1, 20, 10,
        help="Minimum candles between consecutive signals"
    )
    
    # Risk management
    st.sidebar.markdown("**Risk Management**")
    config['sl_atr_multiplier'] = st.sidebar.slider(
        "Stop Loss (ATR Multiplier)", 0.5, 3.0, 1.5, 0.1,
        help="Stop loss as ATR multiplier"
    )
    
    config['tp_atr_multiplier'] = st.sidebar.slider(
        "Take Profit (ATR Multiplier)", 1.0, 5.0, 2.0, 0.1,
        help="Take profit as ATR multiplier"
    )
    
    config['atr_period'] = st.sidebar.slider(
        "ATR Period", 5, 30, 14,
        help="Period for ATR calculation"
    )
    
    # Debug settings
    st.sidebar.markdown("**Debug Settings**")
    config['debug_mode'] = st.sidebar.checkbox("Debug Mode", value=True)
    config['log_level'] = st.sidebar.selectbox("Log Level", ["INFO", "DEBUG", "WARNING"], index=0)
    
    return config

def plot_signals_interactive(df: pd.DataFrame, signals: List[Dict], vwap: pd.Series, 
                           level_type: str, detector: FakeoutDetector):
    """Create interactive plot with signals."""
    
    # Calculate levels for plotting
    df_with_levels = detector.calculate_key_levels(df, level_type)
    
    # Create subplot with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Action & Signals', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#2ca02c',
        decreasing_line_color='#d62728'
    ), row=1, col=1)
    
    # Add VWAP
    fig.add_trace(go.Scatter(
        x=df.index,
        y=vwap,
        mode='lines',
        name='VWAP',
        line=dict(color='purple', width=2),
        opacity=0.8
    ), row=1, col=1)
    
    # Add level lines
    if level_type == 'pdh_pdl':
        if 'pdh' in df_with_levels.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df_with_levels['pdh'],
                mode='lines',
                name='PDH',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.6
            ), row=1, col=1)
        if 'pdl' in df_with_levels.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df_with_levels['pdl'],
                mode='lines',
                name='PDL',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.6
            ), row=1, col=1)
    
    # Add signal markers
    for signal in signals:
        color = '#2ca02c' if signal['signal_type'] == 'long_fakeout' else '#d62728'
        symbol = 'triangle-up' if signal['signal_type'] == 'long_fakeout' else 'triangle-down'
        
        # Entry point
        fig.add_trace(go.Scatter(
            x=[signal['timestamp']],
            y=[signal['entry']],
            mode='markers',
            marker=dict(symbol=symbol, size=15, color=color, line=dict(width=2, color='white')),
            name=f"{signal['signal_type']} Entry",
            showlegend=False
        ), row=1, col=1)
        
        # Stop loss
        fig.add_trace(go.Scatter(
            x=[signal['timestamp']],
            y=[signal['stop_loss']],
            mode='markers',
            marker=dict(symbol='x', size=10, color='red', line=dict(width=1)),
            name='Stop Loss',
            showlegend=False
        ), row=1, col=1)
        
        # Take profit
        fig.add_trace(go.Scatter(
            x=[signal['timestamp']],
            y=[signal['take_profit']],
            mode='markers',
            marker=dict(symbol='star', size=10, color='green', line=dict(width=1)),
            name='Take Profit',
            showlegend=False
        ), row=1, col=1)
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='rgba(100, 100, 100, 0.3)'
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'Fakeout Signals - {level_type.upper()}',
        xaxis_title='Time',
        yaxis_title='Price',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def display_signals_table(signals: List[Dict]):
    """Display signals in a formatted table."""
    if not signals:
        st.warning("No signals detected with current parameters.")
        return
    
    # Convert to DataFrame
    df_signals = pd.DataFrame(signals)
    
    # Format timestamp
    df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Add color coding
    def color_signal_type(val):
        if val == 'long_fakeout':
            return 'background-color: #d4edda; color: #155724;'
        else:
            return 'background-color: #f8d7da; color: #721c24;'
    
    # Display with styling
    st.dataframe(
        df_signals[['timestamp', 'signal_type', 'entry', 'stop_loss', 'take_profit', 'level_value']].style
        .applymap(color_signal_type, subset=['signal_type'])
        .format({
            'entry': '{:.2f}',
            'stop_loss': '{:.2f}',
            'take_profit': '{:.2f}',
            'level_value': '{:.2f}'
        }),
        use_container_width=True
    )

def display_metrics(signals: List[Dict]):
    """Display key metrics about detected signals."""
    if not signals:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(signals))
    
    with col2:
        long_signals = len([s for s in signals if s['signal_type'] == 'long_fakeout'])
        st.metric("Long Signals", long_signals)
    
    with col3:
        short_signals = len([s for s in signals if s['signal_type'] == 'short_fakeout'])
        st.metric("Short Signals", short_signals)
    
    with col4:
        if signals:
            avg_risk_reward = np.mean([
                abs(s['take_profit'] - s['entry']) / abs(s['stop_loss'] - s['entry'])
                for s in signals
            ])
            st.metric("Avg Risk/Reward", f"{avg_risk_reward:.2f}")

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Fakeout Detector</h1>', unsafe_allow_html=True)
    st.markdown("Detect intraday fakeout reversals around key levels with interactive analysis.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Data configuration
        df, vwap = create_sample_data_with_controls()
        
        # Detector configuration
        config = create_detector_config()
        
        # Level type selection
        st.subheader("📊 Level Type")
        level_type = st.selectbox(
            "Select Level Type",
            ["pdh_pdl", "vwap", "support_resistance"],
            help="Type of levels to use for fakeout detection"
        )
        
        # Detection button
        st.subheader("🚀 Detection")
        detect_button = st.button("🔍 Detect Fakeouts", type="primary")
    
    # Main content
    if detect_button:
        # Initialize detector
        detector = FakeoutDetector(config)
        
        # Detect signals
        with st.spinner("Detecting fakeout signals..."):
            signals = detector.detect_fakeout_signals(df, vwap, level_type)
        
        # Display results
        st.success(f"✅ Detection complete! Found {len(signals)} signals.")
        
        # Metrics
        display_metrics(signals)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Chart", "📋 Signals Table", "📈 Analysis"])
        
        with tab1:
            st.subheader("Interactive Price Chart with Signals")
            fig = plot_signals_interactive(df, signals, vwap, level_type, detector)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Detected Signals")
            display_signals_table(signals)
        
        with tab3:
            st.subheader("Signal Analysis")
            
            if signals:
                # Signal distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    signal_types = [s['signal_type'] for s in signals]
                    fig_pie = px.pie(
                        values=[signal_types.count('long_fakeout'), signal_types.count('short_fakeout')],
                        names=['Long Fakeouts', 'Short Fakeouts'],
                        title="Signal Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Risk/Reward analysis
                    risk_rewards = [
                        abs(s['take_profit'] - s['entry']) / abs(s['stop_loss'] - s['entry'])
                        for s in signals
                    ]
                    
                    fig_hist = px.histogram(
                        x=risk_rewards,
                        title="Risk/Reward Distribution",
                        labels={'x': 'Risk/Reward Ratio', 'y': 'Count'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Signal details
                st.subheader("Signal Details")
                for i, signal in enumerate(signals[:5]):  # Show first 5 signals
                    signal_class = "long-signal" if signal['signal_type'] == 'long_fakeout' else "short-signal"
                    st.markdown(f"""
                    <div class="signal-card {signal_class}">
                        <strong>Signal {i+1}:</strong> {signal['signal_type'].replace('_', ' ').title()}<br>
                        <strong>Time:</strong> {signal['timestamp']}<br>
                        <strong>Entry:</strong> {signal['entry']:.2f} | 
                        <strong>SL:</strong> {signal['stop_loss']:.2f} | 
                        <strong>TP:</strong> {signal['take_profit']:.2f}<br>
                        <strong>Level:</strong> {signal['level_value']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Show data preview when not detecting
        st.subheader("📊 Data Preview")
        st.write(f"Generated {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Show price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=vwap,
            mode='lines',
            name='VWAP',
            line=dict(color='purple', width=2)
        ))
        
        fig.update_layout(
            title="Sample Data Preview",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Click '🔍 Detect Fakeouts' in the sidebar to start detection!")

if __name__ == "__main__":
    main() 