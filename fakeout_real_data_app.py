"""
Streamlit App for Fakeout Detection with Real Data
Interactive platform for detecting fakeout reversals using real market data.
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

from fakeout_detector_integration import FakeoutDetectorIntegration, run_fakeout_analysis
from core.data_handler import DataHandler

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fakeout Detector - Real Data",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sidebar_config():
    """Create sidebar configuration for real data analysis."""
    st.sidebar.header("🎛️ Analysis Configuration")
    
    # Symbol selection
    st.sidebar.subheader("📊 Symbols")
    symbols_input = st.sidebar.text_area(
        "Enter symbols (one per line)",
        value="NIFTY\nBANKNIFTY",
        help="Enter symbols to analyze, one per line"
    )
    symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
    
    # Date range
    st.sidebar.subheader("📅 Date Range")
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )
    
    days_back = st.sidebar.slider(
        "Days to analyze",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to look back"
    )
    
    start_date = end_date - timedelta(days=days_back)
    
    # Time range
    st.sidebar.subheader("⏰ Time Range")
    start_time = st.sidebar.time_input(
        "Start Time", 
        value=datetime.strptime("09:15", "%H:%M").time()
    )
    end_time = st.sidebar.time_input(
        "End Time", 
        value=datetime.strptime("15:30", "%H:%M").time()
    )
    
    # Data interval
    st.sidebar.subheader("📊 Data Settings")
    interval = st.sidebar.selectbox(
        "Data Interval",
        ["minute", "5minute", "15minute", "30minute"],
        index=0,
        help="Candle interval for analysis"
    )
    
    # Level type
    level_type = st.sidebar.selectbox(
        "Level Type",
        ["pdh_pdl", "vwap", "support_resistance"],
        index=0,
        help="Type of levels to use for fakeout detection"
    )
    
    return symbols, start_date, end_date, start_time, end_time, interval, level_type

def create_detector_config():
    """Create detector configuration controls."""
    st.subheader("🔧 Detection Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wick_threshold = st.slider(
            "Wick Threshold (%)", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="Minimum wick size as percentage of price"
        )
        
        lookback_window = st.slider(
            "Lookback Window", 
            min_value=5, 
            max_value=100, 
            value=20, 
            step=5,
            help="Number of candles to look back for level calculation"
        )
        
        min_candles_between = st.slider(
            "Min Candles Between Signals", 
            min_value=1, 
            max_value=50, 
            value=10, 
            step=1,
            help="Minimum candles between consecutive signals"
        )
    
    with col2:
        atr_period = st.slider(
            "ATR Period", 
            min_value=5, 
            max_value=50, 
            value=14, 
            step=1,
            help="Period for Average True Range calculation"
        )
        
        sl_multiplier = st.slider(
            "Stop Loss ATR Multiplier", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            help="ATR multiplier for stop loss calculation"
        )
        
        tp_multiplier = st.slider(
            "Take Profit ATR Multiplier", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.5, 
            step=0.1,
            help="ATR multiplier for take profit calculation"
        )
    
    debug_mode = st.checkbox("Debug Mode", value=True, help="Enable detailed logging")
    
    return {
        'wick_threshold_pct': wick_threshold,
        'lookback_window': lookback_window,
        'min_candles_between_signals': min_candles_between,
        'atr_period': atr_period,
        'sl_multiplier': sl_multiplier,
        'tp_multiplier': tp_multiplier,
        'debug_mode': debug_mode,
        'level_tolerance_pct': 0.5
    }

def display_analysis_summary(results: Dict):
    """Display analysis summary with metrics."""
    summary = results['summary']
    
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Symbols", summary['total_symbols'])
    
    with col2:
        st.metric("Symbols with Signals", summary['symbols_with_signals'])
    
    with col3:
        st.metric("Total Signals", summary['total_signals'])
    
    with col4:
        if summary['total_signals'] > 0:
            st.metric("Avg Signals/Symbol", f"{summary['avg_signals_per_symbol']:.1f}")
        else:
            st.metric("Avg Signals/Symbol", "0")
    
    # Signal distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Long Signals", summary['total_long_signals'])
    
    with col2:
        st.metric("Short Signals", summary['total_short_signals'])
    
    # Additional metrics
    if summary['total_signals'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if summary['total_short_signals'] > 0:
                ratio = summary['long_short_ratio']
                st.metric("Long/Short Ratio", f"{ratio:.2f}")
            else:
                st.metric("Long/Short Ratio", "∞")
        
        with col2:
            st.metric("Total Data Points", summary['total_data_points'])

def display_top_signals(results: Dict):
    """Display top signals in a formatted table."""
    top_signals = results['top_signals']
    
    if not top_signals:
        st.warning("No signals detected with current parameters.")
        return
    
    st.subheader("🎯 Top Recent Signals")
    
    # Convert to DataFrame for display
    signals_data = []
    for signal in top_signals:
        signals_data.append({
            'Symbol': signal['symbol'],
            'Type': signal['signal_type'].replace('_', ' ').title(),
            'Time': signal['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'Entry': f"{signal['entry']:.2f}",
            'Stop Loss': f"{signal['stop_loss']:.2f}",
            'Take Profit': f"{signal['take_profit']:.2f}",
            'Level': f"{signal['level_value']:.2f}"
        })
    
    df_signals = pd.DataFrame(signals_data)
    
    # Color coding function
    def color_signal_type(val):
        if 'Long' in val:
            return 'background-color: #d4edda; color: #155724;'
        else:
            return 'background-color: #f8d7da; color: #721c24;'
    
    # Display with styling
    st.dataframe(
        df_signals.style.applymap(color_signal_type, subset=['Type']),
        use_container_width=True
    )

def plot_symbol_signals(symbol: str, analysis_result: Dict, integration: FakeoutDetectorIntegration):
    """Plot signals for a specific symbol."""
    if 'error' in analysis_result:
        st.warning(f"Cannot plot {symbol}: {analysis_result['error']}")
        return
    
    if not analysis_result['signals']:
        st.info(f"No signals detected for {symbol}")
        return
    
    st.subheader(f"📈 {symbol} - Signal Analysis")
    
    # Get the detector instance to access plotting
    detector = integration.detector
    
    # Show signal details
    signals = analysis_result['signals']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Signals", len(signals))
        st.metric("Long Signals", len([s for s in signals if s['signal_type'] == 'long_fakeout']))
    
    with col2:
        st.metric("Short Signals", len([s for s in signals if s['signal_type'] == 'short_fakeout']))
        if signals:
            avg_rr = np.mean([
                abs(s['take_profit'] - s['entry']) / abs(s['stop_loss'] - s['entry'])
                for s in signals
            ])
            st.metric("Avg Risk/Reward", f"{avg_rr:.2f}")
    
    # Show recent signals
    st.subheader("Recent Signals")
    for i, signal in enumerate(signals[:5]):
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
    
    # Create and display the chart
    st.subheader("📊 Price Chart with Signals")
    
    try:
        # Get the data for this symbol
        data_dict = integration.fetch_data_for_analysis(
            [symbol], 
            analysis_result['date_range']['start'], 
            analysis_result['date_range']['end'], 
            analysis_result['interval']
        )
        
        if symbol in data_dict and not data_dict[symbol].empty:
            df = data_dict[symbol]
            vwap = integration.calculate_vwap_for_data(df)
            
            # Create the plot
            fig = detector.plot_signals(df, signals, vwap, analysis_result['level_type'])
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, height=600)
                
                # Add chart controls
                st.markdown("**Chart Features:**")
                st.markdown("""
                - **Candlesticks**: Price action with green/red colors
                - **VWAP**: Purple line showing volume-weighted average price
                - **PDH/PDL**: Red/green dashed lines for previous day high/low
                - **Signals**: Triangle markers for entry points (▲ long, ▼ short)
                - **Stop Loss**: Red X markers
                - **Take Profit**: Green star markers
                - **Volume**: Gray bars at the bottom
                """)
            else:
                st.warning("Chart creation failed")
        else:
            st.warning("No data available for charting")
            
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.info("Chart plotting requires additional data access")

def main():
    """Main Streamlit app for real data analysis."""
    st.set_page_config(
        page_title="Fakeout Detector - Real Data",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .signal-card {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid;
    }
    .long-signal {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .short-signal {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Fakeout Detector - Real Data Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data configuration
        st.subheader("📊 Data Settings")
        symbols_input = st.text_area(
            "Symbols (one per line)",
            value="NIFTY\nBANKNIFTY",
            help="Enter symbols to analyze, one per line"
        )
        
        symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=7),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        interval = st.selectbox(
            "Interval",
            options=["minute", "5minute", "15minute"],
            index=0,
            help="Data interval for analysis"
        )
        
        level_type = st.selectbox(
            "Level Type",
            options=["pdh_pdl", "vwap", "support_resistance"],
            index=0,
            help="Type of levels to use for fakeout detection"
        )
        
        # Analysis button
        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Signals", type="primary", use_container_width=True)
    
    # Main content
    if analyze_button and symbols:
        st.header("📈 Analysis Results")
        
        # Create integration
        try:
            integration = FakeoutDetectorIntegration()
            
            # Get detector configuration
            detector_config = create_detector_config()
            
            # Run analysis
            with st.spinner("Analyzing signals..."):
                # Convert dates to string format
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                results = integration.analyze_multiple_symbols(
                    symbols, 
                    start_date_str, 
                    end_date_str, 
                    level_type, 
                    interval, 
                    detector_config
                )
            
            if results:
                # Create summary and top signals
                try:
                    summary = integration.create_analysis_summary(results)
                    top_signals = integration.get_top_signals(results)
                    
                    # Create the complete results structure
                    complete_results = {
                        'analysis_results': results,
                        'top_signals': top_signals,
                        'summary': summary
                    }
                    
                    # Display summary
                    display_analysis_summary(complete_results)
                    
                    # Display top signals
                    display_top_signals(complete_results)
                    
                    # Individual symbol analysis
                    st.header("📊 Individual Symbol Analysis")
                    
                    for symbol in symbols:
                        if symbol in results and 'error' not in results[symbol]:
                            plot_symbol_signals(symbol, results[symbol], integration)
                            st.markdown("---")
                        elif symbol in results and 'error' in results[symbol]:
                            st.error(f"Error analyzing {symbol}: {results[symbol]['error']}")
                        else:
                            st.warning(f"No data available for {symbol}")
                            
                except Exception as e:
                    st.error(f"Error creating summary: {str(e)}")
                    # Fallback: show raw results
                    st.subheader("📊 Raw Analysis Results")
                    for symbol, result in results.items():
                        if 'error' in result:
                            st.error(f"{symbol}: {result['error']}")
                        elif 'signals' in result:
                            st.success(f"{symbol}: {len(result['signals'])} signals found")
                        else:
                            st.warning(f"{symbol}: No data")
            else:
                st.error("No analysis results available")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please check your data handler configuration and try again.")
    
    elif analyze_button and not symbols:
        st.warning("Please enter at least one symbol to analyze.")
    
    else:
        # Show instructions
        st.info("""
        **Instructions:**
        1. Enter symbols to analyze in the sidebar
        2. Set your date range and interval
        3. Choose level type for fakeout detection
        4. Configure detection parameters
        5. Click "Analyze Signals" to start
        """)
        
        # Show sample configuration
        st.subheader("🔧 Sample Configuration")
        st.json({
            "symbols": ["NIFTY", "BANKNIFTY"],
            "date_range": "Last 7 days",
            "interval": "minute",
            "level_type": "pdh_pdl",
            "detection_params": {
                "wick_threshold_pct": 0.2,
                "lookback_window": 20,
                "min_candles_between_signals": 10
            }
        })

if __name__ == "__main__":
    main() 