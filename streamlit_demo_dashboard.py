#!/usr/bin/env python3
"""
🚀 STREAMLIT TRADING DASHBOARD - API DATA WITH TIMELINE
======================================================

Real-time trading dashboard using Zerodha API data with date and stock selection.

Run with: streamlit run streamlit_demo_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import sys
import os
import warnings
import time as time_module
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

# Page configuration
st.set_page_config(
    page_title="🎯 Trading Dashboard with Timeline",
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

def get_nifty50_symbols():
    """Get Nifty 50 stock symbols."""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'ASIANPAINT.NS',
        'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
        'ULTRACEMCO.NS', 'TITAN.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'HCLTECH.NS',
        'POWERGRID.NS', 'BAJFINANCE.NS', 'NTPC.NS', 'TECHM.NS', 'ONGC.NS',
        'TATAMOTORS.NS', 'COALINDIA.NS', 'BAJAJFINSV.NS', 'DIVISLAB.NS', 'GRASIM.NS',
        'DRREDDY.NS', 'CIPLA.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'ADANIPORTS.NS',
        'TATASTEEL.NS', 'HINDALCO.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
        'UPL.NS', 'BPCL.NS', 'APOLLOHOSP.NS', 'TATACONSUM.NS', 'INDUSINDBK.NS',
        'SHREECEM.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'ADANIENT.NS', 'VEDL.NS'
    ]

def load_api_data(symbols, start_date, end_date):
    """Load data from Zerodha API."""
    try:
        # Import data handler
        from core.data_handler import DataHandler
        
        # Initialize data handler
        data_handler = DataHandler()
        
        # Fetch data from API using the correct method
        api_data = data_handler.get_historical_data(
            symbols=symbols,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d'),
            interval="minute",
            refresh_cache=True  # Always fetch fresh data
        )
        
        if api_data is not None and len(api_data) > 0:
            # Handle both single DataFrame and dict of DataFrames
            if isinstance(api_data, dict):
                st.success(f"✅ Fetched API data for {len(api_data)} stocks")
            else:
                st.success(f"✅ Fetched API data for 1 stock")
            return api_data
        else:
            st.error("❌ Failed to fetch data from API")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading API data: {e}")
        return None

# Order book functionality removed - not available in current setup
# def get_order_book(symbol):
#     """Get order book data for a symbol."""
#     pass

def calculate_technical_indicators(df):
    """Calculate comprehensive intraday technical indicators."""
    data = df.copy()
    
    try:
        # Basic Moving Averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # Exponential Moving Averages
        data['ema_5'] = data['close'].ewm(span=5).mean()
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        
        # VWAP (Volume Weighted Average Price)
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Daily VWAP reset (for intraday)
        try:
            # Create a temporary date column for grouping
            temp_dates = pd.Series(data.index.date, index=data.index)
            data['daily_vwap'] = data.groupby(temp_dates).apply(
                lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
            ).droplevel(0)
        except Exception:
            # Fallback to regular VWAP if daily reset fails
            data['daily_vwap'] = data['vwap']
        
        # RSI (Multiple timeframes)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_9'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(9).mean() / (-delta.where(delta < 0, 0)).rolling(9).mean()))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Stochastic Oscillator
        low_14 = data['low'].rolling(14).min()
        high_14 = data['high'].rolling(14).max()
        data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands (Multiple periods)
        data['bb_middle_20'] = data['close'].rolling(20).mean()
        bb_std_20 = data['close'].rolling(20).std()
        data['bb_upper_20'] = data['bb_middle_20'] + (bb_std_20 * 2)
        data['bb_lower_20'] = data['bb_middle_20'] - (bb_std_20 * 2)
        
        # Bollinger Bands 10-period for scalping
        data['bb_middle_10'] = data['close'].rolling(10).mean()
        bb_std_10 = data['close'].rolling(10).std()
        data['bb_upper_10'] = data['bb_middle_10'] + (bb_std_10 * 2)
        data['bb_lower_10'] = data['bb_middle_10'] - (bb_std_10 * 2)
        
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift())
        low_close_prev = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = true_range.rolling(14).mean()
        data['atr_5'] = true_range.rolling(5).mean()
        
        # Volume Indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['volume_ema'] = data['volume'].ewm(span=10).mean()
        
        # OBV (On Balance Volume)
        data['obv'] = (data['volume'] * ((data['close'] > data['close'].shift()).astype(int) * 2 - 1)).cumsum()
        
        # Money Flow Index
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        raw_money_flow = typical_price * data['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        money_ratio = positive_flow / negative_flow
        data['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Williams %R
        highest_high = data['high'].rolling(14).max()
        lowest_low = data['low'].rolling(14).min()
        data['williams_r'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        data['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # Pivot Points (Daily)
        data['pivot'] = (data['high'].shift() + data['low'].shift() + data['close'].shift()) / 3
        data['r1'] = 2 * data['pivot'] - data['low'].shift()
        data['s1'] = 2 * data['pivot'] - data['high'].shift()
        data['r2'] = data['pivot'] + (data['high'].shift() - data['low'].shift())
        data['s2'] = data['pivot'] - (data['high'].shift() - data['low'].shift())
        
        # Price Action Indicators
        data['price_change'] = data['close'].pct_change() * 100
        data['high_low_pct'] = ((data['high'] - data['low']) / data['close']) * 100
        data['body_size'] = abs(data['close'] - data['open'])
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        
        # Momentum Indicators
        data['momentum'] = data['close'] / data['close'].shift(10) * 100
        data['roc'] = ((data['close'] - data['close'].shift(12)) / data['close'].shift(12)) * 100
        
        # Trend Strength
        data['adx'] = calculate_adx(data)
        
        # Support and Resistance levels
        data = calculate_support_resistance(data)
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

def calculate_adx(df):
    """Calculate Average Directional Index."""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(14).mean()
        dm_plus_smooth = dm_plus.rolling(14).mean()
        dm_minus_smooth = dm_minus.rolling(14).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()
        
        return adx
    except:
        return pd.Series(index=df.index, dtype=float)

def calculate_support_resistance(df):
    """Calculate dynamic support and resistance levels."""
    try:
        # Local highs and lows
        df['local_high'] = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])]
        df['local_low'] = df['low'][(df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])]
        
        # Fill forward for plotting
        df['resistance'] = df['local_high'].ffill()
        df['support'] = df['local_low'].ffill()
        
        return df
    except:
        return df

def calculate_intraday_strategies(df):
    """Calculate intraday trading strategies and signals."""
    data = df.copy()
    
    try:
        # 1. Scalping Strategy (EMA 5/9 crossover)
        data['scalp_signal'] = 0
        data.loc[(data['ema_5'] > data['ema_9']) & (data['ema_5'].shift() <= data['ema_9'].shift()), 'scalp_signal'] = 1  # Buy
        data.loc[(data['ema_5'] < data['ema_9']) & (data['ema_5'].shift() >= data['ema_9'].shift()), 'scalp_signal'] = -1  # Sell
        
        # 2. Mean Reversion Strategy (Bollinger Bands)
        data['mean_revert_signal'] = 0
        data.loc[(data['close'] < data['bb_lower_20']) & (data['rsi'] < 30), 'mean_revert_signal'] = 1  # Oversold
        data.loc[(data['close'] > data['bb_upper_20']) & (data['rsi'] > 70), 'mean_revert_signal'] = -1  # Overbought
        
        # 3. Breakout Strategy (ATR-based)
        data['breakout_signal'] = 0
        breakout_threshold = data['atr'] * 1.5
        data.loc[(data['close'] > data['close'].shift() + breakout_threshold) & (data['volume'] > data['volume_sma'] * 1.5), 'breakout_signal'] = 1
        data.loc[(data['close'] < data['close'].shift() - breakout_threshold) & (data['volume'] > data['volume_sma'] * 1.5), 'breakout_signal'] = -1
        
        # 4. VWAP Strategy
        data['vwap_signal'] = 0
        data.loc[(data['close'] > data['vwap']) & (data['close'].shift() <= data['vwap'].shift()), 'vwap_signal'] = 1
        data.loc[(data['close'] < data['vwap']) & (data['close'].shift() >= data['vwap'].shift()), 'vwap_signal'] = -1
        
        # 5. RSI Divergence Strategy
        data['rsi_div_signal'] = 0
        # Simple RSI overbought/oversold
        data.loc[(data['rsi'] < 30) & (data['rsi'].shift() >= 30), 'rsi_div_signal'] = 1
        data.loc[(data['rsi'] > 70) & (data['rsi'].shift() <= 70), 'rsi_div_signal'] = -1
        
        # 6. MACD Strategy
        data['macd_signal_trade'] = 0
        data.loc[(data['macd'] > data['macd_signal']) & (data['macd'].shift() <= data['macd_signal'].shift()), 'macd_signal_trade'] = 1
        data.loc[(data['macd'] < data['macd_signal']) & (data['macd'].shift() >= data['macd_signal'].shift()), 'macd_signal_trade'] = -1
        
        # 7. Stochastic Strategy
        data['stoch_signal'] = 0
        data.loc[(data['stoch_k'] < 20) & (data['stoch_k'] > data['stoch_d']), 'stoch_signal'] = 1
        data.loc[(data['stoch_k'] > 80) & (data['stoch_k'] < data['stoch_d']), 'stoch_signal'] = -1
        
        # 8. Volume Spike Strategy
        data['volume_spike_signal'] = 0
        volume_spike = data['volume'] > data['volume_sma'] * 2
        price_up = data['close'] > data['close'].shift()
        price_down = data['close'] < data['close'].shift()
        data.loc[volume_spike & price_up, 'volume_spike_signal'] = 1
        data.loc[volume_spike & price_down, 'volume_spike_signal'] = -1
        
        # 9. Support/Resistance Strategy
        data['sr_signal'] = 0
        near_support = abs(data['close'] - data['support']) / data['close'] < 0.005
        near_resistance = abs(data['close'] - data['resistance']) / data['close'] < 0.005
        data.loc[near_support & (data['rsi'] < 40), 'sr_signal'] = 1
        data.loc[near_resistance & (data['rsi'] > 60), 'sr_signal'] = -1
        
        # 10. Composite Signal (Weighted combination)
        signals = ['scalp_signal', 'mean_revert_signal', 'breakout_signal', 'vwap_signal', 
                  'rsi_div_signal', 'macd_signal_trade', 'stoch_signal', 'volume_spike_signal', 'sr_signal']
        
        # Weights for different strategies
        weights = [0.15, 0.12, 0.15, 0.12, 0.10, 0.12, 0.08, 0.08, 0.08]
        
        data['composite_signal'] = sum(data[signal] * weight for signal, weight in zip(signals, weights))
        
        # Strong signals (threshold-based)
        data['strong_buy'] = data['composite_signal'] > 0.3
        data['strong_sell'] = data['composite_signal'] < -0.3
        data['weak_buy'] = (data['composite_signal'] > 0.1) & (data['composite_signal'] <= 0.3)
        data['weak_sell'] = (data['composite_signal'] < -0.1) & (data['composite_signal'] >= -0.3)
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating strategies: {e}")
        return data

# Order book display functionality removed - not available
# def display_order_book(symbol):
#     """Display order book in sidebar."""
#     st.sidebar.warning("Order book data not available")

def create_strategy_plots(data, selected_time):
    """Create separate plots for different strategy categories."""
    
    # Filter data up to selected time
    if selected_time:
        current_data = data[data.index <= selected_time].copy()
    else:
        current_data = data.copy()
    
    if current_data.empty or len(current_data) < 5:
        st.warning("Insufficient data for strategy plots")
        return
    
    # 1. Scalping Strategy Plot
    st.subheader("🎯 Scalping Strategy (EMA 5/9 Crossover)")
    fig_scalp = go.Figure()
    
    # Price and EMAs
    fig_scalp.add_trace(go.Scatter(x=current_data.index, y=current_data['close'], 
                                   name='Close', line=dict(color='black', width=2)))
    
    # Check if EMA columns exist
    if 'ema_5' in current_data.columns:
        fig_scalp.add_trace(go.Scatter(x=current_data.index, y=current_data['ema_5'], 
                                       name='EMA 5', line=dict(color='blue')))
    if 'ema_9' in current_data.columns:
        fig_scalp.add_trace(go.Scatter(x=current_data.index, y=current_data['ema_9'], 
                                       name='EMA 9', line=dict(color='red')))
    
    # Buy/Sell signals
    if 'scalp_signal' in current_data.columns:
        buy_signals = current_data[current_data['scalp_signal'] == 1]
        sell_signals = current_data[current_data['scalp_signal'] == -1]
    else:
        buy_signals = pd.DataFrame()
        sell_signals = pd.DataFrame()
    
    if not buy_signals.empty:
        fig_scalp.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                                       mode='markers', name='BUY', 
                                       marker=dict(color='green', size=10, symbol='triangle-up')))
    if not sell_signals.empty:
        fig_scalp.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                                       mode='markers', name='SELL', 
                                       marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig_scalp.update_layout(height=400, title="Scalping Strategy Signals")
    st.plotly_chart(fig_scalp, use_container_width=True)
    
    # 2. Mean Reversion Strategy Plot
    st.subheader("🔄 Mean Reversion Strategy (Bollinger Bands + RSI)")
    fig_mr = go.Figure()
    
    # Price and Bollinger Bands
    fig_mr.add_trace(go.Scatter(x=current_data.index, y=current_data['close'], 
                                name='Close', line=dict(color='black', width=2)))
    
    # Check if Bollinger Bands columns exist
    if 'bb_upper_20' in current_data.columns:
        fig_mr.add_trace(go.Scatter(x=current_data.index, y=current_data['bb_upper_20'], 
                                    name='BB Upper', line=dict(color='red', dash='dash')))
        fig_mr.add_trace(go.Scatter(x=current_data.index, y=current_data['bb_middle_20'], 
                                    name='BB Middle', line=dict(color='blue')))
        fig_mr.add_trace(go.Scatter(x=current_data.index, y=current_data['bb_lower_20'], 
                                    name='BB Lower', line=dict(color='green', dash='dash')))
    
    # Mean reversion signals
    if 'mean_revert_signal' in current_data.columns:
        mr_buy_signals = current_data[current_data['mean_revert_signal'] == 1]
        mr_sell_signals = current_data[current_data['mean_revert_signal'] == -1]
    else:
        mr_buy_signals = pd.DataFrame()
        mr_sell_signals = pd.DataFrame()
    
    if not mr_buy_signals.empty:
        fig_mr.add_trace(go.Scatter(x=mr_buy_signals.index, y=mr_buy_signals['close'],
                                    mode='markers', name='OVERSOLD BUY', 
                                    marker=dict(color='green', size=12, symbol='circle')))
    if not mr_sell_signals.empty:
        fig_mr.add_trace(go.Scatter(x=mr_sell_signals.index, y=mr_sell_signals['close'],
                                    mode='markers', name='OVERBOUGHT SELL', 
                                    marker=dict(color='red', size=12, symbol='circle')))
    
    fig_mr.update_layout(height=400, title="Mean Reversion Strategy Signals")
    st.plotly_chart(fig_mr, use_container_width=True)
    
    # 3. VWAP Strategy Plot
    st.subheader("📊 VWAP Strategy")
    fig_vwap = go.Figure()
    
    fig_vwap.add_trace(go.Scatter(x=current_data.index, y=current_data['close'], 
                                  name='Close', line=dict(color='black', width=2)))
    
    # Check if VWAP column exists
    if 'vwap' in current_data.columns:
        fig_vwap.add_trace(go.Scatter(x=current_data.index, y=current_data['vwap'], 
                                      name='VWAP', line=dict(color='purple', width=2)))
    
    # VWAP signals
    if 'vwap_signal' in current_data.columns:
        vwap_buy_signals = current_data[current_data['vwap_signal'] == 1]
        vwap_sell_signals = current_data[current_data['vwap_signal'] == -1]
    else:
        vwap_buy_signals = pd.DataFrame()
        vwap_sell_signals = pd.DataFrame()
    
    if not vwap_buy_signals.empty:
        fig_vwap.add_trace(go.Scatter(x=vwap_buy_signals.index, y=vwap_buy_signals['close'],
                                      mode='markers', name='VWAP BUY', 
                                      marker=dict(color='green', size=10, symbol='diamond')))
    if not vwap_sell_signals.empty:
        fig_vwap.add_trace(go.Scatter(x=vwap_sell_signals.index, y=vwap_sell_signals['close'],
                                      mode='markers', name='VWAP SELL', 
                                      marker=dict(color='red', size=10, symbol='diamond')))
    
    fig_vwap.update_layout(height=400, title="VWAP Strategy Signals")
    st.plotly_chart(fig_vwap, use_container_width=True)
    
    # 4. Volume Analysis Plot
    st.subheader("📈 Volume Analysis")
    fig_vol = go.Figure()
    
    # Volume bars
    if 'volume_sma' in current_data.columns:
        colors = ['green' if vol > sma else 'red' for vol, sma in zip(current_data['volume'], current_data['volume_sma'])]
        fig_vol.add_trace(go.Bar(x=current_data.index, y=current_data['volume'], 
                                 name='Volume', marker_color=colors))
        fig_vol.add_trace(go.Scatter(x=current_data.index, y=current_data['volume_sma'], 
                                     name='Volume SMA', line=dict(color='blue')))
    else:
        fig_vol.add_trace(go.Bar(x=current_data.index, y=current_data['volume'], 
                                 name='Volume', marker_color='blue'))
    
    # Volume spike signals
    if 'volume_spike_signal' in current_data.columns:
        vol_spike_buy = current_data[current_data['volume_spike_signal'] == 1]
        vol_spike_sell = current_data[current_data['volume_spike_signal'] == -1]
    else:
        vol_spike_buy = pd.DataFrame()
        vol_spike_sell = pd.DataFrame()
    
    if not vol_spike_buy.empty:
        fig_vol.add_trace(go.Scatter(x=vol_spike_buy.index, y=vol_spike_buy['volume'],
                                     mode='markers', name='VOLUME SPIKE BUY', 
                                     marker=dict(color='green', size=12, symbol='star')))
    if not vol_spike_sell.empty:
        fig_vol.add_trace(go.Scatter(x=vol_spike_sell.index, y=vol_spike_sell['volume'],
                                     mode='markers', name='VOLUME SPIKE SELL', 
                                     marker=dict(color='red', size=12, symbol='star')))
    
    fig_vol.update_layout(height=400, title="Volume Analysis with Spike Signals")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # 5. Composite Signal Heatmap
    st.subheader("🎯 Composite Strategy Signals")
    
    # Create signal strength visualization - only include existing columns
    available_signals = []
    signal_columns = ['scalp_signal', 'mean_revert_signal', 'breakout_signal', 
                     'vwap_signal', 'rsi_div_signal', 'macd_signal_trade', 
                     'stoch_signal', 'volume_spike_signal', 'sr_signal', 'composite_signal']
    
    for col in signal_columns:
        if col in current_data.columns:
            available_signals.append(col)
    
    if available_signals:
        signal_data = current_data[available_signals].tail(50)
    else:
        st.warning("No strategy signals available for heatmap")
        return
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=signal_data.T.values,
        x=signal_data.index,
        y=signal_data.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=signal_data.T.values,
        texttemplate="%{text:.2f}",
        textfont={"size":10}
    ))
    
    fig_heatmap.update_layout(
        title="Strategy Signals Heatmap (Last 50 periods)",
        height=400,
        xaxis_title="Time",
        yaxis_title="Strategy"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

def get_indicator_status(value, indicator_type):
    """Get indicator status and color coding with trading signals."""
    if pd.isna(value):
        return 'N/A', 'neutral', 'HOLD'
        
    if indicator_type == 'rsi':
        if value < 30:
            return 'OVERSOLD', 'good', 'BUY'
        elif value > 70:
            return 'OVERBOUGHT', 'bad', 'SELL'
        elif value < 40:
            return 'BULLISH ZONE', 'good', 'BUY'
        elif value > 60:
            return 'BEARISH ZONE', 'bad', 'SELL'
        else:
            return 'NEUTRAL', 'neutral', 'HOLD'
    
    elif indicator_type == 'macd':
        if value > 0.5:
            return 'STRONG BULLISH', 'good', 'BUY'
        elif value > 0:
            return 'BULLISH', 'good', 'BUY'
        elif value < -0.5:
            return 'STRONG BEARISH', 'bad', 'SELL'
        elif value < 0:
            return 'BEARISH', 'bad', 'SELL'
        else:
            return 'NEUTRAL', 'neutral', 'HOLD'
    
    elif indicator_type == 'volume_ratio':
        if value > 3.0:
            return 'BREAKOUT VOLUME', 'good', 'BUY'
        elif value > 2.0:
            return 'HIGH VOLUME', 'good', 'BUY'
        elif value > 1.5:
            return 'ABOVE AVERAGE', 'neutral', 'HOLD'
        else:
            return 'NORMAL', 'neutral', 'HOLD'
    
    elif indicator_type == 'bb_position':
        if value < 0.1:
            return 'STRONG SUPPORT', 'good', 'BUY'
        elif value < 0.2:
            return 'NEAR SUPPORT', 'good', 'BUY'
        elif value > 0.9:
            return 'STRONG RESISTANCE', 'bad', 'SELL'
        elif value > 0.8:
            return 'NEAR RESISTANCE', 'bad', 'SELL'
        else:
            return 'MIDDLE RANGE', 'neutral', 'HOLD'
    
    elif indicator_type == 'ma_trend':
        if value == 'BULLISH':
            return 'UPTREND', 'good', 'BUY'
        elif value == 'BEARISH':
            return 'DOWNTREND', 'bad', 'SELL'
        else:
            return 'SIDEWAYS', 'neutral', 'HOLD'
    
    return 'UNKNOWN', 'neutral', 'HOLD'

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
    if 'sma_5' in current_data.columns and not current_data['sma_5'].isna().all():
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['sma_5'], name='SMA 5', line=dict(color='red', width=1)), row=1, col=1)
    if 'sma_20' in current_data.columns and not current_data['sma_20'].isna().all():
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['sma_20'], name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    if 'vwap' in current_data.columns and not current_data['vwap'].isna().all():
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
    if 'volume_sma' in current_data.columns and not current_data['volume_sma'].isna().all():
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['volume_sma'], name='Volume SMA', line=dict(color='blue', width=1)), row=2, col=1)
    
    # RSI
    if 'rsi' in current_data.columns and not current_data['rsi'].isna().all():
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['rsi'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
    
    # MACD
    if 'macd' in current_data.columns and not current_data['macd'].isna().all():
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['macd'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=current_data.index, y=current_data['macd_signal'], name='MACD Signal', line=dict(color='red', width=1)), row=3, col=1)
    
    # Bollinger Bands Position
    if 'bb_upper' in current_data.columns and 'bb_lower' in current_data.columns:
        bb_position = (current_data['close'] - current_data['bb_lower']) / (current_data['bb_upper'] - current_data['bb_lower'])
        bb_position = bb_position.fillna(0.5)  # Fill NaN with neutral position
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
    """Main Streamlit application with API data and timeline feature."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Trading Dashboard with Timeline</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Real-time Analysis with Zerodha API Data")
    
    # Sidebar controls
    st.sidebar.header("🔧 Trading Controls")
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7),
            help="Start date for data fetching"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now().date(),
            help="End date for data fetching"
        )
    
    # Stock selection
    all_symbols = get_nifty50_symbols()
    selected_symbols = st.sidebar.multiselect(
        "Select Stocks",
        options=all_symbols,
        default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS'],
        format_func=lambda x: x.replace('.NS', ''),
        help="Choose stocks for analysis"
    )
    
    # Load data button
    if st.sidebar.button("📊 Load API Data", type="primary"):
        
        if not selected_symbols:
            st.error("❌ Please select at least one stock")
            return
            
        if start_date >= end_date:
            st.error("❌ Start date must be before end date")
            return
        
        with st.spinner("📈 Fetching data from Zerodha API..."):
            all_data = load_api_data(selected_symbols, start_date, end_date)
            
            if all_data is not None and len(all_data) > 0:
                st.session_state['trading_data'] = all_data
                st.session_state['data_date_range'] = (start_date, end_date)
            else:
                st.error("❌ Failed to load API data")
                return
    
    # Main timeline analysis
    if 'trading_data' in st.session_state:
        
        all_data = st.session_state['trading_data']
        date_range = st.session_state.get('data_date_range', (start_date, end_date))
        
        # Stock selector for timeline
        st.markdown("## ⏯️ Timeline Analysis")
        
        # Debug: Check data structure
        if isinstance(all_data, dict):
            available_stocks = list(all_data.keys())
        else:
            # If all_data is a DataFrame, we have only one stock
            available_stocks = ['Single Stock Data']
            all_data = {'Single Stock Data': all_data}
        
        selected_stock = st.selectbox(
            "Select Stock for Timeline Analysis",
            options=available_stocks,
            format_func=lambda x: x.replace('.NS', '') if '.NS' in str(x) else str(x),
            help="Choose stock for step-by-step analysis"
        )
        
        if selected_stock and selected_stock in all_data:
            
            stock_data = all_data[selected_stock]
            
            if not stock_data.empty and len(stock_data) > 20:
                
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
                current_data = calculate_intraday_strategies(current_data)
                
                # Order book disabled - not available
                # display_order_book(selected_stock)
                
                if not current_data.empty:
                    latest = current_data.iloc[-1]
                    
                    # Live indicator values
                    st.markdown("## 📊 Live Indicator Values")
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    # Price
                    price_change = latest['price_change'] if 'price_change' in latest and not pd.isna(latest['price_change']) else 0
                    col1.metric("Price", f"₹{latest['close']:.1f}", f"{price_change:+.1f}%")
                    
                    # RSI
                    rsi_val = latest['rsi'] if 'rsi' in latest and not pd.isna(latest['rsi']) else 50
                    rsi_status, rsi_color, rsi_signal = get_indicator_status(rsi_val, 'rsi')
                    col2.metric("RSI", f"{rsi_val:.1f}", f"{rsi_signal} - {rsi_status}")
                    
                    # MACD
                    macd_val = latest['macd'] if 'macd' in latest and not pd.isna(latest['macd']) else 0
                    if pd.isna(macd_val):
                        macd_val = 0
                    macd_status, macd_color, macd_signal = get_indicator_status(macd_val, 'macd')
                    col3.metric("MACD", f"{macd_val:.3f}", f"{macd_signal} - {macd_status}")
                    
                    # Volume Ratio
                    vol_ratio = latest['volume_ratio'] if 'volume_ratio' in latest and not pd.isna(latest['volume_ratio']) else 1
                    vol_status, vol_color, vol_signal = get_indicator_status(vol_ratio, 'volume_ratio')
                    col4.metric("Volume", f"{vol_ratio:.1f}x", f"{vol_signal} - {vol_status}")
                    
                    # Bollinger Band Position
                    bb_upper = latest['bb_upper_20'] if 'bb_upper_20' in latest else None
                    bb_lower = latest['bb_lower_20'] if 'bb_lower_20' in latest else None
                    if bb_upper is not None and bb_lower is not None and not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                        bb_pos = (latest['close'] - bb_lower) / (bb_upper - bb_lower)
                        bb_status, bb_color, bb_signal = get_indicator_status(bb_pos, 'bb_position')
                        col5.metric("BB Position", f"{bb_pos:.1%}", f"{bb_signal} - {bb_status}")
                    else:
                        col5.metric("BB Position", "N/A", "HOLD - CALCULATING")
                    
                    # Moving Average Signal
                    sma5 = latest['sma_5'] if 'sma_5' in latest and not pd.isna(latest['sma_5']) else latest['close']
                    sma20 = latest['sma_20'] if 'sma_20' in latest and not pd.isna(latest['sma_20']) else latest['close']
                        
                    if sma5 > sma20 and latest['close'] > sma5:
                        ma_signal = "BULLISH"
                    elif sma5 < sma20 and latest['close'] < sma5:
                        ma_signal = "BEARISH"
                    else:
                        ma_signal = "NEUTRAL"
                    
                    ma_status, ma_color, ma_trade_signal = get_indicator_status(ma_signal, 'ma_trend')
                    col6.metric("MA Trend", ma_signal, f"{ma_trade_signal} - {ma_status}")
                    
                    # Color-coded signal summary
                    st.markdown("### 🎯 Current Trading Signals")
                    
                    signals_col1, signals_col2, signals_col3 = st.columns(3)
                    
                    # Collect all signals
                    all_signals = []
                    all_signals.append(('RSI', rsi_signal, rsi_color, f"{rsi_val:.0f}"))
                    all_signals.append(('MACD', macd_signal, macd_color, f"{macd_val:.3f}"))
                    all_signals.append(('Volume', vol_signal, vol_color, f"{vol_ratio:.1f}x"))
                    if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                        all_signals.append(('Bollinger', bb_signal, bb_color, f"{bb_pos:.1%}"))
                    all_signals.append(('MA Trend', ma_trade_signal, ma_color, ma_signal))
                    
                    # Display signals with color coding
                    buy_signals = [s for s in all_signals if s[1] == 'BUY']
                    sell_signals = [s for s in all_signals if s[1] == 'SELL']
                    hold_signals = [s for s in all_signals if s[1] == 'HOLD']
                    
                    with signals_col1:
                        st.markdown("#### 🟢 BUY Signals")
                        if buy_signals:
                            for name, signal, color, value in buy_signals:
                                st.markdown(f"🟢 **{name}**: {value}")
                        else:
                            st.markdown("*No buy signals*")
                    
                    with signals_col2:
                        st.markdown("#### 🔴 SELL Signals")
                        if sell_signals:
                            for name, signal, color, value in sell_signals:
                                st.markdown(f"🔴 **{name}**: {value}")
                        else:
                            st.markdown("*No sell signals*")
                    
                    with signals_col3:
                        st.markdown("#### 🟡 HOLD/Neutral")
                        if hold_signals:
                            for name, signal, color, value in hold_signals:
                                st.markdown(f"🟡 **{name}**: {value}")
                        else:
                            st.markdown("*No neutral signals*")
                    
                    # Overall recommendation
                    buy_count = len(buy_signals)
                    sell_count = len(sell_signals)
                    
                    if buy_count > sell_count and buy_count >= 2:
                        overall_signal = "🟢 **OVERALL: BUY BIAS**"
                        signal_strength = f"Strength: {buy_count}/{len(all_signals)} indicators bullish"
                    elif sell_count > buy_count and sell_count >= 2:
                        overall_signal = "🔴 **OVERALL: SELL BIAS**"
                        signal_strength = f"Strength: {sell_count}/{len(all_signals)} indicators bearish"
                    else:
                        overall_signal = "🟡 **OVERALL: NEUTRAL/MIXED**"
                        signal_strength = f"Mixed signals: {buy_count} buy, {sell_count} sell"
                    
                    st.markdown(f"### {overall_signal}")
                    st.markdown(f"*{signal_strength}*")
                    
                    # Strategy signals summary
                    if 'composite_signal' in latest:
                        composite_val = latest['composite_signal']
                        st.markdown("---")
                        st.markdown("### 📊 Strategy Signals Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            strong_buy = latest['strong_buy'] if 'strong_buy' in latest else False
                            if strong_buy:
                                st.markdown('<span class="indicator-good">🚀 STRONG BUY</span>', unsafe_allow_html=True)
                            else:
                                st.markdown("🚀 Strong Buy: No")
                        
                        with col2:
                            weak_buy = latest['weak_buy'] if 'weak_buy' in latest else False
                            if weak_buy:
                                st.markdown('<span class="indicator-neutral">📈 WEAK BUY</span>', unsafe_allow_html=True)
                            else:
                                st.markdown("📈 Weak Buy: No")
                        
                        with col3:
                            weak_sell = latest['weak_sell'] if 'weak_sell' in latest else False
                            if weak_sell:
                                st.markdown('<span class="indicator-neutral">📉 WEAK SELL</span>', unsafe_allow_html=True)
                            else:
                                st.markdown("📉 Weak Sell: No")
                        
                        with col4:
                            strong_sell = latest['strong_sell'] if 'strong_sell' in latest else False
                            if strong_sell:
                                st.markdown('<span class="indicator-bad">🔻 STRONG SELL</span>', unsafe_allow_html=True)
                            else:
                                st.markdown("🔻 Strong Sell: No")
                        
                        # Composite signal strength bar
                        st.markdown("**Composite Signal Strength**")
                        if composite_val > 0:
                            st.progress(min(composite_val, 1.0))
                            st.markdown(f"Bullish Signal: {composite_val:.3f}")
                        elif composite_val < 0:
                            st.progress(min(abs(composite_val), 1.0))
                            st.markdown(f"Bearish Signal: {composite_val:.3f}")
                        else:
                            st.progress(0.0)
                            st.markdown("Neutral Signal: 0.000")
                    
                    # Create and display timeline chart (moved here - below indicators)
                    st.markdown("---")
                    fig = create_timeline_chart(selected_stock, stock_data, current_index)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add comprehensive strategy plots
                        st.markdown("---")
                        st.markdown("## 🎯 Intraday Trading Strategies")
                        st.markdown("*Comprehensive analysis of multiple trading strategies with signals*")
                        
                        create_strategy_plots(current_data, stock_data.index[current_index])
                        
                        # Trading insights based on current indicators
                        st.markdown("### 🎯 Current Trading Insights")
                        
                        insights = []
                        
                        if rsi_val < 30:
                            insights.append("🟢 **RSI Oversold**: Potential buying opportunity")
                        elif rsi_val > 70:
                            insights.append("🔴 **RSI Overbought**: Consider taking profits")
                        
                        if macd_val > 0:
                            insights.append("🟢 **MACD Bullish**: Positive momentum")
                        elif macd_val < 0:
                            insights.append("🔴 **MACD Bearish**: Negative momentum")
                        
                        if vol_ratio > 2.0:
                            insights.append("📊 **High Volume**: Strong institutional interest")
                        
                        if ma_signal == "BULLISH":
                            insights.append("📈 **MA Trend Bullish**: Price above moving averages")
                        elif ma_signal == "BEARISH":
                            insights.append("📉 **MA Trend Bearish**: Price below moving averages")
                        
                        if not insights:
                            insights.append("📊 **Neutral Market**: No strong signals detected")
                        
                        for insight in insights:
                            st.markdown(insight)
                    
                    # Auto-play functionality
                    if auto_play and current_index < max_index:
                        time_module.sleep(1.0 / speed)  # Delay based on speed
                        st.rerun()
                
            else:
                st.error("Insufficient data for timeline analysis")
        
        # Enhanced explanations at the bottom
        st.markdown("---")
        st.markdown("## 📚 Complete Trading Indicator Guide")
        
        # Create tabs for different categories
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Momentum", "📊 Volume", "🎯 Support/Resistance", "📋 Strategy Guide"])
        
        with tab1:
            st.markdown("""
            ### 📈 Momentum Indicators
            
            #### 🔵 RSI (Relative Strength Index)
            **What it measures**: Price momentum and overbought/oversold conditions
            
            **Trading Signals**:
            - 🟢 **BUY Zones**: 
              - RSI < 30: **Strong Oversold** - High probability bounce
              - RSI 30-40: **Bullish Zone** - Good entry after pullback
            - 🔴 **SELL Zones**:
              - RSI > 70: **Strong Overbought** - Take profits/short
              - RSI 60-70: **Bearish Zone** - Prepare for reversal
            - 🟡 **NEUTRAL**: RSI 40-60 - No clear signal, wait for breakout
            
            **Best Practice**: Combine with trend - buy oversold in uptrend, sell overbought in downtrend
            
            #### 🔵 MACD (Moving Average Convergence Divergence)
            **What it measures**: Trend changes and momentum shifts
            
            **Trading Signals**:
            - 🟢 **BUY Signals**:
              - MACD > 0: **Bullish momentum** - Uptrend confirmed
              - MACD > 0.5: **Strong bullish** - High confidence buy
              - MACD crosses above signal line: **Buy trigger**
            - 🔴 **SELL Signals**:
              - MACD < 0: **Bearish momentum** - Downtrend confirmed  
              - MACD < -0.5: **Strong bearish** - High confidence sell
              - MACD crosses below signal line: **Sell trigger**
            
            **Best Practice**: Wait for histogram confirmation and volume support
            """)
        
        with tab2:
            st.markdown("""
            ### 📊 Volume Analysis
            
            #### 🔵 Volume Ratio (Current vs Average)
            **What it measures**: Interest level and breakout confirmation
            
            **Trading Signals**:
            - 🟢 **BUY Signals**:
              - Volume > 3.0x: **Breakout volume** - Strong institutional buying
              - Volume > 2.0x: **High volume** - Confirms price moves
            - 🟡 **NEUTRAL**:
              - Volume 1.5-2.0x: **Above average** - Monitor for direction
              - Volume < 1.5x: **Normal** - No special significance
            
            **Key Rules**:
            - ✅ **Volume confirms price**: High volume + price up = bullish
            - ❌ **Volume divergence**: High volume + price down = bearish
            - ⚠️ **Low volume moves**: Often false breakouts, be cautious
            
            **Best Practice**: Never trade breakouts without volume confirmation
            """)
        
        with tab3:
            st.markdown("""
            ### 🎯 Support & Resistance
            
            #### 🔵 Bollinger Bands Position
            **What it measures**: Price position relative to volatility bands
            
            **Trading Signals**:
            - 🟢 **BUY Zones**:
              - Position < 10%: **Strong support** - High probability bounce
              - Position < 20%: **Near support** - Good risk/reward entry
            - 🔴 **SELL Zones**:
              - Position > 90%: **Strong resistance** - Take profits
              - Position > 80%: **Near resistance** - Prepare for reversal
            - 🟡 **NEUTRAL**: 20-80% - **Middle range** - Wait for clearer signals
            
            #### 🔵 Moving Average Trend
            **What it measures**: Overall trend direction and strength
            
            **Trading Signals**:
            - 🟢 **BUY Setup**: Price > SMA5 > SMA20 = **Strong uptrend**
            - 🔴 **SELL Setup**: Price < SMA5 < SMA20 = **Strong downtrend**
            - 🟡 **SIDEWAYS**: Mixed MA alignment = **Choppy market**
            
            **Best Practice**: Trade in direction of higher timeframe trend
            """)
        
        with tab4:
            st.markdown("""
            ### 📋 Complete Trading Strategy
            
            #### 🎯 Signal Confirmation Rules
            **Never trade on single indicator - use confluence**:
            
            **🟢 HIGH CONFIDENCE BUY**:
            1. RSI oversold (< 30) + MACD bullish (> 0)
            2. High volume (> 2x) + Near BB support (< 20%)
            3. Price above rising moving averages
            4. Multiple timeframes aligned bullish
            
            **🔴 HIGH CONFIDENCE SELL**:
            1. RSI overbought (> 70) + MACD bearish (< 0)
            2. High volume (> 2x) + Near BB resistance (> 80%)
            3. Price below falling moving averages
            4. Multiple timeframes aligned bearish
            
            #### ⏰ Timeline Trading Tips
            **Market Sessions** (Indian Market):
            - **9:15-10:00 AM**: Opening volatility - wait for direction
            - **10:00-11:30 AM**: Primary trend establishment
            - **11:30-2:00 PM**: Midday consolidation - lower volume
            - **2:00-3:30 PM**: Closing moves - institutional activity
            
            **Best Entry Times**:
            - 🟢 **Breakouts**: First 30 minutes after 10:00 AM
            - 🟢 **Reversals**: Support/resistance tests 11:00 AM - 2:00 PM
            - 🟢 **Momentum**: Final hour (2:30-3:30 PM) for strong trends
            
            #### 🛡️ Risk Management
            **Position Sizing**:
            - Strong signals (3+ confirmations): 2-3% risk
            - Medium signals (2 confirmations): 1-2% risk
            - Weak signals (1 confirmation): 0.5-1% risk
            
            **Stop Loss Rules**:
            - Below recent swing low (long positions)
            - Above recent swing high (short positions)
            - Never risk more than 2% per trade
            
            **Take Profit Strategy**:
            - First target: 1:1 risk/reward ratio
            - Second target: 1:2 risk/reward ratio
            - Trail stops on remaining position
            """)
    
    else:
        st.info("👆 Select date range and stocks, then click 'Load API Data' to start analysis!")

if __name__ == "__main__":
    main() 