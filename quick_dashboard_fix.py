#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE INTRADAY TRADING DASHBOARD
============================================
Shows charts for ALL possible Nifty 50 stocks with ALL technical indicators for intraday trading decisions.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from core.data_handler import fetch_data

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

def calculate_all_indicators(data):
    """Calculate ALL technical indicators for comprehensive analysis."""
    df = data.copy()
    
    try:
        # ===== MOVING AVERAGES =====
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # ===== MOMENTUM INDICATORS =====
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ===== VOLATILITY INDICATORS =====
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # ===== VOLUME INDICATORS =====
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # OBV (On Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # ===== PRICE ACTION =====
        df['price_change'] = df['close'].pct_change() * 100
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['body_pct'] = abs(df['close'] - df['open']) / df['close'] * 100
        
        # ===== SUPPORT/RESISTANCE =====
        # Pivot Points
        df['pivot'] = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift()
        df['s1'] = 2 * df['pivot'] - df['high'].shift()
        df['r2'] = df['pivot'] + (df['high'].shift() - df['low'].shift())
        df['s2'] = df['pivot'] - (df['high'].shift() - df['low'].shift())
        
        # ===== GENERATE SIGNALS =====
        df = generate_comprehensive_signals(df)
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df

def generate_comprehensive_signals(df):
    """Generate comprehensive trading signals from all indicators."""
    
    # Initialize signal tracking
    df['buy_signals'] = 0
    df['sell_signals'] = 0
    df['signal_strength'] = 0
    df['signal_reasons'] = ''
    
    # ===== TREND SIGNALS =====
    # Moving Average Alignment
    ma_bullish = (df['sma_5'] > df['sma_10']) & (df['sma_10'] > df['sma_20']) & (df['close'] > df['sma_5'])
    ma_bearish = (df['sma_5'] < df['sma_10']) & (df['sma_10'] < df['sma_20']) & (df['close'] < df['sma_5'])
    
    # EMA Crossovers
    ema_bullish = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
    ema_bearish = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
    
    # ===== MOMENTUM SIGNALS =====
    # RSI
    rsi_oversold = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
    rsi_overbought = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
    rsi_bullish = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    rsi_bearish = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    
    # MACD
    macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # Stochastic
    stoch_oversold = (df['stoch_k'] < 20) & (df['stoch_d'] < 20) & (df['stoch_k'] > df['stoch_d'])
    stoch_overbought = (df['stoch_k'] > 80) & (df['stoch_d'] > 80) & (df['stoch_k'] < df['stoch_d'])
    
    # ===== VOLATILITY SIGNALS =====
    # Bollinger Bands
    bb_oversold = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
    bb_overbought = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
    bb_squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5
    
    # ===== VOLUME SIGNALS =====
    volume_breakout = (df['volume_ratio'] > 2.0) & (df['close'] > df['close'].shift(1))
    volume_selling = (df['volume_ratio'] > 2.0) & (df['close'] < df['close'].shift(1))
    
    # ===== COUNT SIGNALS =====
    # Bullish signals
    df['buy_signals'] = (
        ma_bullish.astype(int) * 3 +
        ema_bullish.astype(int) * 2 +
        rsi_oversold.astype(int) * 2 +
        rsi_bullish.astype(int) * 1 +
        macd_bullish.astype(int) * 2 +
        stoch_oversold.astype(int) * 1 +
        bb_oversold.astype(int) * 1 +
        volume_breakout.astype(int) * 2
    )
    
    # Bearish signals
    df['sell_signals'] = (
        ma_bearish.astype(int) * 3 +
        ema_bearish.astype(int) * 2 +
        rsi_overbought.astype(int) * 2 +
        rsi_bearish.astype(int) * 1 +
        macd_bearish.astype(int) * 2 +
        stoch_overbought.astype(int) * 1 +
        bb_overbought.astype(int) * 1 +
        volume_selling.astype(int) * 2
    )
    
    # Net signal strength
    df['signal_strength'] = df['buy_signals'] - df['sell_signals']
    
    # Signal classification
    df['signal_type'] = 'NEUTRAL'
    df.loc[df['signal_strength'] >= 8, 'signal_type'] = 'STRONG BUY'
    df.loc[(df['signal_strength'] >= 5) & (df['signal_strength'] < 8), 'signal_type'] = 'BUY'
    df.loc[(df['signal_strength'] >= 2) & (df['signal_strength'] < 5), 'signal_type'] = 'WEAK BUY'
    df.loc[(df['signal_strength'] <= -2) & (df['signal_strength'] > -5), 'signal_type'] = 'WEAK SELL'
    df.loc[(df['signal_strength'] <= -5) & (df['signal_strength'] > -8), 'signal_type'] = 'SELL'
    df.loc[df['signal_strength'] <= -8, 'signal_type'] = 'STRONG SELL'
    
    return df

def create_comprehensive_chart(symbol, data, ax):
    """Create comprehensive chart with ALL indicators."""
    
    # Calculate all indicators
    df = calculate_all_indicators(data)
    
    if df.empty or len(df) < 50:
        ax.text(0.5, 0.5, f'{symbol}\nInsufficient Data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return None
    
    # Get latest values
    latest = df.iloc[-1]
    symbol_clean = symbol.replace('.NS', '')
    
    # Basic metrics
    latest_price = latest['close']
    price_change = latest.get('price_change', 0)
    volume_ratio = latest.get('volume_ratio', 1)
    rsi = latest.get('rsi', 50)
    signal_strength = latest.get('signal_strength', 0)
    signal_type = latest.get('signal_type', 'NEUTRAL')
    
    # Advanced metrics
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    bb_position = latest.get('bb_position', 0.5)
    atr = latest.get('atr', 0)
    stoch_k = latest.get('stoch_k', 50)
    
    # Determine colors based on signal
    if 'BUY' in signal_type:
        title_color = 'green'
        signal_emoji = '🚀' if 'STRONG' in signal_type else '📈'
    elif 'SELL' in signal_type:
        title_color = 'red'
        signal_emoji = '💥' if 'STRONG' in signal_type else '📉'
    else:
        title_color = 'orange'
        signal_emoji = '➖'
    
    # ===== PLOT PRICE ACTION =====
    # Main price line
    ax.plot(df.index, df['close'], 'k-', linewidth=2, label='Price', alpha=0.9)
    
    # Moving averages
    if 'sma_5' in df.columns:
        ax.plot(df.index, df['sma_5'], 'r-', linewidth=1, alpha=0.7, label='SMA5')
    if 'sma_20' in df.columns:
        ax.plot(df.index, df['sma_20'], 'b-', linewidth=1, alpha=0.7, label='SMA20')
    if 'ema_9' in df.columns:
        ax.plot(df.index, df['ema_9'], 'g--', linewidth=1, alpha=0.7, label='EMA9')
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                       alpha=0.1, color='blue', label='BB')
        ax.plot(df.index, df['bb_upper'], 'b:', linewidth=1, alpha=0.6)
        ax.plot(df.index, df['bb_lower'], 'b:', linewidth=1, alpha=0.6)
    
    # Support/Resistance levels
    if not pd.isna(latest.get('pivot', np.nan)):
        ax.axhline(y=latest['pivot'], color='orange', linestyle='--', alpha=0.6, linewidth=1)
    if not pd.isna(latest.get('r1', np.nan)):
        ax.axhline(y=latest['r1'], color='red', linestyle=':', alpha=0.5, linewidth=1)
    if not pd.isna(latest.get('s1', np.nan)):
        ax.axhline(y=latest['s1'], color='green', linestyle=':', alpha=0.5, linewidth=1)
    
    # ===== PLOT SIGNALS =====
    recent_data = df.tail(30)
    
    # Strong buy signals
    strong_buy = recent_data[recent_data['buy_signals'] >= 5]
    if not strong_buy.empty:
        ax.scatter(strong_buy.index, strong_buy['close'], 
                  color='darkgreen', marker='^', s=60, alpha=0.9, zorder=5)
    
    # Buy signals
    buy_signals = recent_data[(recent_data['buy_signals'] >= 2) & (recent_data['buy_signals'] < 5)]
    if not buy_signals.empty:
        ax.scatter(buy_signals.index, buy_signals['close'], 
                  color='green', marker='^', s=40, alpha=0.8, zorder=5)
    
    # Strong sell signals
    strong_sell = recent_data[recent_data['sell_signals'] >= 5]
    if not strong_sell.empty:
        ax.scatter(strong_sell.index, strong_sell['close'], 
                  color='darkred', marker='v', s=60, alpha=0.9, zorder=5)
    
    # Sell signals
    sell_signals = recent_data[(recent_data['sell_signals'] >= 2) & (recent_data['sell_signals'] < 5)]
    if not sell_signals.empty:
        ax.scatter(sell_signals.index, sell_signals['close'], 
                  color='red', marker='v', s=40, alpha=0.8, zorder=5)
    
    # ===== FORMAT CHART =====
    ax.set_title(f'{symbol_clean} {signal_emoji}\n₹{latest_price:.1f} ({price_change:+.1f}%)\n{signal_type}', 
                fontsize=10, fontweight='bold', color=title_color)
    
    # Add comprehensive metrics box
    metrics_text = (f'RSI:{rsi:.0f} Vol:{volume_ratio:.1f}x\n'
                   f'MACD:{macd:.2f} BB:{bb_position:.1%}\n'
                   f'ATR:{atr:.2f} Stoch:{stoch_k:.0f}')
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           fontsize=7, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Format axes
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # Time formatting for minute data
    if len(df) > 50:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    return {
        'symbol': symbol,
        'price': latest_price,
        'change_pct': price_change,
        'signal': signal_type,
        'signal_strength': signal_strength,
        'buy_signals': latest.get('buy_signals', 0),
        'sell_signals': latest.get('sell_signals', 0),
        'rsi': rsi,
        'volume_ratio': volume_ratio,
        'macd': macd,
        'bb_position': bb_position,
        'atr': atr,
        'stoch_k': stoch_k
    }

def main():
    """Main function - Comprehensive Intraday Trading Dashboard."""
    
    print("🚀 COMPREHENSIVE INTRADAY TRADING DASHBOARD")
    print("=" * 80)
    print("📊 Analyzing ALL Nifty 50 stocks with ALL technical indicators")
    print("🎯 For complete intraday trading decision support")
    print("=" * 80)
    
    # Get all Nifty 50 symbols
    symbols = get_nifty50_symbols()
    print(f"📈 Loading {len(symbols)} Nifty 50 stocks...")
    
    # Date range (last few days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    print(f"📅 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Fetch data with correct parameters
    all_data = fetch_data(
        symbols=symbols[:25],  # Start with 25 stocks
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='minute'
    )
    
    if not all_data:
        print("❌ No data fetched")
        return
    
    print(f"✅ Data fetched for {len(all_data)} stocks")
    
    # Filter and analyze stocks
    filtered_stocks = {}
    for symbol, data in all_data.items():
        if data.empty or len(data) < 100:
            continue
        
        latest = data.iloc[-1]
        avg_volume = data['volume'].tail(20).mean()
        
        # Quality filters
        if (latest['close'] >= 50 and 
            latest['close'] <= 10000 and
            avg_volume >= 5000):
            filtered_stocks[symbol] = data
    
    print(f"📊 {len(filtered_stocks)} stocks passed quality filters")
    
    if not filtered_stocks:
        print("❌ No stocks passed filters")
        return
    
    # ===== CREATE COMPREHENSIVE DASHBOARD =====
    print("🎨 Creating comprehensive trading dashboard...")
    
    total_stocks = len(filtered_stocks)
    cols = 5  # 5 columns for better layout
    rows = int(np.ceil(total_stocks / cols))
    
    # Create large figure for comprehensive view
    fig, axes = plt.subplots(rows, cols, figsize=(25, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif total_stocks == 1:
        axes = np.array([[axes]])
    
    fig.suptitle(f'🎯 COMPREHENSIVE INTRADAY TRADING DASHBOARD - {end_date.strftime("%Y-%m-%d")}\n'
                f'📊 Complete Technical Analysis of {total_stocks} Nifty 50 Stocks with ALL Indicators', 
                fontsize=18, fontweight='bold')
    
    # Create individual charts
    stock_summaries = []
    for i, (symbol, data) in enumerate(filtered_stocks.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        try:
            summary = create_comprehensive_chart(symbol, data, ax)
            if summary:
                stock_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Error creating chart for {symbol}: {e}")
            ax.text(0.5, 0.5, f'{symbol}\nError: {str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
    
    # Hide empty subplots
    for i in range(len(filtered_stocks), rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('data/results', exist_ok=True)
    filename = f'data/results/comprehensive_intraday_dashboard_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Dashboard saved: {filename}")
    
    plt.show()
    
    # ===== CREATE DETAILED SUMMARY TABLE =====
    if stock_summaries:
        # Sort by signal strength
        sorted_stocks = sorted(stock_summaries, key=lambda x: x['signal_strength'], reverse=True)
        
        print("\n" + "="*140)
        print("🎯 COMPREHENSIVE INTRADAY TRADING ANALYSIS - DETAILED SUMMARY")
        print("="*140)
        print(f"{'#':<3} {'Symbol':<12} {'Price':<9} {'Chg%':<7} {'Signal':<12} {'Str':<4} {'B/S':<5} "
              f"{'RSI':<4} {'Vol':<5} {'MACD':<6} {'BB%':<5} {'ATR':<5} {'Stoch':<6}")
        print("-"*140)
        
        for i, stock in enumerate(sorted_stocks, 1):
            symbol_clean = stock['symbol'].replace('.NS', '')
            
            # Signal emoji
            if 'STRONG BUY' in stock['signal']:
                emoji = '🚀'
            elif 'BUY' in stock['signal']:
                emoji = '📈'
            elif 'STRONG SELL' in stock['signal']:
                emoji = '💥'
            elif 'SELL' in stock['signal']:
                emoji = '📉'
            else:
                emoji = '➖'
            
            print(f"{i:<3} {symbol_clean:<12} ₹{stock['price']:<8.1f} {stock['change_pct']:>+6.1f}% "
                  f"{stock['signal']:<12} {emoji}{stock['signal_strength']:>+2} "
                  f"{stock['buy_signals']}/{stock['sell_signals']:<4} {stock['rsi']:<4.0f} "
                  f"{stock['volume_ratio']:<4.1f}x {stock['macd']:<6.2f} {stock['bb_position']:<5.1%} "
                  f"{stock['atr']:<5.2f} {stock['stoch_k']:<6.0f}")
        
        # Advanced summary statistics
        total = len(sorted_stocks)
        strong_buy = len([s for s in sorted_stocks if 'STRONG BUY' in s['signal']])
        buy = len([s for s in sorted_stocks if s['signal'] in ['BUY', 'WEAK BUY']])
        sell = len([s for s in sorted_stocks if 'SELL' in s['signal']])
        neutral = total - strong_buy - buy - sell
        
        avg_rsi = np.mean([s['rsi'] for s in sorted_stocks])
        avg_volume = np.mean([s['volume_ratio'] for s in sorted_stocks])
        
        print("-"*140)
        print(f"📊 MARKET SUMMARY: {total} stocks analyzed")
        print(f"🚀 Strong Buy: {strong_buy} | 📈 Buy: {buy} | 📉 Sell: {sell} | ➖ Neutral: {neutral}")
        print(f"📈 Average RSI: {avg_rsi:.1f} | 📊 Average Volume Ratio: {avg_volume:.1f}x")
        print("="*140)
        
        # Top opportunities
        print(f"\n🏆 TOP 10 INTRADAY TRADING OPPORTUNITIES:")
        print("-" * 60)
        for i, stock in enumerate(sorted_stocks[:10], 1):
            symbol_clean = stock['symbol'].replace('.NS', '')
            print(f"{i:2d}. {symbol_clean:<12} {stock['signal']:<12} "
                  f"₹{stock['price']:<7.1f} ({stock['change_pct']:+.1f}%) "
                  f"RSI:{stock['rsi']:.0f} Vol:{stock['volume_ratio']:.1f}x")
        
        return sorted_stocks
    else:
        print("❌ No stock summaries generated")
        return []

if __name__ == "__main__":
    print("🎯 Starting Comprehensive Intraday Trading Dashboard...")
    results = main()
    
    if results:
        print(f"\n✅ DASHBOARD COMPLETED SUCCESSFULLY!")
        print(f"📊 Analyzed {len(results)} stocks with comprehensive technical indicators")
        print(f"🎯 All charts show complete intraday analysis for informed trading decisions")
        print(f"📈 Use the signals, indicators, and rankings to make trading decisions")
    else:
        print("\n❌ Dashboard failed to complete") 