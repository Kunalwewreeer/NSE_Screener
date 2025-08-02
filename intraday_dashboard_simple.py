#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE INTRADAY TRADING DASHBOARD
============================================

Displays ALL Nifty 50 stocks with complete technical analysis for intraday trading decisions.
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

from core.data_handler import fetch_data, DataHandler

def get_nifty50_symbols():
    """Get Nifty 50 stock symbols."""
    nifty50 = [
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
    return nifty50

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators."""
    df = data.copy()
    
    # Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
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
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price momentum
    df['price_change'] = df['close'].pct_change() * 100
    df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
    
    # Generate trading signals
    df = generate_signals(df)
    
    return df

def generate_signals(df):
    """Generate comprehensive trading signals."""
    
    # Initialize signal columns
    df['buy_signals'] = 0
    df['sell_signals'] = 0
    df['signal_strength'] = 0
    df['signal_type'] = 'NEUTRAL'
    
    # Moving Average signals
    ma_bullish = (df['sma_5'] > df['sma_20']) & (df['close'] > df['sma_5'])
    ma_bearish = (df['sma_5'] < df['sma_20']) & (df['close'] < df['sma_5'])
    
    # RSI signals
    rsi_oversold = df['rsi'] < 30
    rsi_overbought = df['rsi'] > 70
    rsi_bullish = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    rsi_bearish = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    
    # MACD signals
    macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # Bollinger Band signals
    bb_squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
    bb_breakout_up = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
    bb_breakout_down = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
    
    # Volume confirmation
    volume_confirmation = df['volume_ratio'] > 1.5
    
    # Count bullish signals
    df['buy_signals'] = (
        ma_bullish.astype(int) + 
        rsi_bullish.astype(int) + 
        macd_bullish.astype(int) + 
        bb_breakout_up.astype(int) +
        (rsi_oversold & volume_confirmation).astype(int)
    )
    
    # Count bearish signals
    df['sell_signals'] = (
        ma_bearish.astype(int) + 
        rsi_bearish.astype(int) + 
        macd_bearish.astype(int) + 
        bb_breakout_down.astype(int) +
        (rsi_overbought & volume_confirmation).astype(int)
    )
    
    # Calculate net signal strength
    df['signal_strength'] = df['buy_signals'] - df['sell_signals']
    
    # Determine signal type
    df.loc[df['signal_strength'] >= 3, 'signal_type'] = 'STRONG BUY'
    df.loc[df['signal_strength'] == 2, 'signal_type'] = 'BUY'
    df.loc[df['signal_strength'] == 1, 'signal_type'] = 'WEAK BUY'
    df.loc[df['signal_strength'] == -1, 'signal_type'] = 'WEAK SELL'
    df.loc[df['signal_strength'] == -2, 'signal_type'] = 'SELL'
    df.loc[df['signal_strength'] <= -3, 'signal_type'] = 'STRONG SELL'
    
    return df

def create_stock_chart(symbol, data, ax):
    """Create a comprehensive chart for a single stock."""
    
    # Calculate indicators
    df = calculate_technical_indicators(data)
    
    if df.empty or len(df) < 50:
        ax.text(0.5, 0.5, f'{symbol}\nInsufficient Data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return None
    
    # Get latest values
    latest = df.iloc[-1]
    latest_price = latest['close']
    price_change = latest['price_change']
    volume_ratio = latest.get('volume_ratio', 1)
    rsi = latest.get('rsi', 50)
    signal_strength = latest.get('signal_strength', 0)
    signal_type = latest.get('signal_type', 'NEUTRAL')
    
    # Determine colors
    if 'BUY' in signal_type:
        title_color = 'green'
        signal_color = 'green'
    elif 'SELL' in signal_type:
        title_color = 'red'
        signal_color = 'red'
    else:
        title_color = 'orange'
        signal_color = 'orange'
    
    # Plot price
    ax.plot(df.index, df['close'], 'k-', linewidth=1.5, label='Price')
    
    # Plot moving averages
    if 'sma_5' in df.columns:
        ax.plot(df.index, df['sma_5'], 'r-', linewidth=1, alpha=0.7, label='SMA5')
    if 'sma_20' in df.columns:
        ax.plot(df.index, df['sma_20'], 'b-', linewidth=1, alpha=0.7, label='SMA20')
    
    # Plot Bollinger Bands
    if 'bb_upper' in df.columns:
        ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                       alpha=0.1, color='blue')
        ax.plot(df.index, df['bb_upper'], 'b--', linewidth=0.8, alpha=0.6)
        ax.plot(df.index, df['bb_lower'], 'b--', linewidth=0.8, alpha=0.6)
    
    # Highlight recent signals
    recent_data = df.tail(20)
    
    # Buy signals
    buy_points = recent_data[recent_data['buy_signals'] >= 2]
    if not buy_points.empty:
        ax.scatter(buy_points.index, buy_points['close'], 
                  color='green', marker='^', s=40, alpha=0.8, zorder=5)
    
    # Sell signals
    sell_points = recent_data[recent_data['sell_signals'] >= 2]
    if not sell_points.empty:
        ax.scatter(sell_points.index, sell_points['close'],
                  color='red', marker='v', s=40, alpha=0.8, zorder=5)
    
    # Format chart
    symbol_clean = symbol.replace('.NS', '')
    ax.set_title(f'{symbol_clean}\n₹{latest_price:.1f} ({price_change:+.1f}%)\n{signal_type}', 
                fontsize=9, fontweight='bold', color=title_color)
    
    # Add metrics
    metrics_text = f'RSI:{rsi:.0f} Vol:{volume_ratio:.1f}x'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
           fontsize=7, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Format axes
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, alpha=0.3)
    
    # Format time axis for minute data
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
        'rsi': rsi,
        'volume_ratio': volume_ratio,
        'macd': latest.get('macd', 0),
        'bb_position': (latest_price - latest.get('bb_lower', latest_price)) / (latest.get('bb_upper', latest_price) - latest.get('bb_lower', latest_price)) if latest.get('bb_upper', 0) != latest.get('bb_lower', 0) else 0.5
    }

def create_summary_table(stock_summaries):
    """Create and display summary table."""
    
    # Sort by signal strength
    sorted_stocks = sorted(stock_summaries, key=lambda x: x['signal_strength'], reverse=True)
    
    print("\n" + "="*130)
    print("🎯 COMPREHENSIVE INTRADAY TRADING DASHBOARD")
    print("="*130)
    print(f"{'Rank':<4} {'Symbol':<12} {'Price':<9} {'Change':<8} {'Signal':<12} {'Str':<4} {'RSI':<4} {'Vol':<6} {'MACD':<6} {'BB%':<5}")
    print("-"*130)
    
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
        
        print(f"{i:<4} {symbol_clean:<12} ₹{stock['price']:<8.1f} {stock['change_pct']:>+6.1f}% "
              f"{stock['signal']:<12} {emoji}{stock['signal_strength']:>+2} {stock['rsi']:<4.0f} "
              f"{stock['volume_ratio']:<5.1f}x {stock['macd']:<6.2f} {stock['bb_position']:<5.1%}")
    
    # Summary stats
    total = len(sorted_stocks)
    strong_buy = len([s for s in sorted_stocks if 'STRONG BUY' in s['signal']])
    buy = len([s for s in sorted_stocks if s['signal'] == 'BUY' or s['signal'] == 'WEAK BUY'])
    sell = len([s for s in sorted_stocks if 'SELL' in s['signal']])
    neutral = total - strong_buy - buy - sell
    
    print("-"*130)
    print(f"📊 SUMMARY: {total} stocks | 🚀 {strong_buy} Strong Buy | 📈 {buy} Buy | 📉 {sell} Sell | ➖ {neutral} Neutral")
    print("="*130)
    
    return sorted_stocks

def main():
    """Main function to run the comprehensive dashboard."""
    
    print("🚀 LAUNCHING COMPREHENSIVE INTRADAY TRADING DASHBOARD")
    print("="*70)
    
    # Get symbols
    symbols = get_nifty50_symbols()
    print(f"📊 Analyzing {len(symbols)} Nifty 50 stocks...")
    
    # Get date range (last 5 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    # Fetch data
    print(f"📈 Fetching minute data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    all_data = fetch_data(
        symbols=symbols[:20],  # Start with first 20 for testing
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='minute'
    )
    
    if not all_data:
        print("❌ No data fetched")
        return
    
    print(f"✅ Data fetched for {len(all_data)} stocks")
    
    # Filter stocks
    filtered_stocks = {}
    for symbol, data in all_data.items():
        if data.empty or len(data) < 100:
            continue
        
        latest = data.iloc[-1]
        avg_volume = data['volume'].tail(20).mean()
        
        # Basic filters
        if (latest['close'] >= 50 and 
            latest['close'] <= 5000 and
            avg_volume >= 10000):
            filtered_stocks[symbol] = data
    
    print(f"📊 {len(filtered_stocks)} stocks passed filters")
    
    if not filtered_stocks:
        print("❌ No stocks passed filters")
        return
    
    # Create dashboard
    print("🎨 Creating comprehensive charts...")
    
    total_stocks = len(filtered_stocks)
    cols = 4
    rows = int(np.ceil(total_stocks / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'🎯 COMPREHENSIVE INTRADAY TRADING DASHBOARD - {end_date.strftime("%Y-%m-%d")}\n'
                f'📊 Technical Analysis of {total_stocks} Nifty 50 Stocks', 
                fontsize=16, fontweight='bold')
    
    # Create charts
    stock_summaries = []
    for i, (symbol, data) in enumerate(filtered_stocks.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        try:
            summary = create_stock_chart(symbol, data, ax)
            if summary:
                stock_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Error creating chart for {symbol}: {e}")
            ax.text(0.5, 0.5, f'{symbol}\nError', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
    
    # Hide empty subplots
    for i in range(len(filtered_stocks), rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data/results/comprehensive_dashboard_{timestamp}.png'
    os.makedirs('data/results', exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Dashboard saved: {filename}")
    
    plt.show()
    
    # Create summary table
    if stock_summaries:
        sorted_stocks = create_summary_table(stock_summaries)
        
        # Show top opportunities
        print(f"\n🏆 TOP 5 TRADING OPPORTUNITIES:")
        for i, stock in enumerate(sorted_stocks[:5], 1):
            symbol_clean = stock['symbol'].replace('.NS', '')
            print(f"{i}. {symbol_clean}: {stock['signal']} (₹{stock['price']:.1f}, {stock['change_pct']:+.1f}%, RSI:{stock['rsi']:.0f})")
        
        return sorted_stocks
    else:
        print("❌ No stock summaries generated")
        return []

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Dashboard completed! Analyzed {len(results)} stocks with comprehensive technical indicators.")
    else:
        print("\n❌ Dashboard failed to complete.") 