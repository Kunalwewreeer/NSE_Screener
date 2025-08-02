#!/usr/bin/env python3
"""
Simple script to plot stock data and verify data fetching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from core.data_handler import DataHandler
import argparse

def plot_stock_data(symbol, start_date, end_date, interval="minute"):
    """Plot stock data to verify data fetching."""
    print(f"📊 Plotting {symbol} data")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⏱️  Interval: {interval}")
    
    # Initialize data handler
    data_handler = DataHandler()
    
    try:
        # Fetch data
        print(f"🔍 Fetching {interval} data...")
        data = data_handler.get_historical_data(
            symbols=symbol,
            from_date=start_date,
            to_date=end_date,
            interval=interval,
            refresh_cache=True
        )
        
        # Handle both DataFrame and dict formats
        if isinstance(data, dict):
            # If data is a dict, it contains multiple symbols
            if symbol in data:
                data = data[symbol]
            else:
                print("❌ Symbol not found in data")
                return
        
        if data.empty:
            print("❌ No data received")
            return
        
        print(f"✅ Fetched {len(data)} records")
        print(f"📈 Data shape: {data.shape}")
        print(f"📅 Time range: {data.index.min()} to {data.index.max()}")
        print(f"💰 Price range: ₹{data['low'].min():.2f} - ₹{data['high'].max():.2f}")
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: OHLC Candlestick-like chart
        ax1.plot(data.index, data['close'], label='Close', linewidth=1, color='blue', alpha=0.8)
        ax1.plot(data.index, data['high'], label='High', linewidth=0.5, color='green', alpha=0.6)
        ax1.plot(data.index, data['low'], label='Low', linewidth=0.5, color='red', alpha=0.6)
        
        ax1.set_title(f'{symbol} - {interval.capitalize()} Data ({start_date} to {end_date})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume
        ax2.bar(data.index, data['volume'], alpha=0.7, color='purple', width=0.8)
        ax2.set_title('Volume', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{symbol.replace('.NS', '')}_{interval}_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved as: {filename}")
        
        # Show plot
        plt.show()
        
        # Print data summary
        print(f"\n📋 Data Summary:")
        print(f"   Total records: {len(data)}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"   Price range: ₹{data['low'].min():.2f} - ₹{data['high'].max():.2f}")
        print(f"   Volume range: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        
        # Show sample data
        print(f"\n📊 Sample Data (first 5 records):")
        print(data.head().to_string())
        
        # Show data info
        print(f"\n🔍 Data Info:")
        print(data.info())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def plot_multiple_intervals(symbol, start_date, end_date):
    """Plot data for multiple intervals to compare."""
    intervals = ["minute", "5minute", "15minute", "day"]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    data_handler = DataHandler()
    
    for i, interval in enumerate(intervals):
        try:
            print(f"🔍 Fetching {interval} data...")
            data = data_handler.get_historical_data(
                symbols=symbol,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
                refresh_cache=False  # Use cached data
            )
            
            # Handle both DataFrame and dict formats
            if isinstance(data, dict):
                if symbol in data:
                    data = data[symbol]
                else:
                    print(f"   ❌ {interval}: Symbol not found")
                    continue
            
            if not data.empty:
                ax = axes[i]
                ax.plot(data.index, data['close'], linewidth=1, color='blue', alpha=0.8)
                ax.set_title(f'{interval.capitalize()} - {len(data)} records', fontsize=12, fontweight='bold')
                ax.set_ylabel('Price (₹)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                print(f"   ✅ {interval}: {len(data)} records")
            else:
                print(f"   ❌ {interval}: No data")
                
        except Exception as e:
            print(f"   ❌ {interval}: Error - {e}")
    
    plt.suptitle(f'{symbol} - Data Comparison ({start_date} to {end_date})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{symbol.replace('.NS', '')}_comparison_{start_date}_to_{end_date}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison plot saved as: {filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot stock data to verify data fetching')
    parser.add_argument('--symbol', default='RELIANCE.NS', help='Stock symbol (default: RELIANCE.NS)')
    parser.add_argument('--start-date', default=None, help='Start date (YYYY-MM-DD, default: 2 days ago)')
    parser.add_argument('--end-date', default=None, help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--interval', default='minute', choices=['minute', '5minute', '15minute', '30minute', '60minute', 'day'], 
                       help='Data interval (default: minute)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple intervals')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    print("🚀 STOCK DATA PLOTTING TOOL")
    print("=" * 50)
    
    if args.compare:
        plot_multiple_intervals(args.symbol, args.start_date, args.end_date)
    else:
        plot_stock_data(args.symbol, args.start_date, args.end_date, args.interval)
    
    print("\n✅ Plotting completed!")

if __name__ == "__main__":
    main() 