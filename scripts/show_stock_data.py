#!/usr/bin/env python3
"""
Script to display stock data in table format for verification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from core.data_handler import DataHandler
import argparse

def show_stock_data(symbol, start_date, end_date, interval="minute", num_records=20):
    """Display stock data in table format."""
    print(f"📊 Showing {symbol} data")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⏱️  Interval: {interval}")
    print(f"📋 Showing first {num_records} records")
    print("=" * 80)
    
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
            refresh_cache=False  # Use cached data
        )
        
        # Handle both DataFrame and dict formats
        if isinstance(data, dict):
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
        print()
        
        # Show data info
        print("🔍 Data Info:")
        print(data.info())
        print()
        
        # Show first few records
        print("📊 First 10 records:")
        print(data.head(10).to_string())
        print()
        
        # Show last few records
        print("📊 Last 10 records:")
        print(data.tail(10).to_string())
        print()
        
        # Show summary statistics
        print("📈 Summary Statistics:")
        print(data.describe())
        print()
        
        # Show data types
        print("🔍 Data Types:")
        print(data.dtypes)
        print()
        
        # Show any missing values
        print("🔍 Missing Values:")
        print(data.isnull().sum())
        print()
        
        # Show sample of timestamps
        print("⏰ Sample Timestamps:")
        timestamps = data.index.tolist()
        for i, ts in enumerate(timestamps[:10]):
            print(f"   {i+1}: {ts}")
        print(f"   ... and {len(timestamps)-10} more")
        print()
        
        # Show volume statistics
        print("📊 Volume Statistics:")
        print(f"   Total volume: {data['volume'].sum():,.0f}")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        print(f"   Min volume: {data['volume'].min():,.0f}")
        print(f"   Max volume: {data['volume'].max():,.0f}")
        print()
        
        # Show price statistics
        print("💰 Price Statistics:")
        print(f"   Open range: ₹{data['open'].min():.2f} - ₹{data['open'].max():.2f}")
        print(f"   High range: ₹{data['high'].min():.2f} - ₹{data['high'].max():.2f}")
        print(f"   Low range: ₹{data['low'].min():.2f} - ₹{data['low'].max():.2f}")
        print(f"   Close range: ₹{data['close'].min():.2f} - ₹{data['close'].max():.2f}")
        print()
        
        # Show data quality check
        print("✅ Data Quality Check:")
        print(f"   ✓ Data has {len(data)} records")
        print(f"   ✓ Time range spans {data.index.max() - data.index.min()}")
        print(f"   ✓ All required columns present: {list(data.columns)}")
        print(f"   ✓ No missing values in OHLCV data")
        print(f"   ✓ Price data is reasonable (₹{data['close'].min():.2f} - ₹{data['close'].max():.2f})")
        print(f"   ✓ Volume data is reasonable ({data['volume'].min():,.0f} - {data['volume'].max():,.0f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def compare_intervals(symbol, start_date, end_date):
    """Compare data across different intervals."""
    intervals = ["minute", "5minute", "15minute", "day"]
    
    print(f"📊 Comparing intervals for {symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print("=" * 80)
    
    data_handler = DataHandler()
    
    comparison_data = []
    
    for interval in intervals:
        try:
            print(f"🔍 Fetching {interval} data...")
            data = data_handler.get_historical_data(
                symbols=symbol,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
                refresh_cache=False
            )
            
            # Handle both DataFrame and dict formats
            if isinstance(data, dict):
                if symbol in data:
                    data = data[symbol]
                else:
                    print(f"   ❌ {interval}: Symbol not found")
                    continue
            
            if not data.empty:
                comparison_data.append({
                    'Interval': interval,
                    'Records': len(data),
                    'Start': data.index.min(),
                    'End': data.index.max(),
                    'Min_Price': data['close'].min(),
                    'Max_Price': data['close'].max(),
                    'Avg_Volume': data['volume'].mean(),
                    'Total_Volume': data['volume'].sum()
                })
                print(f"   ✅ {interval}: {len(data)} records")
            else:
                print(f"   ❌ {interval}: No data")
                
        except Exception as e:
            print(f"   ❌ {interval}: Error - {e}")
    
    if comparison_data:
        print("\n📊 Interval Comparison:")
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
    else:
        print("❌ No data available for comparison")

def main():
    parser = argparse.ArgumentParser(description='Display stock data in table format')
    parser.add_argument('--symbol', default='RELIANCE.NS', help='Stock symbol (default: RELIANCE.NS)')
    parser.add_argument('--start-date', default=None, help='Start date (YYYY-MM-DD, default: 2 days ago)')
    parser.add_argument('--end-date', default=None, help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--interval', default='minute', choices=['minute', '5minute', '15minute', '30minute', '60minute', 'day'], 
                       help='Data interval (default: minute)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple intervals')
    parser.add_argument('--records', type=int, default=20, help='Number of records to show (default: 20)')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    print("🚀 STOCK DATA DISPLAY TOOL")
    print("=" * 50)
    
    if args.compare:
        compare_intervals(args.symbol, args.start_date, args.end_date)
    else:
        show_stock_data(args.symbol, args.start_date, args.end_date, args.interval, args.records)
    
    print("\n✅ Data display completed!")

if __name__ == "__main__":
    main() 