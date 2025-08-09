#!/usr/bin/env python3
"""
Runner script for VWAP Reversal Strategy Backtest
Quick execution with default parameters for testing the strategy.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_vwap_backtester import SimpleVWAPBacktester
from core.data_handler import DataHandler

def run_quick_backtest():
    """Run a quick backtest with default parameters."""
    print("🚀 Starting Quick VWAP Reversal Backtest...")
    
    # Initialize backtester
    backtester = SimpleVWAPBacktester()
    data_handler = DataHandler()
    
    # Default parameters
    universe_type = "nifty50"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    top_k = 5
    initial_capital = 100000
    
    print(f"📊 Universe: {universe_type}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🎯 Top K: {top_k}")
    print(f"💰 Initial Capital: ₹{initial_capital:,.2f}")
    
    # Get universe
    universe = data_handler.get_stocks_by_universe(universe_type)
    print(f"📈 Universe size: {len(universe)} stocks")
    
    # Run backtest
    results = backtester.run_backtest(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        initial_capital=initial_capital
    )
    
    if results:
        print("\n✅ Backtest completed successfully!")
        print("\n📊 Performance Summary:")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Final Capital: ₹{results['final_capital']:,.2f}")
        
        # Show some recent trades
        if results['trades']:
            print("\n📋 Recent Trades:")
            for trade in results['trades'][-5:]:
                print(f"   {trade['symbol']} {trade['position']}: {trade['pnl_pct']:.2f}% "
                      f"({trade['entry_date']} -> {trade['exit_date']})")
    else:
        print("❌ Backtest failed!")

def run_custom_backtest():
    """Run a custom backtest with user input."""
    print("🔧 Custom VWAP Reversal Backtest")
    print("=" * 50)
    
    # Get user inputs
    universe_type = input("Enter universe type (nifty50/nifty100/nifty500): ").strip() or "nifty50"
    
    # Date inputs
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ").strip()
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 1 year ago: ").strip()
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Strategy parameters
    try:
        top_k = int(input("Enter number of top stocks (1-10, default 5): ").strip() or "5")
        top_k = max(1, min(10, top_k))
    except ValueError:
        top_k = 5
    
    try:
        initial_capital = float(input("Enter initial capital (₹, default 100000): ").strip() or "100000")
    except ValueError:
        initial_capital = 100000
    
    print(f"\n📊 Configuration:")
    print(f"   Universe: {universe_type}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Top K: {top_k}")
    print(f"   Initial Capital: ₹{initial_capital:,.2f}")
    
    # Initialize and run
    backtester = SimpleVWAPBacktester()
    data_handler = DataHandler()
    
    universe = data_handler.get_stocks_by_universe(universe_type)
    print(f"📈 Universe size: {len(universe)} stocks")
    
    results = backtester.run_backtest(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        initial_capital=initial_capital
    )
    
    if results:
        print("\n✅ Backtest completed successfully!")
        print("\n📊 Performance Summary:")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Final Capital: ₹{results['final_capital']:,.2f}")
    else:
        print("❌ Backtest failed!")

if __name__ == "__main__":
    print("🔍 VWAP Reversal Strategy Backtester")
    print("=" * 50)
    print("1. Quick backtest (default parameters)")
    print("2. Custom backtest (user input)")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "2":
        run_custom_backtest()
    else:
        run_quick_backtest() 