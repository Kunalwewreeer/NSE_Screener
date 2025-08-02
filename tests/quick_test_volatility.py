#!/usr/bin/env python3
"""
Quick test for volatility breakout strategy with 1-minute data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.data_handler import DataHandler
import pandas as pd
import numpy as np

def quick_test():
    """Quick test of volatility breakout strategy."""
    print("🚀 Quick Volatility Breakout Test")
    print("=" * 50)
    
    # Strategy config
    config = {
        'volatility_period': 20,
        'volatility_multiplier': 2.0,
        'momentum_period': 5,
        'volume_threshold': 1.5,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }
    
    # Initialize strategy
    strategy = VolatilityBreakoutStrategy("QuickTest", config)
    print(f"✅ Strategy initialized: {strategy.name}")
    
    # Test with RELIANCE
    symbol = "RELIANCE.NS"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    print(f"\n📊 Testing {symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⏱️  Timeframe: 1-minute")
    
    try:
        # Get data handler
        data_handler = DataHandler()
        
        # Fetch 1-minute data
        print("🔍 Fetching 1-minute data...")
        data = data_handler.get_historical_data(
            symbols=symbol,
            from_date=start_date,
            to_date=end_date,
            interval="minute",
            refresh_cache=True
        )
        
        if data.empty:
            print("❌ No data received")
            return
        
        print(f"✅ Fetched {len(data)} records")
        print(f"📈 Data shape: {data.shape}")
        
        # Calculate indicators
        data_with_indicators = data_handler.calculate_technical_indicators(data)
        print("✅ Technical indicators calculated")
        
        # Generate signals
        print("🔍 Generating signals...")
        signals = strategy.generate_signals(data_with_indicators)
        
        print(f"📊 Generated {len(signals)} signals")
        
        if signals:
            print("\n📋 Signal Summary:")
            buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
            sell_signals = [s for s in signals if s['signal_type'] == 'SELL']
            
            print(f"   📈 Buy signals: {len(buy_signals)}")
            print(f"   📉 Sell signals: {len(sell_signals)}")
            
            if buy_signals:
                print(f"\n📈 Sample Buy Signal:")
                signal = buy_signals[0]
                print(f"   Price: ₹{signal['price']:.2f}")
                print(f"   Stop Loss: ₹{signal['stop_loss']:.2f}")
                print(f"   Take Profit: ₹{signal['take_profit']:.2f}")
                print(f"   Volatility: {signal['metadata']['volatility']:.4f}")
                print(f"   Volume Ratio: {signal['metadata']['volume_ratio']:.2f}")
            
            if sell_signals:
                print(f"\n📉 Sample Sell Signal:")
                signal = sell_signals[0]
                print(f"   Price: ₹{signal['price']:.2f}")
                print(f"   Stop Loss: ₹{signal['stop_loss']:.2f}")
                print(f"   Take Profit: ₹{signal['take_profit']:.2f}")
                print(f"   Volatility: {signal['metadata']['volatility']:.4f}")
                print(f"   Volume Ratio: {signal['metadata']['volume_ratio']:.2f}")
        else:
            print("⚠️  No signals generated")
            print("   This is normal if:")
            print("   - Market conditions don't meet breakout criteria")
            print("   - Volatility is within normal ranges")
            print("   - No significant price movements detected")
        
        print(f"\n✅ Test completed successfully!")
        print(f"🎯 Strategy is ready for backtesting")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 