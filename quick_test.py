#!/usr/bin/env python3
"""
Quick test to verify system components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.test_strategy import TestStrategy
from core.data_handler import DataHandler
from utils.logger import get_logger

logger = get_logger(__name__)

def quick_test():
    """Quick test of system components."""
    
    print("🧪 QUICK SYSTEM TEST")
    print("=" * 30)
    
    # Test 1: Data Handler
    print("1. Testing Data Handler...")
    try:
        data_handler = DataHandler()
        data = data_handler.get_historical_data(
            symbols=['RELIANCE.NS'],
            from_date='2025-01-01',
            to_date='2025-01-15',
            interval='day'
        )
        print(f"   ✅ Data loaded: {len(data)} records")
    except Exception as e:
        print(f"   ❌ Data handler failed: {e}")
        return
    
    # Test 2: Test Strategy
    print("2. Testing Test Strategy...")
    try:
        strategy = TestStrategy("Test", {'signal_frequency': 1.0})  # 100% frequency
        print(f"   ✅ Strategy initialized: {strategy.name}")
        
        # Test signal generation
        if isinstance(data, dict) and 'RELIANCE.NS' in data:
            df = data['RELIANCE.NS']
        else:
            df = data
            
        signals = strategy.generate_signals(df)
        print(f"   ✅ Signal generation: {len(signals)} signals")
        
        if signals:
            print(f"   📊 Sample signal: {signals[0]}")
        
    except Exception as e:
        print(f"   ❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 QUICK TEST PASSED!")
    print("   ✅ Data loading: Working")
    print("   ✅ Strategy initialization: Working")
    print("   ✅ Signal generation: Working")

if __name__ == "__main__":
    quick_test() 