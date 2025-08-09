#!/usr/bin/env python3
"""
Test script to verify fakeout detector fixes for signal balance and charting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_balance():
    """Test that both long and short signals are detected."""
    print("🧪 Testing Signal Balance")
    print("=" * 40)
    
    # Create sample data with explicit fakeout patterns
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='5min')
    
    np.random.seed(42)
    base_price = 18500
    
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            price = data[-1]['close'] + np.random.normal(0, 5)
        
        # Create OHLCV
        open_price = price
        high_price = price + abs(np.random.normal(0, 8))
        low_price = price - abs(np.random.normal(0, 8))
        close_price = price + np.random.normal(0, 5)
        volume = np.random.randint(50000, 200000)
        
        # Add explicit fakeout patterns
        if i == 20:  # Resistance fakeout (short signal) - clear pattern
            high_price = price + 40  # Break above resistance
            close_price = price - 25  # Close below with wick
            open_price = price + 5
        elif i == 40:  # Support fakeout (long signal) - clear pattern
            low_price = price - 40   # Break below support
            close_price = price + 25  # Close above with wick
            open_price = price - 5
        elif i == 60:  # Another resistance fakeout
            high_price = price + 35
            close_price = price - 20
            open_price = price + 3
        elif i == 80:  # Another support fakeout
            low_price = price - 35
            close_price = price + 20
            open_price = price - 3
        
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
    
    print(f"✅ Created test data: {len(df)} candles")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # Test with different configurations
    configs = [
        {'wick_threshold_pct': 0.1, 'debug_mode': True, 'lookback_window': 10},  # Very lenient
        {'wick_threshold_pct': 0.2, 'debug_mode': True, 'lookback_window': 15},  # Lenient
        {'wick_threshold_pct': 0.3, 'debug_mode': True, 'lookback_window': 20},  # Standard
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Test Configuration {i+1} ---")
        print(f"Config: {config}")
        
        from fakeout_detector import FakeoutDetector
        
        detector = FakeoutDetector(config)
        signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
        
        long_signals = [s for s in signals if s['signal_type'] == 'long_fakeout']
        short_signals = [s for s in signals if s['signal_type'] == 'short_fakeout']
        
        print(f"Total signals: {len(signals)}")
        print(f"Long signals: {len(long_signals)}")
        print(f"Short signals: {len(short_signals)}")
        
        if signals:
            print("Sample signals:")
            for j, signal in enumerate(signals[:3]):
                print(f"  {j+1}. {signal['signal_type']} at {signal['timestamp']}")
                print(f"     Entry: {signal['entry']:.2f}, SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}")
        
        # Check if we have both types
        if long_signals and short_signals:
            print("✅ SUCCESS: Both long and short signals detected!")
        elif signals:
            print("⚠️  WARNING: Only one type of signal detected")
        else:
            print("❌ FAILURE: No signals detected")

def test_charting():
    """Test that charting works correctly."""
    print("\n🧪 Testing Charting Functionality")
    print("=" * 40)
    
    # Create sample data
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='5min')
    
    np.random.seed(42)
    base_price = 18500
    
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            price = data[-1]['close'] + np.random.normal(0, 10)
        
        # Create OHLCV
        open_price = price
        high_price = price + abs(np.random.normal(0, 15))
        low_price = price - abs(np.random.normal(0, 15))
        close_price = price + np.random.normal(0, 8)
        volume = np.random.randint(50000, 200000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Test detector
    from fakeout_detector import FakeoutDetector
    
    detector = FakeoutDetector({'debug_mode': True})
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    print(f"✅ Detected {len(signals)} signals")
    
    # Test charting
    if signals:
        try:
            fig = detector.plot_signals(df, signals, vwap, 'pdh_pdl')
            if fig:
                print("✅ SUCCESS: Chart created successfully!")
                print(f"   Chart type: {type(fig)}")
                print(f"   Chart data: {len(fig.data)} traces")
            else:
                print("❌ FAILURE: Chart creation returned None")
        except Exception as e:
            print(f"❌ FAILURE: Chart creation failed: {e}")
    else:
        print("⚠️  WARNING: No signals to chart")

def main():
    """Run all tests."""
    print("🚀 Fakeout Detector Fixes Test Suite")
    print("=" * 60)
    
    # Test signal balance
    test_signal_balance()
    
    # Test charting
    test_charting()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print("✅ Signal balance test completed")
    print("✅ Charting test completed")
    print("\n🎯 If both tests passed, the fixes are working!")
    print("📋 You can now run the real data app with better signal detection.")
    print("=" * 60)

if __name__ == "__main__":
    main() 