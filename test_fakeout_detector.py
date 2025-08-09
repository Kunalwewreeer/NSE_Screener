#!/usr/bin/env python3
"""
Test script for the Fakeout Detector
Demonstrates usage with sample data and different configurations.
"""

import pandas as pd
import numpy as np
from fakeout_detector import FakeoutDetector, create_sample_data

def test_basic_detection():
    """Test basic fakeout detection with sample data."""
    print("🧪 Testing Basic Fakeout Detection")
    print("=" * 50)
    
    # Create sample data
    df, vwap = create_sample_data()
    print(f"Created sample data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Initialize detector with default config
    detector = FakeoutDetector()
    
    # Detect signals
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    # Print results
    detector.print_debug_summary()
    
    return detector, df, vwap, signals

def test_custom_config():
    """Test with custom configuration."""
    print("\n🧪 Testing Custom Configuration")
    print("=" * 50)
    
    # Create sample data
    df, vwap = create_sample_data()
    
    # Custom config for more sensitive detection
    config = {
        'wick_threshold_pct': 0.2,  # Lower wick threshold
        'confirmation_threshold_pct': 0.3,  # Lower confirmation threshold
        'level_tolerance_pct': 0.05,  # Tighter level tolerance
        'lookback_window': 10,  # Shorter lookback
        'min_candles_between_signals': 3,  # Allow more frequent signals
        'sl_atr_multiplier': 1.0,  # Tighter stop loss
        'tp_atr_multiplier': 1.5,  # Closer take profit
        'atr_period': 10,
        'debug_mode': True,
        'log_level': 'INFO'
    }
    
    detector = FakeoutDetector(config)
    
    # Test different level types
    level_types = ['pdh_pdl', 'vwap', 'support_resistance']
    
    for level_type in level_types:
        print(f"\n--- Testing {level_type.upper()} ---")
        signals = detector.detect_fakeout_signals(df, vwap, level_type)
        print(f"Found {len(signals)} signals with {level_type}")
        
        if signals:
            # Show first signal details
            signal = signals[0]
            print(f"Sample signal: {signal['signal_type']} at {signal['timestamp']}")
            print(f"Entry: {signal['entry']:.2f}, SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}")

def test_real_data_integration():
    """Test integration with real data format."""
    print("\n🧪 Testing Real Data Integration")
    print("=" * 50)
    
    # Simulate real data format (like from your data handler)
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='1min')
    
    # Create realistic price data with fakeouts
    np.random.seed(123)
    base_price = 18500  # Nifty-like price
    
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            # Add some trend and volatility
            trend = np.sin(i / 100) * 10  # Cyclical trend
            noise = np.random.normal(0, 5)  # Random noise
            price = data[-1]['close'] + trend + noise
        
        # Create OHLCV
        open_price = price
        high_price = price + abs(np.random.normal(0, 8))
        low_price = price - abs(np.random.normal(0, 8))
        close_price = price + np.random.normal(0, 5)
        volume = np.random.randint(50000, 200000)
        
        # Add fakeout patterns
        if i % 30 == 15:  # Every 30 minutes
            if np.random.choice([True, False]):
                # Resistance fakeout
                high_price += 20
                close_price = price - 15  # Close below high (wick)
            else:
                # Support fakeout
                low_price -= 20
                close_price = price + 15  # Close above low (wick)
        
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
    
    print(f"Created realistic data: {len(df)} 1-minute candles")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # Test detection
    detector = FakeoutDetector({
        'wick_threshold_pct': 0.5,
        'debug_mode': True,
        'min_candles_between_signals': 10
    })
    
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    print(f"\nDetection Results:")
    print(f"Total signals: {len(signals)}")
    
    if signals:
        print("\nSignal Details:")
        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            print(f"{i+1}. {signal['signal_type']} at {signal['timestamp']}")
            print(f"   Entry: {signal['entry']:.2f}, SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}")
            print(f"   Level: {signal['level_value']:.2f}")
    
    return detector, df, vwap, signals

def main():
    """Run all tests."""
    print("🚀 Fakeout Detector Test Suite")
    print("=" * 60)
    
    # Test 1: Basic detection
    detector1, df1, vwap1, signals1 = test_basic_detection()
    
    # Test 2: Custom configuration
    test_custom_config()
    
    # Test 3: Real data integration
    detector3, df3, vwap3, signals3 = test_real_data_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Basic detection: {len(signals1)} signals")
    print(f"✅ Real data integration: {len(signals3)} signals")
    print("\n🎯 All tests completed successfully!")
    
    # Optional: Plot the results
    if signals3:
        print("\n📈 Plotting results...")
        detector3.plot_signals(df3, signals3, vwap3, 'pdh_pdl')

if __name__ == "__main__":
    main() 