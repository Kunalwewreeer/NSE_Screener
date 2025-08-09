#!/usr/bin/env python3
"""
Test script for Fakeout Detector

Demonstrates the fakeout detection functionality with sample data and various configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the strategies directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))

from fakeout_detector import FakeoutDetector, detect_fakeout_signals

def create_sample_data(days: int = 5, minutes_per_day: int = 390) -> pd.DataFrame:
    """
    Create sample OHLCV data with fakeout patterns for testing.
    
    Args:
        days: Number of trading days to generate
        minutes_per_day: Minutes per trading day
        
    Returns:
        DataFrame with OHLCV data and fakeout patterns
    """
    print("📊 Creating sample data with fakeout patterns...")
    
    # Generate datetime index
    start_date = datetime(2024, 1, 1, 9, 15)
    timestamps = []
    
    for day in range(days):
        for minute in range(minutes_per_day):
            timestamp = start_date + timedelta(days=day, minutes=minute)
            timestamps.append(timestamp)
    
    # Create base price movement
    np.random.seed(42)  # For reproducible results
    base_price = 100
    price_changes = np.random.normal(0, 0.001, len(timestamps))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Add some volatility
        volatility = 0.002
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = price * (1 + np.random.normal(0, volatility * 0.5))
        close_price = price * (1 + np.random.normal(0, volatility * 0.5))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Add fakeout patterns manually
    df = _add_fakeout_patterns(df)
    
    print(f"✅ Created sample data: {len(df)} candles across {days} days")
    return df

def _add_fakeout_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add manual fakeout patterns to the sample data.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with added fakeout patterns
    """
    df = df.copy()
    
    # Pattern 1: Resistance fakeout (short signal)
    # Around candle 500
    if len(df) > 500:
        # Create a resistance level
        resistance_level = df.iloc[490:500]['high'].max()
        
        # Breakout candle with wick
        df.iloc[500, df.columns.get_loc('high')] = resistance_level * 1.02  # Breakout
        df.iloc[500, df.columns.get_loc('close')] = resistance_level * 0.99  # Close below
        df.iloc[500, df.columns.get_loc('volume')] = df.iloc[500]['volume'] * 2  # High volume
        
        # Confirmation candle
        df.iloc[501, df.columns.get_loc('close')] = resistance_level * 0.97  # Further below
        
        print(f"🔴 Added resistance fakeout pattern at candle 500, level: {resistance_level:.2f}")
    
    # Pattern 2: Support fakeout (long signal)
    # Around candle 1000
    if len(df) > 1000:
        # Create a support level
        support_level = df.iloc[990:1000]['low'].min()
        
        # Breakout candle with wick
        df.iloc[1000, df.columns.get_loc('low')] = support_level * 0.98  # Breakout
        df.iloc[1000, df.columns.get_loc('close')] = support_level * 1.01  # Close above
        df.iloc[1000, df.columns.get_loc('volume')] = df.iloc[1000]['volume'] * 2  # High volume
        
        # Confirmation candle
        df.iloc[1001, df.columns.get_loc('close')] = support_level * 1.03  # Further above
        
        print(f"🟢 Added support fakeout pattern at candle 1000, level: {support_level:.2f}")
    
    return df

def test_basic_detection():
    """Test basic fakeout detection functionality."""
    print("\n🧪 Testing Basic Fakeout Detection")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(days=3, minutes_per_day=200)
    
    # Test with default configuration
    config = {
        'debug_mode': True,
        'log_level': 'INFO',
        'breakout_threshold': 0.01,  # 1% breakout
        'wick_threshold': 0.2,  # 20% wick
        'confirmation_candles': 1,  # 1 confirmation candle
        'sl_multiplier': 1.5,
        'tp_multiplier': 2.0,
        'plot_signals': True
    }
    
    print(f"📋 Configuration: {config}")
    
    # Detect fakeout signals
    fakeout_signals, breakout_signals = detect_fakeout_signals(
        df, config=config, level_type='pdh_pdl', plot=True
    )
    
    # Display results
    print(f"\n📊 Results:")
    print(f"Breakout signals detected: {len(breakout_signals)}")
    print(f"Fakeout signals confirmed: {len(fakeout_signals)}")
    
    if fakeout_signals:
        print("\n🎯 Fakeout Signals:")
        for i, signal in enumerate(fakeout_signals, 1):
            print(f"Signal {i}:")
            print(f"  Type: {signal['signal_type']}")
            print(f"  Entry Time: {signal['entry_time']}")
            print(f"  Entry Price: {signal['entry_price']:.2f}")
            print(f"  SL Price: {signal['sl_price']:.2f}")
            print(f"  TP Price: {signal['tp_price']:.2f}")
            print(f"  Level: {signal['level']:.2f}")
            print(f"  ATR: {signal['atr']:.4f}")
            print(f"  Volume Ratio: {signal['volume_ratio']:.2f}")
            print(f"  Wick Ratio: {signal['wick_ratio']:.2f}")
            print()

def test_different_configurations():
    """Test fakeout detection with different configurations."""
    print("\n🧪 Testing Different Configurations")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(days=2, minutes_per_day=150)
    
    # Test configurations
    configs = [
        {
            'name': 'Conservative',
            'breakout_threshold': 0.02,  # 2% breakout
            'wick_threshold': 0.4,  # 40% wick
            'confirmation_candles': 2,  # 2 confirmation candles
            'volume_spike_threshold': 2.0,  # Higher volume requirement
        },
        {
            'name': 'Aggressive',
            'breakout_threshold': 0.005,  # 0.5% breakout
            'wick_threshold': 0.1,  # 10% wick
            'confirmation_candles': 1,  # 1 confirmation candle
            'volume_spike_threshold': 1.2,  # Lower volume requirement
        },
        {
            'name': 'VWAP Levels',
            'breakout_threshold': 0.01,  # 1% breakout
            'wick_threshold': 0.3,  # 30% wick
            'confirmation_candles': 1,  # 1 confirmation candle
        }
    ]
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} Configuration:")
        
        # Remove name from config for detector
        detector_config = {k: v for k, v in config.items() if k != 'name'}
        detector_config.update({
            'debug_mode': True,
            'log_level': 'INFO',
            'plot_signals': False  # Disable plotting for multiple tests
        })
        
        # Test with PDH/PDL levels
        fakeout_signals, breakout_signals = detect_fakeout_signals(
            df, config=detector_config, level_type='pdh_pdl', plot=False
        )
        
        print(f"  Breakout signals: {len(breakout_signals)}")
        print(f"  Fakeout signals: {len(fakeout_signals)}")
        
        # Test with VWAP levels for VWAP config
        if config['name'] == 'VWAP Levels':
            fakeout_signals_vwap, breakout_signals_vwap = detect_fakeout_signals(
                df, config=detector_config, level_type='vwap', plot=False
            )
            print(f"  VWAP Breakout signals: {len(breakout_signals_vwap)}")
            print(f"  VWAP Fakeout signals: {len(fakeout_signals_vwap)}")

def test_custom_levels():
    """Test fakeout detection with custom levels."""
    print("\n🧪 Testing Custom Levels")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(days=1, minutes_per_day=100)
    
    # Add custom levels
    df['pdh'] = df['high'].rolling(window=10).max()  # Custom resistance
    df['pdl'] = df['low'].rolling(window=10).min()   # Custom support
    
    print("📊 Added custom PDH/PDL levels")
    
    # Test detection
    config = {
        'debug_mode': True,
        'log_level': 'INFO',
        'breakout_threshold': 0.01,
        'wick_threshold': 0.3,
        'confirmation_candles': 1,
        'plot_signals': True
    }
    
    fakeout_signals, breakout_signals = detect_fakeout_signals(
        df, config=config, level_type='custom', plot=True
    )
    
    print(f"📊 Results with custom levels:")
    print(f"Breakout signals: {len(breakout_signals)}")
    print(f"Fakeout signals: {len(fakeout_signals)}")

def main():
    """Run all tests."""
    print("🚀 Fakeout Detector Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_basic_detection()
        
        # Test different configurations
        test_different_configurations()
        
        # Test custom levels
        test_custom_levels()
        
        print("\n✅ All tests completed successfully!")
        print("\n📝 Usage Examples:")
        print("1. Basic usage: detect_fakeout_signals(df)")
        print("2. Custom config: detect_fakeout_signals(df, config=my_config)")
        print("3. VWAP levels: detect_fakeout_signals(df, level_type='vwap')")
        print("4. Custom levels: detect_fakeout_signals(df, level_type='custom')")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 