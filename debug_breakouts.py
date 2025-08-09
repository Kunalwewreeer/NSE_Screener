#!/usr/bin/env python3
"""
Detailed debug script for breakout detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_breakout_detection():
    """Debug the breakout detection process step by step."""
    print("🔍 Debugging Breakout Detection")
    print("=" * 40)
    
    # Create test data with very explicit fakeouts
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='5min')
    
    data = []
    base_price = 18500
    
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            price = data[-1]['close'] + np.random.normal(0, 2)
        
        # Create OHLCV
        open_price = price
        high_price = price + abs(np.random.normal(0, 5))
        low_price = price - abs(np.random.normal(0, 5))
        close_price = price + np.random.normal(0, 3)
        volume = np.random.randint(50000, 200000)
        
        # Add very explicit fakeout patterns
        if i == 20:  # Resistance fakeout - very clear
            open_price = price + 5
            high_price = price + 50  # Big breakout
            close_price = price - 30  # Big wick
            low_price = price - 10
        elif i == 40:  # Support fakeout - very clear
            open_price = price - 5
            low_price = price - 50   # Big breakout
            close_price = price + 30  # Big wick
            high_price = price + 10
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ Created test data: {len(df)} candles")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # Test level calculation
    from fakeout_detector import FakeoutDetector
    
    detector = FakeoutDetector({'debug_mode': True, 'lookback_window': 10, 'wick_threshold_pct': 0.1})
    
    # Calculate levels
    df_with_levels = detector.calculate_key_levels(df, 'pdh_pdl')
    
    print(f"\n📊 Level Analysis:")
    print(f"PDH range: {df_with_levels['pdh'].min():.2f} - {df_with_levels['pdh'].max():.2f}")
    print(f"PDL range: {df_with_levels['pdl'].min():.2f} - {df_with_levels['pdl'].max():.2f}")
    
    # Check specific fakeout points
    print(f"\n🎯 Checking Fakeout Points:")
    
    # Check resistance fakeout at index 20
    idx_20 = df_with_levels.index[20]
    candle_20 = df_with_levels.loc[idx_20]
    print(f"\nIndex 20 (Resistance Fakeout):")
    print(f"  High: {candle_20['high']:.2f}")
    print(f"  Close: {candle_20['close']:.2f}")
    print(f"  PDH Level: {candle_20['pdh']:.2f}")
    print(f"  High > PDH: {candle_20['high'] > candle_20['pdh']}")
    print(f"  Close < PDH: {candle_20['close'] < candle_20['pdh']}")
    print(f"  Wick size: {candle_20['high'] - candle_20['close']:.2f}")
    print(f"  Wick %: {((candle_20['high'] - candle_20['close']) / candle_20['high'] * 100):.2f}%")
    
    # Check support fakeout at index 40
    idx_40 = df_with_levels.index[40]
    candle_40 = df_with_levels.loc[idx_40]
    print(f"\nIndex 40 (Support Fakeout):")
    print(f"  Low: {candle_40['low']:.2f}")
    print(f"  Close: {candle_40['close']:.2f}")
    print(f"  PDL Level: {candle_40['pdl']:.2f}")
    print(f"  Low < PDL: {candle_40['low'] < candle_40['pdl']}")
    print(f"  Close > PDL: {candle_40['close'] > candle_40['pdl']}")
    print(f"  Wick size: {candle_40['close'] - candle_40['low']:.2f}")
    print(f"  Wick %: {((candle_40['close'] - candle_40['low']) / candle_40['close'] * 100):.2f}%")
    
    # Test breakout detection manually
    print(f"\n🔍 Manual Breakout Detection:")
    
    # Test resistance breakout
    resistance_breakouts = detector.detect_breakout_candle(df_with_levels, df_with_levels['pdh'], 'resistance')
    print(f"Resistance breakouts found: {resistance_breakouts.sum()}")
    
    # Test support breakout
    support_breakouts = detector.detect_breakout_candle(df_with_levels, df_with_levels['pdl'], 'support')
    print(f"Support breakouts found: {support_breakouts.sum()}")
    
    # Show all breakout points
    resistance_points = resistance_breakouts[resistance_breakouts].index
    support_points = support_breakouts[support_breakouts].index
    
    print(f"\n📍 All Resistance Breakout Points:")
    for point in resistance_points:
        candle = df_with_levels.loc[point]
        print(f"  {point}: High={candle['high']:.2f}, Close={candle['close']:.2f}, Level={candle['pdh']:.2f}")
    
    print(f"\n📍 All Support Breakout Points:")
    for point in support_points:
        candle = df_with_levels.loc[point]
        print(f"  {point}: Low={candle['low']:.2f}, Close={candle['close']:.2f}, Level={candle['pdl']:.2f}")
    
    # Test full signal detection
    print(f"\n🎯 Testing Full Signal Detection:")
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    print(f"Total signals detected: {len(signals)}")
    for i, signal in enumerate(signals):
        print(f"  Signal {i+1}: {signal['signal_type']} at {signal['timestamp']}")

if __name__ == "__main__":
    debug_breakout_detection() 