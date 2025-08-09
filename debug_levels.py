#!/usr/bin/env python3
"""
Debug script to understand level calculation issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_level_calculation():
    """Debug the level calculation process."""
    print("🔍 Debugging Level Calculation")
    print("=" * 40)
    
    # Create simple test data
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='5min')
    
    data = []
    base_price = 18500
    
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
        
        # Add explicit fakeout at specific points
        if i == 20:  # Resistance fakeout
            high_price = price + 40
            close_price = price - 25
            open_price = price + 5
        elif i == 40:  # Support fakeout
            low_price = price - 40
            close_price = price + 25
            open_price = price - 5
        
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
    
    detector = FakeoutDetector({'debug_mode': True, 'lookback_window': 10})
    
    # Calculate levels
    df_with_levels = detector.calculate_key_levels(df, 'pdh_pdl')
    
    print(f"\n📊 Level Analysis:")
    print(f"Original columns: {list(df.columns)}")
    print(f"With levels columns: {list(df_with_levels.columns)}")
    
    # Check for NaN values
    print(f"\n🔍 NaN Check:")
    print(f"PDH NaN count: {df_with_levels['pdh'].isna().sum()}")
    print(f"PDL NaN count: {df_with_levels['pdl'].isna().sum()}")
    
    # Show some sample values
    print(f"\n📈 Sample Values:")
    for i in [0, 10, 20, 30, 40, 50]:
        if i < len(df_with_levels):
            print(f"Index {i}: PDH={df_with_levels['pdh'].iloc[i]:.2f}, PDL={df_with_levels['pdl'].iloc[i]:.2f}")
    
    # Test breakout detection
    print(f"\n🎯 Testing Breakout Detection:")
    
    # Test resistance breakout
    resistance_breakouts = detector.detect_breakout_candle(df_with_levels, df_with_levels['pdh'], 'resistance')
    print(f"Resistance breakouts found: {resistance_breakouts.sum()}")
    
    # Test support breakout
    support_breakouts = detector.detect_breakout_candle(df_with_levels, df_with_levels['pdl'], 'support')
    print(f"Support breakouts found: {support_breakouts.sum()}")
    
    # Show specific breakout points
    resistance_points = resistance_breakouts[resistance_breakouts].index
    support_points = support_breakouts[support_breakouts].index
    
    print(f"\n📍 Resistance Breakout Points:")
    for point in resistance_points[:3]:
        print(f"  {point}: High={df_with_levels.loc[point, 'high']:.2f}, Close={df_with_levels.loc[point, 'close']:.2f}, Level={df_with_levels.loc[point, 'pdh']:.2f}")
    
    print(f"\n📍 Support Breakout Points:")
    for point in support_points[:3]:
        print(f"  {point}: Low={df_with_levels.loc[point, 'low']:.2f}, Close={df_with_levels.loc[point, 'close']:.2f}, Level={df_with_levels.loc[point, 'pdl']:.2f}")

if __name__ == "__main__":
    debug_level_calculation() 