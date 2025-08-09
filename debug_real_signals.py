#!/usr/bin/env python3
"""
Debug script for real data signal detection and charting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_real_data_signals():
    """Debug signal detection with real data patterns."""
    print("🔍 Debugging Real Data Signal Detection")
    print("=" * 50)
    
    # Create realistic market data with clear fakeouts
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='1min')
    
    data = []
    base_price = 18500
    
    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            # More realistic price movement
            price = data[-1]['close'] + np.random.normal(0, 2)
        
        # Create OHLCV with more realistic patterns
        open_price = price
        high_price = price + abs(np.random.normal(0, 8))
        low_price = price - abs(np.random.normal(0, 8))
        close_price = price + np.random.normal(0, 5)
        volume = np.random.randint(50000, 200000)
        
        # Add very clear fakeout patterns at specific times
        if i == 120:  # 11:15 AM - Resistance fakeout
            open_price = price + 5
            high_price = price + 60  # Big breakout above resistance
            close_price = price - 40  # Big wick below
            low_price = price - 15
        elif i == 240:  # 1:20 PM - Support fakeout
            open_price = price - 5
            low_price = price - 60   # Big breakout below support
            close_price = price + 40  # Big wick above
            high_price = price + 15
        elif i == 360:  # 3:15 PM - Another resistance fakeout
            open_price = price + 3
            high_price = price + 50
            close_price = price - 35
            low_price = price - 10
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ Created realistic test data: {len(df)} candles")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # Test with different configurations
    configs = [
        {'wick_threshold_pct': 0.1, 'debug_mode': True, 'lookback_window': 20, 'min_candles_between_signals': 5},
        {'wick_threshold_pct': 0.2, 'debug_mode': True, 'lookback_window': 30, 'min_candles_between_signals': 10},
        {'wick_threshold_pct': 0.3, 'debug_mode': True, 'lookback_window': 50, 'min_candles_between_signals': 15},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Test Configuration {i+1} ---")
        print(f"Config: {config}")
        
        from fakeout_detector import FakeoutDetector
        
        detector = FakeoutDetector(config)
        
        # Calculate VWAP
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Test signal detection
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
        
        # Test charting
        if signals:
            print(f"\n🎨 Testing Chart Creation:")
            try:
                fig = detector.plot_signals(df, signals, vwap, 'pdh_pdl')
                if fig:
                    print("✅ SUCCESS: Chart created successfully!")
                    print(f"   Chart type: {type(fig)}")
                    print(f"   Chart data: {len(fig.data)} traces")
                    
                    # Save chart for inspection
                    fig.write_html("test_chart.html")
                    print("   Chart saved as test_chart.html")
                else:
                    print("❌ FAILURE: Chart creation returned None")
            except Exception as e:
                print(f"❌ FAILURE: Chart creation failed: {e}")
        
        # Check if we have both types
        if long_signals and short_signals:
            print("✅ SUCCESS: Both long and short signals detected!")
        elif signals:
            print("⚠️  WARNING: Only one type of signal detected")
        else:
            print("❌ FAILURE: No signals detected")

def test_chart_integration():
    """Test chart integration with Streamlit."""
    print("\n🎨 Testing Chart Integration")
    print("=" * 30)
    
    # Create sample data
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
        
        # Add fakeout patterns
        if i == 20:  # Resistance fakeout
            high_price = price + 40
            close_price = price - 25
        elif i == 40:  # Support fakeout
            low_price = price - 40
            close_price = price + 25
        
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
    
    detector = FakeoutDetector({'debug_mode': True, 'lookback_window': 15})
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    print(f"✅ Detected {len(signals)} signals")
    
    if signals:
        # Test chart creation
        fig = detector.plot_signals(df, signals, vwap, 'pdh_pdl')
        if fig:
            print("✅ Chart created successfully!")
            print(f"   Chart has {len(fig.data)} traces")
            
            # Test Streamlit integration
            try:
                import streamlit as st
                print("✅ Streamlit available for chart display")
            except ImportError:
                print("⚠️  Streamlit not available")
        else:
            print("❌ Chart creation failed")

if __name__ == "__main__":
    debug_real_data_signals()
    test_chart_integration() 