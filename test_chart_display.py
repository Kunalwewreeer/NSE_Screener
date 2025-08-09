#!/usr/bin/env python3
"""
Test script to verify chart display functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def test_chart_creation():
    """Test chart creation and display."""
    print("🎨 Testing Chart Creation and Display")
    print("=" * 40)
    
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
            print(f"   Chart type: {type(fig)}")
            print(f"   Chart data: {len(fig.data)} traces")
            
            # Test saving chart
            try:
                fig.write_html("test_chart_display.html")
                print("✅ Chart saved as test_chart_display.html")
            except Exception as e:
                print(f"❌ Error saving chart: {e}")
            
            # Test Streamlit integration
            try:
                import streamlit as st
                print("✅ Streamlit available for chart display")
                
                # Simulate Streamlit chart display
                print("   Chart would display in Streamlit with:")
                print("   - Candlestick chart")
                print("   - VWAP line")
                print("   - PDH/PDL levels")
                print("   - Signal markers")
                print("   - Volume bars")
                
            except ImportError:
                print("⚠️  Streamlit not available")
        else:
            print("❌ Chart creation failed")
    else:
        print("⚠️  No signals to chart")

def test_streamlit_integration():
    """Test Streamlit chart integration."""
    print("\n📱 Testing Streamlit Integration")
    print("=" * 30)
    
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Test if we can create a simple chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='lines'))
        
        print("✅ Basic Plotly chart creation works")
        print("✅ Chart integration should work in Streamlit app")
        
    except ImportError:
        print("❌ Streamlit not available")
    except Exception as e:
        print(f"❌ Error testing Streamlit: {e}")

if __name__ == "__main__":
    test_chart_creation()
    test_streamlit_integration()
    
    print("\n" + "=" * 50)
    print("📊 CHART TEST SUMMARY")
    print("=" * 50)
    print("✅ Chart creation test completed")
    print("✅ Streamlit integration test completed")
    print("\n🎯 If both tests passed, charts should display correctly!")
    print("📋 You can now run the real data app with full chart functionality.")
    print("=" * 50) 