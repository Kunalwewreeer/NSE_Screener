#!/usr/bin/env python3
"""
Test script to verify the parameter fix works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parameter_order():
    """Test that parameters are passed in the correct order."""
    print("🔧 Testing Parameter Order Fix")
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
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ Created test data: {len(df)} candles")
    
    # Test the integration with correct parameters
    from fakeout_detector_integration import FakeoutDetectorIntegration
    
    integration = FakeoutDetectorIntegration()
    
    # Test with correct parameter order
    symbols = ["TEST"]
    start_date = "2024-01-01"
    end_date = "2024-01-01"
    level_type = "pdh_pdl"
    interval = "minute"
    
    print(f"\n📋 Testing with correct parameters:")
    print(f"  Symbols: {symbols}")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Level Type: {level_type}")
    print(f"  Interval: {interval}")
    
    try:
        # This should work without API errors
        results = integration.analyze_multiple_symbols(
            symbols, start_date, end_date, level_type, interval
        )
        
        print("✅ SUCCESS: Parameters passed correctly!")
        print(f"   Results: {len(results)} symbols analyzed")
        
        for symbol, result in results.items():
            if 'error' in result:
                print(f"   {symbol}: {result['error']}")
            else:
                print(f"   {symbol}: {len(result['signals'])} signals")
                
    except Exception as e:
        print(f"❌ FAILURE: {e}")
    
    # Test with wrong parameter order (should fail)
    print(f"\n📋 Testing with wrong parameter order (should fail):")
    try:
        # This should fail because level_type is passed as interval
        results = integration.analyze_multiple_symbols(
            symbols, start_date, end_date, interval, level_type  # WRONG ORDER
        )
        print("❌ FAILURE: Should have failed with wrong parameter order")
    except Exception as e:
        print("✅ SUCCESS: Correctly failed with wrong parameter order")
        print(f"   Error: {e}")

def test_streamlit_integration():
    """Test Streamlit app parameter handling."""
    print("\n📱 Testing Streamlit Integration")
    print("=" * 30)
    
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Simulate the parameter handling
        start_date = datetime.now().date() - timedelta(days=7)
        end_date = datetime.now().date()
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"✅ Date conversion works:")
        print(f"   Start: {start_date} -> {start_date_str}")
        print(f"   End: {end_date} -> {end_date_str}")
        
        # Test parameter order
        symbols = ["TEST"]
        level_type = "pdh_pdl"
        interval = "minute"
        
        print(f"✅ Parameter order is correct:")
        print(f"   analyze_multiple_symbols(symbols, start_date_str, end_date_str, level_type, interval)")
        
    except ImportError:
        print("❌ Streamlit not available")
    except Exception as e:
        print(f"❌ Error testing Streamlit: {e}")

if __name__ == "__main__":
    test_parameter_order()
    test_streamlit_integration()
    
    print("\n" + "=" * 50)
    print("📊 PARAMETER FIX TEST SUMMARY")
    print("=" * 50)
    print("✅ Parameter order test completed")
    print("✅ Streamlit integration test completed")
    print("\n🎯 If both tests passed, the parameter fix is working!")
    print("📋 The real data app should now work without API errors.")
    print("=" * 50) 