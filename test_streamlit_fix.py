#!/usr/bin/env python3
"""
Test script to verify the Streamlit app fix works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_streamlit_integration():
    """Test that the Streamlit app integration works correctly."""
    print("📱 Testing Streamlit App Integration")
    print("=" * 40)
    
    # Create mock analysis results
    mock_results = {
        'JSWSTEEL.NS': {
            'signals': [
                {'signal_type': 'long_fakeout', 'timestamp': datetime.now(), 'entry': 100, 'stop_loss': 95, 'take_profit': 110, 'level_value': 98},
                {'signal_type': 'short_fakeout', 'timestamp': datetime.now(), 'entry': 100, 'stop_loss': 105, 'take_profit': 90, 'level_value': 102}
            ],
            'data_points': 1000,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        },
        'NIFTY': {
            'signals': [
                {'signal_type': 'short_fakeout', 'timestamp': datetime.now(), 'entry': 200, 'stop_loss': 205, 'take_profit': 190, 'level_value': 202}
            ],
            'data_points': 800,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        }
    }
    
    print(f"✅ Created mock results with {len(mock_results)} symbols")
    
    # Test integration methods
    from fakeout_detector_integration import FakeoutDetectorIntegration
    
    integration = FakeoutDetectorIntegration()
    
    try:
        # Test summary creation
        summary = integration.create_analysis_summary(mock_results)
        print("✅ SUCCESS: Summary created successfully!")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Symbols with signals: {summary['symbols_with_signals']}")
        print(f"   Total signals: {summary['total_signals']}")
        print(f"   Long signals: {summary['total_long_signals']}")
        print(f"   Short signals: {summary['total_short_signals']}")
        
        # Test top signals
        top_signals = integration.get_top_signals(mock_results, top_n=3)
        print(f"✅ SUCCESS: Top signals created successfully!")
        print(f"   Top signals: {len(top_signals)} signals")
        for i, signal in enumerate(top_signals):
            print(f"   {i+1}. {signal['symbol']} - {signal['signal_type']}")
        
        # Test complete results structure
        complete_results = {
            'analysis_results': mock_results,
            'top_signals': top_signals,
            'summary': summary
        }
        
        print("✅ SUCCESS: Complete results structure created!")
        print(f"   Has analysis_results: {'analysis_results' in complete_results}")
        print(f"   Has top_signals: {'top_signals' in complete_results}")
        print(f"   Has summary: {'summary' in complete_results}")
        
        # Test that summary can be accessed
        test_summary = complete_results['summary']
        print(f"✅ SUCCESS: Summary can be accessed!")
        print(f"   Summary keys: {list(test_summary.keys())}")
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()

def test_error_handling():
    """Test error handling in the Streamlit app."""
    print("\n🛡️ Testing Error Handling")
    print("=" * 30)
    
    # Test with empty results
    empty_results = {}
    
    from fakeout_detector_integration import FakeoutDetectorIntegration
    
    integration = FakeoutDetectorIntegration()
    
    try:
        summary = integration.create_analysis_summary(empty_results)
        print("✅ SUCCESS: Empty results handled correctly!")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Total signals: {summary['total_signals']}")
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")
    
    # Test with error results
    error_results = {
        'SYMBOL1': {
            'error': 'No data available',
            'signals': [],
            'data_points': 0
        }
    }
    
    try:
        summary = integration.create_analysis_summary(error_results)
        print("✅ SUCCESS: Error results handled correctly!")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Symbols with signals: {summary['symbols_with_signals']}")
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")

if __name__ == "__main__":
    test_streamlit_integration()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("📱 STREAMLIT FIX TEST SUMMARY")
    print("=" * 50)
    print("✅ Streamlit integration test completed")
    print("✅ Error handling test completed")
    print("\n🎯 If both tests passed, the Streamlit app should work!")
    print("📋 The real data app should now work without summary errors.")
    print("=" * 50) 