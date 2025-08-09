#!/usr/bin/env python3
"""
Test script to verify the summary fix works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_summary_creation():
    """Test that summary creation works correctly."""
    print("📊 Testing Summary Creation Fix")
    print("=" * 40)
    
    # Create mock analysis results
    mock_results = {
        'SYMBOL1': {
            'signals': [
                {'signal_type': 'long_fakeout', 'timestamp': datetime.now(), 'entry': 100, 'stop_loss': 95, 'take_profit': 110},
                {'signal_type': 'short_fakeout', 'timestamp': datetime.now(), 'entry': 100, 'stop_loss': 105, 'take_profit': 90},
                {'signal_type': 'long_fakeout', 'timestamp': datetime.now(), 'entry': 102, 'stop_loss': 97, 'take_profit': 112}
            ],
            'data_points': 1000,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        },
        'SYMBOL2': {
            'signals': [
                {'signal_type': 'short_fakeout', 'timestamp': datetime.now(), 'entry': 200, 'stop_loss': 205, 'take_profit': 190},
                {'signal_type': 'long_fakeout', 'timestamp': datetime.now(), 'entry': 200, 'stop_loss': 195, 'take_profit': 210}
            ],
            'data_points': 800,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        },
        'SYMBOL3': {
            'signals': [],
            'data_points': 500,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        }
    }
    
    print(f"✅ Created mock results with {len(mock_results)} symbols")
    
    # Test summary creation
    from fakeout_detector_integration import FakeoutDetectorIntegration
    
    integration = FakeoutDetectorIntegration()
    
    try:
        summary = integration.create_analysis_summary(mock_results)
        
        print("✅ SUCCESS: Summary created successfully!")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Symbols with signals: {summary['symbols_with_signals']}")
        print(f"   Total signals: {summary['total_signals']}")
        print(f"   Long signals: {summary['total_long_signals']}")
        print(f"   Short signals: {summary['total_short_signals']}")
        print(f"   Total data points: {summary['total_data_points']}")
        print(f"   Avg signals per symbol: {summary['avg_signals_per_symbol']:.2f}")
        print(f"   Long/Short ratio: {summary['long_short_ratio']:.2f}")
        
        # Test top signals
        top_signals = integration.get_top_signals(mock_results, top_n=3)
        print(f"\n✅ Top signals: {len(top_signals)} signals")
        for i, signal in enumerate(top_signals):
            print(f"   {i+1}. {signal['symbol']} - {signal['signal_type']} at {signal['timestamp']}")
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()

def test_empty_results():
    """Test summary creation with empty results."""
    print("\n📊 Testing Empty Results")
    print("=" * 30)
    
    # Create empty results
    empty_results = {
        'SYMBOL1': {
            'signals': [],
            'data_points': 100,
            'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
            'interval': 'minute',
            'level_type': 'pdh_pdl'
        }
    }
    
    from fakeout_detector_integration import FakeoutDetectorIntegration
    
    integration = FakeoutDetectorIntegration()
    
    try:
        summary = integration.create_analysis_summary(empty_results)
        
        print("✅ SUCCESS: Empty summary created successfully!")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Symbols with signals: {summary['symbols_with_signals']}")
        print(f"   Total signals: {summary['total_signals']}")
        print(f"   Avg signals per symbol: {summary['avg_signals_per_symbol']}")
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")

if __name__ == "__main__":
    test_summary_creation()
    test_empty_results()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY FIX TEST SUMMARY")
    print("=" * 50)
    print("✅ Summary creation test completed")
    print("✅ Empty results test completed")
    print("\n🎯 If both tests passed, the summary fix is working!")
    print("📋 The real data app should now work without summary errors.")
    print("=" * 50) 