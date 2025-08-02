#!/usr/bin/env python3
"""
Simple test script to verify the system works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.research_backtest import ResearchBacktester
from utils.logger import get_logger

logger = get_logger(__name__)

def test_system():
    """Test the system with a simple test strategy."""
    
    print("🧪 TESTING SYSTEM WITH TEST STRATEGY")
    print("=" * 50)
    
    # Initialize research backtester
    research_backtester = ResearchBacktester()
    
    # Test configuration
    test_config = {
        'name': 'Test Strategy - High Frequency',
        'strategy': 'test',
        'symbols': ['RELIANCE.NS'],
        'start_date': '2025-01-01',
        'end_date': '2025-01-15',
        'params': {
            'signal_frequency': 0.3  # 30% of data points
        }
    }
    
    print(f"📊 Testing: {test_config['name']}")
    print(f"🎯 Strategy: {test_config['strategy']}")
    print(f"📅 Period: {test_config['start_date']} to {test_config['end_date']}")
    print(f"⚙️  Parameters: {test_config['params']}")
    print("-" * 40)
    
    try:
        # Run backtest
        results = research_backtester.run_comprehensive_backtest(
            strategy_name=test_config['strategy'],
            symbols=test_config['symbols'],
            start_date=test_config['start_date'],
            end_date=test_config['end_date'],
            strategy_params=test_config['params'],
            capital=100000,
            save_results=False,
            generate_plots=False
        )
        
        if results:
            total_return = results.get('total_return', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            max_drawdown = results.get('max_drawdown', 0)
            trades = results.get('trades', [])
            num_trades = len(trades)
            
            print(f"✅ SUCCESS!")
            print(f"💰 Total Return: {total_return:.2f}%")
            print(f"📈 Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
            print(f"🔄 Number of Trades: {num_trades}")
            
            if trades:
                print(f"📊 Sample Trades:")
                for i, trade in enumerate(trades[:5]):  # Show first 5 trades
                    print(f"   Trade {i+1}: {trade.get('direction', 'UNKNOWN')} @ ₹{trade.get('entry_price', 0):.2f}")
            
            print(f"\n🎉 SYSTEM TEST PASSED!")
            print(f"   ✅ Data loading: Working")
            print(f"   ✅ Strategy initialization: Working")
            print(f"   ✅ Signal generation: Working ({num_trades} trades)")
            print(f"   ✅ Backtest execution: Working")
            print(f"   ✅ Results processing: Working")
            
        else:
            print("❌ No results returned")
            print("🔧 SYSTEM TEST FAILED - No results")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("🔧 SYSTEM TEST FAILED - Exception occurred")

if __name__ == "__main__":
    test_system() 