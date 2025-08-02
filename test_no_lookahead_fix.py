# 🧪 TEST NO-LOOKAHEAD FIX
print("\n🧪 Testing No-Lookahead Scanner Fix")
print("=" * 50)

# Import required modules
import sys
import os
sys.path.append('.')

from core.data_handler import fetch_data
from utils.helpers import load_yaml

# Load the scanner
exec(open('orb_trading_scanner_no_lookahead.py').read())

def test_data_structure():
    """Test that opportunities contain both filtered and full day data."""
    print("\n🔍 Testing data structure...")
    
    # Create scanner
    scanner = ORBTradingScannerNoLookahead()
    
    # Test with a small symbol set
    test_symbols = ['RELIANCE.NS', 'TCS.NS']
    
    try:
        # Fetch data
        print(f"📊 Fetching data for {test_symbols}...")
        all_stock_data = fetch_data(test_symbols, '2025-01-13', '2025-01-15', 'minute')
        
        if not all_stock_data:
            print("❌ No data fetched")
            return
        
        print(f"✅ Fetched data for {len(all_stock_data)} symbols")
        
        # Test one symbol
        test_symbol = test_symbols[0]
        if test_symbol in all_stock_data:
            result = scanner.analyze_stock_no_lookahead(test_symbol, all_stock_data)
            
            if result:
                print(f"\n✅ Analysis successful for {test_symbol}")
                print(f"📊 Available data points: {len(result['data'])}")
                print(f"📊 Full day data points: {len(result['full_day_data'])}")
                print(f"⏰ Evaluation time: {result['evaluation_time']}")
                print(f"🎯 Signal: {result['signal_type']} (Score: {result['trading_score']})")
                
                # Check data integrity
                if len(result['full_day_data']) > len(result['data']):
                    print("✅ Full day data contains more points than available data (correct)")
                    
                    # Show time ranges
                    available_start = result['data'].index[0]
                    available_end = result['data'].index[-1]
                    full_start = result['full_day_data'].index[0]
                    full_end = result['full_day_data'].index[-1]
                    
                    print(f"📅 Available data: {available_start.strftime('%H:%M')} to {available_end.strftime('%H:%M')}")
                    print(f"📅 Full day data: {full_start.strftime('%H:%M')} to {full_end.strftime('%H:%M')}")
                    
                    if available_end < full_end:
                        print("✅ Available data ends before full day data (no lookahead)")
                    else:
                        print("⚠️ Potential lookahead issue detected")
                        
                else:
                    print("⚠️ Full day data not larger than available data")
                    
                return result
            else:
                print(f"❌ No result for {test_symbol}")
        else:
            print(f"❌ {test_symbol} not in fetched data")
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def test_plotting():
    """Test that plotting works with full day data."""
    print("\n🎨 Testing plotting functionality...")
    
    # Run the test
    result = test_data_structure()
    
    if result:
        print(f"\n📈 Testing plot for {result['symbol']}...")
        try:
            scanner = ORBTradingScannerNoLookahead()
            scanner.plot_no_lookahead_setup(result, rank=1)
            print("✅ Plot generated successfully")
        except Exception as e:
            print(f"❌ Plot error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No result to plot")

if __name__ == "__main__":
    print("🚀 Running tests...")
    test_plotting()
    print("\n✅ Tests completed!") 