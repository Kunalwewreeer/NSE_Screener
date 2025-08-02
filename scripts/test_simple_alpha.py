#!/usr/bin/env python3
"""
Test script for Simple Alpha Strategy with annotated plots.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from strategies.simple_alpha import SimpleAlphaStrategy
from core.data_handler import DataHandler
from utils.plotting import TradingPlotter
from utils.logger import get_logger

logger = get_logger(__name__)

def test_simple_alpha_strategy():
    """Test the simple alpha strategy with plotting."""
    print("🚀 Testing Simple Alpha Strategy with Annotated Plots")
    print("=" * 60)
    
    # Initialize components
    data_handler = DataHandler()
    plotter = TradingPlotter()
    
    # Strategy parameters
    strategy_config = {
        'fast_ma': 5,
        'slow_ma': 20,
        'volume_threshold': 1.5,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'min_price_change': 0.005
    }
    
    strategy = SimpleAlphaStrategy("SimpleAlpha_Test", strategy_config)
    
    # Data parameters
    symbol = "RELIANCE.NS"
    start_date = "2025-07-25"
    end_date = "2025-07-27"
    
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⚙️  Strategy: {strategy.name}")
    print(f"🔧 Parameters: {strategy_config}")
    print()
    
    try:
        # Fetch data
        print("🔍 Fetching data...")
        data = data_handler.get_historical_data(
            symbols=symbol,
            from_date=start_date,
            to_date=end_date,
            interval="minute",
            refresh_cache=False
        )
        
        if isinstance(data, dict):
            data = data[symbol]
        
        print(f"✅ Fetched {len(data)} records")
        print(f"📈 Data range: {data.index.min()} to {data.index.max()}")
        print()
        
        # Generate signals
        print("🎯 Generating signals...")
        all_signals = []
        
        # Process data in rolling windows to simulate real-time
        window_size = 50  # Minimum data points needed for indicators
        
        for i in range(window_size, len(data)):
            window_data = data.iloc[:i+1]
            signals = strategy.generate_signals(window_data)
            
            if signals:
                # Add timestamp to signals
                for signal in signals:
                    signal['timestamp'] = data.index[i]
                all_signals.extend(signals)
        
        print(f"✅ Generated {len(all_signals)} signals")
        
        # Analyze signals
        buy_signals = [s for s in all_signals if s.get('signal_type') == 'BUY']
        sell_signals = [s for s in all_signals if s.get('signal_type') == 'SELL']
        
        print(f"📊 Signal Analysis:")
        print(f"   Buy signals: {len(buy_signals)}")
        print(f"   Sell signals: {len(sell_signals)}")
        print(f"   Total signals: {len(all_signals)}")
        
        if buy_signals:
            print(f"   Buy price range: ₹{min(s['price'] for s in buy_signals):.2f} - ₹{max(s['price'] for s in buy_signals):.2f}")
        if sell_signals:
            print(f"   Sell price range: ₹{min(s['price'] for s in sell_signals):.2f} - ₹{max(s['price'] for s in sell_signals):.2f}")
        print()
        
        # Calculate indicators for plotting
        print("📈 Calculating indicators for plotting...")
        data_with_indicators = strategy.calculate_indicators(data)
        
        # Plot main strategy signals
        print("🎨 Generating main strategy plot...")
        plotter.plot_strategy_signals(
            data=data_with_indicators,
            signals=all_signals,
            strategy_name=strategy.name,
            save_path=f"{strategy.name}_main_signals.png"
        )
        
        # Plot random day analysis
        if len(all_signals) > 0:
            print("🎨 Generating random day analysis...")
            plotter.plot_random_day_analysis(
                data=data_with_indicators,
                signals=all_signals,
                strategy_name=strategy.name,
                num_days=2
            )
        
        # Show detailed signal information
        print("\n📋 Detailed Signal Information:")
        for i, signal in enumerate(all_signals[:10]):  # Show first 10 signals
            print(f"   Signal {i+1}: {signal.get('signal_type')} at ₹{signal.get('price'):.2f}")
            print(f"      Time: {signal.get('timestamp')}")
            print(f"      Stop Loss: ₹{signal.get('stop_loss'):.2f}")
            print(f"      Take Profit: ₹{signal.get('take_profit'):.2f}")
            print(f"      Strength: {signal.get('strength'):.2f}")
            print(f"      Reason: {signal.get('reason', 'N/A')}")
            print()
        
        if len(all_signals) > 10:
            print(f"   ... and {len(all_signals) - 10} more signals")
        
        print("✅ Simple Alpha Strategy test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_strategy_parameters():
    """Test different parameter combinations."""
    print("\n🔧 Testing Different Parameter Combinations")
    print("=" * 50)
    
    data_handler = DataHandler()
    plotter = TradingPlotter()
    
    # Test parameters
    param_combinations = [
        {'fast_ma': 3, 'slow_ma': 15, 'volume_threshold': 1.2, 'name': 'Aggressive'},
        {'fast_ma': 5, 'slow_ma': 20, 'volume_threshold': 1.5, 'name': 'Balanced'},
        {'fast_ma': 8, 'slow_ma': 25, 'volume_threshold': 2.0, 'name': 'Conservative'}
    ]
    
    symbol = "RELIANCE.NS"
    start_date = "2025-07-25"
    end_date = "2025-07-27"
    
    # Fetch data once
    data = data_handler.get_historical_data(
        symbols=symbol,
        from_date=start_date,
        to_date=end_date,
        interval="minute",
        refresh_cache=False
    )
    
    if isinstance(data, dict):
        data = data[symbol]
    
    results = {}
    
    for params in param_combinations:
        strategy_name = f"SimpleAlpha_{params['name']}"
        strategy_config = {
            'fast_ma': params['fast_ma'],
            'slow_ma': params['slow_ma'],
            'volume_threshold': params['volume_threshold'],
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'min_price_change': 0.005
        }
        
        strategy = SimpleAlphaStrategy(strategy_name, strategy_config)
        
        print(f"\n🔧 Testing {strategy_name}...")
        print(f"   Fast MA: {params['fast_ma']}, Slow MA: {params['slow_ma']}, Volume Threshold: {params['volume_threshold']}")
        
        # Generate signals
        all_signals = []
        window_size = 50
        
        for i in range(window_size, len(data)):
            window_data = data.iloc[:i+1]
            signals = strategy.generate_signals(window_data)
            
            if signals:
                for signal in signals:
                    signal['timestamp'] = data.index[i]
                all_signals.extend(signals)
        
        results[strategy_name] = {
            'signals': all_signals,
            'config': strategy_config,
            'data': strategy.calculate_indicators(data)
        }
        
        buy_signals = [s for s in all_signals if s.get('signal_type') == 'BUY']
        sell_signals = [s for s in all_signals if s.get('signal_type') == 'SELL']
        
        print(f"   ✅ Generated {len(all_signals)} signals ({len(buy_signals)} buy, {len(sell_signals)} sell)")
    
    # Compare strategies
    print("\n📊 Strategy Comparison:")
    for name, result in results.items():
        signals = result['signals']
        buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
        sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
        
        print(f"   {name}: {len(signals)} total signals ({len(buy_signals)} buy, {len(sell_signals)} sell)")
    
    # Plot comparison
    if len(results) > 1:
        print("\n🎨 Generating strategy comparison plot...")
        data_dict = {name: result['data'] for name, result in results.items()}
        signals_dict = {name: result['signals'] for name, result in results.items()}
        
        plotter.plot_strategy_comparison(
            data_dict=data_dict,
            signals_dict=signals_dict,
            strategy_names=list(results.keys())
        )

def main():
    """Main function."""
    print("🚀 SIMPLE ALPHA STRATEGY TEST")
    print("=" * 50)
    
    # Test basic strategy
    test_simple_alpha_strategy()
    
    # Test parameter variations
    test_strategy_parameters()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 