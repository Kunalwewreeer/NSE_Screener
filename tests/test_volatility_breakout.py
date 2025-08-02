#!/usr/bin/env python3
"""
Test script for Volatility Breakout Strategy with 1-minute data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.data_handler import DataHandler
from utils.logger import get_logger

logger = get_logger(__name__)

def test_volatility_breakout_strategy():
    """Test the volatility breakout strategy with 1-minute data."""
    print("=" * 60)
    print("TESTING VOLATILITY BREAKOUT STRATEGY")
    print("=" * 60)
    
    # Initialize data handler
    data_handler = DataHandler()
    
    # Strategy configuration for volatility breakout
    strategy_config = {
        'volatility_period': 20,
        'volatility_multiplier': 2.0,
        'momentum_period': 5,
        'volume_threshold': 1.5,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'min_volatility': 0.005,
        'max_volatility': 0.05
    }
    
    # Initialize strategy
    strategy = VolatilityBreakoutStrategy("Test_VolatilityBreakout", strategy_config)
    
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy_config}")
    
    # Test symbols (high volatility stocks)
    test_symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS"]
    
    # Date range for testing (recent data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")  # 5 days of data
    
    print(f"\nTesting period: {start_date} to {end_date}")
    print(f"Timeframe: 1-minute")
    
    for symbol in test_symbols:
        print(f"\n{'='*40}")
        print(f"Testing {symbol}")
        print(f"{'='*40}")
        
        try:
            # Fetch 1-minute data
            print(f"Fetching 1-minute data for {symbol}...")
            data = data_handler.get_historical_data(
                symbols=symbol,
                from_date=start_date,
                to_date=end_date,
                interval="minute",
                refresh_cache=True  # Force refresh for testing
            )
            
            if data.empty:
                print(f"❌ No data received for {symbol}")
                continue
            
            print(f"✅ Fetched {len(data)} records for {symbol}")
            print(f"📊 Data shape: {data.shape}")
            print(f"📅 Time range: {data.index.min()} to {data.index.max()}")
            print(f"💰 Price range: ₹{data['low'].min():.2f} - ₹{data['high'].max():.2f}")
            
            # Calculate technical indicators
            data_with_indicators = data_handler.calculate_technical_indicators(data)
            print(f"📈 Technical indicators calculated")
            
            # Generate signals
            print(f"🔍 Generating volatility breakout signals...")
            signals = strategy.generate_signals(data_with_indicators)
            
            print(f"📊 Generated {len(signals)} signals")
            
            if signals:
                print(f"\n📋 Signal Details:")
                for i, signal in enumerate(signals[:5]):  # Show first 5 signals
                    print(f"  Signal {i+1}:")
                    print(f"    Type: {signal['signal_type']}")
                    print(f"    Price: ₹{signal['price']:.2f}")
                    print(f"    Stop Loss: ₹{signal['stop_loss']:.2f}")
                    print(f"    Take Profit: ₹{signal['take_profit']:.2f}")
                    print(f"    Strength: {signal['strength']:.2f}")
                    print(f"    Volatility: {signal['metadata']['volatility']:.4f}")
                    print(f"    Momentum: {signal['metadata']['momentum']:.4f}")
                    print(f"    Volume Ratio: {signal['metadata']['volume_ratio']:.2f}")
                    print()
            else:
                print(f"⚠️  No signals generated for {symbol}")
                print(f"   This could be due to:")
                print(f"   - Insufficient volatility")
                print(f"   - No breakout conditions met")
                print(f"   - Data quality issues")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            logger.error(f"Error testing {symbol}: {e}")
    
    # Test with multiple symbols
    print(f"\n{'='*60}")
    print("TESTING MULTIPLE SYMBOLS")
    print(f"{'='*60}")
    
    try:
        # Fetch data for multiple symbols
        print(f"Fetching 1-minute data for multiple symbols...")
        multi_data = data_handler.get_historical_data(
            symbols=test_symbols,
            from_date=start_date,
            to_date=end_date,
            interval="minute",
            refresh_cache=False  # Use cached data
        )
        
        if isinstance(multi_data, dict):
            print(f"✅ Fetched data for {len(multi_data)} symbols")
            
            total_signals = 0
            for symbol, data in multi_data.items():
                if not data.empty:
                    data_with_indicators = data_handler.calculate_technical_indicators(data)
                    signals = strategy.generate_signals(data_with_indicators)
                    total_signals += len(signals)
                    print(f"  {symbol}: {len(signals)} signals")
            
            print(f"\n📊 Total signals across all symbols: {total_signals}")
        else:
            print(f"❌ Unexpected data format: {type(multi_data)}")
            
    except Exception as e:
        print(f"❌ Error testing multiple symbols: {e}")
        logger.error(f"Error testing multiple symbols: {e}")

def test_strategy_parameters():
    """Test different parameter combinations for the volatility breakout strategy."""
    print(f"\n{'='*60}")
    print("TESTING STRATEGY PARAMETERS")
    print(f"{'='*60}")
    
    # Different parameter sets to test
    parameter_sets = [
        {
            'name': 'Conservative',
            'config': {
                'volatility_period': 30,
                'volatility_multiplier': 2.5,
                'momentum_period': 10,
                'volume_threshold': 2.0,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.03
            }
        },
        {
            'name': 'Aggressive',
            'config': {
                'volatility_period': 15,
                'volatility_multiplier': 1.5,
                'momentum_period': 3,
                'volume_threshold': 1.2,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05
            }
        },
        {
            'name': 'Balanced',
            'config': {
                'volatility_period': 20,
                'volatility_multiplier': 2.0,
                'momentum_period': 5,
                'volume_threshold': 1.5,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }
    ]
    
    data_handler = DataHandler()
    
    # Test with RELIANCE data
    symbol = "RELIANCE.NS"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    try:
        # Fetch data once
        data = data_handler.get_historical_data(
            symbols=symbol,
            from_date=start_date,
            to_date=end_date,
            interval="minute",
            refresh_cache=False
        )
        
        if data.empty:
            print(f"❌ No data available for testing parameters")
            return
        
        data_with_indicators = data_handler.calculate_technical_indicators(data)
        
        for param_set in parameter_sets:
            print(f"\n🔧 Testing {param_set['name']} parameters:")
            print(f"   Config: {param_set['config']}")
            
            strategy = VolatilityBreakoutStrategy(f"Test_{param_set['name']}", param_set['config'])
            signals = strategy.generate_signals(data_with_indicators)
            
            print(f"   📊 Signals generated: {len(signals)}")
            
            if signals:
                buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
                sell_signals = [s for s in signals if s['signal_type'] == 'SELL']
                print(f"   📈 Buy signals: {len(buy_signals)}")
                print(f"   📉 Sell signals: {len(sell_signals)}")
                
                # Calculate average signal strength
                avg_strength = sum(s['strength'] for s in signals) / len(signals)
                print(f"   💪 Average signal strength: {avg_strength:.2f}")
    
    except Exception as e:
        print(f"❌ Error testing parameters: {e}")
        logger.error(f"Error testing parameters: {e}")

def main():
    """Main function to run all tests."""
    print("🚀 VOLATILITY BREAKOUT STRATEGY TEST")
    print("=" * 60)
    
    # Test basic strategy functionality
    test_volatility_breakout_strategy()
    
    # Test different parameter combinations
    test_strategy_parameters()
    
    print(f"\n{'='*60}")
    print("TEST COMPLETED")
    print(f"{'='*60}")
    print("✅ Volatility breakout strategy tested with 1-minute data")
    print("📊 Check the logs for detailed signal information")
    print("🎯 The strategy is now ready for backtesting and live trading")

if __name__ == "__main__":
    main() 