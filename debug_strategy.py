#!/usr/bin/env python3
"""
Debug script to examine data loading and signal generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import pandas as pd
from datetime import datetime, timedelta
from core.data_handler import DataHandler
from strategies.simple_alpha import SimpleAlphaStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

def debug_data_and_signals():
    """Debug data loading and signal generation."""
    
    print("🔍 DEBUGGING DATA LOADING AND SIGNAL GENERATION")
    print("=" * 60)
    
    # Initialize data handler
    data_handler = DataHandler()
    
    # Test different date ranges
    test_ranges = [
        ('2025-01-01', '2025-01-15'),
        ('2025-01-01', '2025-01-05'),
        ('2025-01-10', '2025-01-15'),
    ]
    
    for start_date, end_date in test_ranges:
        print(f"\n📅 Testing Date Range: {start_date} to {end_date}")
        print("-" * 40)
        
        # Test different timeframes
        timeframes = ['day', '5minute', 'minute']
        
        for timeframe in timeframes:
            print(f"\n⏰ Timeframe: {timeframe}")
            
            try:
                # Load data
                data = data_handler.get_historical_data(
                    symbols=['RELIANCE.NS'],
                    from_date=start_date,
                    to_date=end_date,
                    interval=timeframe
                )
                
                if 'RELIANCE.NS' in data and not data['RELIANCE.NS'].empty:
                    df = data['RELIANCE.NS']
                    print(f"   📊 Data loaded: {len(df)} records")
                    print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
                    print(f"   📋 Columns: {list(df.columns)}")
                    
                    # Show sample data
                    print(f"   📈 Sample data (first 3 rows):")
                    print(df.head(3)[['open', 'high', 'low', 'close', 'volume']].to_string())
                    
                    # Test strategy signal generation
                    test_strategies = [
                        ('Simple Alpha', SimpleAlphaStrategy, {
                            'fast_ma': 3,
                            'slow_ma': 10,
                            'volume_threshold': 0.5,
                            'stop_loss_pct': 0.02,
                            'take_profit_pct': 0.03,
                            'min_price_change': 0.001
                        }),
                        ('Volatility Breakout', VolatilityBreakoutStrategy, {
                            'volatility_period': 10,
                            'volatility_multiplier': 1.5,
                            'momentum_period': 3,
                            'volume_threshold': 0.5,
                            'stop_loss_pct': 0.01,
                            'take_profit_pct': 0.02,
                            'min_volatility': 0.001,
                            'max_volatility': 0.1
                        })
                    ]
                    
                    for strategy_name, strategy_class, params in test_strategies:
                        print(f"\n   🎯 Testing {strategy_name}:")
                        
                        try:
                            strategy = strategy_class(f"Debug_{strategy_name}", params)
                            signals = strategy.generate_signals(df)
                            
                            print(f"      📊 Generated {len(signals)} signals")
                            
                            if signals:
                                for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                                    print(f"      Signal {i+1}: {signal.get('signal_type', 'UNKNOWN')} @ ₹{signal.get('price', 0):.2f}")
                                    if 'reason' in signal:
                                        print(f"         Reason: {signal['reason']}")
                            else:
                                print(f"      ❌ No signals generated")
                                
                                # Debug why no signals
                                if strategy_name == 'Simple Alpha':
                                    # Check MA crossover conditions
                                    df_with_indicators = strategy.calculate_indicators(df)
                                    if len(df_with_indicators) >= 10:
                                        current = df_with_indicators.iloc[-1]
                                        prev = df_with_indicators.iloc[-2]
                                        
                                        print(f"      🔍 Debug info:")
                                        print(f"         Fast MA: {current.get('fast_ma', 'N/A'):.2f} vs {prev.get('fast_ma', 'N/A'):.2f}")
                                        print(f"         Slow MA: {current.get('slow_ma', 'N/A'):.2f} vs {prev.get('slow_ma', 'N/A'):.2f}")
                                        print(f"         Volume ratio: {current.get('volume_ratio', 'N/A'):.2f}")
                                        print(f"         Price momentum: {current.get('price_momentum', 'N/A'):.4f}")
                                        
                                        # Check conditions
                                        ma_crossover = (current.get('fast_ma', 0) > current.get('slow_ma', 0) and 
                                                      prev.get('fast_ma', 0) <= prev.get('slow_ma', 0))
                                        volume_confirmed = current.get('volume_ratio', 0) > params['volume_threshold']
                                        price_momentum = abs(current.get('price_momentum', 0)) > params['min_price_change']
                                        
                                        print(f"         MA Crossover: {ma_crossover}")
                                        print(f"         Volume Confirmed: {volume_confirmed}")
                                        print(f"         Price Momentum: {price_momentum}")
                                
                        except Exception as e:
                            print(f"      ❌ Error testing {strategy_name}: {e}")
                
                else:
                    print(f"   ❌ No data loaded for {timeframe}")
                    
            except Exception as e:
                print(f"   ❌ Error loading data: {e}")

def debug_cache():
    """Debug cache contents."""
    print("\n🗂️  DEBUGGING CACHE CONTENTS")
    print("=" * 60)
    
    import glob
    import pickle
    
    cache_files = glob.glob("data/cache/*.pkl")
    print(f"Found {len(cache_files)} cache files:")
    
    for cache_file in cache_files[:5]:  # Show first 5 files
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                for symbol, df in data.items():
                    print(f"   📁 {cache_file}: {symbol} - {len(df)} records")
                    if not df.empty:
                        print(f"      Date range: {df.index.min()} to {df.index.max()}")
                        print(f"      Columns: {list(df.columns)}")
            else:
                print(f"   📁 {cache_file}: {type(data)} - {len(data) if hasattr(data, '__len__') else 'unknown'} records")
                
        except Exception as e:
            print(f"   ❌ Error reading {cache_file}: {e}")

if __name__ == "__main__":
    debug_data_and_signals()
    debug_cache() 