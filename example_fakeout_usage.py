#!/usr/bin/env python3
"""
Example usage of the Fakeout Detector

This script demonstrates how to use the fakeout detection system
with different configurations and level types.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the strategies directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))

from fakeout_detector import detect_fakeout_signals

def create_sample_data():
    """Create sample OHLCV data with fakeout patterns."""
    print("📊 Creating sample data...")
    
    # Generate datetime index for 1 day of 1-minute data
    start_date = datetime(2024, 1, 1, 9, 15)
    timestamps = [start_date + timedelta(minutes=i) for i in range(390)]  # 6.5 hours
    
    # Create price data with some patterns
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    
    for i in range(1, len(timestamps)):
        # Add some trend and volatility
        change = np.random.normal(0, 0.001) + (0.0001 if i < 200 else -0.0001)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        volatility = 0.002
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = price * (1 + np.random.normal(0, volatility * 0.5))
        close_price = price * (1 + np.random.normal(0, volatility * 0.5))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Add a fakeout pattern manually
    resistance_level = df.iloc[100:120]['high'].max()
    
    # Breakout candle (around minute 150)
    df.iloc[150, df.columns.get_loc('high')] = resistance_level * 1.02
    df.iloc[150, df.columns.get_loc('close')] = resistance_level * 0.99
    df.iloc[150, df.columns.get_loc('volume')] = df.iloc[150]['volume'] * 2
    
    # Confirmation candle
    df.iloc[151, df.columns.get_loc('close')] = resistance_level * 0.97
    
    print(f"✅ Created sample data with fakeout pattern at resistance level: {resistance_level:.2f}")
    return df

def main():
    """Demonstrate fakeout detection with different configurations."""
    print("🎯 Fakeout Detector Example Usage")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Example 1: Basic detection with default settings
    print("\n📊 Example 1: Basic Detection")
    print("-" * 30)
    
    fakeout_signals, breakout_signals = detect_fakeout_signals(
        df, 
        config={'log_level': 'INFO', 'plot_signals': True},
        level_type='pdh_pdl',
        plot=True
    )
    
    print(f"Breakout signals: {len(breakout_signals)}")
    print(f"Fakeout signals: {len(fakeout_signals)}")
    
    # Example 2: Conservative configuration
    print("\n📊 Example 2: Conservative Configuration")
    print("-" * 30)
    
    conservative_config = {
        'breakout_threshold': 0.02,  # 2% breakout
        'wick_threshold': 0.4,  # 40% wick
        'confirmation_candles': 2,  # 2 confirmation candles
        'volume_spike_threshold': 2.0,  # Higher volume requirement
        'log_level': 'INFO',
        'plot_signals': False
    }
    
    fakeout_signals_cons, breakout_signals_cons = detect_fakeout_signals(
        df, 
        config=conservative_config,
        level_type='pdh_pdl',
        plot=False
    )
    
    print(f"Conservative breakout signals: {len(breakout_signals_cons)}")
    print(f"Conservative fakeout signals: {len(fakeout_signals_cons)}")
    
    # Example 3: Aggressive configuration
    print("\n📊 Example 3: Aggressive Configuration")
    print("-" * 30)
    
    aggressive_config = {
        'breakout_threshold': 0.005,  # 0.5% breakout
        'wick_threshold': 0.1,  # 10% wick
        'confirmation_candles': 1,  # 1 confirmation candle
        'volume_spike_threshold': 1.2,  # Lower volume requirement
        'log_level': 'INFO',
        'plot_signals': False
    }
    
    fakeout_signals_agg, breakout_signals_agg = detect_fakeout_signals(
        df, 
        config=aggressive_config,
        level_type='pdh_pdl',
        plot=False
    )
    
    print(f"Aggressive breakout signals: {len(breakout_signals_agg)}")
    print(f"Aggressive fakeout signals: {len(fakeout_signals_agg)}")
    
    # Example 4: VWAP levels
    print("\n📊 Example 4: VWAP Levels")
    print("-" * 30)
    
    fakeout_signals_vwap, breakout_signals_vwap = detect_fakeout_signals(
        df, 
        config={'log_level': 'INFO', 'plot_signals': False},
        level_type='vwap',
        plot=False
    )
    
    print(f"VWAP breakout signals: {len(breakout_signals_vwap)}")
    print(f"VWAP fakeout signals: {len(fakeout_signals_vwap)}")
    
    # Display signal details
    if fakeout_signals:
        print("\n🎯 Signal Details:")
        for i, signal in enumerate(fakeout_signals, 1):
            print(f"Signal {i}:")
            print(f"  Type: {signal['signal_type']}")
            print(f"  Entry Time: {signal['entry_time']}")
            print(f"  Entry Price: {signal['entry_price']:.2f}")
            print(f"  SL Price: {signal['sl_price']:.2f}")
            print(f"  TP Price: {signal['tp_price']:.2f}")
            print(f"  Level: {signal['level']:.2f}")
            print(f"  ATR: {signal['atr']:.4f}")
            print(f"  Volume Ratio: {signal['volume_ratio']:.2f}")
            print(f"  Wick Ratio: {signal['wick_ratio']:.2f}")
            print()
    
    print("✅ Example completed successfully!")
    print("\n📝 Key Features:")
    print("- Modular design with customizable parameters")
    print("- Multiple level types (PDH/PDL, VWAP, custom)")
    print("- Extensive debugging and logging")
    print("- Interactive plotting with Plotly")
    print("- Risk management with ATR-based SL/TP")

if __name__ == "__main__":
    main() 