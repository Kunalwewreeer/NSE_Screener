#!/usr/bin/env python3
"""
Enhanced plotting utilities for trading strategies with annotated signals.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingPlotter:
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        
    def plot_strategy_signals(self, data: pd.DataFrame, signals: List[Dict], 
                            strategy_name: str = "Strategy", save_path: Optional[str] = None):
        """
        Plot price data with annotated buy/sell signals.
        
        Args:
            data: DataFrame with OHLCV data
            signals: List of signal dictionaries with 'timestamp', 'signal_type', 'price'
            strategy_name: Name of the strategy for title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Main price chart
        ax1 = axes[0]
        ax2 = axes[1]  # Volume
        ax3 = axes[2]  # Indicators
        
        # Plot price data
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=1, color='blue', alpha=0.8)
        ax1.plot(data.index, data['high'], label='High', linewidth=0.5, color='green', alpha=0.4)
        ax1.plot(data.index, data['low'], label='Low', linewidth=0.5, color='red', alpha=0.4)
        
        # Add moving averages if available
        if 'fast_ma' in data.columns and 'slow_ma' in data.columns:
            ax1.plot(data.index, data['fast_ma'], label=f'Fast MA ({data["fast_ma"].iloc[-1]:.2f})', 
                    linewidth=1.5, color='orange', alpha=0.8)
            ax1.plot(data.index, data['slow_ma'], label=f'Slow MA ({data["slow_ma"].iloc[-1]:.2f})', 
                    linewidth=1.5, color='purple', alpha=0.8)
        
        # Annotate signals
        buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
        sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
        
        # Plot buy signals
        for signal in buy_signals:
            timestamp = signal.get('timestamp')
            price = signal.get('price', 0)
            if timestamp in data.index:
                ax1.scatter(timestamp, price, color='green', s=100, marker='^', 
                           alpha=0.8, zorder=5, label='Buy Signal' if signal == buy_signals[0] else "")
                ax1.annotate(f'BUY\n₹{price:.2f}', 
                           xy=(timestamp, price), xytext=(10, 10),
                           textcoords='offset points', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='green'))
        
        # Plot sell signals
        for signal in sell_signals:
            timestamp = signal.get('timestamp')
            price = signal.get('price', 0)
            if timestamp in data.index:
                ax1.scatter(timestamp, price, color='red', s=100, marker='v', 
                           alpha=0.8, zorder=5, label='Sell Signal' if signal == sell_signals[0] else "")
                ax1.annotate(f'SELL\n₹{price:.2f}', 
                           xy=(timestamp, price), xytext=(10, -20),
                           textcoords='offset points', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        ax1.set_title(f'{strategy_name} - Trading Signals', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(data.index, data['volume'], alpha=0.7, color='purple', width=0.8)
        if 'volume_ma' in data.columns:
            ax2.plot(data.index, data['volume_ma'], color='orange', linewidth=1.5, label='Volume MA')
        ax2.set_title('Volume', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Indicators chart
        if 'price_momentum' in data.columns:
            ax3.plot(data.index, data['price_momentum'], color='blue', linewidth=1, label='Price Momentum')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        if 'volatility' in data.columns:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(data.index, data['volatility'], color='red', linewidth=1, label='Volatility', alpha=0.7)
            ax3_twin.set_ylabel('Volatility', fontsize=10, color='red')
        
        ax3.set_title('Indicators', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Momentum', fontsize=10)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {save_path}")
        
        plt.show()
        
        # Print signal summary
        print(f"\n📊 Signal Summary for {strategy_name}:")
        print(f"   Buy signals: {len(buy_signals)}")
        print(f"   Sell signals: {len(sell_signals)}")
        print(f"   Total signals: {len(signals)}")
        
        if buy_signals:
            print(f"   Buy price range: ₹{min(s['price'] for s in buy_signals):.2f} - ₹{max(s['price'] for s in buy_signals):.2f}")
        if sell_signals:
            print(f"   Sell price range: ₹{min(s['price'] for s in sell_signals):.2f} - ₹{max(s['price'] for s in sell_signals):.2f}")
    
    def plot_random_day_analysis(self, data: pd.DataFrame, signals: List[Dict], 
                               strategy_name: str = "Strategy", num_days: int = 3):
        """
        Plot random days with buy/sell signals for detailed analysis.
        """
        # Get unique trading days
        trading_days = data.index.date.unique()
        
        if len(trading_days) < num_days:
            num_days = len(trading_days)
        
        # Select random days
        selected_days = np.random.choice(trading_days, num_days, replace=False)
        
        for i, day in enumerate(selected_days):
            day_data = data[data.index.date == day]
            day_signals = [s for s in signals if s.get('timestamp').date() == day]
            
            if len(day_data) == 0:
                continue
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Price chart
            ax1.plot(day_data.index, day_data['close'], linewidth=1.5, color='blue', alpha=0.8)
            
            # Add moving averages
            if 'fast_ma' in day_data.columns and 'slow_ma' in day_data.columns:
                ax1.plot(day_data.index, day_data['fast_ma'], linewidth=1.5, color='orange', alpha=0.8, label='Fast MA')
                ax1.plot(day_data.index, day_data['slow_ma'], linewidth=1.5, color='purple', alpha=0.8, label='Slow MA')
            
            # Annotate signals for this day
            for signal in day_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                signal_type = signal.get('signal_type', 'UNKNOWN')
                
                if signal_type == 'BUY':
                    ax1.scatter(timestamp, price, color='green', s=150, marker='^', zorder=5)
                    ax1.annotate(f'BUY\n₹{price:.2f}', 
                               xy=(timestamp, price), xytext=(10, 15),
                               textcoords='offset points', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='green'))
                elif signal_type == 'SELL':
                    ax1.scatter(timestamp, price, color='red', s=150, marker='v', zorder=5)
                    ax1.annotate(f'SELL\n₹{price:.2f}', 
                               xy=(timestamp, price), xytext=(10, -25),
                               textcoords='offset points', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='red'))
            
            ax1.set_title(f'{strategy_name} - {day.strftime("%Y-%m-%d")} (Day {i+1})', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (₹)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            ax2.bar(day_data.index, day_data['volume'], alpha=0.7, color='purple')
            if 'volume_ma' in day_data.columns:
                ax2.plot(day_data.index, day_data['volume_ma'], color='orange', linewidth=1.5)
            ax2.set_title('Volume', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            
            plt.tight_layout()
            
            # Save plot
            filename = f"{strategy_name.replace(' ', '_')}_day_{i+1}_{day.strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Day {i+1} plot saved as: {filename}")
            
            plt.show()
            
            # Print day summary
            day_buy_signals = [s for s in day_signals if s.get('signal_type') == 'BUY']
            day_sell_signals = [s for s in day_signals if s.get('signal_type') == 'SELL']
            
            print(f"\n📊 Day {i+1} Summary ({day.strftime('%Y-%m-%d')}):")
            print(f"   Trading hours: {day_data.index.min().strftime('%H:%M')} - {day_data.index.max().strftime('%H:%M')}")
            print(f"   Price range: ₹{day_data['low'].min():.2f} - ₹{day_data['high'].max():.2f}")
            print(f"   Buy signals: {len(day_buy_signals)}")
            print(f"   Sell signals: {len(day_sell_signals)}")
            print(f"   Total volume: {day_data['volume'].sum():,.0f}")
    
    def plot_strategy_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                               signals_dict: Dict[str, List[Dict]], 
                               strategy_names: List[str]):
        """
        Compare multiple strategies on the same data.
        """
        fig, axes = plt.subplots(len(strategy_names), 1, figsize=(16, 4*len(strategy_names)))
        
        if len(strategy_names) == 1:
            axes = [axes]
        
        for i, strategy_name in enumerate(strategy_names):
            if strategy_name not in data_dict:
                continue
                
            data = data_dict[strategy_name]
            signals = signals_dict.get(strategy_name, [])
            
            ax = axes[i]
            
            # Plot price
            ax.plot(data.index, data['close'], linewidth=1, color='blue', alpha=0.8)
            
            # Plot signals
            for signal in signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                signal_type = signal.get('signal_type', 'UNKNOWN')
                
                if signal_type == 'BUY':
                    ax.scatter(timestamp, price, color='green', s=80, marker='^', zorder=5)
                elif signal_type == 'SELL':
                    ax.scatter(timestamp, price, color='red', s=80, marker='v', zorder=5)
            
            ax.set_title(f'{strategy_name} - {len(signals)} signals', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price (₹)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show() 