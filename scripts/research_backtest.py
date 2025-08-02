#!/usr/bin/env python3
"""
Research-oriented backtesting script with comprehensive analysis and enhanced plotting.
Supports multiple strategies with detailed visualization and performance metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from backtest.backtester import Backtester
from strategies import ORBStrategy, MomentumStrategy, VolatilityBreakoutStrategy, SimpleAlphaStrategy
from strategies.test_strategy import TestStrategy
from utils.plotting import TradingPlotter
from utils.logger import get_logger

logger = get_logger(__name__)

class ResearchBacktester:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize research backtester with configuration."""
        self.config = self._load_config(config_path)
        self.backtester = Backtester(config_path)
        self.strategies_config = self._load_strategies_config()
    
    def _load_strategies_config(self) -> Dict:
        """Load strategies configuration from YAML file."""
        try:
            strategies_path = "config/strategies.yaml"
            with open(strategies_path, 'r') as file:
                strategies_config = yaml.safe_load(file)
            logger.info(f"Loaded strategies configuration from {strategies_path}")
            return strategies_config
        except Exception as e:
            logger.error(f"Error loading strategies config: {e}")
            return {}
        self.plotter = TradingPlotter()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default config
            return {
                'data': {'timeframe': 'day'},
                'trading': {'default_capital': 100000},
                'backtesting': {'enable_plots': True, 'save_results': True}
            }
    
    def get_strategy(self, strategy_name: str, strategy_params: Dict) -> Any:
        """Get strategy instance based on name."""
        strategy_name_lower = strategy_name.lower()
        
        if strategy_name_lower == "orb":
            return ORBStrategy(f"Research_{strategy_name}", strategy_params)
        elif strategy_name_lower == "momentum":
            return MomentumStrategy(f"Research_{strategy_name}", strategy_params)
        elif strategy_name_lower == "volatility_breakout":
            return VolatilityBreakoutStrategy(f"Research_{strategy_name}", strategy_params)
        elif strategy_name_lower == "simple_alpha":
            return SimpleAlphaStrategy(f"Research_{strategy_name}", strategy_params)
        elif strategy_name_lower == "test":
            return TestStrategy(f"Research_{strategy_name}", strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def run_comprehensive_backtest(self, strategy_name: str, symbols: List[str], 
                                 start_date: str, end_date: str, 
                                 strategy_params: Dict, capital: float = 100000,
                                 save_results: bool = True, generate_plots: bool = True) -> Dict:
        """Run comprehensive backtest with detailed analysis."""
        print(f"🚀 Running Comprehensive Backtest")
        print(f"📊 Strategy: {strategy_name}")
        print(f"🎯 Symbols: {symbols}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Capital: ₹{capital:,.0f}")
        print("=" * 60)
        
        try:
            # Initialize strategy
            strategy = self.get_strategy(strategy_name, strategy_params)
            print(f"✅ Strategy initialized: {strategy.name}")
            
            # Run backtest
            print("🔍 Running backtest...")
            results = self.backtester.run_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital
            )
            
            print("✅ Backtest completed successfully!")
            
            # Generate plots if requested
            if generate_plots and results:
                print("🎨 Generating comprehensive plots...")
                self._generate_comprehensive_plots(results, strategy, symbols, start_date, end_date)
            
            # Save results if requested
            if save_results and results:
                self._save_results(results, strategy_name, symbols, start_date, end_date)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive backtest: {e}")
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_comprehensive_plots(self, results: Dict, strategy: Any, 
                                    symbols: List[str], start_date: str, end_date: str):
        """Generate comprehensive plots for analysis."""
        try:
            # Get equity curve
            equity_curve = results.get('equity_curve', pd.DataFrame())
            if equity_curve.empty:
                print("⚠️  No equity curve data available for plotting")
                return
            
            # Convert to list of dicts if it's a DataFrame
            if isinstance(equity_curve, pd.DataFrame):
                # Reset index to include date/timestamp as a column
                equity_curve_reset = equity_curve.reset_index()
                portfolio_history = equity_curve_reset.to_dict('records')
            else:
                portfolio_history = equity_curve
            
            # 1. Stock Price Analysis with Strategy Parameters
            self._plot_stock_price_analysis(results, strategy, symbols, start_date, end_date)
            
            # 2. Portfolio Performance Plot
            self._plot_portfolio_performance(portfolio_history, strategy.name, start_date, end_date)
            
            # 3. Drawdown Analysis
            self._plot_drawdown_analysis(portfolio_history, strategy.name, start_date, end_date)
            
            # 4. Trade Analysis
            trades = results.get('trades', [])
            if trades:
                self._plot_trade_analysis(trades, strategy.name, start_date, end_date)
            
            # 5. Signal Analysis
            signals = results.get('signals', [])
            if signals:
                self._plot_signal_analysis(signals, strategy.name, start_date, end_date)
            
            # 6. Monthly Returns Heatmap
            self._plot_monthly_returns_heatmap(portfolio_history, strategy.name, start_date, end_date)
            
            # 7. Risk-Return Scatter
            self._plot_risk_return_scatter(portfolio_history, strategy.name, start_date, end_date)
            
            # 8. Random Day Analysis (if we have minute data)
            if strategy.__class__.__name__ in ['VolatilityBreakoutStrategy', 'SimpleAlphaStrategy']:
                self._plot_random_day_analysis(results, strategy, symbols, start_date, end_date)
            
            print("✅ All plots generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            print(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_portfolio_performance(self, portfolio_history: List[Dict], 
                                  strategy_name: str, start_date: str, end_date: str):
        """Plot portfolio performance over time."""
        if not portfolio_history:
            print("⚠️  No portfolio history data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        
        # Determine the value column
        value_col = 'portfolio_value' if 'portfolio_value' in df.columns else 'capital'
        
        if value_col not in df.columns:
            print(f"⚠️  No {value_col} column found in portfolio history")
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name == 'date':
            # If the index is already named 'date', convert it to datetime
            df.index = pd.to_datetime(df.index)
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️  No valid datetime index found in portfolio history")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Portfolio value
        ax1.plot(df.index, df[value_col], linewidth=2, color='blue', alpha=0.8)
        ax1.set_title(f'{strategy_name} - Portfolio Performance ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add performance metrics
        if len(df) > 1:
            initial_value = df[value_col].iloc[0]
            final_value = df[value_col].iloc[-1]
            total_return = ((final_value - initial_value) / initial_value) * 100
            ax1.text(0.02, 0.98, f'Total Return: {total_return:.2f}%', 
                    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Returns
        if len(df) > 1:
            returns = df[value_col].pct_change().dropna()
            ax2.plot(returns.index, returns * 100, linewidth=1, color='green', alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Daily Returns (%)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Returns (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"data/results/{strategy_name}_portfolio_performance_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Portfolio performance plot saved: {filename}")
        
        plt.show()
    
    def _plot_drawdown_analysis(self, portfolio_history: List[Dict], 
                              strategy_name: str, start_date: str, end_date: str):
        """Plot drawdown analysis."""
        if not portfolio_history:
            return
        
        df = pd.DataFrame(portfolio_history)
        value_col = 'portfolio_value' if 'portfolio_value' in df.columns else 'capital'
        
        if value_col not in df.columns:
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name == 'date':
            # If the index is already named 'date', convert it to datetime
            df.index = pd.to_datetime(df.index)
        
        # Calculate drawdown
        peak = df[value_col].expanding().max()
        drawdown = (df[value_col] - peak) / peak * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Portfolio value with peak
        ax1.plot(df.index, df[value_col], linewidth=2, color='blue', alpha=0.8, label='Portfolio Value')
        ax1.plot(df.index, peak, linewidth=1, color='red', alpha=0.6, label='Peak Value')
        ax1.fill_between(df.index, df[value_col], peak, alpha=0.3, color='red')
        ax1.set_title(f'{strategy_name} - Drawdown Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown percentage
        ax2.fill_between(df.index, drawdown, 0, alpha=0.7, color='red')
        ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{strategy_name}_drawdown_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Drawdown analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_trade_analysis(self, trades: List[Dict], strategy_name: str, 
                           start_date: str, end_date: str):
        """Plot trade analysis."""
        if not trades:
            return
        
        df_trades = pd.DataFrame(trades)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade P&L distribution
        if 'pnl' in df_trades.columns:
            axes[0, 0].hist(df_trades['pnl'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Trade P&L Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('P&L (₹)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Win/Loss ratio
        if 'pnl' in df_trades.columns:
            wins = (df_trades['pnl'] > 0).sum()
            losses = (df_trades['pnl'] <= 0).sum()
            total = len(df_trades)
            
            labels = ['Wins', 'Losses']
            sizes = [wins, losses]
            colors = ['green', 'red']
            
            axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title(f'Win/Loss Ratio (Total: {total})', fontweight='bold')
        
        # Trade duration
        if 'entry_time' in df_trades.columns and 'exit_time' in df_trades.columns:
            df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
            df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
            df_trades['duration'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600  # hours
            
            axes[1, 0].hist(df_trades['duration'], bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Trade Duration Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Duration (hours)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Cumulative P&L
        if 'pnl' in df_trades.columns:
            cumulative_pnl = df_trades['pnl'].cumsum()
            axes[1, 1].plot(cumulative_pnl.index, cumulative_pnl, linewidth=2, color='green')
            axes[1, 1].set_title('Cumulative P&L', fontweight='bold')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative P&L (₹)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{strategy_name}_trade_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Trade analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_signal_analysis(self, signals: List[Dict], strategy_name: str, 
                            start_date: str, end_date: str):
        """Plot signal analysis."""
        if not signals:
            return
        
        df_signals = pd.DataFrame(signals)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Signal distribution by type
        if 'signal_type' in df_signals.columns:
            signal_counts = df_signals['signal_type'].value_counts()
            axes[0, 0].bar(signal_counts.index, signal_counts.values, color=['green', 'red'])
            axes[0, 0].set_title('Signal Distribution by Type', fontweight='bold')
            axes[0, 0].set_ylabel('Count')
        
        # Signal strength distribution
        if 'strength' in df_signals.columns:
            axes[0, 1].hist(df_signals['strength'], bins=15, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 1].set_title('Signal Strength Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Signal Strength')
            axes[0, 1].set_ylabel('Frequency')
        
        # Signals over time
        if 'timestamp' in df_signals.columns:
            df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
            df_signals['date'] = df_signals['timestamp'].dt.date
            
            daily_signals = df_signals.groupby('date').size()
            axes[1, 0].plot(daily_signals.index, daily_signals.values, marker='o', linewidth=2)
            axes[1, 0].set_title('Daily Signal Count', fontweight='bold')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Number of Signals')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Price at signal
        if 'price' in df_signals.columns:
            axes[1, 1].scatter(df_signals.index, df_signals['price'], 
                             c=df_signals['signal_type'].map({'BUY': 'green', 'SELL': 'red'}),
                             alpha=0.7, s=50)
            axes[1, 1].set_title('Signal Prices', fontweight='bold')
            axes[1, 1].set_xlabel('Signal Number')
            axes[1, 1].set_ylabel('Price (₹)')
        
        plt.tight_layout()
        
        filename = f"data/results/{strategy_name}_signal_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Signal analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_monthly_returns_heatmap(self, portfolio_history: List[Dict], 
                                    strategy_name: str, start_date: str, end_date: str):
        """Plot monthly returns heatmap."""
        if not portfolio_history:
            return
        
        df = pd.DataFrame(portfolio_history)
        value_col = 'portfolio_value' if 'portfolio_value' in df.columns else 'capital'
        
        if value_col not in df.columns:
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name == 'date':
            # If the index is already named 'date', convert it to datetime
            df.index = pd.to_datetime(df.index)
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("⚠️  No valid datetime index for monthly returns heatmap")
            return
        
        # Calculate daily returns
        daily_returns = df[value_col].pct_change().dropna()
        
        if len(daily_returns) < 30:  # Need at least a month of data
            print("⚠️  Insufficient data for monthly returns heatmap")
            return
        
        # Resample to monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if monthly_returns.empty:
            print("⚠️  No monthly returns data available")
            return
        
        # Create heatmap data
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        if pivot_table.empty:
            print("⚠️  No data for monthly returns heatmap")
            return
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table * 100, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Monthly Returns (%)'})
        plt.title(f'{strategy_name} - Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        filename = f"data/results/{strategy_name}_monthly_returns_heatmap_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Monthly returns heatmap saved: {filename}")
        
        plt.show()
    
    def _plot_risk_return_scatter(self, portfolio_history: List[Dict], 
                                strategy_name: str, start_date: str, end_date: str):
        """Plot risk-return scatter plot."""
        if not portfolio_history:
            return
        
        df = pd.DataFrame(portfolio_history)
        value_col = 'portfolio_value' if 'portfolio_value' in df.columns else 'capital'
        
        if value_col not in df.columns:
            return
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name == 'date':
            # If the index is already named 'date', convert it to datetime
            df.index = pd.to_datetime(df.index)
        
        # Calculate rolling metrics
        returns = df[value_col].pct_change().dropna()
        
        if len(returns) < 30:
            return
        
        # 30-day rolling metrics
        rolling_return = returns.rolling(30).mean() * 252  # Annualized
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)  # Annualized
        
        plt.figure(figsize=(10, 8))
        plt.scatter(rolling_vol * 100, rolling_return * 100, alpha=0.6, s=50)
        plt.xlabel('Volatility (%)')
        plt.ylabel('Return (%)')
        plt.title(f'{strategy_name} - Risk-Return Scatter (30-day rolling)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add risk-free rate line
        risk_free_rate = 0.06  # 6% annual
        plt.axhline(y=risk_free_rate * 100, color='red', linestyle='--', alpha=0.7, label=f'Risk-free rate ({risk_free_rate*100:.1f}%)')
        plt.legend()
        
        filename = f"data/results/{strategy_name}_risk_return_scatter_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Risk-return scatter plot saved: {filename}")
        
        plt.show()
    
    def _plot_stock_price_analysis(self, results: Dict, strategy: Any, 
                                 symbols: List[str], start_date: str, end_date: str):
        """Plot stock price analysis with strategy parameters and indicators."""
        try:
            # Get the original data from results
            data = results.get('data', {})
            if not data:
                print("⚠️  No stock data available for price analysis")
                return
            
            # Get strategy parameters
            strategy_params = getattr(strategy, '__dict__', {})
            
            # Plot for each symbol
            for symbol in symbols:
                if symbol not in data:
                    continue
                
                symbol_data = data[symbol]
                if symbol_data.empty:
                    continue
                
                # Calculate technical indicators based on strategy type
                strategy_type = strategy.__class__.__name__
                
                if strategy_type == 'SimpleAlphaStrategy':
                    self._plot_simple_alpha_analysis(symbol_data, strategy, symbol, start_date, end_date)
                elif strategy_type == 'VolatilityBreakoutStrategy':
                    self._plot_volatility_breakout_analysis(symbol_data, strategy, symbol, start_date, end_date)
                elif strategy_type == 'ORBStrategy':
                    self._plot_orb_analysis(symbol_data, strategy, symbol, start_date, end_date)
                elif strategy_type == 'MomentumStrategy':
                    self._plot_momentum_analysis(symbol_data, strategy, symbol, start_date, end_date)
                else:
                    self._plot_generic_analysis(symbol_data, strategy, symbol, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error in stock price analysis: {e}")
            print(f"❌ Error in stock price analysis: {e}")
    
    def _plot_simple_alpha_analysis(self, data: pd.DataFrame, strategy: Any, 
                                  symbol: str, start_date: str, end_date: str):
        """Plot Simple Alpha strategy analysis with MA crossovers and volume."""
        # Calculate indicators
        fast_ma = getattr(strategy, 'fast_ma', 5)
        slow_ma = getattr(strategy, 'slow_ma', 20)
        volume_threshold = getattr(strategy, 'volume_threshold', 1.5)
        
        df = data.copy()
        df['fast_ma'] = df['close'].rolling(window=fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_ma).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_momentum'] = df['close'].pct_change().rolling(window=5).mean()
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(df.index, df['fast_ma'], label=f'Fast MA ({fast_ma})', linewidth=1, color='orange', alpha=0.7)
        ax1.plot(df.index, df['slow_ma'], label=f'Slow MA ({slow_ma})', linewidth=1, color='red', alpha=0.7)
        
        # Add buy/sell signals if available
        signals = self._get_signals_for_symbol(symbol)
        if signals:
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            for signal in buy_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='green', s=100, marker='^', alpha=0.8, zorder=5)
                    ax1.annotate(f'BUY\n₹{price:.2f}', xy=(timestamp, price), xytext=(10, 10),
                                textcoords='offset points', fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', color='green'))
            
            for signal in sell_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='red', s=100, marker='v', alpha=0.8, zorder=5)
                    ax1.annotate(f'SELL\n₹{price:.2f}', xy=(timestamp, price), xytext=(10, -20),
                                textcoords='offset points', fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', color='red'))
        
        ax1.set_title(f'{symbol} - Simple Alpha Strategy Analysis ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume with threshold
        ax2 = axes[1]
        ax2.bar(df.index, df['volume'], alpha=0.6, color='gray', label='Volume')
        ax2.axhline(y=volume_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Volume Threshold ({volume_threshold}x)')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Price Momentum
        ax3 = axes[2]
        ax3.plot(df.index, df['price_momentum'] * 100, color='purple', alpha=0.7, label='Price Momentum')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Momentum (%)', fontsize=10)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"data/results/{symbol}_simple_alpha_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Simple Alpha analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_volatility_breakout_analysis(self, data: pd.DataFrame, strategy: Any, 
                                         symbol: str, start_date: str, end_date: str):
        """Plot Volatility Breakout strategy analysis."""
        # Get strategy parameters
        lookback_period = getattr(strategy, 'lookback_period', 20)
        volatility_threshold = getattr(strategy, 'volatility_threshold', 0.002)
        
        df = data.copy()
        
        # Calculate volatility indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=lookback_period).std()
        df['volatility_ma'] = df['volatility'].rolling(window=lookback_period).mean()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['atr'] = df['price_range'].rolling(window=lookback_period).mean()
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price and volatility bands
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)
        
        # Add volatility bands
        df['upper_band'] = df['close'] * (1 + df['volatility'])
        df['lower_band'] = df['close'] * (1 - df['volatility'])
        ax1.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.2, color='gray', label='Volatility Bands')
        
        # Add signals
        signals = self._get_signals_for_symbol(symbol)
        if signals:
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            for signal in buy_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='green', s=100, marker='^', alpha=0.8, zorder=5)
            
            for signal in sell_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='red', s=100, marker='v', alpha=0.8, zorder=5)
        
        ax1.set_title(f'{symbol} - Volatility Breakout Analysis ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatility
        ax2 = axes[1]
        ax2.plot(df.index, df['volatility'] * 100, color='orange', alpha=0.7, label='Volatility')
        ax2.axhline(y=volatility_threshold * 100, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({volatility_threshold*100:.1f}%)')
        ax2.set_ylabel('Volatility (%)', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ATR
        ax3 = axes[2]
        ax3.plot(df.index, df['atr'] * 100, color='purple', alpha=0.7, label='ATR')
        ax3.set_ylabel('ATR (%)', fontsize=10)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{symbol}_volatility_breakout_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Volatility Breakout analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_orb_analysis(self, data: pd.DataFrame, strategy: Any, 
                          symbol: str, start_date: str, end_date: str):
        """Plot ORB strategy analysis."""
        # Get strategy parameters
        lookback_period = getattr(strategy, 'lookback_period', 30)
        breakout_threshold = getattr(strategy, 'breakout_threshold', 0.01)
        
        df = data.copy()
        
        # Calculate ORB indicators
        df['high_ma'] = df['high'].rolling(window=lookback_period).mean()
        df['low_ma'] = df['low'].rolling(window=lookback_period).mean()
        df['range'] = df['high_ma'] - df['low_ma']
        df['breakout_level'] = df['high_ma'] + (df['range'] * breakout_threshold)
        df['breakdown_level'] = df['low_ma'] - (df['range'] * breakout_threshold)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Price and ORB levels
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(df.index, df['breakout_level'], label='Breakout Level', linewidth=1, color='green', alpha=0.7)
        ax1.plot(df.index, df['breakdown_level'], label='Breakdown Level', linewidth=1, color='red', alpha=0.7)
        
        # Add signals
        signals = self._get_signals_for_symbol(symbol)
        if signals:
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            for signal in buy_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='green', s=100, marker='^', alpha=0.8, zorder=5)
            
            for signal in sell_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='red', s=100, marker='v', alpha=0.8, zorder=5)
        
        ax1.set_title(f'{symbol} - ORB Strategy Analysis ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Range
        ax2 = axes[1]
        ax2.plot(df.index, df['range'], color='purple', alpha=0.7, label='Price Range')
        ax2.set_ylabel('Range (₹)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{symbol}_orb_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 ORB analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_momentum_analysis(self, data: pd.DataFrame, strategy: Any, 
                              symbol: str, start_date: str, end_date: str):
        """Plot Momentum strategy analysis."""
        # Get strategy parameters
        lookback_period = getattr(strategy, 'lookback_period', 20)
        momentum_threshold = getattr(strategy, 'momentum_threshold', 0.02)
        rsi_period = getattr(strategy, 'rsi_period', 14)
        
        df = data.copy()
        
        # Calculate momentum indicators
        df['returns'] = df['close'].pct_change()
        df['momentum'] = df['returns'].rolling(window=lookback_period).sum()
        df['momentum_ma'] = df['momentum'].rolling(window=lookback_period).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)
        
        # Add signals
        signals = self._get_signals_for_symbol(symbol)
        if signals:
            buy_signals = [s for s in signals if s.get('signal_type') == 'BUY']
            sell_signals = [s for s in signals if s.get('signal_type') == 'SELL']
            
            for signal in buy_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='green', s=100, marker='^', alpha=0.8, zorder=5)
            
            for signal in sell_signals:
                timestamp = signal.get('timestamp')
                price = signal.get('price', 0)
                if timestamp in df.index:
                    ax1.scatter(timestamp, price, color='red', s=100, marker='v', alpha=0.8, zorder=5)
        
        ax1.set_title(f'{symbol} - Momentum Strategy Analysis ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Momentum
        ax2 = axes[1]
        ax2.plot(df.index, df['momentum'] * 100, color='orange', alpha=0.7, label='Momentum')
        ax2.axhline(y=momentum_threshold * 100, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({momentum_threshold*100:.1f}%)')
        ax2.axhline(y=-momentum_threshold * 100, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Momentum (%)', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RSI
        ax3 = axes[2]
        ax3.plot(df.index, df['rsi'], color='purple', alpha=0.7, label='RSI')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{symbol}_momentum_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Momentum analysis plot saved: {filename}")
        
        plt.show()
    
    def _plot_generic_analysis(self, data: pd.DataFrame, strategy: Any, 
                             symbol: str, start_date: str, end_date: str):
        """Plot generic analysis for unknown strategies."""
        df = data.copy()
        
        # Calculate basic indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Price and moving averages
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(df.index, df['sma_20'], label='SMA 20', linewidth=1, color='orange', alpha=0.7)
        ax1.plot(df.index, df['sma_50'], label='SMA 50', linewidth=1, color='red', alpha=0.7)
        
        ax1.set_title(f'{symbol} - Generic Analysis ({start_date} to {end_date})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = axes[1]
        ax2.bar(df.index, df['volume'], alpha=0.6, color='gray', label='Volume')
        ax2.plot(df.index, df['volume_ma'], color='red', alpha=0.7, label='Volume MA')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"data/results/{symbol}_generic_analysis_{start_date}_to_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Generic analysis plot saved: {filename}")
        
        plt.show()
    
    def _get_signals_for_symbol(self, symbol: str) -> List[Dict]:
        """Get signals for a specific symbol from the backtest results."""
        # This would need to be implemented to extract signals from the backtest results
        # For now, return empty list
        return []
    
    def _plot_random_day_analysis(self, results: Dict, strategy: Any, 
                                symbols: List[str], start_date: str, end_date: str):
        """Plot random day analysis for minute-level strategies."""
        try:
            # This would require access to the original data and signals
            # For now, we'll create a placeholder
            print("📊 Random day analysis available for minute-level strategies")
            print("   (Requires access to original data and signals)")
            
        except Exception as e:
            logger.error(f"Error in random day analysis: {e}")
    
    def _save_results(self, results: Dict, strategy_name: str, symbols: List[str], 
                     start_date: str, end_date: str):
        """Save backtest results to file."""
        try:
            # Create results directory if it doesn't exist
            Path("data/results").mkdir(parents=True, exist_ok=True)
            
            # Save results
            filename = f"data/results/{strategy_name}_results_{start_date}_to_{end_date}.json"
            
            # Convert DataFrames to dict for JSON serialization
            results_copy = results.copy()
            if 'equity_curve' in results_copy and isinstance(results_copy['equity_curve'], pd.DataFrame):
                results_copy['equity_curve'] = results_copy['equity_curve'].to_dict('records')
            
            import json
            with open(filename, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            print(f"❌ Error saving results: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Research-oriented backtesting with comprehensive analysis')
    parser.add_argument('--strategy', required=True, 
                       choices=['orb', 'momentum', 'volatility_breakout', 'simple_alpha', 'test'],
                       help='Strategy to test')
    parser.add_argument('--symbols', nargs='+', default=['RELIANCE.NS'], 
                       help='List of symbols to test')
    parser.add_argument('--start-date', default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--no-save', action='store_true', help='Disable result saving')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Load strategy parameters
    try:
        with open('config/strategies.yaml', 'r') as f:
            strategies_config = yaml.safe_load(f)
        
        strategy_config = strategies_config.get(args.strategy, {})
        strategy_params = strategy_config.get('parameters', {})
        
    except Exception as e:
        logger.warning(f"Could not load strategy config: {e}")
        strategy_params = {}
    
    # Initialize research backtester
    research_backtester = ResearchBacktester(args.config)
    
    # Run comprehensive backtest
    results = research_backtester.run_comprehensive_backtest(
        strategy_name=args.strategy,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_params=strategy_params,
        capital=args.capital,
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )
    
    if results:
        print("\n✅ Research backtest completed successfully!")
        print(f"📊 Strategy: {args.strategy}")
        print(f"🎯 Symbols: {args.symbols}")
        print(f"📅 Period: {args.start_date} to {args.end_date}")
        print(f"💰 Capital: ₹{args.capital:,.0f}")
    else:
        print("\n❌ Research backtest failed!")

if __name__ == "__main__":
    main() 