#!/usr/bin/env python3
"""
Example usage of the trading system components.
This script demonstrates how to use the various parts of the system.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_handler import DataHandler
from strategies.orb import ORBStrategy
from strategies.momentum import MomentumStrategy
from core.portfolio import Portfolio
from core.broker import BrokerFactory
from core.clock import Clock
from core.metrics import PerformanceMetrics
from utils.logger import get_logger

logger = get_logger(__name__)


def example_data_handler():
    """Example of using the data handler."""
    print("\n=== Data Handler Example ===")
    
    # Initialize data handler
    data_handler = DataHandler()
    
    # Get market status
    market_status = data_handler.get_market_status()
    print(f"Market Status: {market_status}")
    
    # Get Nifty 50 symbols
    nifty50 = data_handler.get_nifty50_stocks()
    print(f"Number of Nifty 50 stocks: {len(nifty50)}")
    print(f"First 5 symbols: {nifty50[:5]}")
    
    # Fetch real data for demonstration
    print("\nFetching real data for demonstration...")
    try:
        sample_data = data_handler.get_historical_data(
            symbols="RELIANCE.NS",
            from_date="2023-01-01",
            to_date="2023-01-31",  # Shorter period for demonstration
            interval="day"
        )
        print(f"Data shape: {sample_data.shape}")
        print(f"Data preview:\n{sample_data.head()}")
        
        # Calculate technical indicators
        data_with_indicators = data_handler.calculate_technical_indicators(sample_data)
        print(f"Technical indicators calculated: {[col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
        
    except Exception as e:
        print(f"Data fetching failed: {e}")
        print("This is expected if tokens are not configured or symbols are invalid.")


def example_strategies():
    """Example of using strategies."""
    print("\n=== Strategy Example ===")
    
    # Initialize data handler for sample data generation
    data_handler = DataHandler()
    
    # Fetch real data
    try:
        sample_data = data_handler.get_historical_data(
            symbols="RELIANCE.NS",
            from_date="2023-01-01",
            to_date="2023-01-31",  # Shorter period for demonstration
            interval="day"
        )
        print(f"Fetched data: {sample_data.shape[0]} days of data")
    except Exception as e:
        print(f"Data fetching failed: {e}")
        # Create dummy data for demonstration
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 30),
            'high': np.random.uniform(150, 250, 30),
            'low': np.random.uniform(50, 150, 30),
            'close': np.random.uniform(100, 200, 30),
            'volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
        print(f"Using dummy data: {sample_data.shape[0]} days of data")
    
    # ORB Strategy
    orb_config = {
        'lookback_period': 30,
        'breakout_threshold': 0.005,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.06
    }
    orb_strategy = ORBStrategy("Example_ORB", orb_config)
    
    # Generate signals
    signals = orb_strategy.generate_signals(sample_data)
    signal_count = len(signals[signals['signal'] != 0])
    print(f"ORB Strategy generated {signal_count} signals")
    
    # Momentum Strategy
    momentum_config = {
        'lookback_period': 20,
        'momentum_threshold': 0.02,
        'rsi_period': 14,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.09
    }
    momentum_strategy = MomentumStrategy("Example_Momentum", momentum_config)
    
    # Generate signals
    signals = momentum_strategy.generate_signals(sample_data)
    signal_count = len(signals[signals['signal'] != 0])
    print(f"Momentum Strategy generated {signal_count} signals")


def example_portfolio():
    """Example of using portfolio management."""
    print("\n=== Portfolio Example ===")
    
    # Initialize portfolio
    config = {
        'max_position_size': 0.1,
        'slippage': 0.001,
        'transaction_cost': 0.0005
    }
    portfolio = Portfolio(100000, config)
    
    print(f"Initial capital: ₹{portfolio.initial_capital:,.2f}")
    
    # Place some orders
    orders = [
        ("RELIANCE", "buy", 50, 2500.0),
        ("TCS", "buy", 30, 3500.0),
        ("INFY", "buy", 100, 1500.0)
    ]
    
    for symbol, direction, quantity, price in orders:
        order = portfolio.place_order(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )
        print(f"Order: {symbol} {direction} {quantity} @ ₹{price:.2f}")
    
    print(f"Current capital: ₹{portfolio.current_capital:,.2f}")
    
    # Get position summary
    position_summary = portfolio.get_position_summary()
    print(f"Total positions: {position_summary['total_positions']}")
    print(f"Total exposure: ₹{position_summary['total_exposure']:,.2f}")


def example_broker():
    """Example of using broker interface."""
    print("\n=== Broker Example ===")
    
    # Paper broker
    config = {'capital': 100000}
    paper_broker = BrokerFactory.create_broker('paper', config)
    
    # Place orders
    orders = [
        ("HDFCBANK", "buy", 100, 1600.0),
        ("AXISBANK", "sell", 50, 800.0)
    ]
    
    for symbol, direction, quantity, price in orders:
        order = paper_broker.place_order(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price
        )
        print(f"Order: {order}")
    
    # Get positions
    positions = paper_broker.get_positions()
    print(f"Positions: {positions}")


def example_clock():
    """Example of using clock/time management."""
    print("\n=== Clock Example ===")
    
    # Live clock
    clock = Clock(mode='live')
    print(f"Current time: {clock.get_current_time()}")
    print(f"Market status: {clock.get_market_status()}")
    print(f"Is market open: {clock.is_market_open()}")
    
    # Backtest clock
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 12, 31)
    backtest_clock = Clock(mode='backtest', start_time=start_time, end_time=end_time)
    
    print(f"Backtest period: {backtest_clock.start_time} to {backtest_clock.end_time}")
    print(f"Elapsed time: {backtest_clock.get_elapsed_time()}")
    print(f"Remaining time: {backtest_clock.get_remaining_time()}")


def example_performance_metrics():
    """Example of using performance metrics."""
    print("\n=== Performance Metrics Example ===")
    
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 252)), index=dates)
    
    # Calculate returns
    returns = metrics.calculate_returns(prices)
    
    # Calculate various metrics
    sharpe = metrics.calculate_sharpe_ratio(returns)
    sortino = metrics.calculate_sortino_ratio(returns)
    max_dd, dd_start, dd_end = metrics.calculate_max_drawdown(prices)
    var_95 = metrics.calculate_var(returns, 0.05)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"VaR (95%): {var_95:.2%}")
    
    # Sample trades
    sample_trades = [
        {'pnl': 1000, 'return': 0.05},
        {'pnl': -500, 'return': -0.02},
        {'pnl': 2000, 'return': 0.08},
        {'pnl': -300, 'return': -0.01},
        {'pnl': 1500, 'return': 0.06}
    ]
    
    win_rate = metrics.calculate_win_rate(sample_trades)
    profit_factor = metrics.calculate_profit_factor(sample_trades)
    avg_trade = metrics.calculate_average_trade(sample_trades)
    
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade: ₹{avg_trade['avg_trade']:.2f}")


def main():
    """Run all examples."""
    print("TRADING SYSTEM EXAMPLES")
    print("=" * 50)
    
    try:
        example_data_handler()
        example_strategies()
        example_portfolio()
        example_broker()
        example_clock()
        example_performance_metrics()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("The trading system is working correctly.")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Example failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 