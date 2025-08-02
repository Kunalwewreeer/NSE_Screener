#!/usr/bin/env python3
"""
Simple test script to demonstrate the trading system functionality.
This script shows how to use the various components of the system.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_handler import DataHandler
from core.strategy import BaseStrategy
from strategies.orb import ORBStrategy
from strategies.momentum import MomentumStrategy
from core.portfolio import Portfolio
from core.broker import BrokerFactory
from core.clock import Clock
from core.metrics import PerformanceMetrics
from utils.logger import get_logger
from utils.file_utils import load_yaml

logger = get_logger(__name__)


def test_data_handler():
    """Test the data handler functionality."""
    print("\n" + "="*50)
    print("TESTING DATA HANDLER")
    print("="*50)
    
    try:
        # Initialize data handler
        data_handler = DataHandler()
        
        # Test market status
        market_status = data_handler.get_market_status()
        print(f"Market Status: {market_status}")
        
        # Test Nifty 50 symbols
        nifty50 = data_handler.get_nifty50_stocks()
        print(f"Nifty 50 stocks: {len(nifty50)} symbols")
        print(f"First 5 symbols: {nifty50[:5]}")
        
        # Test real data fetching
        print("\nTesting real data fetching...")
        try:
            sample_data = data_handler.get_historical_data(
                symbols="RELIANCE.NS",
                from_date="2023-01-01",
                to_date="2023-01-31",  # Shorter period for testing
                interval="day"
            )
            print(f"Fetched data shape: {sample_data.shape}")
            print(f"Data columns: {list(sample_data.columns)}")
            print(f"First few rows:\n{sample_data.head()}")
            
            # Test technical indicators calculation
            data_with_indicators = data_handler.calculate_technical_indicators(sample_data)
            print(f"\nTechnical indicators added: {[col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
            
        except Exception as e:
            print(f"Data fetching test failed: {e}")
            print("This is expected if tokens are not configured or symbols are invalid.")
            # Create dummy data for testing indicators
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            sample_data = pd.DataFrame({
                'open': np.random.uniform(100, 200, 30),
                'high': np.random.uniform(150, 250, 30),
                'low': np.random.uniform(50, 150, 30),
                'close': np.random.uniform(100, 200, 30),
                'volume': np.random.uniform(1000000, 5000000, 30)
            }, index=dates)
            
            # Test technical indicators calculation
            data_with_indicators = data_handler.calculate_technical_indicators(sample_data)
            print(f"\nTechnical indicators added: {[col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data handler test failed: {e}")
        return False


def test_strategies():
    """Test the strategy functionality."""
    print("\n" + "="*50)
    print("TESTING STRATEGIES")
    print("="*50)
    
    try:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(150, 250, 50),
            'low': np.random.uniform(50, 150, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        # Test ORB Strategy
        orb_config = {
            'lookback_period': 30,
            'breakout_threshold': 0.005,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06
        }
        orb_strategy = ORBStrategy("Test_ORB", orb_config)
        
        # Generate signals
        signals = orb_strategy.generate_signals(sample_data)
        print(f"ORB Strategy signals generated: {len(signals[signals['signal'] != 0])} signals")
        
        # Test Momentum Strategy
        momentum_config = {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'rsi_period': 14,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.09
        }
        momentum_strategy = MomentumStrategy("Test_Momentum", momentum_config)
        
        # Generate signals
        signals = momentum_strategy.generate_signals(sample_data)
        print(f"Momentum Strategy signals generated: {len(signals[signals['signal'] != 0])} signals")
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        return False


def test_portfolio():
    """Test the portfolio functionality."""
    print("\n" + "="*50)
    print("TESTING PORTFOLIO")
    print("="*50)
    
    try:
        # Initialize portfolio
        config = {
            'max_position_size': 0.1,
            'slippage': 0.001,
            'transaction_cost': 0.0005
        }
        portfolio = Portfolio(100000, config)
        
        # Test order placement
        order = portfolio.place_order(
            symbol="RELIANCE",
            direction="buy",
            quantity=100,
            price=2500.0,
            timestamp=datetime.now()
        )
        
        print(f"Order placed: {order}")
        print(f"Portfolio capital after order: ₹{portfolio.current_capital:,.2f}")
        
        # Test position summary
        position_summary = portfolio.get_position_summary()
        print(f"Position summary: {position_summary}")
        
        # Test performance summary
        performance = portfolio.get_performance_summary()
        print(f"Performance summary: {performance}")
        
        return True
        
    except Exception as e:
        logger.error(f"Portfolio test failed: {e}")
        return False


def test_broker():
    """Test the broker functionality."""
    print("\n" + "="*50)
    print("TESTING BROKER")
    print("="*50)
    
    try:
        # Test paper broker
        config = {'capital': 100000}
        paper_broker = BrokerFactory.create_broker('paper', config)
        
        # Place test order
        order = paper_broker.place_order(
            symbol="TCS",
            direction="buy",
            quantity=50,
            price=3500.0
        )
        
        print(f"Paper broker order: {order}")
        
        # Get positions
        positions = paper_broker.get_positions()
        print(f"Paper broker positions: {positions}")
        
        return True
        
    except Exception as e:
        logger.error(f"Broker test failed: {e}")
        return False


def test_clock():
    """Test the clock functionality."""
    print("\n" + "="*50)
    print("TESTING CLOCK")
    print("="*50)
    
    try:
        # Test live clock
        clock = Clock(mode='live')
        
        print(f"Current time: {clock.get_current_time()}")
        print(f"Market status: {clock.get_market_status()}")
        print(f"Is market open: {clock.is_market_open()}")
        
        # Test backtest clock
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 12, 31)
        backtest_clock = Clock(mode='backtest', start_time=start_time, end_time=end_time)
        
        print(f"Backtest start: {backtest_clock.start_time}")
        print(f"Backtest end: {backtest_clock.end_time}")
        print(f"Is finished: {backtest_clock.is_finished()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Clock test failed: {e}")
        return False


def test_performance_metrics():
    """Test the performance metrics functionality."""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE METRICS")
    print("="*50)
    
    try:
        # Initialize metrics
        metrics = PerformanceMetrics()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 252)), index=dates)
        
        # Calculate returns
        returns = metrics.calculate_returns(prices)
        
        # Calculate metrics
        sharpe = metrics.calculate_sharpe_ratio(returns)
        max_dd, dd_start, dd_end = metrics.calculate_max_drawdown(prices)
        
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Drawdown period: {dd_start} to {dd_end}")
        
        # Test sample trades
        sample_trades = [
            {'pnl': 1000, 'return': 0.05},
            {'pnl': -500, 'return': -0.02},
            {'pnl': 2000, 'return': 0.08},
            {'pnl': -300, 'return': -0.01}
        ]
        
        win_rate = metrics.calculate_win_rate(sample_trades)
        profit_factor = metrics.calculate_profit_factor(sample_trades)
        
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance metrics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("TRADING SYSTEM TEST SUITE")
    print("="*80)
    
    tests = [
        ("Data Handler", test_data_handler),
        ("Strategies", test_strategies),
        ("Portfolio", test_portfolio),
        ("Broker", test_broker),
        ("Clock", test_clock),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "PASSED" if success else "FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"{test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The trading system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 