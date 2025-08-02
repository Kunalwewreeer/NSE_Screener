#!/usr/bin/env python3
"""
Comprehensive test script for all strategies with debugging information.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import pandas as pd
from datetime import datetime, timedelta
from scripts.research_backtest import ResearchBacktester
from utils.logger import get_logger

logger = get_logger(__name__)

def test_all_strategies():
    """Test all strategies with different date ranges and parameters."""
    
    # Test configurations
    test_configs = [
        {
            'name': 'Simple Alpha - Recent',
            'strategy': 'simple_alpha',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'fast_ma': 3,  # More sensitive
                'slow_ma': 10,  # Shorter period
                'volume_threshold': 0.5,  # Lower threshold
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'min_price_change': 0.001  # Lower threshold
            }
        },
        {
            'name': 'Volatility Breakout - Recent',
            'strategy': 'volatility_breakout',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'volatility_period': 10,  # Shorter period
                'volatility_multiplier': 1.5,  # More sensitive
                'momentum_period': 3,  # Shorter period
                'volume_threshold': 0.5,  # Lower threshold
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02,
                'min_volatility': 0.001,  # Lower threshold
                'max_volatility': 0.1  # Higher threshold
            }
        },
        {
            'name': 'ORB Strategy - Recent',
            'strategy': 'orb',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'lookback_period': 15,  # Shorter period
                'breakout_threshold': 0.003,  # Lower threshold
                'volume_threshold': 1.0,  # Lower threshold
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'min_range_pct': 0.001  # Lower threshold
            }
        },
        {
            'name': 'Momentum Strategy - Recent',
            'strategy': 'momentum',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'lookback_period': 10,  # Shorter period
                'momentum_threshold': 0.01,  # Lower threshold
                'volume_threshold': 1.0,  # Lower threshold
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'rsi_period': 10,  # Shorter period
                'rsi_overbought': 75,  # More lenient
                'rsi_oversold': 25  # More lenient
            }
        }
    ]
    
    # Initialize research backtester
    research_backtester = ResearchBacktester()
    
    print("🚀 Starting Comprehensive Strategy Testing")
    print("=" * 60)
    
    results_summary = []
    
    for config in test_configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"🎯 Strategy: {config['strategy']}")
        print(f"📅 Period: {config['start_date']} to {config['end_date']}")
        print(f"⚙️  Parameters: {config['params']}")
        print("-" * 40)
        
        try:
            # Run backtest
            results = research_backtester.run_comprehensive_backtest(
                strategy_name=config['strategy'],
                symbols=config['symbols'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                strategy_params=config['params'],
                capital=100000,
                save_results=False,  # Don't save for testing
                generate_plots=False  # Don't generate plots for testing
            )
            
            if results:
                # Extract key metrics
                total_return = results.get('total_return', 0)
                sharpe_ratio = results.get('sharpe_ratio', 0)
                max_drawdown = results.get('max_drawdown', 0)
                trades = results.get('trades', [])
                num_trades = len(trades)
                
                print(f"✅ Success!")
                print(f"💰 Total Return: {total_return:.2f}%")
                print(f"📈 Sharpe Ratio: {sharpe_ratio:.3f}")
                print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
                print(f"🔄 Number of Trades: {num_trades}")
                
                if trades:
                    print(f"📊 Sample Trades:")
                    for i, trade in enumerate(trades[:3]):  # Show first 3 trades
                        print(f"   Trade {i+1}: {trade.get('signal_type', 'UNKNOWN')} @ ₹{trade.get('price', 0):.2f}")
                
                results_summary.append({
                    'name': config['name'],
                    'strategy': config['strategy'],
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'num_trades': num_trades,
                    'status': 'SUCCESS'
                })
                
            else:
                print("❌ No results returned")
                results_summary.append({
                    'name': config['name'],
                    'strategy': config['strategy'],
                    'status': 'NO_RESULTS'
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'name': config['name'],
                'strategy': config['strategy'],
                'status': f'ERROR: {str(e)}'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TESTING SUMMARY")
    print("=" * 60)
    
    for result in results_summary:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['name']}")
        
        if result['status'] == 'SUCCESS':
            print(f"   📊 Return: {result['total_return']:.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"Trades: {result['num_trades']}")
        else:
            print(f"   ❌ {result['status']}")
    
    # Find best performing strategy
    successful_results = [r for r in results_summary if r['status'] == 'SUCCESS']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['total_return'])
        print(f"\n🏆 Best Performer: {best_result['name']}")
        print(f"   📈 Return: {best_result['total_return']:.2f}%")
        print(f"   📊 Sharpe: {best_result['sharpe_ratio']:.3f}")
        print(f"   🔄 Trades: {best_result['num_trades']}")
    
    return results_summary

if __name__ == "__main__":
    test_all_strategies() 