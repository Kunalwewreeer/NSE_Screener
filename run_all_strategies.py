#!/usr/bin/env python3
"""
Simple script to run all strategies and show results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.research_backtest import ResearchBacktester
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Run all strategies with different configurations."""
    
    # Initialize research backtester
    research_backtester = ResearchBacktester()
    
    # Test configurations
    test_configs = [
        {
            'name': 'Simple Alpha - Daily',
            'strategy': 'simple_alpha',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'fast_ma': 3,
                'slow_ma': 8,
                'volume_threshold': 0.3,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.03,
                'min_price_change': 0.0005
            }
        },
        {
            'name': 'Volatility Breakout - Minute',
            'strategy': 'volatility_breakout',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'volatility_period': 10,
                'volatility_multiplier': 1.2,
                'momentum_period': 3,
                'volume_threshold': 0.3,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02,
                'min_volatility': 0.001,
                'max_volatility': 0.1
            }
        },
        {
            'name': 'ORB Strategy - Minute',
            'strategy': 'orb',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'lookback_period': 15,
                'breakout_threshold': 0.003,
                'volume_threshold': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'min_range_pct': 0.001
            }
        },
        {
            'name': 'Momentum Strategy - Daily',
            'strategy': 'momentum',
            'symbols': ['RELIANCE.NS'],
            'start_date': '2025-01-01',
            'end_date': '2025-01-15',
            'params': {
                'lookback_period': 10,
                'momentum_threshold': 0.01,
                'volume_threshold': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'rsi_period': 10,
                'rsi_overbought': 75,
                'rsi_oversold': 25
            }
        }
    ]
    
    print("🚀 RUNNING ALL STRATEGIES")
    print("=" * 60)
    
    results = []
    
    for config in test_configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"🎯 Strategy: {config['strategy']}")
        print(f"📅 Period: {config['start_date']} to {config['end_date']}")
        print("-" * 40)
        
        try:
            # Run backtest
            result = research_backtester.run_comprehensive_backtest(
                strategy_name=config['strategy'],
                symbols=config['symbols'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                strategy_params=config['params'],
                capital=100000,
                save_results=False,
                generate_plots=False
            )
            
            if result:
                total_return = result.get('total_return', 0)
                sharpe_ratio = result.get('sharpe_ratio', 0)
                max_drawdown = result.get('max_drawdown', 0)
                trades = result.get('trades', [])
                num_trades = len(trades)
                
                print(f"✅ Success!")
                print(f"💰 Total Return: {total_return:.2f}%")
                print(f"📈 Sharpe Ratio: {sharpe_ratio:.3f}")
                print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
                print(f"🔄 Number of Trades: {num_trades}")
                
                if trades:
                    print(f"📊 Sample Trades:")
                    for i, trade in enumerate(trades[:3]):
                        print(f"   Trade {i+1}: {trade.get('signal_type', 'UNKNOWN')} @ ₹{trade.get('price', 0):.2f}")
                
                results.append({
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
                results.append({
                    'name': config['name'],
                    'strategy': config['strategy'],
                    'status': 'NO_RESULTS'
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': config['name'],
                'strategy': config['strategy'],
                'status': f'ERROR: {str(e)}'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['name']}")
        
        if result['status'] == 'SUCCESS':
            print(f"   📊 Return: {result['total_return']:.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"Trades: {result['num_trades']}")
        else:
            print(f"   ❌ {result['status']}")
    
    # Find best performer
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['total_return'])
        print(f"\n🏆 Best Performer: {best_result['name']}")
        print(f"   📈 Return: {best_result['total_return']:.2f}%")
        print(f"   📊 Sharpe: {best_result['sharpe_ratio']:.3f}")
        print(f"   🔄 Trades: {best_result['num_trades']}")

if __name__ == "__main__":
    main() 