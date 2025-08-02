#!/usr/bin/env python3
"""
Backtest script for Volatility Breakout Strategy with 1-minute data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
from research_backtest import ResearchBacktester

def main():
    parser = argparse.ArgumentParser(description='Run Volatility Breakout Strategy Backtest')
    parser.add_argument('--symbols', nargs='+', default=['RELIANCE.NS'], 
                       help='List of symbols to test (default: RELIANCE.NS)')
    parser.add_argument('--start-date', default=None, 
                       help='Start date (YYYY-MM-DD, default: 5 days ago)')
    parser.add_argument('--end-date', default=None, 
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    parser.add_argument('--volatility-period', type=int, default=20,
                       help='Volatility calculation period (default: 20)')
    parser.add_argument('--volatility-multiplier', type=float, default=2.0,
                       help='Volatility multiplier for breakout threshold (default: 2.0)')
    parser.add_argument('--momentum-period', type=int, default=5,
                       help='Momentum calculation period (default: 5)')
    parser.add_argument('--volume-threshold', type=float, default=1.5,
                       help='Volume threshold for confirmation (default: 1.5)')
    parser.add_argument('--stop-loss', type=float, default=0.02,
                       help='Stop loss percentage (default: 0.02)')
    parser.add_argument('--take-profit', type=float, default=0.04,
                       help='Take profit percentage (default: 0.04)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results to files')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    # Strategy parameters
    strategy_params = {
        'volatility_period': args.volatility_period,
        'volatility_multiplier': args.volatility_multiplier,
        'momentum_period': args.momentum_period,
        'volume_threshold': args.volume_threshold,
        'stop_loss_pct': args.stop_loss,
        'take_profit_pct': args.take_profit,
        'min_volatility': 0.005,
        'max_volatility': 0.05
    }
    
    print("🚀 VOLATILITY BREAKOUT STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Strategy: Volatility Breakout")
    print(f"Symbols: {args.symbols}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Timeframe: 1-minute")
    print(f"Initial Capital: ₹{args.capital:,.2f}")
    print(f"Parameters: {strategy_params}")
    
    # Initialize research backtester
    researcher = ResearchBacktester()
    
    try:
        # Run comprehensive backtest
        results = researcher.run_comprehensive_backtest(
            strategy_name="volatility_breakout",
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy_params=strategy_params,
            initial_capital=args.capital,
            plot_results=not args.no_plots,
            save_results=args.save_results
        )
        
        if results:
            print("\n✅ Backtest completed successfully!")
            print("📊 Check the results directory for detailed analysis and plots")
        else:
            print("\n❌ Backtest failed to produce results")
            
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 