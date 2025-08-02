#!/usr/bin/env python3
"""
Main entry point for running backtests.

Usage:
    python run_backtest.py --strategy orb --symbols RELIANCE,TCS,INFY --start-date 2023-01-01 --end-date 2024-01-01
    python run_backtest.py --strategy momentum --symbols NIFTY50 --start-date 2023-01-01 --end-date 2024-01-01
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtester import Backtester
from strategies.orb import ORBStrategy
from strategies.momentum import MomentumStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from utils.logger import get_logger
from utils.file_utils import load_yaml

logger = get_logger(__name__)


def get_strategy(strategy_name: str, config: Dict[str, Any]):
    """
    Get strategy instance by name.
    
    Args:
        strategy_name: Name of the strategy
        config: Configuration dictionary
        
    Returns:
        Strategy instance
    """
    strategy_config = config.get('strategies', {}).get(strategy_name, {})
    
    if strategy_name.lower() == 'orb':
        return ORBStrategy(f"ORB_{datetime.now().strftime('%Y%m%d_%H%M%S')}", strategy_config)
    elif strategy_name.lower() == 'momentum':
        return MomentumStrategy(f"Momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')}", strategy_config)
    elif strategy_name.lower() == 'volatility_breakout':
        return VolatilityBreakoutStrategy(f"VolatilityBreakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}", strategy_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def get_nifty50_symbols() -> List[str]:
    """
    Get Nifty 50 symbols with proper NSE suffixes.
    
    Returns:
        List of Nifty 50 symbols with .NS suffix
    """
    return [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
        "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "UPL.NS", "WIPRO.NS"
    ]


def main():
    """Main function to run backtests."""
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')
    
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['orb', 'momentum', 'volatility_breakout'],
                       help='Strategy to test')
    
    parser.add_argument('--symbols', type=str, required=True,
                       help='Comma-separated list of symbols (use .NS suffix for NSE stocks) or "NIFTY50" for all Nifty 50 stocks')
    
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date in YYYY-MM-DD format')
    
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path (default: config/config.yaml)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to files')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_yaml(args.config)
        
        # Parse symbols
        if args.symbols.upper() == 'NIFTY50':
            symbols = get_nifty50_symbols()
            logger.info(f"Using all Nifty 50 symbols: {len(symbols)} stocks")
        else:
            symbols = [s.strip() for s in args.symbols.split(',')]
            logger.info(f"Using symbols: {symbols}")
        
        # Validate dates
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > datetime.now():
            logger.warning("End date is in the future, using current date")
            end_date = datetime.now()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize backtester
        backtester = Backtester(args.config)
        
        # Get strategy
        strategy = get_strategy(args.strategy, config)
        
        # Run backtest
        logger.info("Starting backtest...")
        results = backtester.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )
        
        if not results:
            logger.error("Backtest failed to produce results")
            return 1
        
        # Generate report
        report = backtester.generate_report()
        print("\n" + "="*80)
        print("BACKTEST REPORT")
        print("="*80)
        print(report)
        
        # Save results if requested
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{args.output_dir}/{args.strategy}_{timestamp}"
            
            # Save report
            report_path = f"{base_filename}_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
            
            # Save results
            results_path = f"{base_filename}_results.csv"
            backtester.save_results(results_path)
            logger.info(f"Results saved to {results_path}")
        
        # Generate plots if requested
        if args.plot:
            plot_path = f"{args.output_dir}/{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_plots.png"
            backtester.plot_results(save_path=plot_path)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Strategy: {args.strategy}")
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Initial Capital: ₹{args.capital:,.2f}")
        print(f"Final Capital: ₹{results.get('final_capital', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Total Trades: {len(results.get('trades', []))}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 