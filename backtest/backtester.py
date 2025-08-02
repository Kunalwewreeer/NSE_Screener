"""
Main backtester for running full backtests with strategies and configuration.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from core.data_handler import DataHandler
from core.strategy import BaseStrategy
from core.portfolio import Portfolio
from core.broker import BrokerFactory
from core.clock import Clock
from core.metrics import PerformanceMetrics
from utils.logger import get_logger
from utils.file_utils import load_yaml, save_csv

logger = get_logger(__name__)


class Backtester:
    """
    Main backtester class for running trading strategy backtests.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the backtester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml(config_path)
        self.data_handler = DataHandler(config_path)
        self.performance_metrics = PerformanceMetrics()
        
        # Results storage
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
        logger.info("Initialized backtester")
    
    def run_backtest(self, strategy: BaseStrategy, symbols: List[str], 
                    start_date: str, end_date: str, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy: Strategy instance to test
            symbols: List of symbols to trade
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {strategy.name}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ₹{initial_capital:,.2f}")
        
        # Initialize components
        portfolio = Portfolio(initial_capital, self.config['trading'])
        broker = BrokerFactory.create_broker('paper', self.config['trading'])
        clock = Clock(mode='backtest', 
                     start_time=datetime.strptime(start_date, "%Y-%m-%d"),
                     end_time=datetime.strptime(end_date, "%Y-%m-%d"))
        
        # Load data for all symbols
        all_data = self._load_data(symbols, start_date, end_date, strategy)
        if all_data.empty:
            logger.error("No data loaded for backtest")
            return {}
        
        # Run the backtest
        results = self._run_backtest_loop(strategy, portfolio, broker, clock, all_data)
        
        # Calculate performance metrics
        performance = self._calculate_performance(portfolio, all_data)
        
        # Store results
        self.results = {
            'strategy_name': strategy.name,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': portfolio.current_capital,
            'total_return': performance.get('total_return_pct', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown': performance.get('max_drawdown', 0),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance': performance,
            'data': {symbol: all_data[all_data['symbol'] == symbol] for symbol in symbols}  # Include original data for analysis
        }
        
        logger.info(f"Backtest completed. Final Capital: ₹{portfolio.current_capital:,.2f}")
        logger.info(f"Total Return: {performance.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        
        return self.results
    
    def _load_data(self, symbols: List[str], start_date: str, end_date: str, strategy=None) -> pd.DataFrame:
        """
        Load historical data for all symbols using robust, cached fetching.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all symbol data
        """
        try:
            # Determine the appropriate timeframe based on strategy type
            strategy_type = getattr(strategy, '__class__', None).__name__ if strategy else None
            
            # Use minute data for all strategies for better signal generation
            if strategy_type in ['VolatilityBreakoutStrategy', 'TestStrategy']:
                interval = 'minute'  # Use 1-minute data for these strategies
            elif strategy_type in ['SimpleAlphaStrategy', 'ORBStrategy', 'MomentumStrategy']:
                interval = 'minute'  # Use minute data for all strategies
            else:
                interval = 'minute'  # Default to minute data for better granularity
            
            # Fetch data for all symbols using the new robust fetcher
            data_dict = self.data_handler.get_historical_data(
                symbols=symbols,
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )
            
            all_data = []
            
            # Process each symbol's data
            for symbol, data in data_dict.items():
                if not data.empty:
                    # Ensure we have a copy to avoid SettingWithCopyWarning
                    data = data.copy()
                    # Add symbol column
                    data['symbol'] = symbol
                    
                    # Calculate technical indicators
                    try:
                        data = self.data_handler.calculate_technical_indicators(data)
                    except Exception as e:
                        logger.warning(f"Could not calculate technical indicators for {symbol}: {e}")
                        # Continue without technical indicators
                    
                    all_data.append(data)
                    logger.debug(f"Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            
            if not all_data:
                logger.error("No valid data found for any symbol")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=False)
            
            # Ensure index is properly formatted as datetime
            try:
                # Convert index to datetime if it's not already
                if not isinstance(combined_data.index, pd.DatetimeIndex):
                    combined_data.index = pd.to_datetime(combined_data.index)
                combined_data = combined_data.sort_index()
            except Exception as e:
                logger.error(f"Error sorting data by index: {e}")
                # Try alternative approach
                try:
                    # If index is problematic, reset and use timestamp column
                    if 'timestamp' in combined_data.columns:
                        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
                        combined_data.set_index('timestamp', inplace=True)
                        combined_data = combined_data.sort_index()
                    else:
                        # Create a simple numeric index
                        combined_data = combined_data.reset_index(drop=True)
                        combined_data.index = pd.to_datetime(combined_data.index)
                except Exception as e2:
                    logger.error(f"Error fixing index: {e2}")
                    raise ValueError(f"Failed to process data index: {e}")
            
            logger.info(f"Loaded total {len(combined_data)} records for {len(symbols)} symbols")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data for symbols {symbols}: {e}")
    
    def _run_backtest_loop(self, strategy: BaseStrategy, portfolio: Portfolio, 
                          broker: Any, clock: Clock, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the main backtest loop.
        
        Args:
            strategy: Strategy instance
            portfolio: Portfolio instance
            broker: Broker instance
            clock: Clock instance
            data: Historical data
            
        Returns:
            Dictionary with backtest results
        """
        # Always use minute data processing for better signal generation
        # Determine if we're using minute data or daily data
        strategy_type = getattr(strategy, '__class__', None).__name__ if strategy else None
        is_minute_data = strategy_type in ['VolatilityBreakoutStrategy', 'TestStrategy', 'SimpleAlphaStrategy', 'ORBStrategy', 'MomentumStrategy']
        
        # Check if data has minute-level timestamps (more granular than daily)
        data_index = data.index if hasattr(data, 'index') else pd.DatetimeIndex([])
        has_minute_data = len(data_index) > 0 and (data_index.max() - data_index.min()).days < len(data_index)
        
        # Always use minute data processing for better signal generation
        if True:  # Always use minute processing
            # For minute data, process each timestamp
            data_sorted = data.sort_index()
            portfolio_values = []
            
            # Create a rolling window for strategy data
            window_size = 50  # Number of historical data points to include
            
            for timestamp, row in tqdm(data_sorted.iterrows(), desc="Running backtest", total=len(data_sorted)):
                # Update clock
                clock.current_time = timestamp
                
                # Skip weekends and non-market hours
                if not clock.is_market_open():
                    continue
                
                symbol = row.get('symbol', 'UNKNOWN')
                
                # Create rolling window of data for this symbol
                symbol_all_data = data_sorted[data_sorted['symbol'] == symbol]
                
                # Get data up to current timestamp (including current)
                symbol_data = symbol_all_data.loc[:timestamp]
                
                # Limit to window_size if we have too much data
                if len(symbol_data) > window_size:
                    symbol_data = symbol_data.tail(window_size)
                
                # Generate signals for this symbol with historical context
                signals = strategy.generate_signals(symbol_data)
                
                # Process signals
                if signals:
                    logger.info(f"Processing {len(signals)} signals for {symbol} at {timestamp}")
                self._process_signals(strategy, portfolio, broker, signals, symbol, timestamp, symbol_data)
                
                # Update portfolio value every 100 records to avoid excessive updates
                if len(portfolio_values) % 100 == 0:
                    current_prices = {symbol: row['close']}
                    portfolio.update_unrealized_pnl(current_prices)
                    portfolio_value = portfolio.get_portfolio_value(current_prices)
                    
                    portfolio_values.append({
                        'date': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': portfolio.current_capital,
                        'unrealized_pnl': portfolio.unrealized_pnl
                    })
        else:
            # For daily data, group by date
            daily_data = data.groupby(data.index.date)
            portfolio_values = []
            
            for date, day_data in tqdm(daily_data, desc="Running backtest"):
                # Update clock
                clock.current_time = datetime.combine(date, datetime.min.time())
                
                # Skip weekends
                if not clock.is_market_open_today():
                    continue
                
                # Process each symbol for the day
                for symbol in day_data['symbol'].unique():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    
                    # Generate signals for this symbol
                    signals = strategy.generate_signals(symbol_data)
                    
                    # Process signals
                    if signals:
                        logger.info(f"Processing {len(signals)} signals for {symbol} on {date}")
                    self._process_signals(strategy, portfolio, broker, signals, symbol, clock.current_time, symbol_data)
                
                # Update portfolio value
                current_prices = {}
                for symbol in day_data['symbol'].unique():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        current_prices[symbol] = symbol_data['close'].iloc[-1]
                
                portfolio.update_unrealized_pnl(current_prices)
                portfolio_value = portfolio.get_portfolio_value(current_prices)
                
                portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': portfolio.current_capital,
                    'unrealized_pnl': portfolio.unrealized_pnl
                })
        
        # Store equity curve
        self.equity_curve = pd.DataFrame(portfolio_values)
        if not self.equity_curve.empty:
            self.equity_curve.set_index('date', inplace=True)
        
        logger.info(f"Backtest completed. Portfolio values: {len(portfolio_values)}, Trades: {len(self.trades)}")
        
        return {'portfolio_values': portfolio_values}
    
    def _process_signals(self, strategy: BaseStrategy, portfolio: Portfolio, 
                        broker: Any, signals: List[Dict], symbol: str, timestamp: datetime, data: pd.DataFrame = None):
        """
        Process trading signals for a symbol.
        
        Args:
            strategy: Strategy instance
            portfolio: Portfolio instance
            broker: Broker instance
            signals: List of signal dictionaries
            symbol: Symbol being processed
            timestamp: Current timestamp
        """
        if not signals or (hasattr(signals, 'empty') and signals.empty):
            return
        
        for signal in signals:
            # Extract signal information
            signal_type = signal.get('signal_type', 'UNKNOWN')
            if signal_type == 'UNKNOWN':
                continue
            
            # Create signal dictionary
            processed_signal = {
                'symbol': symbol,
                'direction': 'buy' if signal_type == 'BUY' else 'sell',
                'entry_price': signal.get('price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'signal_strength': signal.get('strength', 1.0),
                'timestamp': timestamp
            }
            
            # Validate signal with data context
            validation_data = data if data is not None else pd.DataFrame()
            validation_result = not hasattr(strategy, 'validate_signal') or strategy.validate_signal(processed_signal, validation_data)
            logger.info(f"Signal validation result: {validation_result} for {signal_type} signal")
            
            if validation_result:
                # Calculate position size (simplified)
                position_size = 8  # Reduced position size to fit within risk limits
                
                # Check risk limits
                position_value = position_size * processed_signal['entry_price']
                risk_limit = portfolio.current_capital * 0.1  # Max 10% per trade
                logger.info(f"Position value: ₹{position_value:.2f}, Risk limit: ₹{risk_limit:.2f}, Capital: ₹{portfolio.current_capital:.2f}")
                
                if position_value > risk_limit:
                    logger.info(f"Risk limit exceeded, skipping order")
                    continue
                
                # Place order
                logger.info(f"Placing order: {symbol} {processed_signal['direction']} {position_size} @ ₹{processed_signal['entry_price']:.2f}")
                order = broker.place_order(
                    symbol=symbol,
                    direction=processed_signal['direction'],
                    quantity=position_size,
                    price=processed_signal['entry_price'],
                    order_type="MARKET"
                )
                logger.info(f"Order result: {order.get('status', 'UNKNOWN')}")
                
                if order.get('status') == 'COMPLETE':
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'direction': processed_signal['direction'],
                        'quantity': position_size,
                        'entry_price': processed_signal['entry_price'],
                        'stop_loss': processed_signal['stop_loss'],
                        'take_profit': processed_signal['take_profit'],
                        'signal_strength': processed_signal['signal_strength']
                    }
                    
                    self.trades.append(trade)
                    logger.info(f"Executed trade: {symbol} {processed_signal['direction']} {position_size} @ ₹{processed_signal['entry_price']:.2f}")
    
    def _calculate_performance(self, portfolio: Portfolio, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: Portfolio instance
            data: Historical data
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve.empty:
            # Calculate returns from equity curve
            equity_series = self.equity_curve['portfolio_value']
            returns = self.performance_metrics.calculate_returns(equity_series)
            
            # Calculate performance metrics
            performance = self.performance_metrics.calculate_performance_metrics(
                returns=returns,
                prices=equity_series,
                trades=self.trades
            )
        else:
            # Fallback to portfolio performance
            performance = portfolio.get_performance_summary()
        
        return performance
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        if not self.results:
            return "No backtest results available."
        
        # Generate performance report
        performance = self.results.get('performance', {})
        report = self.performance_metrics.generate_performance_report(performance)
        
        # Add backtest summary
        summary = [
            "\n" + "=" * 60,
            "BACKTEST SUMMARY",
            "=" * 60,
            f"Strategy: {self.results.get('strategy_name', 'Unknown')}",
            f"Symbols: {', '.join(self.results.get('symbols', []))}",
            f"Period: {self.results.get('start_date', '')} to {self.results.get('end_date', '')}",
            f"Initial Capital: ₹{self.results.get('initial_capital', 0):,.2f}",
            f"Final Capital: ₹{self.results.get('final_capital', 0):,.2f}",
            f"Total Trades: {len(self.trades)}",
            f"Win Rate: {performance.get('trades', {}).get('win_rate_pct', 0):.1f}%",
            f"Profit Factor: {performance.get('trades', {}).get('profit_factor', 0):.2f}",
            "=" * 60
        ]
        
        full_report = report + "\n" + "\n".join(summary)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            logger.info(f"Report saved to {save_path}")
        
        return full_report
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Generate plots for backtest results.
        
        Args:
            save_path: Optional path to save plots
        """
        if self.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results: {self.results.get("strategy_name", "Unknown Strategy")}', fontsize=16)
        
        # 1. Equity Curve
        axes[0, 0].plot(self.equity_curve.index, self.equity_curve['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value (₹)')
        axes[0, 0].grid(True)
        
        # 2. Daily Returns
        if not self.equity_curve.empty:
            returns = self.equity_curve['portfolio_value'].pct_change().dropna()
            axes[0, 1].hist(returns, bins=50, alpha=0.7)
            axes[0, 1].set_title('Distribution of Daily Returns')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # 3. Drawdown
        if not self.equity_curve.empty:
            equity = self.equity_curve['portfolio_value']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown (%)')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True)
        
        # 4. Trade Analysis
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            if 'pnl' in trade_df.columns:
                axes[1, 1].scatter(range(len(trade_df)), trade_df['pnl'], alpha=0.6)
                axes[1, 1].axhline(y=0, color='red', linestyle='--')
                axes[1, 1].set_title('Trade P&L')
                axes[1, 1].set_xlabel('Trade Number')
                axes[1, 1].set_ylabel('P&L (₹)')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, file_path: str):
        """
        Save backtest results to file.
        
        Args:
            file_path: Path to save results
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save equity curve
        if not self.equity_curve.empty:
            equity_path = file_path.replace('.csv', '_equity.csv')
            self.equity_curve.to_csv(equity_path)
            logger.info(f"Equity curve saved to {equity_path}")
        
        # Save trades
        if self.trades:
            trades_path = file_path.replace('.csv', '_trades.csv')
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Trades saved to {trades_path}")
        
        # Save summary
        summary_path = file_path.replace('.csv', '_summary.csv')
        summary_data = {
            'metric': list(self.results['performance'].keys()),
            'value': list(self.results['performance'].values())
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
    
    def reset(self):
        """Reset backtester state."""
        self.results = {}
        self.trades = []
        self.equity_curve = []
        logger.info("Backtester reset") 