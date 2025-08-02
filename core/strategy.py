"""
Base strategy class with method stubs for signal generation, position sizing, and risk management.
All trading strategies should inherit from this class.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config
        self.positions = {}  # Current positions
        self.signals = []  # Historical signals
        self.performance = {}  # Performance metrics
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        This method must be implemented by all strategy classes.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signal columns (buy, sell, hold)
        """
        pass
    
    def position_sizing(self, signal: Dict[str, Any], capital: float, 
                       current_positions: Dict[str, Any]) -> float:
        """
        Calculate position size based on signal and risk parameters.
        Can be overridden by subclasses for custom position sizing.
        
        Args:
            signal: Signal dictionary with entry price, direction, etc.
            capital: Available capital
            current_positions: Current open positions
            
        Returns:
            Position size in number of shares
        """
        # Default position sizing: 10% of capital per position
        max_position_size = self.config.get('max_position_size', 0.1)
        risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Calculate position size based on risk
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        
        if entry_price > 0 and stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / risk_per_share
            else:
                position_size = (capital * max_position_size) / entry_price
        else:
            position_size = (capital * max_position_size) / entry_price
        
        # Ensure position size doesn't exceed maximum
        max_shares = (capital * max_position_size) / entry_price
        position_size = min(position_size, max_shares)
        
        return max(0, position_size)
    
    def calculate_stop_loss(self, signal: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calculate stop loss level for a signal.
        Can be overridden by subclasses for custom stop loss logic.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            Stop loss price
        """
        entry_price = signal.get('entry_price', 0)
        direction = signal.get('direction', 'long')
        
        if direction == 'long':
            # Default: 2% below entry price
            stop_loss = entry_price * (1 - 0.02)
        else:
            # Default: 2% above entry price
            stop_loss = entry_price * (1 + 0.02)
        
        return stop_loss
    
    def calculate_take_profit(self, signal: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calculate take profit level for a signal.
        Can be overridden by subclasses for custom take profit logic.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            Take profit price
        """
        entry_price = signal.get('entry_price', 0)
        direction = signal.get('direction', 'long')
        
        if direction == 'long':
            # Default: 6% above entry price (3:1 risk-reward)
            take_profit = entry_price * (1 + 0.06)
        else:
            # Default: 6% below entry price (3:1 risk-reward)
            take_profit = entry_price * (1 - 0.06)
        
        return take_profit
    
    def validate_signal(self, signal: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Validate if a signal meets all criteria.
        Can be overridden by subclasses for custom validation logic.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Basic validation
        required_fields = ['symbol', 'direction', 'entry_price', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Check if entry price is reasonable
        if signal['entry_price'] <= 0:
            logger.warning(f"Invalid entry price: {signal['entry_price']}")
            return False
        
        # Check if we have enough data (very lenient for testing)
        if len(data) < 1:
            logger.warning("Insufficient data for signal validation")
            return False
        
        return True
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            trade_result: Dictionary with trade results
        """
        if 'trades' not in self.performance:
            self.performance['trades'] = []
        
        self.performance['trades'].append(trade_result)
        
        # Calculate running metrics
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics from trade history."""
        if 'trades' not in self.performance or not self.performance['trades']:
            return
        
        trades = self.performance['trades']
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        # Risk metrics
        returns = [t.get('return', 0) for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Update performance dictionary
        self.performance.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance.copy()
    
    def reset(self) -> None:
        """Reset strategy state (positions, signals, performance)."""
        self.positions = {}
        self.signals = []
        self.performance = {}
        logger.info(f"Reset strategy: {self.name}")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})" 