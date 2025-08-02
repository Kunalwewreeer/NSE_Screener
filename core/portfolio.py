"""
Portfolio management class to track positions, capital, risk limits, and transaction costs.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class Portfolio:
    """
    Portfolio management class for tracking positions, capital, and performance.
    """
    
    def __init__(self, initial_capital: float, config: Dict[str, Any]):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital amount
            config: Portfolio configuration dictionary
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config
        
        # Position tracking
        self.positions = {}  # Current open positions
        self.closed_positions = []  # Historical closed positions
        
        # Transaction tracking
        self.transactions = []  # All buy/sell transactions
        self.daily_nav = []  # Daily NAV tracking
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)
        self.slippage = config.get('slippage', 0.001)
        self.transaction_cost = config.get('transaction_cost', 0.0005)
        
        # Performance tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        logger.info(f"Initialized portfolio with capital: ₹{initial_capital:,.2f}")
    
    def place_order(self, symbol: str, direction: str, quantity: int, 
                   price: float, timestamp: datetime, order_type: str = "MARKET") -> Dict[str, Any]:
        """
        Place a buy/sell order and update portfolio.
        
        Args:
            symbol: Stock symbol
            direction: 'buy' or 'sell'
            quantity: Number of shares
            price: Order price
            timestamp: Order timestamp
            order_type: Type of order (MARKET, LIMIT, etc.)
            
        Returns:
            Dictionary with order details and execution results
        """
        # Calculate transaction costs
        transaction_value = quantity * price
        brokerage = transaction_value * self.transaction_cost
        slippage_cost = transaction_value * self.slippage
        total_cost = brokerage + slippage_cost
        
        # Calculate effective price
        if direction == 'buy':
            effective_price = price * (1 + self.slippage)
            total_cost += transaction_value * self.slippage
        else:
            effective_price = price * (1 - self.slippage)
            total_cost += transaction_value * self.slippage
        
        # Create transaction record
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'effective_price': effective_price,
            'transaction_value': transaction_value,
            'brokerage': brokerage,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'order_type': order_type
        }
        
        # Update portfolio based on direction
        if direction == 'buy':
            self._execute_buy_order(transaction)
        else:
            self._execute_sell_order(transaction)
        
        # Record transaction
        self.transactions.append(transaction)
        
        logger.info(f"Executed {direction} order: {symbol} {quantity} @ ₹{price:.2f}")
        return transaction
    
    def _execute_buy_order(self, transaction: Dict[str, Any]) -> None:
        """Execute a buy order and update positions."""
        symbol = transaction['symbol']
        quantity = transaction['quantity']
        effective_price = transaction['effective_price']
        total_cost = transaction['total_cost']
        
        # Check if we have enough capital
        required_capital = (quantity * effective_price) + total_cost
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital for buy order: {symbol}")
            return
        
        # Update capital
        self.current_capital -= required_capital
        
        # Update positions
        if symbol in self.positions:
            # Average down/up existing position
            existing_pos = self.positions[symbol]
            total_quantity = existing_pos['quantity'] + quantity
            total_value = (existing_pos['quantity'] * existing_pos['avg_price']) + (quantity * effective_price)
            avg_price = total_value / total_quantity
            
            self.positions[symbol].update({
                'quantity': total_quantity,
                'avg_price': avg_price,
                'last_update': transaction['timestamp']
            })
        else:
            # New position
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': effective_price,
                'entry_date': transaction['timestamp'],
                'last_update': transaction['timestamp']
            }
    
    def _execute_sell_order(self, transaction: Dict[str, Any]) -> None:
        """Execute a sell order and update positions."""
        symbol = transaction['symbol']
        quantity = transaction['quantity']
        effective_price = transaction['effective_price']
        total_cost = transaction['total_cost']
        
        # Check if we have the position
        if symbol not in self.positions:
            logger.warning(f"No position to sell: {symbol}")
            return
        
        position = self.positions[symbol]
        if position['quantity'] < quantity:
            logger.warning(f"Insufficient shares to sell: {symbol}")
            return
        
        # Calculate PnL
        entry_value = quantity * position['avg_price']
        exit_value = quantity * effective_price
        gross_pnl = exit_value - entry_value
        net_pnl = gross_pnl - total_cost
        
        # Update capital
        self.current_capital += exit_value - total_cost
        
        # Update position
        remaining_quantity = position['quantity'] - quantity
        if remaining_quantity > 0:
            # Partial exit
            self.positions[symbol]['quantity'] = remaining_quantity
        else:
            # Full exit - close position
            self._close_position(symbol, transaction['timestamp'], net_pnl)
        
        # Update PnL
        self.total_pnl += net_pnl
        self.realized_pnl += net_pnl
    
    def _close_position(self, symbol: str, timestamp: datetime, pnl: float) -> None:
        """Close a position and record it in closed positions."""
        position = self.positions[symbol]
        
        closed_position = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': timestamp,
            'quantity': position['quantity'],
            'entry_price': position['avg_price'],
            'exit_price': position.get('exit_price', 0),
            'pnl': pnl,
            'return_pct': (pnl / (position['quantity'] * position['avg_price'])) * 100
        }
        
        self.closed_positions.append(closed_position)
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol}, PnL: ₹{pnl:.2f}")
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]) -> None:
        """
        Update unrealized PnL based on current market prices.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        """
        self.unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                entry_value = position['quantity'] * position['avg_price']
                unrealized_pnl = position_value - entry_value
                self.unrealized_pnl += unrealized_pnl
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position['quantity'] * current_prices[symbol]
                portfolio_value += position_value
        
        return portfolio_value
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of current positions.
        
        Returns:
            Dictionary with position summary
        """
        summary = {
            'total_positions': len(self.positions),
            'positions': {},
            'total_exposure': 0.0
        }
        
        for symbol, position in self.positions.items():
            position_value = position['quantity'] * position['avg_price']
            summary['positions'][symbol] = {
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'value': position_value,
                'entry_date': position['entry_date']
            }
            summary['total_exposure'] += position_value
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get portfolio performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p['pnl'] > 0])
        losing_trades = len([p for p in self.closed_positions if p['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        total_return = (self.total_pnl / self.initial_capital) * 100
        
        # Calculate Sharpe ratio (simplified)
        if self.closed_positions:
            returns = [p['return_pct'] for p in self.closed_positions]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio
        }
    
    def check_risk_limits(self, new_position_value: float) -> bool:
        """
        Check if a new position would exceed risk limits.
        
        Args:
            new_position_value: Value of the new position
            
        Returns:
            True if within risk limits, False otherwise
        """
        # Check maximum position size
        portfolio_value = self.get_portfolio_value({})
        max_position_value = portfolio_value * self.max_position_size
        
        if new_position_value > max_position_value:
            logger.warning(f"Position size exceeds limit: {new_position_value} > {max_position_value}")
            return False
        
        # Check portfolio risk
        total_exposure = sum(pos['quantity'] * pos['avg_price'] for pos in self.positions.values())
        total_exposure += new_position_value
        
        if total_exposure > portfolio_value * (1 + self.max_portfolio_risk):
            logger.warning(f"Portfolio risk exceeds limit")
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.closed_positions = []
        self.transactions = []
        self.daily_nav = []
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        logger.info("Portfolio reset to initial state")
    
    def __str__(self) -> str:
        """String representation of the portfolio."""
        return f"Portfolio(Capital: ₹{self.current_capital:,.2f}, Positions: {len(self.positions)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the portfolio."""
        return f"Portfolio(initial_capital={self.initial_capital}, current_capital={self.current_capital}, positions={len(self.positions)})" 