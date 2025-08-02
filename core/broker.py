"""
Abstract broker interface for paper and live trading with KiteConnect backend.
"""
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from kiteconnect import KiteConnect
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseBroker(ABC):
    """
    Abstract base class for broker interfaces.
    """
    
    @abstractmethod
    def place_order(self, symbol: str, direction: str, quantity: int, 
                   price: float, order_type: str = "MARKET") -> Dict[str, Any]:
        """Place an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_holdings(self) -> Dict[str, Any]:
        """Get current holdings."""
        pass


class PaperBroker(BaseBroker):
    """
    Paper trading broker for backtesting and simulation.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize paper broker.
        
        Args:
            initial_capital: Initial capital for paper trading
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        
        logger.info(f"Initialized paper broker with capital: ₹{initial_capital:,.2f}")
    
    def place_order(self, symbol: str, direction: str, quantity: int, 
                   price: float, order_type: str = "MARKET") -> Dict[str, Any]:
        """
        Place a paper trading order.
        
        Args:
            symbol: Stock symbol
            direction: 'buy' or 'sell'
            quantity: Number of shares
            price: Order price
            order_type: Type of order (MARKET, LIMIT, etc.)
            
        Returns:
            Dictionary with order details
        """
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        # Simulate order execution
        execution_price = price  # For paper trading, assume execution at order price
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'order_type': order_type,
            'status': 'COMPLETE',
            'timestamp': datetime.now(),
            'filled_quantity': quantity,
            'pending_quantity': 0
        }
        
        # Update positions and capital
        if direction == 'buy':
            cost = quantity * execution_price
            if cost <= self.current_capital:
                self.current_capital -= cost
                if symbol in self.positions:
                    # Average down/up
                    existing_pos = self.positions[symbol]
                    total_quantity = existing_pos['quantity'] + quantity
                    total_value = (existing_pos['quantity'] * existing_pos['avg_price']) + cost
                    self.positions[symbol] = {
                        'quantity': total_quantity,
                        'avg_price': total_value / total_quantity
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': execution_price
                    }
            else:
                order['status'] = 'REJECTED'
                order['rejection_reason'] = 'Insufficient capital'
        else:
            # Sell order
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                proceeds = quantity * execution_price
                self.current_capital += proceeds
                
                remaining_quantity = self.positions[symbol]['quantity'] - quantity
                if remaining_quantity > 0:
                    self.positions[symbol]['quantity'] = remaining_quantity
                else:
                    del self.positions[symbol]
            else:
                order['status'] = 'REJECTED'
                order['rejection_reason'] = 'Insufficient shares'
        
        self.orders[order_id] = order
        
        logger.info(f"Paper order executed: {symbol} {direction} {quantity} @ ₹{execution_price:.2f}")
        return order
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get paper order status."""
        return self.orders.get(order_id, {'status': 'NOT_FOUND'})
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel paper order."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            logger.info(f"Paper order cancelled: {order_id}")
            return True
        return False
    
    def get_positions(self) -> Dict[str, Any]:
        """Get paper trading positions."""
        return {
            'net': self.positions,
            'day': {}  # Paper trading doesn't track day positions
        }
    
    def get_holdings(self) -> Dict[str, Any]:
        """Get paper trading holdings."""
        holdings = []
        for symbol, position in self.positions.items():
            holdings.append({
                'tradingsymbol': symbol,
                'quantity': position['quantity'],
                'average_price': position['avg_price']
            })
        return holdings
    
    def get_margins(self) -> Dict[str, Any]:
        """Get paper trading margins."""
        return {
            'equity': {
                'available': self.current_capital,
                'used': 0,
                'net': self.current_capital
            }
        }


class LiveBroker(BaseBroker):
    """
    Live trading broker using Zerodha KiteConnect.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live broker with KiteConnect.
        
        Args:
            config: Broker configuration dictionary
        """
        self.config = config
        self.kite = None
        self._initialize_kite()
        
    def _initialize_kite(self):
        """Initialize KiteConnect connection."""
        try:
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            access_token = os.getenv("ACCESS_TOKEN")
            
            if not all([api_key, api_secret]):
                logger.error("Zerodha API credentials not found")
                return
            
            self.kite = KiteConnect(api_key=api_key)
            
            if access_token:
                self.kite.set_access_token(access_token)
                logger.info("Live broker initialized with KiteConnect")
            else:
                logger.warning("Access token not found for live trading")
                
        except Exception as e:
            logger.error(f"Error initializing live broker: {e}")
    
    def place_order(self, symbol: str, direction: str, quantity: int, 
                   price: float, order_type: str = "MARKET") -> Dict[str, Any]:
        """
        Place a live order through KiteConnect.
        
        Args:
            symbol: Stock symbol
            direction: 'buy' or 'sell'
            quantity: Number of shares
            price: Order price
            order_type: Type of order (MARKET, LIMIT, etc.)
            
        Returns:
            Dictionary with order details
        """
        if not self.kite:
            logger.error("Kite connection not initialized")
            return {'status': 'ERROR', 'message': 'Broker not connected'}
        
        try:
            # Convert direction to Kite format
            transaction_type = "BUY" if direction == "buy" else "SELL"
            
            # Place order
            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": "CNC",  # Cash and Carry
                "order_type": order_type
            }
            
            if order_type == "LIMIT":
                order_params["price"] = price
            
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                **order_params
            )
            
            logger.info(f"Live order placed: {symbol} {direction} {quantity} @ ₹{price:.2f}, Order ID: {order_id}")
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'order_type': order_type,
                'status': 'PENDING',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error placing live order: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get live order status from KiteConnect."""
        if not self.kite:
            return {'status': 'ERROR', 'message': 'Broker not connected'}
        
        try:
            order_history = self.kite.order_history(order_id)
            if order_history:
                latest_status = order_history[-1]
                return {
                    'order_id': order_id,
                    'status': latest_status.get('status'),
                    'filled_quantity': latest_status.get('filled_quantity', 0),
                    'pending_quantity': latest_status.get('pending_quantity', 0),
                    'average_price': latest_status.get('average_price', 0)
                }
            else:
                return {'status': 'NOT_FOUND'}
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel live order through KiteConnect."""
        if not self.kite:
            logger.error("Kite connection not initialized")
            return False
        
        try:
            self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            logger.info(f"Live order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Any]:
        """Get live positions from KiteConnect."""
        if not self.kite:
            return {'net': {}, 'day': {}}
        
        try:
            positions = self.kite.positions()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {'net': {}, 'day': {}}
    
    def get_holdings(self) -> Dict[str, Any]:
        """Get live holdings from KiteConnect."""
        if not self.kite:
            return []
        
        try:
            holdings = self.kite.holdings()
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return []
    
    def get_margins(self) -> Dict[str, Any]:
        """Get live margins from KiteConnect."""
        if not self.kite:
            return {}
        
        try:
            margins = self.kite.margins()
            return margins
            
        except Exception as e:
            logger.error(f"Error getting margins: {e}")
            return {}


class BrokerFactory:
    """
    Factory class to create broker instances.
    """
    
    @staticmethod
    def create_broker(broker_type: str, config: Dict[str, Any]) -> BaseBroker:
        """
        Create a broker instance based on type.
        
        Args:
            broker_type: 'paper' or 'live'
            config: Broker configuration
            
        Returns:
            Broker instance
        """
        if broker_type.lower() == 'paper':
            initial_capital = config.get('capital', 100000)
            return PaperBroker(initial_capital)
        elif broker_type.lower() == 'live':
            return LiveBroker(config)
        else:
            raise ValueError(f"Unknown broker type: {broker_type}") 