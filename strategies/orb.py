"""
Opening Range Breakout (ORB) Strategy

This strategy identifies the opening range (high-low) for a specified period after market open
and generates buy/sell signals when price breaks above/below this range.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from core.strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy.
    
    The strategy works as follows:
    1. Define an opening range period (e.g., first 30 minutes)
    2. Calculate high and low of this range
    3. Generate buy signal when price breaks above the high
    4. Generate sell signal when price breaks below the low
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize ORB strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # ORB specific parameters
        self.lookback_period = config.get('lookback_period', 30)  # minutes
        self.breakout_threshold = config.get('breakout_threshold', 0.005)  # 0.5%
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.06)  # 6%
        self.min_volume = config.get('min_volume', 1000000)  # Minimum volume filter
        
        # State tracking
        self.opening_ranges = {}  # Store opening ranges for each symbol
        self.breakout_levels = {}  # Store breakout levels
        
        logger.info(f"Initialized ORB strategy: {name}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ORB trading signals.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            
        Returns:
            DataFrame with signal columns
        """
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Initialize signal columns
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['signal_strength'] = 0.0
        df['entry_price'] = 0.0
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        
        # Group by date to process each trading day separately
        for date, day_data in df.groupby(df.index.date):
            self._process_trading_day(day_data, df)
        
        return df
    
    def _process_trading_day(self, day_data: pd.DataFrame, full_df: pd.DataFrame) -> None:
        """
        Process a single trading day for ORB signals.
        
        Args:
            day_data: Data for a single trading day
            full_df: Full DataFrame to update with signals
        """
        if len(day_data) < 2:
            return
        
        # Get symbol (assuming single symbol in data)
        symbol = day_data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in day_data.columns else 'UNKNOWN'
        
        # Calculate opening range
        opening_range = self._calculate_opening_range(day_data)
        if opening_range is None:
            return
        
        # Store opening range for this symbol and date
        date_key = f"{symbol}_{day_data.index[0].date()}"
        self.opening_ranges[date_key] = opening_range
        
        # Calculate breakout levels
        high_breakout = opening_range['high'] * (1 + self.breakout_threshold)
        low_breakout = opening_range['low'] * (1 - self.breakout_threshold)
        
        self.breakout_levels[date_key] = {
            'high_breakout': high_breakout,
            'low_breakout': low_breakout,
            'opening_range': opening_range
        }
        
        # Generate signals for the rest of the day
        self._generate_breakout_signals(day_data, full_df, date_key)
    
    def _calculate_opening_range(self, day_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculate opening range for the specified period.
        
        Args:
            day_data: Data for a single trading day
            
        Returns:
            Dictionary with high, low, and volume of opening range
        """
        # Get market open time (9:15 AM IST)
        market_open = day_data.index[0].replace(hour=9, minute=15, second=0, microsecond=0)
        
        # Calculate end time for opening range
        range_end = market_open + timedelta(minutes=self.lookback_period)
        
        # Get data within opening range period
        opening_data = day_data[
            (day_data.index >= market_open) & 
            (day_data.index <= range_end)
        ]
        
        if len(opening_data) < 2:
            logger.warning(f"Insufficient data for opening range calculation")
            return None
        
        # Calculate opening range
        opening_range = {
            'high': opening_data['high'].max(),
            'low': opening_data['low'].min(),
            'volume': opening_data['volume'].sum(),
            'start_time': market_open,
            'end_time': range_end
        }
        
        # Volume filter
        if opening_range['volume'] < self.min_volume:
            logger.debug(f"Opening range volume {opening_range['volume']} below minimum {self.min_volume}")
            return None
        
        # Range filter (minimum range size)
        range_size = (opening_range['high'] - opening_range['low']) / opening_range['low']
        if range_size < 0.005:  # Minimum 0.5% range
            logger.debug(f"Opening range too small: {range_size:.4f}")
            return None
        
        return opening_range
    
    def _generate_breakout_signals(self, day_data: pd.DataFrame, full_df: pd.DataFrame, date_key: str) -> None:
        """
        Generate breakout signals for the trading day.
        
        Args:
            day_data: Data for a single trading day
            full_df: Full DataFrame to update with signals
            date_key: Key for accessing breakout levels
        """
        if date_key not in self.breakout_levels:
            return
        
        breakout_levels = self.breakout_levels[date_key]
        high_breakout = breakout_levels['high_breakout']
        low_breakout = breakout_levels['low_breakout']
        opening_range = breakout_levels['opening_range']
        
        # Track if we've already generated signals for this day
        buy_signal_generated = False
        sell_signal_generated = False
        
        # Process each bar after the opening range period
        range_end_time = opening_range['end_time']
        
        for idx, row in day_data[day_data.index > range_end_time].iterrows():
            current_price = row['close']
            current_volume = row['volume']
            
            # Check for high breakout (buy signal)
            if not buy_signal_generated and current_price > high_breakout:
                # Volume confirmation
                if current_volume > opening_range['volume'] * 0.5:  # At least 50% of opening range volume
                    signal_strength = self._calculate_signal_strength(current_price, high_breakout, opening_range)
                    
                    # Update DataFrame
                    full_df.loc[idx, 'signal'] = 1
                    full_df.loc[idx, 'signal_strength'] = signal_strength
                    full_df.loc[idx, 'entry_price'] = current_price
                    full_df.loc[idx, 'stop_loss'] = current_price * (1 - self.stop_loss_pct)
                    full_df.loc[idx, 'take_profit'] = current_price * (1 + self.take_profit_pct)
                    
                    buy_signal_generated = True
                    logger.info(f"ORB Buy signal: Price {current_price:.2f} broke above {high_breakout:.2f}")
            
            # Check for low breakout (sell signal)
            elif not sell_signal_generated and current_price < low_breakout:
                # Volume confirmation
                if current_volume > opening_range['volume'] * 0.5:
                    signal_strength = self._calculate_signal_strength(low_breakout, current_price, opening_range)
                    
                    # Update DataFrame
                    full_df.loc[idx, 'signal'] = -1
                    full_df.loc[idx, 'signal_strength'] = signal_strength
                    full_df.loc[idx, 'entry_price'] = current_price
                    full_df.loc[idx, 'stop_loss'] = current_price * (1 + self.stop_loss_pct)
                    full_df.loc[idx, 'take_profit'] = current_price * (1 - self.take_profit_pct)
                    
                    sell_signal_generated = True
                    logger.info(f"ORB Sell signal: Price {current_price:.2f} broke below {low_breakout:.2f}")
    
    def _calculate_signal_strength(self, current_price: float, breakout_level: float, 
                                 opening_range: Dict[str, float]) -> float:
        """
        Calculate signal strength based on breakout magnitude and volume.
        
        Args:
            current_price: Current price
            breakout_level: Breakout level (high or low)
            opening_range: Opening range data
            
        Returns:
            Signal strength (0-1)
        """
        # Calculate breakout magnitude
        if current_price > breakout_level:
            # High breakout
            breakout_magnitude = (current_price - breakout_level) / breakout_level
        else:
            # Low breakout
            breakout_magnitude = (breakout_level - current_price) / breakout_level
        
        # Normalize breakout magnitude (0-1 scale)
        normalized_magnitude = min(breakout_magnitude / 0.02, 1.0)  # 2% = full strength
        
        # Volume factor
        volume_factor = min(opening_range['volume'] / self.min_volume, 2.0) / 2.0
        
        # Combine factors
        signal_strength = (normalized_magnitude * 0.7) + (volume_factor * 0.3)
        
        return min(signal_strength, 1.0)
    
    def position_sizing(self, signal: Dict[str, Any], capital: float, 
                       current_positions: Dict[str, Any]) -> float:
        """
        Calculate position size for ORB strategy.
        
        Args:
            signal: Signal dictionary
            capital: Available capital
            current_positions: Current open positions
            
        Returns:
            Position size in number of shares
        """
        # Get signal strength for position sizing
        signal_strength = signal.get('signal_strength', 0.5)
        
        # Base position size from parent class
        base_position_size = super().position_sizing(signal, capital, current_positions)
        
        # Adjust based on signal strength
        adjusted_position_size = base_position_size * signal_strength
        
        # Additional risk management for ORB
        # Reduce position size if we have multiple positions
        if len(current_positions) > 0:
            adjusted_position_size *= 0.8  # Reduce by 20%
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, signal: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calculate stop loss for ORB strategy.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            Stop loss price
        """
        entry_price = signal.get('entry_price', 0)
        direction = signal.get('direction', 'long')
        
        if direction == 'long':
            # For long positions, stop loss below entry
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        else:
            # For short positions, stop loss above entry
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        
        return stop_loss
    
    def calculate_take_profit(self, signal: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calculate take profit for ORB strategy.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            Take profit price
        """
        entry_price = signal.get('entry_price', 0)
        direction = signal.get('direction', 'long')
        
        if direction == 'long':
            # For long positions, take profit above entry
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            # For short positions, take profit below entry
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        return take_profit
    
    def validate_signal(self, signal: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Validate ORB signal.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            True if signal is valid
        """
        # Basic validation from parent class
        if not super().validate_signal(signal, data):
            return False
        
        # ORB specific validation
        signal_strength = signal.get('signal_strength', 0)
        if signal_strength < 0.3:  # Minimum signal strength
            logger.warning(f"ORB signal strength too low: {signal_strength}")
            return False
        
        # Check if we already have a position in this symbol
        symbol = signal.get('symbol', '')
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}")
            return False
        
        return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and statistics.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'strategy_name': self.name,
            'strategy_type': 'ORB',
            'parameters': {
                'lookback_period': self.lookback_period,
                'breakout_threshold': self.breakout_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'min_volume': self.min_volume
            },
            'opening_ranges_count': len(self.opening_ranges),
            'breakout_levels_count': len(self.breakout_levels)
        } 