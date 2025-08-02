"""
Momentum Strategy

This strategy identifies stocks with strong momentum and generates buy/sell signals
based on price momentum, volume, and relative strength.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from core.strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.
    
    The strategy works as follows:
    1. Calculate momentum indicators (RSI, MACD, price momentum)
    2. Identify stocks with strong positive/negative momentum
    3. Generate buy signals for strong positive momentum
    4. Generate sell signals for strong negative momentum
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize momentum strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Momentum specific parameters
        self.lookback_period = config.get('lookback_period', 20)  # days
        self.momentum_threshold = config.get('momentum_threshold', 0.02)  # 2%
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 1.5x average volume
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3%
        self.take_profit_pct = config.get('take_profit_pct', 0.09)  # 9%
        
        # State tracking
        self.momentum_signals = {}  # Store momentum signals for each symbol
        
        logger.info(f"Initialized momentum strategy: {name}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum trading signals.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            
        Returns:
            DataFrame with signal columns
        """
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Initialize signal columns
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['signal_strength'] = 0.0
        df['entry_price'] = 0.0
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        
        # Generate signals
        self._generate_momentum_signals(df)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(self.lookback_period)
        
        # Volume momentum
        df['volume_sma'] = df['volume'].rolling(window=self.lookback_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Moving averages
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        df['ma_cross'] = df['sma_short'] - df['sma_long']
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=self.lookback_period).std()
        
        # Relative strength (compared to market)
        # For simplicity, we'll use a simple momentum calculation
        df['relative_strength'] = df['price_momentum'] - df['price_momentum'].rolling(window=50).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> None:
        """
        Generate momentum signals based on calculated indicators.
        
        Args:
            df: DataFrame with indicators
        """
        for idx, row in df.iterrows():
            # Skip if we don't have enough data for indicators
            if pd.isna(row['price_momentum']) or pd.isna(row['rsi']):
                continue
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(row)
            
            # Generate buy signal for strong positive momentum
            if momentum_score > 0.7 and self._validate_buy_conditions(row):
                df.loc[idx, 'signal'] = 1
                df.loc[idx, 'signal_strength'] = momentum_score
                df.loc[idx, 'entry_price'] = row['close']
                df.loc[idx, 'stop_loss'] = row['close'] * (1 - self.stop_loss_pct)
                df.loc[idx, 'take_profit'] = row['close'] * (1 + self.take_profit_pct)
                
                logger.info(f"Momentum Buy signal: Score {momentum_score:.2f}, Price {row['close']:.2f}")
            
            # Generate sell signal for strong negative momentum
            elif momentum_score < -0.7 and self._validate_sell_conditions(row):
                df.loc[idx, 'signal'] = -1
                df.loc[idx, 'signal_strength'] = abs(momentum_score)
                df.loc[idx, 'entry_price'] = row['close']
                df.loc[idx, 'stop_loss'] = row['close'] * (1 + self.stop_loss_pct)
                df.loc[idx, 'take_profit'] = row['close'] * (1 - self.take_profit_pct)
                
                logger.info(f"Momentum Sell signal: Score {momentum_score:.2f}, Price {row['close']:.2f}")
    
    def _calculate_momentum_score(self, row: pd.Series) -> float:
        """
        Calculate momentum score based on multiple indicators.
        
        Args:
            row: DataFrame row with indicators
            
        Returns:
            Momentum score (-1 to 1)
        """
        score = 0.0
        weights = {
            'price_momentum': 0.3,
            'rsi': 0.2,
            'macd_histogram': 0.2,
            'volume_ratio': 0.15,
            'ma_cross': 0.1,
            'relative_strength': 0.05
        }
        
        # Price momentum component
        if not pd.isna(row['price_momentum']):
            momentum_norm = min(abs(row['price_momentum']) / self.momentum_threshold, 1.0)
            score += weights['price_momentum'] * np.sign(row['price_momentum']) * momentum_norm
        
        # RSI component
        if not pd.isna(row['rsi']):
            if row['rsi'] > self.rsi_overbought:
                rsi_score = -1.0  # Overbought - negative momentum
            elif row['rsi'] < self.rsi_oversold:
                rsi_score = 1.0   # Oversold - positive momentum potential
            else:
                rsi_score = (50 - row['rsi']) / 50  # Normalized RSI
            score += weights['rsi'] * rsi_score
        
        # MACD component
        if not pd.isna(row['macd_histogram']):
            macd_norm = min(abs(row['macd_histogram']) / 0.01, 1.0)  # Normalize MACD
            score += weights['macd_histogram'] * np.sign(row['macd_histogram']) * macd_norm
        
        # Volume component
        if not pd.isna(row['volume_ratio']):
            volume_score = min((row['volume_ratio'] - 1) / (self.volume_threshold - 1), 1.0)
            score += weights['volume_ratio'] * volume_score
        
        # Moving average cross component
        if not pd.isna(row['ma_cross']):
            ma_norm = min(abs(row['ma_cross']) / row['close'] * 100, 1.0)
            score += weights['ma_cross'] * np.sign(row['ma_cross']) * ma_norm
        
        # Relative strength component
        if not pd.isna(row['relative_strength']):
            rs_norm = min(abs(row['relative_strength']) / 0.01, 1.0)
            score += weights['relative_strength'] * np.sign(row['relative_strength']) * rs_norm
        
        return np.clip(score, -1.0, 1.0)
    
    def _validate_buy_conditions(self, row: pd.Series) -> bool:
        """
        Validate buy signal conditions.
        
        Args:
            row: DataFrame row with indicators
            
        Returns:
            True if buy conditions are met
        """
        # Price momentum should be positive
        if row['price_momentum'] <= 0:
            return False
        
        # Volume should be above threshold
        if row['volume_ratio'] < 1.0:
            return False
        
        # RSI should not be overbought
        if row['rsi'] > self.rsi_overbought:
            return False
        
        # MACD should be positive
        if row['macd_histogram'] <= 0:
            return False
        
        # Moving average cross should be positive
        if row['ma_cross'] <= 0:
            return False
        
        return True
    
    def _validate_sell_conditions(self, row: pd.Series) -> bool:
        """
        Validate sell signal conditions.
        
        Args:
            row: DataFrame row with indicators
            
        Returns:
            True if sell conditions are met
        """
        # Price momentum should be negative
        if row['price_momentum'] >= 0:
            return False
        
        # Volume should be above threshold
        if row['volume_ratio'] < 1.0:
            return False
        
        # RSI should not be oversold
        if row['rsi'] < self.rsi_oversold:
            return False
        
        # MACD should be negative
        if row['macd_histogram'] >= 0:
            return False
        
        # Moving average cross should be negative
        if row['ma_cross'] >= 0:
            return False
        
        return True
    
    def position_sizing(self, signal: Dict[str, Any], capital: float, 
                       current_positions: Dict[str, Any]) -> float:
        """
        Calculate position size for momentum strategy.
        
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
        
        # Adjust based on signal strength and volatility
        volatility_factor = 1.0
        if 'volatility' in signal:
            # Reduce position size for high volatility
            volatility = signal['volatility']
            if volatility > 0.03:  # 3% daily volatility
                volatility_factor = 0.7
            elif volatility > 0.05:  # 5% daily volatility
                volatility_factor = 0.5
        
        adjusted_position_size = base_position_size * signal_strength * volatility_factor
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, signal: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calculate stop loss for momentum strategy.
        
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
        Calculate take profit for momentum strategy.
        
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
        Validate momentum signal.
        
        Args:
            signal: Signal dictionary
            data: Market data
            
        Returns:
            True if signal is valid
        """
        # Basic validation from parent class
        if not super().validate_signal(signal, data):
            return False
        
        # Momentum specific validation
        signal_strength = signal.get('signal_strength', 0)
        if signal_strength < 0.5:  # Minimum signal strength for momentum
            logger.warning(f"Momentum signal strength too low: {signal_strength}")
            return False
        
        # Check for recent signals in the same direction
        symbol = signal.get('symbol', '')
        if symbol in self.momentum_signals:
            recent_signals = self.momentum_signals[symbol]
            if len(recent_signals) > 0:
                last_signal = recent_signals[-1]
                # Don't generate same direction signal within 5 days
                if (signal['direction'] == last_signal['direction'] and 
                    (signal['timestamp'] - last_signal['timestamp']).days < 5):
                    logger.warning(f"Recent signal exists for {symbol}")
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
            'strategy_type': 'Momentum',
            'parameters': {
                'lookback_period': self.lookback_period,
                'momentum_threshold': self.momentum_threshold,
                'rsi_period': self.rsi_period,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'volume_threshold': self.volume_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'momentum_signals_count': len(self.momentum_signals)
        } 