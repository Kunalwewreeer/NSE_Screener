#!/usr/bin/env python3
"""
Volatility Breakout Strategy for 1-minute tick data.
Detects breakouts based on volatility expansion and price momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy optimized for 1-minute data.
    
    This strategy identifies breakouts when:
    1. Volatility expands beyond normal levels
    2. Price breaks above/below key levels with momentum
    3. Volume confirms the breakout
    """
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        
        # Strategy parameters
        self.volatility_period = config.get('volatility_period', 20)  # Period for volatility calculation
        self.volatility_multiplier = config.get('volatility_multiplier', 2.0)  # Multiplier for breakout threshold
        self.momentum_period = config.get('momentum_period', 5)  # Period for momentum calculation
        self.volume_threshold = config.get('volume_threshold', 1.5)  # Volume multiplier for confirmation
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4% take profit
        self.min_volatility = config.get('min_volatility', 0.005)  # Minimum volatility threshold
        self.max_volatility = config.get('max_volatility', 0.05)  # Maximum volatility threshold
        
        logger.info(f"Initialized Volatility Breakout strategy: {name}")
        logger.info(f"Parameters: volatility_period={self.volatility_period}, "
                   f"volatility_multiplier={self.volatility_multiplier}, "
                   f"momentum_period={self.momentum_period}")
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators for breakout detection."""
        if df.empty:
            return df
        
        # Calculate rolling volatility (standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_period).std()
        
        # Calculate volatility bands
        df['volatility_ma'] = df['volatility'].rolling(window=self.volatility_period).mean()
        df['volatility_upper'] = df['volatility_ma'] * self.volatility_multiplier
        df['volatility_lower'] = df['volatility_ma'] * 0.5
        
        # Calculate momentum indicators
        df['price_momentum'] = df['close'].pct_change(periods=self.momentum_period)
        df['volume_momentum'] = df['volume'].pct_change(periods=self.momentum_period)
        
        # Calculate support and resistance levels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['close_20'] = df['close'].rolling(window=20).mean()
        
        # Calculate breakout levels
        df['resistance'] = df['high_20'] + (df['volatility_upper'] * df['close_20'])
        df['support'] = df['low_20'] - (df['volatility_upper'] * df['close_20'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Clean up NaN values
        df = df.dropna()
        
        return df
    
    def detect_breakout_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Detect breakout signals based on volatility and price action."""
        signals = []
        
        if df.empty or len(df) < self.volatility_period:
            return signals
        
        # Calculate indicators
        df = self.calculate_volatility_indicators(df)
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check if volatility is expanding (more lenient)
            volatility_expanding = (
                (current_row['volatility'] > current_row['volatility_upper'] or 
                 current_row['volatility'] > prev_row['volatility']) and
                current_row['volatility'] > self.min_volatility and
                current_row['volatility'] < self.max_volatility
            )
            
            # Check for bullish breakout (more lenient conditions)
            bullish_breakout = (
                (current_row['close'] > current_row['resistance'] or current_row['price_momentum'] > 0.001) and
                (current_row['volume_ratio'] > self.volume_threshold or pd.isna(current_row['volume_ratio'])) and
                current_row['close'] > prev_row['close']
            )
            
            # Check for bearish breakout (more lenient conditions)
            bearish_breakout = (
                (current_row['close'] < current_row['support'] or current_row['price_momentum'] < -0.001) and
                (current_row['volume_ratio'] > self.volume_threshold or pd.isna(current_row['volume_ratio'])) and
                current_row['close'] < prev_row['close']
            )
            
            # Generate signals
            if volatility_expanding and bullish_breakout:
                signal = {
                    'timestamp': current_row.name,
                    'symbol': df.name if hasattr(df, 'name') else 'UNKNOWN',
                    'signal_type': 'BUY',
                    'price': current_row['close'],
                    'strength': min(current_row['volume_ratio'] / self.volume_threshold, 3.0),
                    'stop_loss': current_row['close'] * (1 - self.stop_loss_pct),
                    'take_profit': current_row['close'] * (1 + self.take_profit_pct),
                    'reason': f"Vol expansion + bullish breakout (vol: {current_row['volatility']:.4f}, mom: {current_row['price_momentum']:.4f})",
                    'metadata': {
                        'volatility': current_row['volatility'],
                        'momentum': current_row['price_momentum'],
                        'volume_ratio': current_row['volume_ratio'],
                        'breakout_level': current_row['resistance']
                    }
                }
                signals.append(signal)
                logger.info(f"Bullish breakout signal: {signal['symbol']} @ ₹{signal['price']:.2f} "
                           f"(Vol: {signal['metadata']['volatility']:.4f}, "
                           f"Mom: {signal['metadata']['momentum']:.4f})")
            
            elif volatility_expanding and bearish_breakout:
                signal = {
                    'timestamp': current_row.name,
                    'symbol': df.name if hasattr(df, 'name') else 'UNKNOWN',
                    'signal_type': 'SELL',
                    'price': current_row['close'],
                    'strength': min(current_row['volume_ratio'] / self.volume_threshold, 3.0),
                    'stop_loss': current_row['close'] * (1 + self.stop_loss_pct),
                    'take_profit': current_row['close'] * (1 - self.take_profit_pct),
                    'reason': f"Vol expansion + bearish breakout (vol: {current_row['volatility']:.4f}, mom: {current_row['price_momentum']:.4f})",
                    'metadata': {
                        'volatility': current_row['volatility'],
                        'momentum': current_row['price_momentum'],
                        'volume_ratio': current_row['volume_ratio'],
                        'breakout_level': current_row['support']
                    }
                }
                signals.append(signal)
                logger.info(f"Bearish breakout signal: {signal['symbol']} @ ₹{signal['price']:.2f} "
                           f"(Vol: {signal['metadata']['volatility']:.4f}, "
                           f"Mom: {signal['metadata']['momentum']:.4f})")
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate trading signals based on volatility breakout strategy.
        
        Args:
            data: DataFrame with OHLCV data (preferably 1-minute)
            
        Returns:
            List of trading signals
        """
        if data.empty:
            logger.warning("Empty data provided to volatility breakout strategy")
            return []
        
        # Set the symbol name for logging
        if hasattr(data, 'name'):
            data.name = data.name
        elif 'symbol' in data.columns:
            data.name = data['symbol'].iloc[0]
        
        logger.info(f"Generating volatility breakout signals for {data.name if hasattr(data, 'name') else 'UNKNOWN'}")
        logger.info(f"Data shape: {data.shape}, Time range: {data.index.min()} to {data.index.max()}")
        
        # Detect breakout signals
        signals = self.detect_breakout_signals(data)
        
        logger.info(f"Generated {len(signals)} volatility breakout signals")
        return signals
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and parameters."""
        return {
            'name': self.name,
            'type': 'VolatilityBreakout',
            'description': 'Volatility breakout strategy optimized for 1-minute data',
            'parameters': {
                'volatility_period': self.volatility_period,
                'volatility_multiplier': self.volatility_multiplier,
                'momentum_period': self.momentum_period,
                'volume_threshold': self.volume_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'min_volatility': self.min_volatility,
                'max_volatility': self.max_volatility
            },
            'recommended_timeframe': '1-minute',
            'suitable_for': ['High volatility stocks', 'Intraday trading', 'Breakout trading']
        } 