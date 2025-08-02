#!/usr/bin/env python3
"""
Simple Alpha Strategy for testing plotting utilities.
Uses basic moving average crossover with volume confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

class SimpleAlphaStrategy(BaseStrategy):
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.fast_ma = config.get('fast_ma', 5)
        self.slow_ma = config.get('slow_ma', 20)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.04)
        self.min_price_change = config.get('min_price_change', 0.005)
        logger.info(f"Initialized Simple Alpha strategy: {name}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = data.copy()
        
        # Moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['price_change'].rolling(window=5).mean()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals."""
        if len(data) < 2:  # Need at least 2 data points for comparison
            return []
        
        df = self.calculate_indicators(data)
        signals = []
        
        # Get latest data point
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        
        if prev is None:
            return []
        
        # More aggressive signal conditions
        ma_crossover = (current['fast_ma'] > current['slow_ma'] and 
                       prev['fast_ma'] <= prev['slow_ma'])
        
        ma_crossunder = (current['fast_ma'] < current['slow_ma'] and 
                        prev['fast_ma'] >= prev['slow_ma'])
        
        # Relaxed volume condition
        volume_confirmed = current['volume_ratio'] > self.volume_threshold or pd.isna(current['volume_ratio'])
        price_momentum = abs(current['price_momentum']) > self.min_price_change
        
        # Generate buy signal (more aggressive)
        if (ma_crossover or (current['fast_ma'] > current['slow_ma'] and current['price_momentum'] > 0)) and volume_confirmed:
            stop_loss = current['close'] * (1 - self.stop_loss_pct)
            take_profit = current['close'] * (1 + self.take_profit_pct)
            
            signals.append({
                'signal_type': 'BUY',
                'price': current['close'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': min(current['volume_ratio'] / 2 if not pd.isna(current['volume_ratio']) else 1.0, 2.0),
                'reason': f"MA crossover (fast_ma: {current['fast_ma']:.2f}, slow_ma: {current['slow_ma']:.2f})"
            })
        
        # Generate sell signal (more aggressive)
        elif (ma_crossunder or (current['fast_ma'] < current['slow_ma'] and current['price_momentum'] < 0)) and volume_confirmed:
            stop_loss = current['close'] * (1 + self.stop_loss_pct)
            take_profit = current['close'] * (1 - self.take_profit_pct)
            
            signals.append({
                'signal_type': 'SELL',
                'price': current['close'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': min(current['volume_ratio'] / 2 if not pd.isna(current['volume_ratio']) else 1.0, 2.0),
                'reason': f"MA crossunder (fast_ma: {current['fast_ma']:.2f}, slow_ma: {current['slow_ma']:.2f})"
            })
        
        return signals

    def get_strategy_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'type': 'Simple Alpha',
            'description': 'Moving average crossover with volume confirmation',
            'parameters': {
                'fast_ma': self.fast_ma,
                'slow_ma': self.slow_ma,
                'volume_threshold': self.volume_threshold,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'min_price_change': self.min_price_change
            }
        } 