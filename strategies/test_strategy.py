#!/usr/bin/env python3
"""
Test Strategy for debugging signal generation.
Generates signals on every data point to verify the system works.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

class TestStrategy(BaseStrategy):
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.signal_frequency = config.get('signal_frequency', 0.1)  # 10% of data points
        logger.info(f"Initialized Test Strategy: {name}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple indicators."""
        df = data.copy()
        
        # Simple moving average
        df['sma_5'] = df['close'].rolling(window=5).mean()
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate test signals."""
        logger.info(f"Test strategy generate_signals called with {len(data)} records")
        
        if len(data) < 1:
            logger.info(f"Not enough data ({len(data)} records), returning empty signals")
            return []
        
        df = self.calculate_indicators(data)
        signals = []
        
        # Get latest data point
        current = df.iloc[-1]
        logger.info(f"Processing data point: {current.name} with close price ₹{current['close']:.2f}")
        
        # Generate signal on every data point for testing
        import random
        signal_type = 'BUY' if random.random() > 0.5 else 'SELL'
        
        signal = {
            'signal_type': signal_type,
            'price': current['close'],
            'stop_loss': current['close'] * (0.98 if signal_type == 'BUY' else 1.02),
            'take_profit': current['close'] * (1.02 if signal_type == 'BUY' else 0.98),
            'strength': 1.0,
            'reason': f"Test signal - {signal_type}"
        }
        
        signals.append(signal)
        logger.info(f"Generated test signal: {signal_type} @ ₹{current['close']:.2f}")
        logger.info(f"Returning {len(signals)} signals")
        return signals
        

    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and parameters."""
        return {
            'name': self.name,
            'type': 'TestStrategy',
            'description': 'Test strategy for debugging signal generation',
            'parameters': {
                'signal_frequency': self.signal_frequency
            },
            'recommended_timeframe': 'any',
            'suitable_for': ['Testing', 'Debugging']
        } 