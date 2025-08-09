#!/usr/bin/env python3
"""
Fakeout Reversal Detection System

Detects intraday fakeout reversals around key levels (PDH/PDL, VWAP, etc.)
with customizable parameters and extensive debugging capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeoutDetector:
    """
    Modular fakeout reversal detection system with extensive debugging.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fakeout detector with configuration.
        
        Args:
            config: Dictionary containing detection parameters
        """
        # Default configuration
        self.config = {
            # Level detection
            'level_lookback': 20,  # Candles to look back for level calculation
            'level_threshold': 0.1,  # % threshold for level significance
            
            # Breakout detection
            'breakout_threshold': 0.05,  # % above/below level to consider breakout
            'wick_threshold': 0.3,  # Minimum wick % of candle body
            'volume_spike_threshold': 1.5,  # Volume spike multiplier
            
            # Confirmation rules
            'confirmation_candles': 2,  # Number of candles to confirm reversal
            'confirmation_threshold': 0.02,  # % move back inside level
            'max_confirmation_time': 10,  # Max minutes to wait for confirmation
            
            # Risk management
            'sl_multiplier': 1.5,  # SL distance as multiple of ATR
            'tp_multiplier': 2.0,  # TP distance as multiple of ATR
            'atr_period': 14,  # ATR calculation period
            
            # Debug settings
            'debug_mode': True,
            'log_level': 'INFO',
            'plot_signals': True
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, self.config['log_level']))
        
        logger.info(f"FakeoutDetector initialized with config: {self.config}")
    
    def calculate_key_levels(self, df: pd.DataFrame, level_type: str = 'pdh_pdl') -> pd.DataFrame:
        """
        Calculate key levels (PDH/PDL, VWAP, etc.) for fakeout detection.
        
        Args:
            df: OHLCV DataFrame with datetime index
            level_type: Type of levels to calculate ('pdh_pdl', 'vwap', 'custom')
            
        Returns:
            DataFrame with calculated levels
        """
        logger.info(f"Calculating {level_type} levels for {len(df)} candles")
        
        df = df.copy()
        
        if level_type == 'pdh_pdl':
            # Previous Day High/Low
            df['date'] = df.index.date
            df['pdh'] = df.groupby('date')['high'].shift(1).rolling(
                window=self.config['level_lookback'], min_periods=1
            ).max()
            df['pdl'] = df.groupby('date')['low'].shift(1).rolling(
                window=self.config['level_lookback'], min_periods=1
            ).min()
            
            logger.debug(f"PDH range: {df['pdh'].min():.2f} - {df['pdh'].max():.2f}")
            logger.debug(f"PDL range: {df['pdl'].min():.2f} - {df['pdl'].max():.2f}")
            
        elif level_type == 'vwap':
            # VWAP as level
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['pdh'] = df['vwap']  # Use VWAP as resistance
            df['pdl'] = df['vwap']  # Use VWAP as support
            
        elif level_type == 'custom':
            # Custom levels - user should add 'pdh' and 'pdl' columns
            if 'pdh' not in df.columns or 'pdl' not in df.columns:
                raise ValueError("Custom level type requires 'pdh' and 'pdl' columns")
        
        # Calculate ATR for SL/TP
        df['atr'] = self._calculate_atr(df, self.config['atr_period'])
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_breakout_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect breakout candles with wicks above resistance or below support.
        
        Args:
            df: DataFrame with OHLCV and level data
            
        Returns:
            DataFrame with breakout signals
        """
        logger.info("Detecting breakout candles...")
        
        df = df.copy()
        
        # Calculate candle properties
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['body'].replace(0, 1)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Detect breakouts
        breakout_signals = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Resistance breakout (above PDH)
            if (current['high'] > current['pdh'] * (1 + self.config['breakout_threshold']) and
                current['wick_ratio'] > self.config['wick_threshold'] and
                current['volume_ratio'] > self.config['volume_spike_threshold']):
                
                logger.debug(f"Resistance breakout detected at {current.name}: "
                           f"High={current['high']:.2f}, PDH={current['pdh']:.2f}, "
                           f"Wick ratio={current['wick_ratio']:.2f}")
                
                breakout_signals.append({
                    'timestamp': current.name,
                    'type': 'resistance_breakout',
                    'level': current['pdh'],
                    'breakout_price': current['high'],
                    'close_price': current['close'],
                    'wick_ratio': current['wick_ratio'],
                    'volume_ratio': current['volume_ratio'],
                    'candle_index': i
                })
            
            # Support breakout (below PDL)
            elif (current['low'] < current['pdl'] * (1 - self.config['breakout_threshold']) and
                  current['wick_ratio'] > self.config['wick_threshold'] and
                  current['volume_ratio'] > self.config['volume_spike_threshold']):
                
                logger.debug(f"Support breakout detected at {current.name}: "
                           f"Low={current['low']:.2f}, PDL={current['pdl']:.2f}, "
                           f"Wick ratio={current['wick_ratio']:.2f}")
                
                breakout_signals.append({
                    'timestamp': current.name,
                    'type': 'support_breakout',
                    'level': current['pdl'],
                    'breakout_price': current['low'],
                    'close_price': current['close'],
                    'wick_ratio': current['wick_ratio'],
                    'volume_ratio': current['volume_ratio'],
                    'candle_index': i
                })
        
        logger.info(f"Detected {len(breakout_signals)} breakout candles")
        return pd.DataFrame(breakout_signals)
    
    def detect_reversal_confirmation(self, df: pd.DataFrame, breakout_signals: pd.DataFrame) -> List[Dict]:
        """
        Detect reversal confirmation after breakout candles.
        
        Args:
            df: OHLCV DataFrame
            breakout_signals: DataFrame with breakout signals
            
        Returns:
            List of confirmed fakeout signals
        """
        logger.info("Detecting reversal confirmations...")
        
        fakeout_signals = []
        
        for _, breakout in breakout_signals.iterrows():
            breakout_idx = breakout['candle_index']
            level = breakout['level']
            breakout_type = breakout['type']
            
            logger.debug(f"Checking confirmation for {breakout_type} at {breakout['timestamp']}")
            
            # Look for confirmation candles
            confirmation_found = False
            confirmation_candles = 0
            
            for i in range(breakout_idx + 1, min(breakout_idx + 1 + self.config['max_confirmation_time'], len(df))):
                current = df.iloc[i]
                
                if breakout_type == 'resistance_breakout':
                    # Check if price closed back below resistance
                    if current['close'] < level * (1 - self.config['confirmation_threshold']):
                        confirmation_candles += 1
                        logger.debug(f"Resistance confirmation candle {confirmation_candles} at {current.name}")
                    else:
                        confirmation_candles = 0  # Reset if not confirmed
                
                elif breakout_type == 'support_breakout':
                    # Check if price closed back above support
                    if current['close'] > level * (1 + self.config['confirmation_threshold']):
                        confirmation_candles += 1
                        logger.debug(f"Support confirmation candle {confirmation_candles} at {current.name}")
                    else:
                        confirmation_candles = 0  # Reset if not confirmed
                
                # Check if we have enough confirmation candles
                if confirmation_candles >= self.config['confirmation_candles']:
                    confirmation_found = True
                    confirmation_idx = i
                    break
            
            if confirmation_found:
                # Generate fakeout signal
                signal = self._generate_fakeout_signal(df, breakout, confirmation_idx, level)
                fakeout_signals.append(signal)
                
                logger.info(f"Fakeout signal confirmed: {signal['signal_type']} at {signal['entry_time']}")
        
        logger.info(f"Generated {len(fakeout_signals)} fakeout signals")
        return fakeout_signals
    
    def _generate_fakeout_signal(self, df: pd.DataFrame, breakout: pd.Series, 
                                confirmation_idx: int, level: float) -> Dict:
        """Generate a complete fakeout signal with entry, SL, TP."""
        
        entry_candle = df.iloc[confirmation_idx]
        atr = entry_candle['atr']
        
        if breakout['type'] == 'resistance_breakout':
            # Short fakeout - price broke above resistance, then reversed
            signal_type = 'short_fakeout'
            entry_price = entry_candle['close']
            sl_price = entry_price + (atr * self.config['sl_multiplier'])
            tp_price = entry_price - (atr * self.config['tp_multiplier'])
            
        else:  # support_breakout
            # Long fakeout - price broke below support, then reversed
            signal_type = 'long_fakeout'
            entry_price = entry_candle['close']
            sl_price = entry_price - (atr * self.config['sl_multiplier'])
            tp_price = entry_price + (atr * self.config['tp_multiplier'])
        
        return {
            'signal_type': signal_type,
            'entry_time': entry_candle.name,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'level': level,
            'breakout_time': breakout['timestamp'],
            'breakout_price': breakout['breakout_price'],
            'atr': atr,
            'volume_ratio': breakout['volume_ratio'],
            'wick_ratio': breakout['wick_ratio'],
            'confirmation_candles': self.config['confirmation_candles'],
            'debug_info': {
                'breakout_candle_idx': breakout['candle_index'],
                'confirmation_candle_idx': confirmation_idx,
                'level_type': 'pdh_pdl'
            }
        }
    
    def detect_fakeout_signals(self, df: pd.DataFrame, level_type: str = 'pdh_pdl') -> Tuple[List[Dict], pd.DataFrame]:
        """
        Main function to detect fakeout reversal signals.
        
        Args:
            df: OHLCV DataFrame with datetime index
            level_type: Type of levels to use ('pdh_pdl', 'vwap', 'custom')
            
        Returns:
            Tuple of (fakeout_signals, breakout_signals_df)
        """
        logger.info(f"Starting fakeout detection for {len(df)} candles with {level_type} levels")
        
        # Calculate key levels
        df_with_levels = self.calculate_key_levels(df, level_type)
        
        # Detect breakout candles
        breakout_signals = self.detect_breakout_candles(df_with_levels)
        
        if breakout_signals.empty:
            logger.warning("No breakout signals detected")
            return [], breakout_signals
        
        # Detect reversal confirmations
        fakeout_signals = self.detect_reversal_confirmation(df_with_levels, breakout_signals)
        
        logger.info(f"Detection complete: {len(fakeout_signals)} fakeout signals found")
        
        return fakeout_signals, breakout_signals
    
    def plot_fakeout_signals(self, df: pd.DataFrame, fakeout_signals: List[Dict], 
                           breakout_signals: pd.DataFrame, title: str = "Fakeout Signals") -> None:
        """
        Plot fakeout signals with price action, levels, and signal markers.
        
        Args:
            df: OHLCV DataFrame
            fakeout_signals: List of fakeout signals
            breakout_signals: DataFrame with breakout signals
            title: Plot title
        """
        if not self.config['plot_signals']:
            return
        
        logger.info(f"Plotting {len(fakeout_signals)} fakeout signals")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, "Volume"),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add VWAP if available
        if 'vwap' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['vwap'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )
        
        # Add PDH/PDL levels
        if 'pdh' in df.columns and 'pdl' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['pdh'],
                    mode='lines',
                    name='PDH',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['pdl'],
                    mode='lines',
                    name='PDL',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add breakout points
        for _, breakout in breakout_signals.iterrows():
            color = 'red' if breakout['type'] == 'resistance_breakout' else 'green'
            fig.add_trace(
                go.Scatter(
                    x=[breakout['timestamp']],
                    y=[breakout['breakout_price']],
                    mode='markers',
                    marker=dict(symbol='triangle-up' if breakout['type'] == 'resistance_breakout' else 'triangle-down',
                              size=10, color=color),
                    name=f"Breakout ({breakout['type']})",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add fakeout signals
        for signal in fakeout_signals:
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[signal['entry_time']],
                    y=[signal['entry_price']],
                    mode='markers',
                    marker=dict(symbol='circle', size=12, 
                              color='blue' if signal['signal_type'] == 'long_fakeout' else 'orange'),
                    name=f"Entry ({signal['signal_type']})",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # SL and TP lines
            fig.add_trace(
                go.Scatter(
                    x=[signal['entry_time'], signal['entry_time']],
                    y=[signal['entry_price'], signal['sl_price']],
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(symbol='x', size=8, color='red'),
                    name=f"SL ({signal['signal_type']})",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[signal['entry_time'], signal['entry_time']],
                    y=[signal['entry_price'], signal['tp_price']],
                    mode='lines+markers',
                    line=dict(color='green', width=2),
                    marker=dict(symbol='triangle-up', size=8, color='green'),
                    name=f"TP ({signal['signal_type']})",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Show plot
        fig.show()
        
        logger.info("Plot displayed successfully")


def detect_fakeout_signals(df: pd.DataFrame, config: Optional[Dict] = None, 
                          level_type: str = 'pdh_pdl', plot: bool = True) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Convenience function to detect fakeout signals with default configuration.
    
    Args:
        df: OHLCV DataFrame with datetime index
        config: Optional configuration dictionary
        level_type: Type of levels to use
        plot: Whether to plot the signals
        
    Returns:
        Tuple of (fakeout_signals, breakout_signals_df)
    """
    detector = FakeoutDetector(config)
    fakeout_signals, breakout_signals = detector.detect_fakeout_signals(df, level_type)
    
    if plot and fakeout_signals:
        detector.plot_fakeout_signals(df, fakeout_signals, breakout_signals)
    
    return fakeout_signals, breakout_signals


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'debug_mode': True,
        'log_level': 'DEBUG',
        'breakout_threshold': 0.03,  # 3% breakout
        'wick_threshold': 0.4,  # 40% wick
        'confirmation_candles': 1,  # 1 confirmation candle
        'sl_multiplier': 1.5,
        'tp_multiplier': 2.0
    }
    
    print("Fakeout Detector initialized with example configuration")
    print("Use detect_fakeout_signals() function to analyze your data") 