"""
Modular Fakeout Detection System
Detects intraday fakeout reversals around key levels with customizable parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeoutDetector:
    """
    Modular fakeout detection system for intraday trading.
    
    Detects fakeout reversals around key levels with customizable parameters.
    Supports various level types: PDH/PDL, VWAP, support/resistance, etc.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fakeout detector with configuration.
        
        Args:
            config: Dictionary with detection parameters
        """
        # Default configuration
        self.config = {
            # Level detection
            'wick_threshold_pct': 0.3,  # Minimum wick percentage for breakout candle
            'confirmation_threshold_pct': 0.5,  # Minimum reversal percentage for confirmation
            'level_tolerance_pct': 0.1,  # Tolerance around level for breakout detection
            
            # Signal parameters
            'lookback_window': 20,  # Candles to look back for level calculation
            'min_candles_between_signals': 10,  # Minimum candles between signals
            
            # Risk management
            'sl_atr_multiplier': 1.5,  # Stop loss as ATR multiplier
            'tp_atr_multiplier': 2.0,  # Take profit as ATR multiplier
            'atr_period': 14,  # ATR calculation period
            
            # Debug settings
            'debug_mode': True,
            'log_level': 'INFO'
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, self.config['log_level']))
        
        # Initialize results storage
        self.signals = []
        self.debug_info = []
    
    def calculate_key_levels(self, df: pd.DataFrame, level_type: str = 'pdh_pdl') -> pd.DataFrame:
        """
        Calculate key levels based on the specified type.
        
        Args:
            df: OHLCV DataFrame with datetime index
            level_type: Type of levels to calculate ('pdh_pdl', 'vwap', 'custom')
            
        Returns:
            DataFrame with additional level columns
        """
        data = df.copy()
        
        if level_type == 'pdh_pdl':
            # Previous Day High/Low - improved calculation for single-day data
            data['date_only'] = data.index.date
            
            # For single-day data, use rolling windows with shift to exclude current candle
            if len(data['date_only'].unique()) == 1:
                # Single day - use rolling windows with shift
                data['pdh'] = data['high'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).max()
                data['pdl'] = data['low'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).min()
            else:
                # Multiple days - use groupby with rolling and shift
                data['pdh'] = data.groupby('date_only')['high'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).max().reset_index(0, drop=True)
                data['pdl'] = data.groupby('date_only')['low'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).min().reset_index(0, drop=True)
            
            # Fill any remaining NaN values with forward fill, then backward fill
            data['pdh'] = data['pdh'].fillna(method='ffill').fillna(method='bfill')
            data['pdl'] = data['pdl'].fillna(method='ffill').fillna(method='bfill')
            
            data = data.drop('date_only', axis=1)
            
        elif level_type == 'vwap':
            # VWAP levels
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            data['vwap_upper'] = data['vwap'] * (1 + self.config['level_tolerance_pct'] / 100)
            data['vwap_lower'] = data['vwap'] * (1 - self.config['level_tolerance_pct'] / 100)
            
        elif level_type == 'support_resistance':
            # Simple support/resistance using rolling high/low with shift
            data['resistance'] = data['high'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).max()
            data['support'] = data['low'].shift(1).rolling(window=self.config['lookback_window'], min_periods=1).min()
        
        return data
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range for stop loss and take profit."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config['atr_period']).mean()
        
        return atr
    
    def detect_breakout_candle(self, df: pd.DataFrame, level: pd.Series, level_type: str) -> pd.Series:
        """
        Detect breakout candles that penetrate the level with wicks.
        
        Args:
            df: OHLCV DataFrame
            level: Series with level values
            level_type: 'resistance' or 'support'
            
        Returns:
            Boolean series indicating breakout candles
        """
        breakout = pd.Series(False, index=df.index)
        
        for i in range(1, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            level_value = level.iloc[i]
            
            if pd.isna(level_value):
                continue
                
            if level_type == 'resistance':
                # Breakout above resistance - more lenient criteria
                if (candle['high'] > level_value and 
                    candle['close'] < level_value and  # Close below level (wick)
                    (candle['high'] - candle['close']) > (candle['high'] * self.config['wick_threshold_pct'] / 100)):
                    
                    breakout.iloc[i] = True
                    if self.config['debug_mode']:
                        logger.info(f"Resistance breakout detected at {candle.name}: "
                                  f"High={candle['high']:.2f}, Close={candle['close']:.2f}, "
                                  f"Level={level_value:.2f}")
            
            elif level_type == 'support':
                # Breakout below support - improved detection
                # Check if low breaks below support and close is above support (wick)
                if (candle['low'] < level_value and 
                    candle['close'] > level_value and  # Close above level (wick)
                    (candle['close'] - candle['low']) > (candle['close'] * self.config['wick_threshold_pct'] / 100)):
                    
                    # Additional check: ensure the wick is significant
                    wick_size = candle['close'] - candle['low']
                    body_size = abs(candle['close'] - candle['open'])
                    
                    if wick_size > body_size * 0.3:  # Wick should be at least 30% of body
                        breakout.iloc[i] = True
                        if self.config['debug_mode']:
                            logger.info(f"Support breakout detected at {candle.name}: "
                                      f"Low={candle['low']:.2f}, Close={candle['close']:.2f}, "
                                      f"Level={level_value:.2f}, Wick={wick_size:.2f}, Body={body_size:.2f}")
        
        return breakout
    
    def detect_reversal_confirmation(self, df: pd.DataFrame, breakout_idx: int, level: pd.Series, level_type: str) -> Optional[int]:
        """
        Detect reversal confirmation after a breakout.
        
        Args:
            df: OHLCV DataFrame
            breakout_idx: Index of breakout candle
            level: Series with level values
            level_type: 'resistance' or 'support'
            
        Returns:
            Index of confirmation candle or None
        """
        if breakout_idx >= len(df) - 1:
            return None
        
        # Look for confirmation in next few candles
        for i in range(breakout_idx + 1, min(breakout_idx + 5, len(df))):
            candle = df.iloc[i]
            level_value = level.iloc[i]
            
            if level_type == 'resistance':
                # Confirmation: close back below resistance
                if candle['close'] < level_value:
                    if self.config['debug_mode']:
                        logger.info(f"Resistance fakeout confirmed at {candle.name}: "
                                  f"Close={candle['close']:.2f}, Level={level_value:.2f}")
                    return i
            
            elif level_type == 'support':
                # Confirmation: close back above support
                if candle['close'] > level_value:
                    if self.config['debug_mode']:
                        logger.info(f"Support fakeout confirmed at {candle.name}: "
                                  f"Close={candle['close']:.2f}, Level={level_value:.2f}")
                    return i
        
        return None
    
    def calculate_entry_sl_tp(self, df: pd.DataFrame, entry_idx: int, signal_type: str) -> Tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            df: OHLCV DataFrame
            entry_idx: Index of entry candle
            signal_type: 'long_fakeout' or 'short_fakeout'
            
        Returns:
            Tuple of (entry, stop_loss, take_profit)
        """
        atr = self.calculate_atr(df)
        entry_candle = df.iloc[entry_idx]
        atr_value = atr.iloc[entry_idx]
        
        if signal_type == 'long_fakeout':
            entry = entry_candle['close']
            stop_loss = entry - (atr_value * self.config['sl_atr_multiplier'])
            take_profit = entry + (atr_value * self.config['tp_atr_multiplier'])
        else:  # short_fakeout
            entry = entry_candle['close']
            stop_loss = entry + (atr_value * self.config['sl_atr_multiplier'])
            take_profit = entry - (atr_value * self.config['tp_atr_multiplier'])
        
        return entry, stop_loss, take_profit
    
    def detect_fakeout_signals(self, df: pd.DataFrame, vwap_series: Optional[pd.Series] = None, 
                              level_type: str = 'pdh_pdl') -> List[Dict]:
        """
        Main function to detect fakeout signals.
        
        Args:
            df: OHLCV DataFrame with datetime index
            vwap_series: Optional VWAP series
            level_type: Type of levels to use ('pdh_pdl', 'vwap', 'support_resistance')
            
        Returns:
            List of signal dictionaries
        """
        logger.info(f"Starting fakeout detection with level_type: {level_type}")
        
        # Calculate key levels
        df_with_levels = self.calculate_key_levels(df, level_type)
        
        # Add VWAP if provided
        if vwap_series is not None:
            df_with_levels['vwap'] = vwap_series
        
        # Calculate ATR
        df_with_levels['atr'] = self.calculate_atr(df_with_levels)
        
        signals = []
        last_signal_idx = -self.config['min_candles_between_signals']
        
        # Detect resistance fakeouts (short signals)
        if level_type == 'pdh_pdl':
            resistance_level = df_with_levels['pdh']
        elif level_type == 'vwap':
            resistance_level = df_with_levels['vwap_upper']
        elif level_type == 'support_resistance':
            resistance_level = df_with_levels['resistance']
        else:
            resistance_level = None
        
        if resistance_level is not None:
            resistance_breakouts = self.detect_breakout_candle(df_with_levels, resistance_level, 'resistance')
            
            for i in range(len(df_with_levels)):
                if resistance_breakouts.iloc[i] and i > last_signal_idx + self.config['min_candles_between_signals']:
                    confirmation_idx = self.detect_reversal_confirmation(df_with_levels, i, resistance_level, 'resistance')
                    
                    if confirmation_idx is not None:
                        entry, sl, tp = self.calculate_entry_sl_tp(df_with_levels, confirmation_idx, 'short_fakeout')
                        
                        signal = {
                            'timestamp': df_with_levels.index[confirmation_idx],
                            'signal_type': 'short_fakeout',
                            'entry': entry,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'level_value': resistance_level.iloc[i],
                            'breakout_idx': i,
                            'confirmation_idx': confirmation_idx,
                            'level_type': level_type
                        }
                        
                        signals.append(signal)
                        last_signal_idx = confirmation_idx
                        
                        logger.info(f"Short fakeout signal generated: Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
        
        # Detect support fakeouts (long signals) - FIXED LOGIC
        if level_type == 'pdh_pdl':
            support_level = df_with_levels['pdl']
        elif level_type == 'vwap':
            support_level = df_with_levels['vwap_lower']
        elif level_type == 'support_resistance':
            support_level = df_with_levels['support']
        else:
            support_level = None
        
        if support_level is not None:
            support_breakouts = self.detect_breakout_candle(df_with_levels, support_level, 'support')
            
            for i in range(len(df_with_levels)):
                if support_breakouts.iloc[i] and i > last_signal_idx + self.config['min_candles_between_signals']:
                    confirmation_idx = self.detect_reversal_confirmation(df_with_levels, i, support_level, 'support')
                    
                    if confirmation_idx is not None:
                        entry, sl, tp = self.calculate_entry_sl_tp(df_with_levels, confirmation_idx, 'long_fakeout')
                        
                        signal = {
                            'timestamp': df_with_levels.index[confirmation_idx],
                            'signal_type': 'long_fakeout',
                            'entry': entry,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'level_value': support_level.iloc[i],
                            'breakout_idx': i,
                            'confirmation_idx': confirmation_idx,
                            'level_type': level_type
                        }
                        
                        signals.append(signal)
                        last_signal_idx = confirmation_idx
                        
                        logger.info(f"Long fakeout signal generated: Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
        
        self.signals = signals
        logger.info(f"Detection complete. Found {len(signals)} signals.")
        
        return signals
    
    def plot_signals(self, df: pd.DataFrame, signals: List[Dict], 
                     vwap_series: Optional[pd.Series] = None, 
                     level_type: str = 'pdh_pdl') -> Optional[go.Figure]:
        """
        Plot the price action with signals and levels.
        
        Args:
            df: OHLCV DataFrame
            signals: List of signal dictionaries
            vwap_series: Optional VWAP series
            level_type: Type of levels used
            
        Returns:
            Plotly figure object or None if plotting fails
        """
        try:
            # Calculate levels for plotting
            df_with_levels = self.calculate_key_levels(df, level_type)
            
            # Create subplot with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action & Signals', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#2ca02c',
                decreasing_line_color='#d62728'
            ), row=1, col=1)
            
            # Add VWAP if provided
            if vwap_series is not None:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=vwap_series,
                    mode='lines',
                    name='VWAP',
                    line=dict(color='purple', width=2),
                    opacity=0.8
                ), row=1, col=1)
            
            # Add level lines
            if level_type == 'pdh_pdl':
                if 'pdh' in df_with_levels.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df_with_levels['pdh'],
                        mode='lines',
                        name='PDH',
                        line=dict(color='red', width=1, dash='dash'),
                        opacity=0.6
                    ), row=1, col=1)
                if 'pdl' in df_with_levels.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df_with_levels['pdl'],
                        mode='lines',
                        name='PDL',
                        line=dict(color='green', width=1, dash='dash'),
                        opacity=0.6
                    ), row=1, col=1)
            
            # Add signal markers
            for signal in signals:
                color = '#2ca02c' if signal['signal_type'] == 'long_fakeout' else '#d62728'
                symbol = 'triangle-up' if signal['signal_type'] == 'long_fakeout' else 'triangle-down'
                
                # Entry point
                fig.add_trace(go.Scatter(
                    x=[signal['timestamp']],
                    y=[signal['entry']],
                    mode='markers',
                    marker=dict(symbol=symbol, size=15, color=color, line=dict(width=2, color='white')),
                    name=f"{signal['signal_type']} Entry",
                    showlegend=False
                ), row=1, col=1)
                
                # Stop loss
                fig.add_trace(go.Scatter(
                    x=[signal['timestamp']],
                    y=[signal['stop_loss']],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='red', line=dict(width=1)),
                    name='Stop Loss',
                    showlegend=False
                ), row=1, col=1)
                
                # Take profit
                fig.add_trace(go.Scatter(
                    x=[signal['timestamp']],
                    y=[signal['take_profit']],
                    mode='markers',
                    marker=dict(symbol='star', size=10, color='green', line=dict(width=1)),
                    name='Take Profit',
                    showlegend=False
                ), row=1, col=1)
            
            # Add volume bars
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.3)'
            ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'Fakeout Signals - {level_type.upper()}',
                xaxis_title='Time',
                yaxis_title='Price',
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return None
    
    def get_signals_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame for easy analysis."""
        if not self.signals:
            return pd.DataFrame()
        
        return pd.DataFrame(self.signals)
    
    def print_debug_summary(self):
        """Print debug summary of detected signals."""
        if not self.signals:
            logger.info("No signals detected.")
            return
        
        logger.info(f"\n=== FAKEOUT DETECTION SUMMARY ===")
        logger.info(f"Total signals: {len(self.signals)}")
        
        long_signals = [s for s in self.signals if s['signal_type'] == 'long_fakeout']
        short_signals = [s for s in self.signals if s['signal_type'] == 'short_fakeout']
        
        logger.info(f"Long fakeouts: {len(long_signals)}")
        logger.info(f"Short fakeouts: {len(short_signals)}")
        
        if self.signals:
            logger.info(f"First signal: {self.signals[0]['timestamp']}")
            logger.info(f"Last signal: {self.signals[-1]['timestamp']}")
        
        logger.info("=" * 40)


def create_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01 09:15:00', '2024-01-01 15:30:00', freq='5min')
    
    # Generate sample price data with some fakeout patterns
    np.random.seed(42)
    base_price = 100
    prices = []
    
    for i in range(len(dates)):
        if i == 0:
            price = base_price
        else:
            price = prices[-1] + np.random.normal(0, 0.5)
        
        # Add some fakeout patterns
        if i % 20 == 10:  # Every 20th candle, create a fakeout
            if np.random.choice([True, False]):
                # Resistance fakeout
                price = price + 2 + np.random.normal(0, 0.3)
            else:
                # Support fakeout
                price = price - 2 + np.random.normal(0, 0.3)
        
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        open_price = price
        high_price = price + abs(np.random.normal(0, 0.3))
        low_price = price - abs(np.random.normal(0, 0.3))
        close_price = price + np.random.normal(0, 0.2)
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate VWAP
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df, vwap


def main():
    """Example usage of the fakeout detector."""
    # Create sample data
    df, vwap = create_sample_data()
    
    # Initialize detector with custom config
    config = {
        'wick_threshold_pct': 0.3,
        'confirmation_threshold_pct': 0.5,
        'level_tolerance_pct': 0.1,
        'lookback_window': 20,
        'min_candles_between_signals': 5,
        'sl_atr_multiplier': 1.5,
        'tp_atr_multiplier': 2.0,
        'atr_period': 14,
        'debug_mode': True,
        'log_level': 'INFO'
    }
    
    detector = FakeoutDetector(config)
    
    # Detect signals
    signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
    
    # Print summary
    detector.print_debug_summary()
    
    # Plot signals
    plot_fig = detector.plot_signals(df, signals, vwap, 'pdh_pdl')
    if plot_fig:
        plot_fig.show()
    
    # Get signals as DataFrame
    signals_df = detector.get_signals_dataframe()
    if not signals_df.empty:
        print("\nSignals DataFrame:")
        print(signals_df)


if __name__ == "__main__":
    main() 