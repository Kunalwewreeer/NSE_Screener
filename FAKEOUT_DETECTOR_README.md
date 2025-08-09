# 🎯 Fakeout Reversal Detection System

A modular and debuggable Python system for detecting intraday fakeout reversals around key levels (PDH/PDL, VWAP, etc.) with extensive customization and visualization capabilities.

## 🚀 Key Features

### 📊 **Modular Design**
- **Configurable Parameters**: All detection thresholds and rules are easily adjustable
- **Multiple Level Types**: Support for PDH/PDL, VWAP, and custom levels
- **Extensive Debugging**: Comprehensive logging and debug information
- **Risk Management**: ATR-based stop-loss and take-profit calculations

### 🎯 **Detection Logic**
- **Breakout Detection**: Identifies candles breaking above resistance or below support
- **Wick Analysis**: Analyzes candle wicks for fakeout confirmation
- **Volume Confirmation**: Requires volume spikes for signal validation
- **Reversal Confirmation**: Waits for price to move back inside the level

### 📈 **Visualization**
- **Interactive Charts**: Plotly-based candlestick charts with signal markers
- **Level Visualization**: Horizontal lines for key levels (PDH/PDL, VWAP)
- **Signal Markers**: Entry, stop-loss, and take-profit points clearly marked
- **Volume Analysis**: Volume bars with breakout confirmation

## 📁 File Structure

```
strategies/
├── fakeout_detector.py      # Main detection system
├── example_fakeout_usage.py # Usage examples
└── FAKEOUT_DETECTOR_README.md # This file
```

## 🏃‍♂️ Quick Start

### 1. Basic Usage
```python
from strategies.fakeout_detector import detect_fakeout_signals

# Detect fakeout signals with default settings
fakeout_signals, breakout_signals = detect_fakeout_signals(df, plot=True)
```

### 2. Custom Configuration
```python
config = {
    'breakout_threshold': 0.02,  # 2% breakout
    'wick_threshold': 0.4,  # 40% wick
    'confirmation_candles': 2,  # 2 confirmation candles
    'volume_spike_threshold': 2.0,  # Volume spike multiplier
    'sl_multiplier': 1.5,  # SL distance as ATR multiple
    'tp_multiplier': 2.0,  # TP distance as ATR multiple
    'log_level': 'DEBUG'
}

fakeout_signals, breakout_signals = detect_fakeout_signals(
    df, config=config, level_type='pdh_pdl', plot=True
)
```

### 3. Different Level Types
```python
# PDH/PDL levels (default)
fakeout_signals, breakout_signals = detect_fakeout_signals(df, level_type='pdh_pdl')

# VWAP levels
fakeout_signals, breakout_signals = detect_fakeout_signals(df, level_type='vwap')

# Custom levels (requires 'pdh' and 'pdl' columns)
df['pdh'] = your_resistance_levels
df['pdl'] = your_support_levels
fakeout_signals, breakout_signals = detect_fakeout_signals(df, level_type='custom')
```

## ⚙️ Configuration Parameters

### Level Detection
```python
'level_lookback': 20,  # Candles to look back for level calculation
'level_threshold': 0.1,  # % threshold for level significance
```

### Breakout Detection
```python
'breakout_threshold': 0.05,  # % above/below level to consider breakout
'wick_threshold': 0.3,  # Minimum wick % of candle body
'volume_spike_threshold': 1.5,  # Volume spike multiplier
```

### Confirmation Rules
```python
'confirmation_candles': 2,  # Number of candles to confirm reversal
'confirmation_threshold': 0.02,  # % move back inside level
'max_confirmation_time': 10,  # Max minutes to wait for confirmation
```

### Risk Management
```python
'sl_multiplier': 1.5,  # SL distance as multiple of ATR
'tp_multiplier': 2.0,  # TP distance as multiple of ATR
'atr_period': 14,  # ATR calculation period
```

### Debug Settings
```python
'debug_mode': True,  # Enable debug mode
'log_level': 'INFO',  # Logging level (DEBUG, INFO, WARNING, ERROR)
'plot_signals': True,  # Enable signal plotting
```

## 📊 Signal Output Format

### Fakeout Signal Dictionary
```python
{
    'signal_type': 'short_fakeout',  # or 'long_fakeout'
    'entry_time': datetime,  # Entry timestamp
    'entry_price': 100.50,  # Entry price
    'sl_price': 101.20,  # Stop-loss price
    'tp_price': 99.80,  # Take-profit price
    'level': 100.00,  # Key level (PDH/PDL/VWAP)
    'breakout_time': datetime,  # Breakout timestamp
    'breakout_price': 100.80,  # Breakout price
    'atr': 0.50,  # Average True Range
    'volume_ratio': 2.1,  # Volume spike ratio
    'wick_ratio': 0.35,  # Wick to body ratio
    'confirmation_candles': 2,  # Number of confirmation candles
    'debug_info': {
        'breakout_candle_idx': 150,
        'confirmation_candle_idx': 152,
        'level_type': 'pdh_pdl'
    }
}
```

### Breakout Signal DataFrame
```python
# DataFrame with columns:
# - timestamp: Breakout timestamp
# - type: 'resistance_breakout' or 'support_breakout'
# - level: Key level price
# - breakout_price: Price at breakout
# - close_price: Close price of breakout candle
# - wick_ratio: Wick to body ratio
# - volume_ratio: Volume spike ratio
# - candle_index: Index of breakout candle
```

## 🎯 Detection Logic

### 1. **Level Calculation**
- **PDH/PDL**: Previous day high/low with rolling window
- **VWAP**: Volume-weighted average price
- **Custom**: User-defined resistance/support levels

### 2. **Breakout Detection**
- **Price Breakout**: High above resistance or low below support
- **Wick Analysis**: Significant wick relative to candle body
- **Volume Confirmation**: Volume spike above threshold
- **Threshold**: Configurable percentage above/below level

### 3. **Reversal Confirmation**
- **Price Reversal**: Close back inside the level
- **Confirmation Candles**: Multiple candles confirming reversal
- **Time Limit**: Maximum time to wait for confirmation
- **Threshold**: Configurable percentage move back inside level

### 4. **Signal Generation**
- **Entry Point**: Confirmation candle close
- **Stop Loss**: ATR-based distance from entry
- **Take Profit**: ATR-based distance from entry
- **Risk Management**: Configurable SL/TP multipliers

## 📈 Visualization Features

### Interactive Charts
- **Candlestick Chart**: Price action with OHLC data
- **Level Lines**: Horizontal lines for key levels
- **Signal Markers**: Entry, SL, TP points
- **Volume Bars**: Volume analysis with breakout confirmation

### Chart Elements
- **Price Action**: Candlestick chart with datetime index
- **Key Levels**: PDH/PDL or VWAP lines
- **Breakout Points**: Triangle markers for breakouts
- **Entry Points**: Circle markers for signal entries
- **SL/TP Lines**: Vertical lines with markers
- **Volume**: Bar chart with volume analysis

## 🔧 Advanced Usage

### Custom Level Types
```python
# Create custom levels
df['pdh'] = df['high'].rolling(window=20).max()  # Custom resistance
df['pdl'] = df['low'].rolling(window=20).min()   # Custom support

# Use custom levels
fakeout_signals, breakout_signals = detect_fakeout_signals(
    df, level_type='custom'
)
```

### Conservative Configuration
```python
conservative_config = {
    'breakout_threshold': 0.02,  # 2% breakout
    'wick_threshold': 0.4,  # 40% wick
    'confirmation_candles': 2,  # 2 confirmation candles
    'volume_spike_threshold': 2.0,  # Higher volume requirement
    'sl_multiplier': 2.0,  # Wider SL
    'tp_multiplier': 3.0,  # Higher TP
}
```

### Aggressive Configuration
```python
aggressive_config = {
    'breakout_threshold': 0.005,  # 0.5% breakout
    'wick_threshold': 0.1,  # 10% wick
    'confirmation_candles': 1,  # 1 confirmation candle
    'volume_spike_threshold': 1.2,  # Lower volume requirement
    'sl_multiplier': 1.0,  # Tighter SL
    'tp_multiplier': 1.5,  # Lower TP
}
```

## 🐛 Debugging and Logging

### Log Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about detection process
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures

### Debug Information
```python
# Enable debug logging
config = {'log_level': 'DEBUG'}

# Debug output includes:
# - Level calculation details
# - Breakout detection criteria
# - Confirmation process
# - Signal generation details
```

### Debug Output Example
```
2024-01-01 10:30:00 - INFO - Calculating pdh_pdl levels for 390 candles
2024-01-01 10:30:00 - DEBUG - PDH range: 100.50 - 101.20
2024-01-01 10:30:00 - DEBUG - PDL range: 99.80 - 100.10
2024-01-01 10:30:00 - INFO - Detecting breakout candles...
2024-01-01 10:30:00 - DEBUG - Resistance breakout detected at 2024-01-01 10:45:00
2024-01-01 10:30:00 - INFO - Detecting reversal confirmations...
2024-01-01 10:30:00 - INFO - Fakeout signal confirmed: short_fakeout at 2024-01-01 10:47:00
```

## 📋 Example Usage

### Basic Example
```python
import pandas as pd
from strategies.fakeout_detector import detect_fakeout_signals

# Load your OHLCV data
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Detect fakeout signals
fakeout_signals, breakout_signals = detect_fakeout_signals(df, plot=True)

# Process signals
for signal in fakeout_signals:
    print(f"Signal: {signal['signal_type']}")
    print(f"Entry: {signal['entry_price']:.2f}")
    print(f"SL: {signal['sl_price']:.2f}")
    print(f"TP: {signal['tp_price']:.2f}")
```

### Advanced Example
```python
# Custom configuration for intraday trading
config = {
    'breakout_threshold': 0.01,  # 1% breakout
    'wick_threshold': 0.3,  # 30% wick
    'confirmation_candles': 1,  # Quick confirmation
    'volume_spike_threshold': 1.5,  # Moderate volume
    'sl_multiplier': 1.5,  # 1.5x ATR SL
    'tp_multiplier': 2.0,  # 2x ATR TP
    'log_level': 'INFO',
    'plot_signals': True
}

# Detect signals with VWAP levels
fakeout_signals, breakout_signals = detect_fakeout_signals(
    df, config=config, level_type='vwap', plot=True
)
```

## 🎯 Best Practices

### For Live Trading
1. **Use Conservative Settings**: Higher thresholds for reliability
2. **Monitor Volume**: Ensure volume confirmation for signals
3. **Check Timeframes**: Use appropriate timeframe for your strategy
4. **Validate Levels**: Ensure key levels are significant
5. **Risk Management**: Always use proper SL/TP

### For Backtesting
1. **Test Multiple Configurations**: Find optimal parameters
2. **Analyze Signal Quality**: Check win rate and profit factor
3. **Monitor Drawdown**: Ensure reasonable risk levels
4. **Validate Assumptions**: Test on different market conditions

### For Development
1. **Enable Debug Logging**: Use DEBUG level for development
2. **Test with Sample Data**: Use provided example scripts
3. **Validate Output**: Check signal format and accuracy
4. **Customize Parameters**: Adjust for your specific needs

## 🔍 Troubleshooting

### Common Issues
- **No Signals Detected**: Lower thresholds or check data quality
- **Too Many Signals**: Increase thresholds or add filters
- **Poor Signal Quality**: Adjust confirmation rules or volume requirements
- **Plotting Issues**: Ensure Plotly is installed and data is valid

### Debug Steps
1. **Check Data Format**: Ensure OHLCV columns and datetime index
2. **Verify Levels**: Check if key levels are calculated correctly
3. **Review Logs**: Use DEBUG level for detailed information
4. **Test Parameters**: Try different configuration settings

## 📚 Dependencies

### Required Packages
```bash
pip install pandas numpy plotly
```

### Optional Packages
```bash
pip install matplotlib  # For additional plotting options
```

## 🚀 Performance Tips

### Optimization
- **Use Appropriate Timeframes**: 1m/5m for intraday, 15m/1h for swing
- **Limit Data Size**: Process reasonable amounts of data
- **Cache Results**: Store calculated levels for reuse
- **Parallel Processing**: Process multiple symbols in parallel

### Memory Management
- **Clean Data**: Remove unnecessary columns and rows
- **Use Efficient Data Types**: Optimize DataFrame dtypes
- **Limit History**: Use appropriate lookback periods
- **Garbage Collection**: Clear unused variables

---

**🎯 Ready for Production!**

The Fakeout Detector provides a robust, modular, and debuggable system for detecting intraday fakeout reversals with extensive customization options and comprehensive visualization capabilities. 