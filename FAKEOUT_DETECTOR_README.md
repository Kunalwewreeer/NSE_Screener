# 📈 Fakeout Detection System

A modular and debuggable Python system for detecting intraday fakeout reversals around key levels (PDH/PDL, VWAP, support/resistance) with interactive Streamlit visualization.

## 🎯 Features

### Core Detection Engine
- **Modular Design**: Easily customizable parameters for different market conditions
- **Multiple Level Types**: Support for PDH/PDL, VWAP, and custom support/resistance levels
- **Debug Logging**: Comprehensive logging at key decision points
- **Risk Management**: ATR-based stop loss and take profit calculations
- **Signal Filtering**: Configurable minimum intervals between signals

### Streamlit Interactive Platform
- **Real-time Visualization**: Interactive candlestick charts with signal markers
- **Parameter Tuning**: Sidebar controls for all detection parameters
- **Data Generation**: Built-in sample data generator with customizable patterns
- **Analysis Dashboard**: Signal distribution, risk/reward analysis, and detailed metrics
- **Export Capabilities**: Easy export of signals and analysis results

## 📁 Project Structure

```
├── fakeout_detector.py          # Core detection engine
├── fakeout_streamlit_app.py     # Interactive Streamlit app
├── test_fakeout_detector.py     # Test suite and examples
├── run_fakeout_app.py          # Streamlit app launcher
├── FAKEOUT_DETECTOR_README.md   # This documentation
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas numpy matplotlib
```

### 2. Launch Streamlit App
```bash
python3 run_fakeout_app.py
```
Or directly:
```bash
streamlit run fakeout_streamlit_app.py
```

### 3. Use Core Detector
```python
from fakeout_detector import FakeoutDetector
import pandas as pd

# Initialize detector
detector = FakeoutDetector()

# Detect signals
signals = detector.detect_fakeout_signals(df, vwap_series, 'pdh_pdl')

# Plot results
detector.plot_signals(df, signals, vwap_series, 'pdh_pdl')
```

## ⚙️ Configuration Parameters

### Level Detection
- `wick_threshold_pct`: Minimum wick percentage for breakout candle (0.1-2.0%)
- `confirmation_threshold_pct`: Minimum reversal percentage for confirmation (0.1-2.0%)
- `level_tolerance_pct`: Tolerance around level for breakout detection (0.05-1.0%)

### Signal Parameters
- `lookback_window`: Candles to look back for level calculation (5-50)
- `min_candles_between_signals`: Minimum candles between consecutive signals (1-20)

### Risk Management
- `sl_atr_multiplier`: Stop loss as ATR multiplier (0.5-3.0)
- `tp_atr_multiplier`: Take profit as ATR multiplier (1.0-5.0)
- `atr_period`: Period for ATR calculation (5-30)

## 📊 Detection Logic

### 1. Breakout Detection
- **Resistance Fakeout**: Price breaks above level with wick, closes below level
- **Support Fakeout**: Price breaks below level with wick, closes above level
- **Wick Requirement**: Minimum wick percentage to confirm false breakout

### 2. Confirmation Logic
- **Reversal Candle**: Next candle closes back inside the level
- **Time Window**: Looks for confirmation within 5 candles after breakout
- **Signal Generation**: Entry at confirmation candle close

### 3. Risk Management
- **ATR-based SL/TP**: Uses Average True Range for dynamic levels
- **Configurable Multipliers**: Adjustable risk/reward ratios
- **Signal Filtering**: Prevents signal clustering

## 🎨 Streamlit App Features

### Interactive Controls
- **Data Configuration**: Date range, time range, candle frequency
- **Detection Parameters**: All core parameters with real-time updates
- **Level Type Selection**: PDH/PDL, VWAP, or support/resistance
- **Sample Data Generation**: Customizable volatility and fakeout patterns

### Visualization
- **Interactive Charts**: Candlestick charts with signal markers
- **Volume Analysis**: Volume bars with price action
- **Level Display**: Key levels shown as dashed lines
- **Signal Markers**: Entry (triangles), SL (X), TP (stars)

### Analysis Dashboard
- **Metrics Overview**: Total signals, long/short distribution
- **Signal Table**: Formatted table with color coding
- **Distribution Charts**: Pie chart for signal types
- **Risk/Reward Analysis**: Histogram of risk/reward ratios

## 🔧 Advanced Usage

### Custom Level Types
```python
# Add custom level calculation
def calculate_custom_levels(df, config):
    # Your custom logic here
    return df_with_levels

# Extend detector
detector.calculate_key_levels = calculate_custom_levels
```

### Integration with Real Data
```python
# Load your OHLCV data
df = your_data_handler.get_data(symbol, start_date, end_date, interval='1min')

# Calculate VWAP
vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

# Detect signals
signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
```

### Custom Signal Processing
```python
# Process signals for backtesting
for signal in signals:
    entry_price = signal['entry']
    stop_loss = signal['stop_loss']
    take_profit = signal['take_profit']
    
    # Your backtesting logic here
    # ...
```

## 📈 Example Results

### Sample Output
```
=== FAKEOUT DETECTION SUMMARY ===
Total signals: 8
Long fakeouts: 3
Short fakeouts: 5
First signal: 2024-01-01 10:30:00
Last signal: 2024-01-01 14:45:00
========================================
```

### Signal Format
```python
{
    'timestamp': '2024-01-01 10:30:00',
    'signal_type': 'short_fakeout',
    'entry': 18550.25,
    'stop_loss': 18580.50,
    'take_profit': 18490.00,
    'level_value': 18545.00,
    'breakout_idx': 45,
    'confirmation_idx': 47,
    'level_type': 'pdh_pdl'
}
```

## 🐛 Debugging Features

### Logging Levels
- **INFO**: Basic signal detection and summary
- **DEBUG**: Detailed breakout and confirmation logic
- **WARNING**: Parameter validation and edge cases

### Debug Output
```
2024-01-01 10:30:00 - INFO - Resistance breakout detected at 2024-01-01 10:25:00: High=18555.25, Close=18542.50, Level=18545.00
2024-01-01 10:30:00 - INFO - Resistance fakeout confirmed at 2024-01-01 10:30:00: Close=18540.25, Level=18545.00
2024-01-01 10:30:00 - INFO - Short fakeout signal generated: Entry=18540.25, SL=18570.50, TP=18480.00
```

## 🔄 Integration with Existing Systems

### Data Handler Integration
```python
from core.data_handler import DataHandler

# Initialize data handler
data_handler = DataHandler()

# Get data
df = data_handler.get_historical_data(['NIFTY'], start_date, end_date, interval='1min')

# Detect signals
detector = FakeoutDetector()
signals = detector.detect_fakeout_signals(df, vwap, 'pdh_pdl')
```

### Backtesting Integration
```python
# Use signals in your backtesting framework
for signal in signals:
    # Add to your backtesting engine
    backtester.add_signal(signal)
```

## 📊 Performance Considerations

### Optimization Tips
- **Data Preprocessing**: Ensure clean OHLCV data with proper datetime index
- **Parameter Tuning**: Start with conservative parameters and adjust based on results
- **Signal Filtering**: Use appropriate minimum intervals to avoid signal clustering
- **Memory Management**: For large datasets, consider chunking data processing

### Scalability
- **Batch Processing**: Process multiple symbols in parallel
- **Caching**: Cache level calculations for repeated analysis
- **Streaming**: Real-time signal detection for live trading

## 🛠️ Troubleshooting

### Common Issues

1. **No Signals Detected**
   - Check parameter thresholds (too strict)
   - Verify data quality and format
   - Ensure sufficient price movement

2. **Too Many Signals**
   - Increase `min_candles_between_signals`
   - Adjust `wick_threshold_pct` higher
   - Tighten `level_tolerance_pct`

3. **Poor Signal Quality**
   - Review level calculation logic
   - Adjust confirmation criteria
   - Check for data anomalies

### Debug Mode
```python
# Enable detailed logging
detector = FakeoutDetector({
    'debug_mode': True,
    'log_level': 'DEBUG'
})
```

## 📚 API Reference

### FakeoutDetector Class

#### Methods
- `__init__(config)`: Initialize with configuration
- `detect_fakeout_signals(df, vwap, level_type)`: Main detection function
- `plot_signals(df, signals, vwap, level_type)`: Plot results
- `get_signals_dataframe()`: Convert signals to DataFrame
- `print_debug_summary()`: Print detection summary

#### Configuration Options
See the Configuration Parameters section above for all available options.

## 🤝 Contributing

### Adding New Features
1. Extend the `FakeoutDetector` class
2. Add corresponding Streamlit controls
3. Update documentation
4. Add tests

### Custom Level Types
1. Implement level calculation function
2. Add to `calculate_key_levels` method
3. Update Streamlit level type selector

## 📄 License

This project is part of the baller trading system. Use responsibly and in accordance with your trading strategy and risk management rules.

---

**Happy Trading! 📈💰** 