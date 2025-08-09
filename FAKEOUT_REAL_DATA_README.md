# 🎯 Fakeout Detector - Real Data Integration

A complete integration of the fakeout detection system with your existing data handler for real market data analysis.

## 🚀 Quick Start

### 1. Launch the Real Data App
```bash
python3 run_fakeout_real_data.py
```
This will start the Streamlit app at `http://localhost:8502`

### 2. Use the Integration Directly
```python
from fakeout_detector_integration import run_fakeout_analysis

# Analyze multiple symbols
results = run_fakeout_analysis(
    symbols=['NIFTY', 'BANKNIFTY'],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    level_type='pdh_pdl',
    interval='minute'
)
```

## 📁 Integration Components

### Core Files
- **`fakeout_detector_integration.py`** - Main integration class
- **`fakeout_real_data_app.py`** - Streamlit app for real data
- **`run_fakeout_real_data.py`** - App launcher
- **`test_fakeout_integration.py`** - Integration tests

### Integration Features
✅ **Real Data Connection** - Uses your existing `DataHandler`
✅ **Multiple Symbols** - Analyze multiple symbols simultaneously
✅ **Configurable Parameters** - All detection parameters adjustable
✅ **Interactive Dashboard** - Streamlit app with real-time analysis
✅ **Signal Filtering** - Get top signals across all symbols
✅ **Risk Management** - ATR-based stop loss and take profit
✅ **Debug Logging** - Comprehensive logging for troubleshooting

## 🎯 Key Features

### Real Data Integration
- **Seamless Connection**: Works with your existing `DataHandler`
- **Multiple Intervals**: Support for minute, 5minute, 15minute data
- **Date Range Selection**: Flexible date and time range selection
- **Symbol Management**: Analyze multiple symbols in one run

### Advanced Analysis
- **Signal Ranking**: Get top signals across all symbols
- **Performance Metrics**: Risk/reward ratios, signal distribution
- **Level Types**: PDH/PDL, VWAP, support/resistance levels
- **Customizable Parameters**: All detection thresholds adjustable

### Interactive Dashboard
- **Real-time Analysis**: Live parameter tuning and analysis
- **Visual Charts**: Interactive candlestick charts with signals
- **Signal Tables**: Formatted tables with color coding
- **Metrics Dashboard**: Summary statistics and performance metrics

## 📊 Usage Examples

### 1. Basic Analysis
```python
from fakeout_detector_integration import run_fakeout_analysis
from datetime import datetime, timedelta

# Simple analysis
results = run_fakeout_analysis(
    symbols=['NIFTY'],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    level_type='pdh_pdl'
)

print(f"Found {results['summary']['total_signals']} signals")
```

### 2. Advanced Configuration
```python
config = {
    'wick_threshold_pct': 0.3,
    'confirmation_threshold_pct': 0.5,
    'level_tolerance_pct': 0.1,
    'lookback_window': 20,
    'min_candles_between_signals': 10,
    'sl_atr_multiplier': 1.5,
    'tp_atr_multiplier': 2.0,
    'atr_period': 14,
    'debug_mode': True,
    'log_level': 'INFO'
}

results = run_fakeout_analysis(
    symbols=['NIFTY', 'BANKNIFTY'],
    start_date=datetime.now() - timedelta(days=5),
    end_date=datetime.now(),
    level_type='vwap',
    interval='minute',
    config=config
)
```

### 3. Direct Integration Class Usage
```python
from fakeout_detector_integration import FakeoutDetectorIntegration

# Initialize integration
integration = FakeoutDetectorIntegration()

# Analyze single symbol
result = integration.analyze_symbol(
    'NIFTY',
    start_date=datetime.now() - timedelta(days=1),
    end_date=datetime.now(),
    level_type='pdh_pdl',
    interval='minute'
)

# Get top signals
top_signals = integration.get_top_signals(
    {'NIFTY': result}, 
    top_n=5
)
```

## 🎨 Streamlit App Features

### Interactive Controls
- **Symbol Input**: Enter multiple symbols (one per line)
- **Date Range**: Flexible date and time selection
- **Interval Selection**: Choose data interval (minute, 5minute, etc.)
- **Level Type**: Select detection level type
- **Parameter Tuning**: Real-time parameter adjustment

### Analysis Dashboard
- **Summary Metrics**: Total signals, long/short distribution
- **Top Signals Table**: Recent signals with details
- **Symbol Details**: Individual symbol analysis
- **Interactive Charts**: Signal distribution and risk/reward analysis

### Real-time Features
- **Live Analysis**: Run analysis with current parameters
- **Parameter Validation**: Real-time parameter checking
- **Error Handling**: Graceful error handling and user feedback
- **Progress Tracking**: Analysis progress indicators

## 📈 Output Format

### Analysis Results
```python
{
    'summary': {
        'total_symbols': 2,
        'symbols_with_signals': 2,
        'total_signals': 15,
        'total_long_signals': 8,
        'total_short_signals': 7,
        'avg_signals_per_symbol': 7.5,
        'long_short_ratio': 1.14
    },
    'analysis_results': {
        'NIFTY': {
            'symbol': 'NIFTY',
            'signals': [...],
            'data_points': 1500,
            'total_signals': 8,
            'long_signals': 5,
            'short_signals': 3,
            'avg_risk_reward': 2.1
        },
        'BANKNIFTY': {
            'symbol': 'BANKNIFTY',
            'signals': [...],
            'data_points': 1500,
            'total_signals': 7,
            'long_signals': 3,
            'short_signals': 4,
            'avg_risk_reward': 1.8
        }
    },
    'top_signals': [
        {
            'symbol': 'NIFTY',
            'timestamp': '2024-01-01 10:30:00',
            'signal_type': 'short_fakeout',
            'entry': 18550.25,
            'stop_loss': 18580.50,
            'take_profit': 18490.00,
            'level_value': 18545.00
        },
        ...
    ]
}
```

## 🔧 Configuration Options

### Detection Parameters
```python
config = {
    # Level detection
    'wick_threshold_pct': 0.3,        # Minimum wick percentage
    'confirmation_threshold_pct': 0.5, # Confirmation threshold
    'level_tolerance_pct': 0.1,       # Level tolerance
    
    # Signal parameters
    'lookback_window': 20,             # Level calculation window
    'min_candles_between_signals': 10, # Signal filtering
    
    # Risk management
    'sl_atr_multiplier': 1.5,         # Stop loss multiplier
    'tp_atr_multiplier': 2.0,         # Take profit multiplier
    'atr_period': 14,                 # ATR calculation period
    
    # Debug settings
    'debug_mode': True,                # Enable debug mode
    'log_level': 'INFO'               # Logging level
}
```

### Level Types
- **`pdh_pdl`**: Previous day high/low levels
- **`vwap`**: Volume-weighted average price levels
- **`support_resistance`**: Rolling support/resistance levels

### Data Intervals
- **`minute`**: 1-minute candles
- **`5minute`**: 5-minute candles
- **`15minute`**: 15-minute candles
- **`30minute`**: 30-minute candles

## 🐛 Troubleshooting

### Common Issues

1. **No Data Available**
   ```
   Error: Instrument token not found for symbol
   ```
   - **Solution**: Check if symbols exist in your data handler
   - **Check**: Verify symbol names match your system

2. **No Signals Detected**
   - **Check**: Parameter thresholds may be too strict
   - **Adjust**: Lower `wick_threshold_pct` or `confirmation_threshold_pct`
   - **Verify**: Ensure sufficient price movement in data

3. **Too Many Signals**
   - **Adjust**: Increase `min_candles_between_signals`
   - **Tighten**: Increase `wick_threshold_pct`
   - **Filter**: Use more conservative parameters

4. **Streamlit App Issues**
   - **Check**: Ensure all dependencies are installed
   - **Verify**: Port 8502 is available
   - **Restart**: Kill existing Streamlit processes

### Debug Mode
```python
# Enable detailed logging
config = {
    'debug_mode': True,
    'log_level': 'DEBUG'
}

# Check logs for detailed information
```

## 📊 Performance Tips

### Optimization
- **Limit Date Range**: Use reasonable date ranges (1-7 days for minute data)
- **Symbol Selection**: Start with 1-2 symbols for testing
- **Parameter Tuning**: Use conservative parameters initially
- **Caching**: Results are cached for repeated analysis

### Memory Management
- **Data Size**: Large date ranges may consume significant memory
- **Symbol Count**: Limit to 5-10 symbols for optimal performance
- **Interval Selection**: Higher intervals (15min, 30min) use less memory

## 🔄 Integration with Existing Systems

### Backtesting Integration
```python
# Use signals in your backtesting framework
for signal in results['top_signals']:
    backtester.add_signal({
        'symbol': signal['symbol'],
        'entry_time': signal['timestamp'],
        'entry_price': signal['entry'],
        'stop_loss': signal['stop_loss'],
        'take_profit': signal['take_profit'],
        'signal_type': signal['signal_type']
    })
```

### Live Trading Integration
```python
# Monitor for new signals
def check_for_signals():
    results = run_fakeout_analysis(
        symbols=['NIFTY'],
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now(),
        level_type='pdh_pdl'
    )
    
    if results['summary']['total_signals'] > 0:
        # Process new signals
        process_signals(results['top_signals'])
```

## 📚 API Reference

### FakeoutDetectorIntegration Class

#### Methods
- `__init__(data_handler)`: Initialize with data handler
- `setup_detector(config)`: Configure the detector
- `analyze_symbol(symbol, start_date, end_date, ...)`: Analyze single symbol
- `analyze_multiple_symbols(symbols, start_date, end_date, ...)`: Analyze multiple symbols
- `get_top_signals(analysis_results, top_n, signal_type)`: Get top signals
- `create_analysis_summary(analysis_results)`: Create summary

### run_fakeout_analysis Function
```python
def run_fakeout_analysis(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    level_type: str = 'pdh_pdl',
    interval: str = 'minute',
    config: Optional[Dict] = None
) -> Dict
```

## 🚀 Deployment

### Production Setup
1. **Install Dependencies**: Ensure all required packages are installed
2. **Configure Data Handler**: Set up your data handler with proper credentials
3. **Test Integration**: Run `test_fakeout_integration.py` to verify setup
4. **Launch App**: Use `run_fakeout_real_data.py` to start the app

### Environment Variables
```bash
# Set your data handler credentials
export KITE_API_KEY="your_api_key"
export KITE_API_SECRET="your_api_secret"
export KITE_ACCESS_TOKEN="your_access_token"
```

## 📄 Dependencies

### Required Packages
```bash
pip install streamlit plotly pandas numpy
```

### Optional Packages
```bash
pip install matplotlib  # For additional plotting
```

---

**🎯 Ready for Real Data Analysis!**

The fakeout detector is now fully integrated with your existing data handler system, providing a powerful platform for detecting intraday fakeout reversals using real market data with comprehensive analysis and visualization capabilities. 