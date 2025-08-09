# 🔍 VWAP Reversal Strategy Backtester

A comprehensive backtesting system for VWAP (Volume Weighted Average Price) reversal strategies with detailed analysis and visualization.

## 🎯 Strategy Overview

The VWAP Reversal Strategy is designed to capitalize on mean reversion opportunities in stock prices relative to their VWAP:

### Core Logic
1. **Daily Stock Selection**: Select top 5 stocks with strongest VWAP reversal signals
2. **Entry Logic**: 
   - **LONG**: Stocks trading below VWAP (negative VWAP distance)
   - **SHORT**: Stocks trading above VWAP (positive VWAP distance)
3. **Exit Logic**: Exit when price converges to VWAP (within 0.5%)
4. **Risk Management**: Stop loss if price moves further away from VWAP (2% threshold)

### VWAP Reversal Signal Calculation
The system calculates reversal signals based on:
- **Current VWAP distance magnitude**
- **Recent VWAP crosses** (price crossing above/below VWAP)
- **VWAP distance change** (acceleration in reversal)

## 📁 Files Overview

### Core Files
- `simple_vwap_backtester.py` - Main backtesting engine
- `run_vwap_backtest.py` - Command-line runner script
- `vwap_reversal_backtester.py` - Advanced version with comprehensive features

### Key Features
- **1-year data download** for comprehensive backtesting
- **Daily stock selection** based on VWAP reversal strength
- **Convergence-based exits** when price returns to VWAP
- **Performance metrics** including Sharpe ratio, drawdown, win rate
- **Interactive visualizations** with Plotly charts
- **Trade analysis** with detailed PnL tracking

## 🚀 Quick Start

### Option 1: Command Line Runner
```bash
# Run with default parameters (Nifty50, 1 year, top 5 stocks)
python run_vwap_backtest.py

# Choose option 1 for quick backtest or option 2 for custom parameters
```

### Option 2: Streamlit Dashboard
```bash
# Run the interactive dashboard
streamlit run simple_vwap_backtester.py
```

### Option 3: Direct Python Execution
```python
from simple_vwap_backtester import SimpleVWAPBacktester
from core.data_handler import DataHandler

# Initialize
backtester = SimpleVWAPBacktester()
data_handler = DataHandler()

# Get universe
universe = data_handler.get_stocks_by_universe("nifty50")

# Run backtest
results = backtester.run_backtest(
    universe=universe,
    start_date="2023-01-01",
    end_date="2024-01-01",
    top_k=5,
    initial_capital=100000
)

# Display results
backtester.plot_results()
```

## ⚙️ Configuration Options

### Universe Selection
- `nifty50` - Nifty 50 stocks
- `nifty100` - Nifty 100 stocks  
- `nifty500` - Nifty 500 stocks

### Strategy Parameters
- **Top K**: Number of stocks to select daily (1-10, default: 5)
- **Initial Capital**: Starting capital for backtest (default: ₹100,000)
- **Date Range**: Customizable start and end dates
- **Exit Threshold**: Convergence threshold (default: 0.5%)
- **Stop Loss**: Maximum adverse move (default: 2%)

### Performance Metrics
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade PnL**: Mean profit/loss per trade

## 📊 Output Analysis

### Equity Curve
- Portfolio value over time
- Daily PnL tracking
- Capital growth visualization

### Trade Analysis
- Individual trade details
- PnL distribution
- Position type analysis (LONG vs SHORT)
- Exit reason breakdown

### Performance Summary
- Key metrics table
- Trade statistics
- Risk metrics

## 🔧 Customization

### Modifying Exit Conditions
```python
# In simulate_trade method, adjust convergence threshold
if abs((current_price - current_vwap) / current_vwap) < 0.005:  # 0.5%
    # Exit logic
```

### Adjusting Signal Calculation
```python
# In get_vwap_reversal_signal method
signal = abs(latest['vwap_distance'])  # Base signal
if latest['vwap_cross_above'] or latest['vwap_cross_below']:
    signal *= 1.5  # Cross bonus
```

### Changing Universe
```python
# Use different universe
universe = data_handler.get_stocks_by_universe("nifty100")
```

## 📈 Strategy Logic Deep Dive

### VWAP Calculation
```python
# Volume Weighted Average Price
vwap = (close * volume).cumsum() / volume.cumsum()
```

### Reversal Signal Components
1. **Distance Magnitude**: `abs(vwap_distance)`
2. **Cross Bonus**: 1.5x multiplier for recent VWAP crosses
3. **Change Bonus**: 1.3x multiplier for strong distance changes

### Entry/Exit Rules
- **LONG Entry**: `vwap_distance < 0` (below VWAP)
- **SHORT Entry**: `vwap_distance > 0` (above VWAP)
- **Exit**: `abs((price - vwap) / vwap) < 0.005` (convergence)
- **Stop Loss**: `abs((price - vwap) / vwap) > 0.02` (adverse move)

## 🎯 Research Applications

### Parameter Optimization
- Test different Top K values (1-10)
- Adjust convergence thresholds (0.1% - 1.0%)
- Modify stop loss levels (1% - 5%)

### Universe Analysis
- Compare performance across different universes
- Analyze sector-specific behavior
- Test with different market cap ranges

### Time Period Analysis
- Test different market conditions
- Analyze seasonal patterns
- Compare bull vs bear market performance

## 📋 Example Results

### Sample Performance Metrics
```
📊 Performance Summary:
   Total Return: 15.23%
   Sharpe Ratio: 1.45
   Max Drawdown: 8.67%
   Win Rate: 62.3%
   Total Trades: 1,247
   Final Capital: ₹115,230
```

### Sample Trade
```
RELIANCE LONG: 2.34% (2023-06-15 -> 2023-06-16)
Entry: ₹2,450 (2.1% below VWAP)
Exit: ₹2,507 (converged to VWAP)
Reason: Converged to VWAP
```

## 🔍 Troubleshooting

### Common Issues
1. **No data available**: Check internet connection and API access
2. **Empty results**: Verify date range and universe selection
3. **Memory issues**: Reduce universe size or date range
4. **Slow performance**: Use smaller universe or shorter period

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

### Required Packages
```
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.22.0
plotly>=5.0.0
tqdm>=4.62.0
```

### Project Structure
```
baller/
├── simple_vwap_backtester.py
├── run_vwap_backtest.py
├── vwap_reversal_backtester.py
├── core/
│   └── data_handler.py
├── utils/
│   └── logger.py
└── VWAP_BACKTEST_README.md
```

## 🎯 Next Steps

### Enhancements
1. **Multi-timeframe analysis** (intraday, daily, weekly)
2. **Advanced risk management** (position sizing, correlation)
3. **Machine learning integration** (signal optimization)
4. **Real-time implementation** (live trading signals)

### Research Extensions
1. **Sector rotation** based on VWAP patterns
2. **Market regime detection** using VWAP behavior
3. **Volatility-adjusted** VWAP calculations
4. **Cross-asset** VWAP analysis

---

**Note**: This backtesting system is designed for research and educational purposes. Always validate results and consider transaction costs, slippage, and market impact in real trading scenarios. 