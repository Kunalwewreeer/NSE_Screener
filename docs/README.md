# Trading System Documentation

## Overview

A comprehensive, modular Python trading system for the Indian market using Zerodha's Kite API. The system supports multiple strategies, backtesting, and research-oriented analysis with annotated plots.

## Features

### 🎯 Strategies
- **Simple Alpha**: Moving average crossover with volume confirmation
- **Volatility Breakout**: 1-minute volatility-based breakout strategy
- **Opening Range Breakout (ORB)**: Opening range breakout strategy
- **Momentum**: Momentum-based trading strategy

### 📊 Data Management
- Real-time data fetching from Zerodha Kite API
- Intelligent caching system
- Support for multiple timeframes (1-minute to daily)
- Nifty 50 and other Indian equity symbols

### 🔬 Research Tools
- Comprehensive backtesting engine
- Annotated plots with buy/sell signals
- Random day analysis
- Strategy comparison tools
- Parameter optimization

### 📈 Visualization
- Enhanced plotting with technical indicators
- Buy/sell signal annotations
- Volume and momentum analysis
- Performance metrics visualization

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd baller

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Zerodha API credentials
```

### 2. Test Data Fetching

```bash
# Test API connectivity
python3 scripts/check_api.py

# Plot stock data
python3 scripts/plot_stock_data.py --symbol RELIANCE.NS --interval minute
```

### 3. Run Strategy Tests

```bash
# Test simple alpha strategy
python3 scripts/test_simple_alpha.py

# Run volatility breakout backtest
python3 scripts/run_volatility_backtest.py --symbol RELIANCE.NS
```

### 4. Run Backtests

```bash
# Standard backtest
python3 scripts/run_backtest.py --symbol RELIANCE.NS --strategy simple_alpha

# Research backtest with comprehensive analysis
python3 scripts/research_backtest.py --strategy volatility_breakout --symbols RELIANCE.NS TCS.NS
```

## Directory Structure

```
baller/
├── core/                 # Core system components
├── strategies/           # Trading strategies
├── backtest/            # Backtesting engine
├── utils/               # Utility modules
├── config/              # Configuration files
├── data/                # Data storage
├── scripts/             # Executable scripts
├── tests/               # Test files
├── docs/                # Documentation
└── notebooks/           # Jupyter notebooks
```

## Configuration

### Main Configuration (`config/config.yaml`)
- API settings
- Data management
- Trading parameters
- Risk management
- Performance metrics

### Strategy Configuration (`config/strategies.yaml`)
- Strategy parameters
- Optimization settings
- Supported timeframes
- Target symbols

## API Integration

### Zerodha Kite API
- Automatic token management
- Rate limiting and chunking
- Error handling and retries
- Real-time data streaming

### Data Caching
- Pickle-based caching
- Automatic cache expiration
- Efficient storage and retrieval

## Strategy Development

### Adding New Strategies

1. Create strategy class inheriting from `BaseStrategy`:

```python
from core.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        # Initialize parameters
        
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        # Implement signal generation logic
        return signals
```

2. Add to `strategies/__init__.py`:

```python
from .my_strategy import MyStrategy

__all__ = [
    # ... existing strategies
    'MyStrategy'
]
```

3. Update configuration in `config/strategies.yaml`

4. Test with `scripts/test_simple_alpha.py`

## Usage Examples

### Basic Data Analysis

```python
from core.data_handler import DataHandler

# Initialize data handler
data_handler = DataHandler()

# Fetch data
data = data_handler.get_historical_data(
    symbols="RELIANCE.NS",
    from_date="2025-07-25",
    to_date="2025-07-27",
    interval="minute"
)
```

### Strategy Testing

```python
from strategies.simple_alpha import SimpleAlphaStrategy

# Initialize strategy
strategy = SimpleAlphaStrategy("Test", {
    'fast_ma': 5,
    'slow_ma': 20,
    'volume_threshold': 1.5
})

# Generate signals
signals = strategy.generate_signals(data)
```

### Backtesting

```python
from backtest.backtester import Backtester

# Initialize backtester
backtester = Backtester(config)

# Run backtest
results = backtester.run_backtest(
    strategy=strategy,
    symbols=["RELIANCE.NS"],
    start_date="2025-07-25",
    end_date="2025-07-27"
)
```

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Maximum drawdown

## Visualization Features

### Annotated Plots
- Buy/sell signals with prices
- Technical indicators overlay
- Volume analysis
- Random day breakdown

### Research Tools
- Strategy comparison
- Parameter impact analysis
- Signal distribution
- Risk-return scatter plots

## Troubleshooting

### Common Issues

1. **API Token Issues**
   ```bash
   python3 token_manager.py
   # Follow interactive token generation
   ```

2. **Data Fetching Errors**
   ```bash
   python3 scripts/check_api.py
   # Verify API connectivity and symbol list
   ```

3. **Strategy Not Generating Signals**
   - Check parameter values
   - Verify data quality
   - Review strategy logic

### Logs
- Check `logs/trading_system.log` for detailed error information
- Enable DEBUG logging in `config/config.yaml` for more details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the documentation in `docs/`
- Review example scripts in `scripts/`
- Examine test files in `tests/`
- Check logs in `logs/` 