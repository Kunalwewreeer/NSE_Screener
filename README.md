# Indian Market Trading System

A scalable, modular Python trading system designed for the Indian equity market using Zerodha's Kite API. This system provides a complete framework for backtesting strategies, live trading, and research.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data handling, strategies, portfolio management, and execution
- **Multiple Strategies**: Built-in ORB (Opening Range Breakout) and Momentum strategies with easy extensibility
- **Comprehensive Backtesting**: Full backtesting engine with performance metrics, risk analysis, and visualization
- **Live Trading Support**: Paper and live trading modes with Zerodha Kite API integration
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio risk controls
- **Performance Analytics**: Advanced metrics including Sharpe ratio, drawdown analysis, and trade attribution
- **Indian Market Focus**: Optimized for NSE/BSE with proper market hours and instrument handling

## 📁 Project Structure

```
📁 project_root/
│
├── config/
│   └── config.yaml         # Global configuration
│
├── core/
│   ├── data_handler.py     # Data fetching and processing
│   ├── strategy.py         # Base strategy class
│   ├── portfolio.py        # Portfolio management
│   ├── metrics.py          # Performance metrics
│   ├── broker.py           # Broker interface (paper/live)
│   └── clock.py            # Time management
│
├── strategies/
│   ├── orb.py              # Opening Range Breakout strategy
│   ├── momentum.py         # Momentum strategy
│   └── pairs.py            # Pairs trading (placeholder)
│
├── backtest/
│   └── backtester.py       # Backtesting engine
│
├── utils/
│   ├── logger.py           # Logging system
│   └── file_utils.py       # File operations
│
├── run_backtest.py         # Backtest entry point
├── test_system.py          # System test suite
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Zerodha Kite account and API credentials

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd trading-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   API_KEY=your_zerodha_api_key
   API_SECRET=your_zerodha_api_secret
   ACCESS_TOKEN=your_access_token
   ```

4. **Configure the system**:
   Edit `config/config.yaml` to set your trading parameters, risk limits, and strategy configurations.

## 🚀 Quick Start

### 1. Test the System

Run the test suite to verify everything is working:

```bash
python test_system.py
```

### 2. Run a Backtest

Test the ORB strategy on Nifty 50 stocks:

```bash
python run_backtest.py \
    --strategy orb \
    --symbols NIFTY50 \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --capital 100000 \
    --plot \
    --save-results
```

Test the Momentum strategy on specific stocks:

```bash
python run_backtest.py \
    --strategy momentum \
    --symbols RELIANCE,TCS,INFY,HDFCBANK \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --capital 100000 \
    --plot
```

## 📊 Available Strategies

### 1. Opening Range Breakout (ORB)

**Concept**: Identifies the opening range (high-low) for a specified period after market open and generates signals when price breaks above/below this range.

**Key Parameters**:
- `lookback_period`: Minutes for opening range calculation (default: 30)
- `breakout_threshold`: Breakout confirmation threshold (default: 0.5%)
- `stop_loss_pct`: Stop loss percentage (default: 2%)
- `take_profit_pct`: Take profit percentage (default: 6%)

**Usage**:
```python
from strategies.orb import ORBStrategy

config = {
    'lookback_period': 30,
    'breakout_threshold': 0.005,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.06
}

strategy = ORBStrategy("My_ORB", config)
```

### 2. Momentum Strategy

**Concept**: Identifies stocks with strong momentum using multiple indicators (RSI, MACD, price momentum, volume) and generates buy/sell signals.

**Key Parameters**:
- `lookback_period`: Days for momentum calculation (default: 20)
- `momentum_threshold`: Minimum momentum threshold (default: 2%)
- `rsi_period`: RSI calculation period (default: 14)
- `volume_threshold`: Volume confirmation threshold (default: 1.5x)

**Usage**:
```python
from strategies.momentum import MomentumStrategy

config = {
    'lookback_period': 20,
    'momentum_threshold': 0.02,
    'rsi_period': 14,
    'stop_loss_pct': 0.03,
    'take_profit_pct': 0.09
}

strategy = MomentumStrategy("My_Momentum", config)
```

## ⚙️ Configuration

The system is configured through `config/config.yaml`:

```yaml
# API Configuration
api:
  zerodha:
    api_key: "${API_KEY}"
    api_secret: "${API_SECRET}"
    access_token: "${ACCESS_TOKEN}"

# Trading Parameters
trading:
  capital: 100000
  max_position_size: 0.1
  slippage: 0.001
  transaction_cost: 0.0005
  risk_per_trade: 0.02

# Strategy Configuration
strategies:
  orb:
    enabled: true
    lookback_period: 30
    breakout_threshold: 0.005
    
  momentum:
    enabled: true
    lookback_period: 20
    momentum_threshold: 0.02
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Returns**: Total return, annualized return, daily returns
- **Risk Metrics**: Volatility, VaR, CVaR, maximum drawdown
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, average trade, consecutive wins/losses
- **Portfolio Metrics**: Recovery factor, best/worst month analysis

## 🔧 Extending the System

### Adding a New Strategy

1. Create a new file in `strategies/` directory
2. Inherit from `BaseStrategy`
3. Implement the `generate_signals()` method
4. Optionally override `position_sizing()`, `calculate_stop_loss()`, etc.

Example:
```python
from core.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)
        # Initialize strategy-specific parameters
    
    def generate_signals(self, data):
        # Implement your signal generation logic
        # Return DataFrame with signal columns
        pass
```

### Adding New Data Sources

Extend the `DataHandler` class to support additional data sources:

```python
class MyDataHandler(DataHandler):
    def get_historical_data(self, symbol, start_date, end_date):
        # Implement custom data fetching
        pass
```

## 🧪 Testing

### Run System Tests
```bash
python test_system.py
```

### Run Strategy Tests
```bash
# Test ORB strategy
python run_backtest.py --strategy orb --symbols RELIANCE --start-date 2023-01-01 --end-date 2023-12-31

# Test Momentum strategy
python run_backtest.py --strategy momentum --symbols TCS --start-date 2023-01-01 --end-date 2023-12-31
```

## 📊 Example Results

### ORB Strategy Backtest Results
```
PERFORMANCE REPORT
============================================================
Total Return: 15.67%
Annual Return: 12.34%
Daily Return (Mean): 0.0005
Daily Return (Std): 0.0189

Risk Metrics:
Volatility (Annual): 30.12%
Max Drawdown: -8.45%
VaR (95%): -0.0234
CVaR (95%): -0.0345

Risk-Adjusted Returns:
Sharpe Ratio: 1.23
Sortino Ratio: 1.45
Calmar Ratio: 1.46
Recovery Factor: 1.85

Trade Statistics:
Total Trades: 45
Win Rate: 62.2%
Profit Factor: 1.67
Average Trade: ₹1,234.56
```

## 🔒 Risk Management

The system includes comprehensive risk management:

- **Position Sizing**: Based on risk per trade and portfolio size
- **Stop Loss**: Automatic stop loss calculation and execution
- **Portfolio Limits**: Maximum position size and portfolio risk limits
- **Transaction Costs**: Realistic slippage and brokerage modeling
- **Correlation Limits**: Prevents over-concentration in correlated positions

## 📝 Logging

The system uses structured logging with different levels:

- **INFO**: General system information
- **DEBUG**: Detailed debugging information
- **WARNING**: Warning messages
- **ERROR**: Error messages

Logs are saved to `logs/trading.log` with rotation and compression.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Run the test suite to verify your setup
3. Check the logs for error messages
4. Open an issue on GitHub

## 🔄 Roadmap

- [ ] Live trading implementation
- [ ] Additional strategies (pairs trading, mean reversion)
- [ ] Web dashboard for monitoring
- [ ] Real-time alerts and notifications
- [ ] Machine learning strategy integration
- [ ] Multi-timeframe analysis
- [ ] Options trading support
- [ ] Portfolio optimization algorithms 