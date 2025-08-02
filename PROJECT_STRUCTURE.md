# Trading System Project Structure

## 📁 Directory Organization

```
baller/
├── 📁 core/                          # Core system components
│   ├── __init__.py
│   ├── data_handler.py              # Data fetching and processing
│   ├── strategy.py                  # Base strategy class
│   ├── portfolio.py                 # Portfolio management
│   └── broker.py                    # Broker interface
│
├── 📁 strategies/                    # Trading strategies
│   ├── __init__.py
│   ├── orb.py                       # Opening Range Breakout
│   ├── momentum.py                  # Momentum strategy
│   ├── volatility_breakout.py       # Volatility breakout (1-min data)
│   └── simple_alpha.py              # Simple alpha (MA crossover)
│
├── 📁 backtest/                      # Backtesting engine
│   ├── __init__.py
│   ├── backtester.py                # Main backtesting engine
│   └── performance.py               # Performance metrics
│
├── 📁 utils/                         # Utility modules
│   ├── __init__.py
│   ├── logger.py                    # Logging system
│   ├── plotting.py                  # Enhanced plotting utilities
│   └── helpers.py                   # Helper functions
│
├── 📁 config/                        # Configuration files
│   ├── config.yaml                  # Main configuration
│   └── strategies.yaml              # Strategy configurations
│
├── 📁 data/                          # Data storage
│   ├── cache/                       # Cached market data
│   ├── results/                     # Backtest results
│   └── nifty50_instruments.csv      # Nifty 50 instrument list
│
├── 📁 scripts/                       # Executable scripts
│   ├── run_backtest.py              # Standard backtest runner
│   ├── research_backtest.py         # Research-oriented backtesting
│   ├── run_volatility_backtest.py   # Volatility strategy backtest
│   ├── test_system.py               # System testing
│   ├── example_usage.py             # Usage examples
│   ├── check_api.py                 # API connectivity check
│   ├── plot_stock_data.py           # Data plotting utility
│   ├── show_stock_data.py           # Data display utility
│   └── test_simple_alpha.py         # Simple alpha test
│
├── 📁 tests/                         # Test files
│   ├── test_volatility_breakout.py  # Volatility strategy tests
│   └── quick_test_volatility.py     # Quick volatility tests
│
├── 📁 docs/                          # Documentation
│   ├── README.md                    # Main documentation
│   ├── API_REFERENCE.md             # API documentation
│   └── STRATEGY_GUIDE.md            # Strategy development guide
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── strategy_analysis.ipynb      # Strategy analysis
│   └── data_exploration.ipynb       # Data exploration
│
├── 📁 logs/                          # Log files
│   └── trading_system.log           # System logs
│
├── .env                              # Environment variables (API keys)
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── token_manager.py                  # Token management
└── PROJECT_STRUCTURE.md              # This file
```

## 🔧 Key Components

### Core System
- **DataHandler**: Fetches and caches market data from Zerodha API
- **BaseStrategy**: Abstract base class for all trading strategies
- **Portfolio**: Manages positions, P&L, and risk
- **Broker**: Interface for paper/live trading

### Strategies
- **ORBStrategy**: Opening Range Breakout
- **MomentumStrategy**: Momentum-based trading
- **VolatilityBreakoutStrategy**: 1-minute volatility breakout
- **SimpleAlphaStrategy**: Moving average crossover

### Backtesting
- **Backtester**: Main backtesting engine supporting multiple timeframes
- **ResearchBacktester**: Advanced backtesting with comprehensive analysis
- **Performance**: Risk and performance metrics

### Utilities
- **Logger**: Centralized logging system
- **Plotting**: Enhanced plotting with annotated signals
- **TokenManager**: Zerodha API token management

## 📊 Data Flow

1. **Data Fetching**: `DataHandler` → Zerodha API → Cache
2. **Signal Generation**: Strategy → Technical Indicators → Signals
3. **Backtesting**: Backtester → Portfolio → Performance Metrics
4. **Visualization**: Plotting → Annotated Charts → Analysis

## 🎯 Strategy Development

### Adding New Strategies
1. Create strategy class inheriting from `BaseStrategy`
2. Implement `generate_signals()` method
3. Add to `strategies/__init__.py`
4. Update backtesting scripts

### Strategy Parameters
- Configurable via YAML files
- Parameter optimization support
- Research-oriented analysis

## 🔑 API Integration

### Zerodha Kite API
- **Token Management**: Automatic refresh and validation
- **Data Fetching**: Historical and live data
- **Trading**: Paper and live trading support
- **Instruments**: Nifty 50 and other symbols

### Data Caching
- **Pickle-based**: Efficient storage and retrieval
- **Automatic Refresh**: Configurable cache expiration
- **Chunked Fetching**: API rate limit management

## 📈 Visualization Features

### Annotated Plots
- **Buy/Sell Signals**: Clear markers with prices
- **Technical Indicators**: Moving averages, volume, momentum
- **Random Day Analysis**: Detailed day-by-day breakdown
- **Strategy Comparison**: Multiple strategies on same data

### Research Tools
- **Parameter Impact**: How parameters affect performance
- **Signal Distribution**: Buy vs sell signal analysis
- **Risk-Return Scatter**: Performance visualization
- **Monthly Returns**: Heatmap analysis

## 🚀 Usage Examples

### Basic Backtest
```bash
python3 scripts/run_backtest.py --symbol RELIANCE.NS --strategy simple_alpha
```

### Research Backtest
```bash
python3 scripts/research_backtest.py --strategy volatility_breakout --symbols RELIANCE.NS TCS.NS
```

### Data Visualization
```bash
python3 scripts/plot_stock_data.py --symbol RELIANCE.NS --interval minute
```

### Strategy Testing
```bash
python3 scripts/test_simple_alpha.py
```

## 🔄 Recent Updates

### Latest Features
- ✅ 1-minute data support for volatility strategies
- ✅ Enhanced plotting with annotated signals
- ✅ Simple alpha strategy for testing
- ✅ Comprehensive directory organization
- ✅ Research-oriented backtesting
- ✅ Token management system

### Current Status
- ✅ Data fetching working correctly
- ✅ All strategies functional
- ✅ Backtesting engine complete
- ✅ Plotting utilities enhanced
- ✅ Directory structure organized

## 📝 Next Steps

1. **Strategy Optimization**: Parameter tuning and optimization
2. **Live Trading**: Paper trading implementation
3. **Risk Management**: Advanced position sizing
4. **Performance Analysis**: More detailed metrics
5. **Documentation**: Complete API reference

## 🛠️ Development Context

### Key Files Modified Recently
- `core/data_handler.py`: Enhanced for 1-minute data
- `strategies/volatility_breakout.py`: New strategy
- `strategies/simple_alpha.py`: Simple alpha strategy
- `utils/plotting.py`: Enhanced plotting utilities
- `backtest/backtester.py`: Minute-level backtesting
- `token_manager.py`: Centralized token management

### Current Working State
- All components functional
- Data fetching verified
- Strategies generating signals
- Plots showing buy/sell annotations
- Directory structure organized

### Environment Setup
- Python 3.8+
- Zerodha Kite API credentials in `.env`
- Required packages in `requirements.txt`
- WSL environment for development 