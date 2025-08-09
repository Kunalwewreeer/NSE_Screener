# Intraday Stock Screener & Trading Platform

A comprehensive Python-based intraday trading platform designed for the Indian equity market. The system features an advanced stock screener with real-time analysis, multiple technical indicators, backtesting capabilities, and a complete trading framework using Zerodha's Kite API.

## Core Platform Features

### Stock Screener Platform (Primary)
**Advanced Stock Screener**: Real-time screening of Nifty 50, 100, and 500 stocks with customizable filters and precise minute-level cutoff times for intraday analysis.

**Multi-Criteria Analysis**: Screen stocks by percentage change, volume ratios, technical scores, VWAP distance, momentum indicators, and recent minute-level returns.

**Live Trading Ready**: Optimized for live deployment with intelligent caching, API rate limiting, and robust error handling for morning trading sessions.

### Technical Analysis & Visualization
**Comprehensive Technical Analysis**: 15+ technical indicators including SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, ADX, Stochastic, and more with live calculation and visualization.

**Interactive Dashboard**: Streamlit-based web interface with real-time data visualization, interactive charts, and timeline analysis for minute-by-minute market progression.

**Signal Generation**: Automated buy/sell signal detection with detailed reasoning and confidence scoring.

### Trading Strategies & Backtesting
**Multiple Trading Strategies**: Built-in strategies including Volatility Breakout, Opening Range Breakout (ORB), VWAP Mean Reversion, and Fakeout Reversal detection.

**Comprehensive Backtesting**: Full backtesting engine with performance metrics, risk analysis, trade attribution, and visual equity curves.

**Strategy Development Framework**: Modular architecture for easy strategy creation and testing with proper risk management.

### Research & Analytics
**Performance Analytics**: Advanced metrics including Sharpe ratio, drawdown analysis, win rates, profit factors, and risk-adjusted returns.

**Pattern Recognition**: Fakeout reversal detection, breakout analysis, and mean reversion identification with customizable parameters.

**No-Lookahead Analysis**: Strict adherence to using only available data up to specified cutoff times, ensuring realistic trading conditions and research integrity.

## Project Architecture

```
project_root/
│
├── core/
│   ├── data_handler.py         # Zerodha API integration and data fetching
│   ├── strategy.py             # Base strategy framework
│   └── portfolio.py            # Portfolio management
│
├── strategies/
│   ├── volatility_breakout.py  # Volatility-based breakout strategy
│   ├── orb_strategy.py         # Opening Range Breakout implementation
│   └── fakeout_detector.py     # Fakeout reversal detection
│
├── stock_screener.py           # Main screening engine with technical analysis
├── run_enhanced_screener.py    # Streamlit dashboard launcher
├── cache_manager.py            # Data caching and management
├── streamlit_demo_dashboard.py # Interactive trading dashboard
│
├── config/
│   ├── config.yaml             # System configuration
│   └── strategies.yaml         # Strategy parameters
│
└── docs/                       # Documentation and guides
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Zerodha Kite account with API access
- WSL environment (recommended for Windows users)

### Setup Instructions

1. **Clone and Install**:
   ```bash
   git clone <repository-url>
   cd baller
   pip install -r requirements.txt
   ```

2. **Configure API Credentials**:
   Create `.env` file:
   ```bash
   API_KEY=your_zerodha_api_key
   API_SECRET=your_zerodha_api_secret
   ACCESS_TOKEN=your_access_token
   ```

3. **Launch the Stock Screener Platform**:
   ```bash
   streamlit run run_enhanced_screener.py --server.port 8503
   ```

## Additional Platform Capabilities

### Interactive Technical Analysis Dashboard
```bash
streamlit run streamlit_demo_dashboard.py
```
Provides detailed single-stock analysis with:
- Timeline progression analysis with live indicator updates
- 10+ intraday trading strategies with signal visualization
- Interactive charting with zoom, pan, and annotation features
- Color-coded signal indicators (green for buy, red for sell)

### Jupyter Notebook Research Environment
Access `signal_analysis.ipynb` for:
- Strategy development and testing
- Random day analysis for pattern validation
- Custom indicator development and backtesting
- Research-oriented parameter optimization

### Fakeout Detection System
Specialized module for detecting and trading fakeout reversals:
- PDH/PDL (Previous Day High/Low) level analysis
- VWAP-based reversal detection
- Custom level support with confirmation rules
- Volume spike validation and risk management

### Portfolio Management & Risk Controls
- Position sizing based on volatility and risk parameters
- Automatic stop-loss and take-profit calculation
- Portfolio-level risk monitoring and limits
- Transaction cost modeling with realistic slippage

## Stock Screener Platform

### Key Features

**Real-Time Screening**: Screen stocks based on percentage change, volume ratios, technical indicator values, and recent minute-level returns with precise cutoff times.

**Advanced Filtering**: Filter stocks by minimum absolute percentage change, volume ratios, interest scores, VWAP distance, and recent volume patterns.

**Multi-Timeframe Analysis**: Analyze stocks across different timeframes with recent minute returns (1min, 2min, 3min, 5min, 10min, 15min, 30min).

**Bidirectional Screening**: Identify both bullish and bearish opportunities with comprehensive scoring algorithms.

**Interactive Visualization**: Plotly-based charts with technical indicators, support/resistance levels, and signal annotations.

### Screening Criteria

- **Percentage Change**: Absolute percentage change from market open
- **Volume Analysis**: Volume ratios compared to historical averages
- **Technical Scores**: Bullish/bearish scoring based on multiple indicators
- **VWAP Distance**: Distance from Volume Weighted Average Price
- **Momentum Indicators**: Recent minute-level price movements
- **Interest Score**: Composite score based on volatility and volume

### Usage Examples

**Morning Trading Setup** (9:50 AM cutoff):
1. Select analysis date and cutoff time (e.g., 09:50)
2. Choose stock universe (Nifty 50/100/500)
3. Set minimum filters (e.g., 2% absolute change, 1.5x volume ratio)
4. Sort by absolute percentage change or recent returns
5. Analyze top 10-20 stocks for trading opportunities

**Intraday Analysis** (Any time cutoff):
1. Set precise cutoff time for analysis
2. Use advanced filters for volume patterns
3. Screen for VWAP mean reversion opportunities
4. Identify fakeout reversal setups

## Trading Strategies

### 1. Volatility Breakout Strategy

Identifies stocks with expanding volatility and momentum breakouts with volume confirmation.

**Key Parameters**:
- Volatility period: 5 minutes
- Breakout threshold: Configurable multiplier
- Volume confirmation: Optional with fallback
- Risk management: 1% stop loss, 2% take profit

### 2. Opening Range Breakout (ORB)

Detects breakouts from the opening range with adaptive range calculation and multi-factor scoring.

**Features**:
- Flexible opening range calculation
- Volume and momentum confirmation
- Timing-based scoring
- No-lookahead compliance

### 3. VWAP Mean Reversion

Backtesting strategy that identifies stocks with high VWAP distance and trades mean reversion.

**Implementation**:
- Select top 5 stocks by absolute VWAP distance
- Long positions below VWAP, short positions above
- Hold until end of day
- Comprehensive P&L tracking

### 4. Fakeout Reversal Detection

Advanced pattern recognition for detecting fakeout reversals around key levels.

**Capabilities**:
- PDH/PDL level detection
- VWAP-based reversals
- Custom level support
- Confirmation rules with volume spikes

## Technical Indicators

The platform calculates 15+ technical indicators in real-time:

- **Trend**: SMA (20, 50), EMA (12, 26), VWAP
- **Momentum**: RSI (14), MACD, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Volatility Index
- **Volume**: OBV, MFI, Volume Ratios
- **Strength**: ADX, CCI
- **Support/Resistance**: Pivot Points, Dynamic Levels

## Live Trading Features

### Data Management
- **Intelligent Caching**: Pickle-based caching with refresh options
- **Data Validation**: Completeness checks and missing data handling
- **API Rate Limiting**: Configurable request delays and concurrent limits

### Performance Optimization
- **Parallel Processing**: ThreadPoolExecutor for multi-stock analysis
- **Memory Management**: Efficient data structures and cleanup
- **Error Recovery**: Robust error handling with fallback mechanisms

### User Interface
- **Real-Time Updates**: Live indicator calculations and signal generation
- **Interactive Charts**: Plotly-based visualizations with zoom and pan
- **Progress Tracking**: Real-time progress bars for screening operations
- **Debug Information**: Comprehensive logging and error reporting

## Configuration

### System Configuration (config/config.yaml)
```yaml
api:
  zerodha:
    rate_limit_delay: 0.1
    max_concurrent_requests: 2

trading:
  capital: 100000
  max_position_size: 0.1
  transaction_cost: 0.0005

screening:
  default_universe: "nifty50"
  min_volume_threshold: 1000
  cache_expiry_hours: 1
```

### Strategy Configuration (config/strategies.yaml)
```yaml
volatility_breakout:
  enabled: true
  volatility_period: 5
  volatility_multiplier: 0.8
  momentum_period: 2
  volume_threshold: 0.1

orb_strategy:
  enabled: true
  lookback_period: 30
  breakout_threshold: 0.005
  volume_confirmation: true
```

## Advanced Features

### Cache Management
- Clear cache for fresh data fetching
- View cache information and statistics
- Validate data completeness across time periods
- Automatic cache refresh for live trading

### Backtesting Engine
- Historical strategy performance analysis
- Risk-adjusted returns calculation
- Trade-by-trade analysis with detailed logs
- Performance visualization with equity curves

### Research Tools
- Random day analysis for strategy validation
- Parameter sensitivity analysis
- Signal quality assessment
- Pattern recognition and validation

## API Integration

### Zerodha Kite API
- Historical data fetching with proper date handling
- Instrument master management
- Real-time quote integration
- Error handling for rate limits and connectivity issues

### Data Processing
- Timezone-aware datetime handling
- Missing data interpolation
- Technical indicator calculation optimization
- Memory-efficient data structures

## Performance Metrics

The platform provides comprehensive performance analysis:

- **Screening Performance**: Processing speed, API efficiency, cache hit rates
- **Strategy Performance**: Win rates, profit factors, risk-adjusted returns
- **System Performance**: Memory usage, processing times, error rates
- **Data Quality**: Completeness metrics, validation results

## Best Practices for Live Trading

### Morning Setup (9:30-10:00 AM)
1. Clear cache for fresh data
2. Set cutoff time to 9:50 AM for initial screening
3. Use minimum 2% absolute change filter
4. Focus on high-volume stocks (>1.5x average volume)
5. Validate data completeness before trading

### Risk Management
- Never risk more than 1-2% per trade
- Use stop losses based on ATR or technical levels
- Diversify across uncorrelated positions
- Monitor position sizes and portfolio exposure

### System Monitoring
- Check API rate limits and usage
- Monitor cache performance and refresh as needed
- Validate data quality and completeness
- Track system performance metrics

## Troubleshooting

### Common Issues
- **API Rate Limits**: Reduce concurrent requests and increase delays
- **Data Completeness**: Refresh cache and validate time ranges
- **Performance Issues**: Clear cache and optimize filter criteria
- **Memory Usage**: Limit analysis to smaller stock universes

### Debug Features
- Comprehensive logging with different levels
- Real-time debug output in Streamlit interface
- Performance timing for bottleneck identification
- Data validation and quality checks

## Platform Architecture & Extensibility

### Modular Design
The platform is built with a modular architecture allowing easy extension and customization:

**Core Modules**:
- `data_handler.py`: Unified API integration with caching and rate limiting
- `strategy.py`: Base strategy framework with risk management
- `portfolio.py`: Portfolio management and position tracking

**Strategy Modules**:
- `volatility_breakout.py`: Volatility expansion detection with momentum confirmation
- `orb_strategy.py`: Opening Range Breakout with adaptive range calculation
- `fakeout_detector.py`: Advanced pattern recognition for reversal detection

**Analysis Modules**:
- `stock_screener.py`: Multi-criteria screening engine with parallel processing
- `cache_manager.py`: Intelligent data caching with refresh and validation
- `metrics.py`: Comprehensive performance analytics and risk metrics

### Extending the Platform

#### Adding New Technical Indicators
```python
def calculate_custom_indicator(data):
    # Example: Custom momentum indicator
    short_ma = data['close'].rolling(5).mean()
    long_ma = data['close'].rolling(20).mean()
    momentum = (short_ma - long_ma) / long_ma * 100
    return momentum

# Integrate into screening indicators
def calculate_screening_indicators(self, data):
    # ... existing indicators ...
    data['custom_momentum'] = calculate_custom_indicator(data)
    return data
```

#### Creating New Trading Strategies
```python
from core.strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.lookback_period = config.get('lookback_period', 20)
        self.deviation_threshold = config.get('deviation_threshold', 2.0)
    
    def generate_signals(self, data):
        # Calculate Bollinger Bands
        sma = data['close'].rolling(self.lookback_period).mean()
        std = data['close'].rolling(self.lookback_period).std()
        upper_band = sma + (std * self.deviation_threshold)
        lower_band = sma - (std * self.deviation_threshold)
        
        # Generate signals
        signals = []
        for i in range(len(data)):
            if data['close'].iloc[i] < lower_band.iloc[i]:
                signals.append({
                    'timestamp': data.index[i],
                    'signal_type': 'BUY',
                    'confidence': 0.8,
                    'reason': f'Price below lower Bollinger Band'
                })
        
        return signals
```

#### Custom Screening Filters
```python
def momentum_divergence_filter(df, min_price_momentum=0.02, max_rsi=30):
    """
    Filter for stocks showing price momentum but oversold RSI
    """
    momentum_condition = df['pct_change_from_open'].abs() > min_price_momentum
    rsi_condition = df['rsi'] < max_rsi
    volume_condition = df['volume_ratio'] > 1.5
    
    return df[momentum_condition & rsi_condition & volume_condition]

# Integrate into screening workflow
def screen_stocks_custom(self, criteria='momentum_divergence'):
    # ... fetch data ...
    if criteria == 'momentum_divergence':
        filtered_stocks = momentum_divergence_filter(screened_data)
    return filtered_stocks
```

#### Adding New Data Sources
```python
class AlternativeDataHandler(DataHandler):
    def get_options_data(self, symbol, expiry_date):
        # Implement options data fetching
        pass
    
    def get_futures_data(self, symbol, expiry_date):
        # Implement futures data fetching
        pass
    
    def get_sector_data(self, sector_name):
        # Implement sector-wise analysis
        pass
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough testing and consider consulting with financial advisors before live trading.

## Support and Documentation

- **Enhanced Screener Guide**: See ENHANCED_SCREENER_README.md
- **Fakeout Detector Guide**: See FAKEOUT_DETECTOR_README.md
- **VWAP Backtest Guide**: See VWAP_BACKTEST_README.md
- **System Documentation**: See docs/ directory

For issues and support:
1. Check the comprehensive documentation
2. Review debug output and logs
3. Validate API credentials and connectivity
4. Test with sample data before live deployment