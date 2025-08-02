#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE INTRADAY TRADING DASHBOARD
=========================================

Shows charts for ALL Nifty 50 stocks with ALL technical indicators
for making informed intraday trading decisions.

Features:
- Multi-stock analysis with all technical indicators
- Real-time signal detection across all timeframes
- Volume analysis and momentum indicators
- Support/Resistance levels
- Volatility and trend strength
- Risk-reward analysis
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from core.data_handler import fetch_data, DataHandler
from utils.helpers import load_yaml
from utils.logger import get_logger

# Set up logging
logger = get_logger(__name__)

class IntradayTradingDashboard:
    """Comprehensive intraday trading dashboard with all technical indicators."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.data_handler = DataHandler()
        self.config = {
            # Moving Averages
            'sma_fast': 9,
            'sma_slow': 21,
            'ema_fast': 12,
            'ema_slow': 26,
            
            # Momentum Indicators
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stoch_k': 14,
            'stoch_d': 3,
            
            # Volatility Indicators
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            
            # Volume Indicators
            'volume_sma': 20,
            'vwap_period': 20,
            
            # Trend Indicators
            'adx_period': 14,
            'cci_period': 20,
            
            # Support/Resistance
            'pivot_lookback': 5,
            'fractal_period': 5
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_strength': 0.1,
            'volume_spike': 2.0,
            'bb_squeeze': 0.02,
            'trend_strength': 25,
            'momentum_threshold': 0.5
        }
        
        plt.style.use('seaborn-v0_8')
        
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators for intraday trading."""
        df = data.copy()
        
        # Price-based indicators
        df = self._add_moving_averages(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_support_resistance(df)
        
        return df
    
    def _add_moving_averages(self, df):
        """Add moving average indicators."""
        # Simple Moving Averages
        df['sma_9'] = df['close'].rolling(window=self.config['sma_fast']).mean()
        df['sma_21'] = df['close'].rolling(window=self.config['sma_slow']).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=self.config['ema_fast']).mean()
        df['ema_26'] = df['close'].ewm(span=self.config['ema_slow']).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _add_momentum_indicators(self, df):
        """Add momentum indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=self.config['macd_fast']).mean()
        ema26 = df['close'].ewm(span=self.config['macd_slow']).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=self.config['macd_signal']).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_min = df['low'].rolling(window=self.config['stoch_k']).min()
        high_max = df['high'].rolling(window=self.config['stoch_k']).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=self.config['stoch_d']).mean()
        
        # Rate of Change
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        return df
    
    def _add_volatility_indicators(self, df):
        """Add volatility indicators."""
        # Bollinger Bands
        sma = df['close'].rolling(window=self.config['bb_period']).mean()
        std = df['close'].rolling(window=self.config['bb_period']).std()
        df['bb_upper'] = sma + (std * self.config['bb_std'])
        df['bb_lower'] = sma - (std * self.config['bb_std'])
        df['bb_middle'] = sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.config['atr_period']).mean()
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df):
        """Add volume indicators."""
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=self.config['volume_sma']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0.0)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        return df
    
    def _add_trend_indicators(self, df):
        """Add trend strength indicators."""
        # ADX (Average Directional Index)
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0.0)
        
        plus_di = 100 * (plus_dm.rolling(window=self.config['adx_period']).mean() / df['atr'])
        minus_di = 100 * (minus_dm.rolling(window=self.config['adx_period']).mean() / df['atr'])
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=self.config['adx_period']).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=self.config['cci_period']).mean()
        mad = tp.rolling(window=self.config['cci_period']).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        return df
    
    def _add_support_resistance(self, df):
        """Add support and resistance levels."""
        # Pivot Points (Classic)
        df['pivot'] = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift()
        df['s1'] = 2 * df['pivot'] - df['high'].shift()
        df['r2'] = df['pivot'] + (df['high'].shift() - df['low'].shift())
        df['s2'] = df['pivot'] - (df['high'].shift() - df['low'].shift())
        
        # Fractal Support/Resistance
        df['fractal_high'] = df['high'].rolling(window=self.config['fractal_period'], center=True).max()
        df['fractal_low'] = df['low'].rolling(window=self.config['fractal_period'], center=True).min()
        
        return df
    
    def generate_trading_signals(self, df):
        """Generate comprehensive trading signals."""
        signals = []
        
        if len(df) < 50:  # Need sufficient data
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Price action signals
        if latest['close'] > latest['sma_9'] > latest['sma_21']:
            signals.append({
                'type': 'BUY',
                'indicator': 'MA_TREND',
                'strength': 'STRONG',
                'reason': 'Price above both moving averages',
                'entry': latest['close'],
                'stop_loss': latest['sma_9'],
                'target': latest['close'] + (latest['close'] - latest['sma_9'])
            })
        
        # RSI signals
        if latest['rsi'] < self.signal_thresholds['rsi_oversold'] and prev['rsi'] >= self.signal_thresholds['rsi_oversold']:
            signals.append({
                'type': 'BUY',
                'indicator': 'RSI_OVERSOLD',
                'strength': 'MEDIUM',
                'reason': f'RSI crossed below {self.signal_thresholds["rsi_oversold"]}',
                'entry': latest['close'],
                'stop_loss': latest['close'] * 0.98,
                'target': latest['close'] * 1.04
            })
        
        # MACD signals
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            signals.append({
                'type': 'BUY',
                'indicator': 'MACD_BULLISH',
                'strength': 'STRONG',
                'reason': 'MACD crossed above signal line',
                'entry': latest['close'],
                'stop_loss': latest['close'] * 0.97,
                'target': latest['close'] * 1.05
            })
        
        # Bollinger Band signals
        if latest['close'] < latest['bb_lower'] and latest['rsi'] < 40:
            signals.append({
                'type': 'BUY',
                'indicator': 'BB_OVERSOLD',
                'strength': 'MEDIUM',
                'reason': 'Price below BB lower band with low RSI',
                'entry': latest['close'],
                'stop_loss': latest['bb_lower'] * 0.99,
                'target': latest['bb_middle']
            })
        
        # Volume breakout signals
        if latest['volume_ratio'] > self.signal_thresholds['volume_spike'] and latest['close'] > prev['close']:
            signals.append({
                'type': 'BUY',
                'indicator': 'VOLUME_BREAKOUT',
                'strength': 'STRONG',
                'reason': f'Volume spike {latest["volume_ratio"]:.1f}x with price up',
                'entry': latest['close'],
                'stop_loss': latest['close'] * 0.96,
                'target': latest['close'] * 1.06
            })
        
        # Trend strength signals
        if latest['adx'] > self.signal_thresholds['trend_strength'] and latest['plus_di'] > latest['minus_di']:
            signals.append({
                'type': 'BUY',
                'indicator': 'STRONG_UPTREND',
                'strength': 'STRONG',
                'reason': f'Strong uptrend (ADX: {latest["adx"]:.1f})',
                'entry': latest['close'],
                'stop_loss': latest['close'] * 0.95,
                'target': latest['close'] * 1.08
            })
        
        return signals
    
    def plot_comprehensive_chart(self, symbol, data, signals, rank=1):
        """Plot comprehensive chart with all indicators."""
        try:
            # Create subplot layout
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(6, 2, height_ratios=[4, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.2)
            
            # Main price chart
            ax1 = fig.add_subplot(gs[0, :])
            
            # Plot price and moving averages
            ax1.plot(data.index, data['close'], 'k-', linewidth=2, label='Close Price', alpha=0.8)
            ax1.plot(data.index, data['sma_9'], 'b-', linewidth=1, label='SMA 9', alpha=0.7)
            ax1.plot(data.index, data['sma_21'], 'r-', linewidth=1, label='SMA 21', alpha=0.7)
            ax1.plot(data.index, data['ema_12'], 'g--', linewidth=1, label='EMA 12', alpha=0.7)
            ax1.plot(data.index, data['vwap'], 'purple', linewidth=1, label='VWAP', alpha=0.7)
            
            # Bollinger Bands
            ax1.fill_between(data.index, data['bb_upper'], data['bb_lower'], alpha=0.1, color='gray')
            ax1.plot(data.index, data['bb_upper'], 'gray', linewidth=1, alpha=0.5)
            ax1.plot(data.index, data['bb_lower'], 'gray', linewidth=1, alpha=0.5)
            
            # Support/Resistance levels
            ax1.axhline(y=data['pivot'].iloc[-1], color='orange', linestyle='--', alpha=0.6, label='Pivot')
            ax1.axhline(y=data['r1'].iloc[-1], color='red', linestyle=':', alpha=0.6, label='R1')
            ax1.axhline(y=data['s1'].iloc[-1], color='green', linestyle=':', alpha=0.6, label='S1')
            
            # Plot signals on main chart
            for signal in signals:
                if signal['type'] == 'BUY':
                    ax1.scatter(data.index[-1], signal['entry'], color='green', s=100, marker='^', 
                              zorder=5, label=f"BUY ({signal['indicator']})")
                elif signal['type'] == 'SELL':
                    ax1.scatter(data.index[-1], signal['entry'], color='red', s=100, marker='v', 
                              zorder=5, label=f"SELL ({signal['indicator']})")
            
            ax1.set_title(f'{symbol} - Comprehensive Intraday Analysis (Rank #{rank})', fontsize=16, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            ax2 = fig.add_subplot(gs[1, :])
            colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
                     for i in range(len(data))]
            ax2.bar(data.index, data['volume'], color=colors, alpha=0.6, width=0.8)
            ax2.plot(data.index, data['volume_sma'], 'blue', linewidth=1, label='Volume SMA')
            ax2.set_title('Volume Analysis', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # RSI chart
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(data.index, data['rsi'], 'purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
            ax3.set_title('RSI (14)', fontsize=12)
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # MACD chart
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.plot(data.index, data['macd'], 'blue', linewidth=2, label='MACD')
            ax4.plot(data.index, data['macd_signal'], 'red', linewidth=1, label='Signal')
            ax4.bar(data.index, data['macd_histogram'], alpha=0.3, color='gray', label='Histogram')
            ax4.set_title('MACD', fontsize=12)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # Stochastic chart
            ax5 = fig.add_subplot(gs[3, 0])
            ax5.plot(data.index, data['stoch_k'], 'blue', linewidth=2, label='%K')
            ax5.plot(data.index, data['stoch_d'], 'red', linewidth=1, label='%D')
            ax5.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax5.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax5.set_title('Stochastic', fontsize=12)
            ax5.set_ylim(0, 100)
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # ADX chart
            ax6 = fig.add_subplot(gs[3, 1])
            ax6.plot(data.index, data['adx'], 'black', linewidth=2, label='ADX')
            ax6.plot(data.index, data['plus_di'], 'green', linewidth=1, label='+DI')
            ax6.plot(data.index, data['minus_di'], 'red', linewidth=1, label='-DI')
            ax6.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
            ax6.set_title('ADX & Directional Indicators', fontsize=12)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # CCI chart
            ax7 = fig.add_subplot(gs[4, 0])
            ax7.plot(data.index, data['cci'], 'orange', linewidth=2)
            ax7.axhline(y=100, color='red', linestyle='--', alpha=0.7)
            ax7.axhline(y=-100, color='green', linestyle='--', alpha=0.7)
            ax7.set_title('CCI (20)', fontsize=12)
            ax7.grid(True, alpha=0.3)
            
            # ATR chart
            ax8 = fig.add_subplot(gs[4, 1])
            ax8.plot(data.index, data['atr'], 'brown', linewidth=2)
            ax8.set_title('Average True Range', fontsize=12)
            ax8.grid(True, alpha=0.3)
            
            # Bollinger Band Width
            ax9 = fig.add_subplot(gs[5, 0])
            ax9.plot(data.index, data['bb_width'], 'gray', linewidth=2)
            ax9.set_title('Bollinger Band Width', fontsize=12)
            ax9.grid(True, alpha=0.3)
            
            # Volume Ratio
            ax10 = fig.add_subplot(gs[5, 1])
            ax10.plot(data.index, data['volume_ratio'], 'cyan', linewidth=2)
            ax10.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Spike Level')
            ax10.set_title('Volume Ratio', fontsize=12)
            ax10.legend(fontsize=8)
            ax10.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting {symbol}: {e}")
    
    def analyze_all_stocks(self, date_str=None, max_stocks=50):
        """Analyze all Nifty 50 stocks with comprehensive indicators."""
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n🚀 COMPREHENSIVE INTRADAY TRADING DASHBOARD")
        print(f"📅 Analysis Date: {date_str}")
        print("=" * 80)
        
        # Get Nifty 50 stocks
        print("📊 Loading Nifty 50 stocks...")
        symbols = self.data_handler.get_nifty50_stocks()
        
        if not symbols:
            print("❌ Failed to load Nifty 50 stocks")
            return
        
        print(f"📊 Fetching minute data for {len(symbols)} stocks...")
        
        # Fetch data
        all_data = self.data_handler.fetch_data(symbols, date_str, date_str, 'minute')
        
        if not all_data:
            print("❌ No data fetched")
            return
        
        print(f"✅ Data fetched for {len(all_data)} stocks")
        
        # Analyze each stock
        all_results = []
        
        for symbol in symbols:
            if symbol not in all_data:
                continue
                
            try:
                data = all_data[symbol]
                if data.empty or len(data) < 50:
                    continue
                
                print(f"  🔍 Analyzing {symbol}...")
                
                # Calculate all indicators
                data_with_indicators = self.calculate_all_indicators(data)
                
                # Generate signals
                signals = self.generate_trading_signals(data_with_indicators)
                
                # Calculate scores
                latest = data_with_indicators.iloc[-1]
                
                # Comprehensive scoring
                score = 0
                score_factors = []
                
                # Trend score
                if latest['close'] > latest['sma_21']:
                    score += 20
                    score_factors.append("Above SMA21")
                
                # Momentum score
                if 30 < latest['rsi'] < 70:
                    score += 15
                    score_factors.append("RSI balanced")
                
                # Volume score
                if latest['volume_ratio'] > 1.5:
                    score += 25
                    score_factors.append("High volume")
                
                # Volatility score
                if latest['atr'] > data_with_indicators['atr'].rolling(20).mean().iloc[-1]:
                    score += 20
                    score_factors.append("High volatility")
                
                # Signal score
                score += len(signals) * 10
                if signals:
                    score_factors.append(f"{len(signals)} signals")
                
                result = {
                    'symbol': symbol,
                    'data': data_with_indicators,
                    'signals': signals,
                    'score': score,
                    'score_factors': score_factors,
                    'current_price': latest['close'],
                    'volume_ratio': latest['volume_ratio'],
                    'rsi': latest['rsi'],
                    'atr': latest['atr'],
                    'trend_strength': latest['adx'] if 'adx' in latest else 0
                }
                
                all_results.append(result)
                
            except Exception as e:
                print(f"  ❌ Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Print summary
        print(f"\n📈 TOP INTRADAY TRADING OPPORTUNITIES")
        print("=" * 120)
        print(f"{'#':<2} {'Symbol':<12} {'Score':<5} {'Price':<8} {'RSI':<5} {'Vol':<4} {'ATR':<6} {'Signals':<8} {'Factors':<40}")
        print("-" * 120)
        
        for i, result in enumerate(all_results[:20], 1):
            signals_str = f"{len(result['signals'])}"
            factors_str = ", ".join(result['score_factors'][:3])[:39]
            
            print(f"{i:<2} {result['symbol']:<12} {result['score']:<5.0f} "
                  f"₹{result['current_price']:<7.1f} {result['rsi']:<4.1f} "
                  f"{result['volume_ratio']:<3.1f}x {result['atr']:<5.2f} "
                  f"{signals_str:<8} {factors_str:<40}")
        
        # Plot top stocks
        print(f"\n📊 Generating comprehensive charts for top stocks...")
        
        for i, result in enumerate(all_results[:10], 1):  # Top 10 stocks
            print(f"📈 Plotting {result['symbol']} (Rank #{i}, Score: {result['score']})...")
            self.plot_comprehensive_chart(result['symbol'], result['data'], result['signals'], rank=i)
        
        return all_results

def main():
    """Main function to run the comprehensive dashboard."""
    print("🚀 Starting Comprehensive Intraday Trading Dashboard...")
    
    dashboard = IntradayTradingDashboard()
    
    # Analyze all stocks
    results = dashboard.analyze_all_stocks(max_stocks=50)
    
    print(f"\n✅ Analysis completed for {len(results)} stocks!")
    print("📊 Charts generated for top 10 opportunities")
    print("🎯 Use the signals and indicators to make informed trading decisions")

if __name__ == "__main__":
    main() 