"""
Fakeout Detector Integration with Real Data
Integrates the fakeout detection system with existing data handlers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from fakeout_detector import FakeoutDetector
from core.data_handler import DataHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeoutDetectorIntegration:
    """
    Integration class for fakeout detection with real data.
    Connects the fakeout detector with existing data handlers.
    """
    
    def __init__(self, data_handler: Optional[DataHandler] = None):
        """
        Initialize the integration with data handler.
        
        Args:
            data_handler: DataHandler instance for fetching real data
        """
        self.data_handler = data_handler or DataHandler()
        self.detector = None
        self.last_analysis = None
    
    def setup_detector(self, config: Optional[Dict] = None) -> FakeoutDetector:
        """
        Setup the fakeout detector with configuration.
        
        Args:
            config: Detector configuration dictionary
            
        Returns:
            Configured FakeoutDetector instance
        """
        default_config = {
            # Level detection
            'wick_threshold_pct': 0.3,
            'confirmation_threshold_pct': 0.5,
            'level_tolerance_pct': 0.1,
            
            # Signal parameters
            'lookback_window': 20,
            'min_candles_between_signals': 10,
            
            # Risk management
            'sl_atr_multiplier': 1.5,
            'tp_atr_multiplier': 2.0,
            'atr_period': 14,
            
            # Debug settings
            'debug_mode': True,
            'log_level': 'INFO'
        }
        
        if config:
            default_config.update(config)
        
        self.detector = FakeoutDetector(default_config)
        return self.detector
    
    def fetch_data_for_analysis(self, symbols: List[str], start_date: str, end_date: str, 
                               interval: str = "minute") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for analysis using the data handler.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (minute, 5minute, etc.)
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        try:
            # Fetch data using the data handler
            data = self.data_handler.get_historical_data(
                symbols, start_date, end_date, interval=interval
            )
            
            # Handle different return types
            if isinstance(data, dict):
                return data
            elif isinstance(data, pd.DataFrame):
                return {symbols[0]: data}
            else:
                logger.error(f"Unexpected data type: {type(data)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return {}
    
    def calculate_vwap_for_data(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP for the given data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            VWAP series
        """
        if df.empty:
            return pd.Series()
        
        # Calculate VWAP with daily reset for intraday data
        df_copy = df.copy()
        df_copy['date_only'] = df_copy.index.date
        
        # Group by date and calculate VWAP for each day
        daily_vwap = df_copy.groupby('date_only').apply(
            lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
        )
        
        # Reset index to match original DataFrame
        vwap_series = daily_vwap.reset_index(level=0, drop=True)
        
        return vwap_series
    
    def analyze_symbol(self, symbol: str, start_date: str, end_date: str, 
                      interval: str = "minute", level_type: str = "pdh_pdl", 
                      detector_config: Optional[Dict] = None) -> Dict:
        """
        Analyze a single symbol for fakeout signals.
        
        Args:
            symbol: Symbol to analyze
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (minute, 5minute, etc.)
            level_type: Type of levels to use (pdh_pdl, vwap, support_resistance)
            detector_config: Optional detector configuration
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Analyzing {symbol} for fakeout signals")
            
            # Fetch data
            data_dict = self.fetch_data_for_analysis([symbol], start_date, end_date, interval)
            
            if symbol not in data_dict or data_dict[symbol].empty:
                return {
                    'error': f"No data available for {symbol}",
                    'signals': [],
                    'date_range': {'start': start_date, 'end': end_date},
                    'interval': interval,
                    'level_type': level_type
                }
            
            df = data_dict[symbol]
            
            # Calculate VWAP
            vwap = self.calculate_vwap_for_data(df)
            
            # Setup detector
            detector = self.setup_detector(detector_config)
            
            # Detect signals
            signals = detector.detect_fakeout_signals(df, vwap, level_type)
            
            return {
                'signals': signals,
                'data_points': len(df),
                'date_range': {'start': start_date, 'end': end_date},
                'interval': interval,
                'level_type': level_type,
                'vwap_available': vwap is not None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'error': f"Error analyzing {symbol}: {str(e)}",
                'signals': [],
                'date_range': {'start': start_date, 'end': end_date},
                'interval': interval,
                'level_type': level_type
            }
    
    def analyze_multiple_symbols(self, symbols: List[str],
                               start_date: str,
                               end_date: str,
                               level_type: str = 'pdh_pdl',
                               interval: str = 'minute',
                               config: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Analyze multiple symbols for fakeout signals.
        
        Args:
            symbols: List of symbols to analyze
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            level_type: Type of levels to use (pdh_pdl, vwap, support_resistance)
            interval: Data interval (minute, 5minute, etc.)
            config: Optional detector configuration
            
        Returns:
            Dictionary with analysis results for each symbol
        """
        logger.info(f"Analyzing {len(symbols)} symbols for fakeout signals")
        
        results = {}
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol, start_date, end_date, interval, level_type, config)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {
                    'error': f"Error analyzing {symbol}: {str(e)}",
                    'signals': [],
                    'date_range': {'start': start_date, 'end': end_date},
                    'interval': interval,
                    'level_type': level_type
                }
        
        return results
    
    def get_top_signals(self, analysis_results: Dict[str, Dict], 
                        top_n: int = 5,
                        signal_type: Optional[str] = None) -> List[Dict]:
        """
        Get top signals from analysis results.
        
        Args:
            analysis_results: Results from analyze_multiple_symbols
            top_n: Number of top signals to return
            signal_type: Filter by signal type ('long_fakeout', 'short_fakeout')
            
        Returns:
            List of top signals with symbol information
        """
        all_signals = []
        
        for symbol, result in analysis_results.items():
            if 'signals' in result and result['signals']:
                for signal in result['signals']:
                    if signal_type is None or signal['signal_type'] == signal_type:
                        signal_with_symbol = signal.copy()
                        signal_with_symbol['symbol'] = symbol
                        all_signals.append(signal_with_symbol)
        
        # Sort by timestamp (most recent first)
        all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_signals[:top_n]
    
    def create_analysis_summary(self, analysis_results: Dict[str, Dict]) -> Dict:
        """
        Create a summary of analysis results.
        
        Args:
            analysis_results: Results from analyze_multiple_symbols
            
        Returns:
            Summary dictionary
        """
        total_signals = 0
        total_long = 0
        total_short = 0
        total_data_points = 0
        symbols_with_signals = 0
        
        for symbol, result in analysis_results.items():
            if 'signals' in result and result['signals']:
                signals = result['signals']
                total_signals += len(signals)
                
                # Count long and short signals
                long_signals = [s for s in signals if s['signal_type'] == 'long_fakeout']
                short_signals = [s for s in signals if s['signal_type'] == 'short_fakeout']
                
                total_long += len(long_signals)
                total_short += len(short_signals)
                symbols_with_signals += 1
            
            if 'data_points' in result:
                total_data_points += result['data_points']
        
        summary = {
            'total_symbols': len(analysis_results),
            'symbols_with_signals': symbols_with_signals,
            'total_signals': total_signals,
            'total_long_signals': total_long,
            'total_short_signals': total_short,
            'total_data_points': total_data_points,
            'analysis_date': datetime.now()
        }
        
        if total_signals > 0:
            summary['avg_signals_per_symbol'] = total_signals / symbols_with_signals
            summary['long_short_ratio'] = total_long / total_short if total_short > 0 else float('inf')
        else:
            summary['avg_signals_per_symbol'] = 0
            summary['long_short_ratio'] = 0
        
        return summary
    
    def plot_symbol_analysis(self, symbol: str, analysis_result: Dict):
        """
        Plot analysis results for a single symbol.
        
        Args:
            symbol: Symbol name
            analysis_result: Analysis result from analyze_symbol
        """
        if 'error' in analysis_result:
            logger.warning(f"Cannot plot {symbol}: {analysis_result['error']}")
            return
        
        # This would integrate with the detector's plotting functionality
        # For now, return the analysis result for external plotting
        return analysis_result


def run_fakeout_analysis(symbols: List[str],
                        start_date: str,
                        end_date: str,
                        level_type: str = 'pdh_pdl',
                        interval: str = 'minute',
                        config: Optional[Dict] = None) -> Dict:
    """
    Run fakeout analysis for multiple symbols.
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        level_type: Type of levels to use (pdh_pdl, vwap, support_resistance)
        interval: Data interval (minute, 5minute, etc.)
        config: Optional detector configuration
        
    Returns:
        Dictionary with analysis results and summary
    """
    try:
        # Create integration instance
        integration = FakeoutDetectorIntegration()
        
        # Analyze symbols
        analysis_results = integration.analyze_multiple_symbols(
            symbols, start_date, end_date, level_type, interval, config
        )
        
        # Get top signals
        top_signals = integration.get_top_signals(analysis_results)
        
        # Create summary
        summary = integration.create_analysis_summary(analysis_results)
        
        return {
            'analysis_results': analysis_results,
            'top_signals': top_signals,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error in fakeout analysis: {e}")
        return {
            'error': f"Analysis failed: {str(e)}",
            'analysis_results': {},
            'top_signals': [],
            'summary': {}
        }


def main():
    """Example usage of the integration."""
    
    # Example configuration
    symbols = ['NIFTY', 'BANKNIFTY']
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
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
    
    # Run analysis
    results = run_fakeout_analysis(
        symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 'pdh_pdl', 'minute', config
    )
    
    # Print summary
    print("\n=== FAKEOUT ANALYSIS SUMMARY ===")
    print(f"Total symbols analyzed: {results['summary']['total_symbols']}")
    print(f"Symbols with signals: {results['summary']['symbols_with_signals']}")
    print(f"Total signals: {results['summary']['total_signals']}")
    print(f"Long signals: {results['summary']['total_long_signals']}")
    print(f"Short signals: {results['summary']['total_short_signals']}")
    
    if results['top_signals']:
        print(f"\nTop 5 recent signals:")
        for i, signal in enumerate(results['top_signals'][:5]):
            print(f"{i+1}. {signal['symbol']} - {signal['signal_type']} at {signal['timestamp']}")
            print(f"   Entry: {signal['entry']:.2f}, SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}")


if __name__ == "__main__":
    main() 