#!/usr/bin/env python3
"""
Helper utility functions for the trading system.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

def load_yaml(file_path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}

def save_yaml(data: Dict, file_path: str) -> bool:
    """Save data to YAML file."""
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving YAML file {file_path}: {e}")
        return False

def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {}

def save_json(data: Dict, file_path: str) -> bool:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def save_csv(data: pd.DataFrame, file_path: str) -> bool:
    """Save DataFrame to CSV file."""
    try:
        data.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving CSV file {file_path}: {e}")
        return False

def ensure_directory(directory_path: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series."""
    return prices.pct_change().dropna()

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns."""
    return (1 + returns).cumprod() - 1

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    drawdown = calculate_drawdown(equity_curve)
    return drawdown.min()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Calculate Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Calculate Sortino ratio."""
    if returns.empty:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

def calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from trades."""
    if not trades:
        return 0
    
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    return winning_trades / len(trades)

def calculate_profit_factor(trades: List[Dict]) -> float:
    """Calculate profit factor from trades."""
    if not trades:
        return 0
    
    gross_profit = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
    gross_loss = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')

def format_currency(amount: float) -> str:
    """Format amount as Indian currency."""
    return f"₹{amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"

def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """Get list of trading days between start and end date."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days (excluding weekends)
    business_days = pd.bdate_range(start=start, end=end)
    
    # Convert to string format
    return [day.strftime('%Y-%m-%d') for day in business_days]

def is_market_open() -> bool:
    """Check if market is currently open (simplified)."""
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's within market hours (9:15 AM to 3:30 PM IST)
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end

def validate_symbol(symbol: str) -> bool:
    """Validate if symbol format is correct."""
    # Basic validation for Indian equity symbols
    if not symbol or '.' not in symbol:
        return False
    
    # Check if it ends with .NS (NSE)
    if not symbol.endswith('.NS'):
        return False
    
    # Check if the symbol part is not empty
    symbol_part = symbol.replace('.NS', '')
    if not symbol_part or len(symbol_part) < 2:
        return False
    
    return True

def clean_symbol(symbol: str) -> str:
    """Clean and standardize symbol format."""
    # Remove extra spaces and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Add .NS suffix if not present
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    
    return symbol

def get_symbol_list(symbols: List[str]) -> List[str]:
    """Clean and validate list of symbols."""
    cleaned_symbols = []
    for symbol in symbols:
        cleaned_symbol = clean_symbol(symbol)
        if validate_symbol(cleaned_symbol):
            cleaned_symbols.append(cleaned_symbol)
        else:
            print(f"Warning: Invalid symbol format: {symbol}")
    
    return cleaned_symbols

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple DataFrames with same index."""
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Merge all DataFrames
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, left_index=True, right_index=True, how='outer')
    
    return merged

def resample_data(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample data to different interval."""
    if data.empty:
        return data
    
    # Define resample rules
    rules = {
        'minute': '1T',
        '5minute': '5T',
        '15minute': '15T',
        '30minute': '30T',
        '60minute': '1H',
        'day': 'D'
    }
    
    rule = rules.get(interval)
    if not rule:
        return data
    
    # Resample OHLCV data
    resampled = data.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators."""
    if data.empty:
        return data
    
    df = data.copy()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def calculate_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    returns = data['close'].pct_change().dropna()
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta relative to market."""
    if returns.empty or market_returns.empty:
        return 0
    
    # Align the series
    aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
    if len(aligned_data) < 30:  # Need at least 30 observations
        return 0
    
    returns_aligned = aligned_data.iloc[:, 0]
    market_aligned = aligned_data.iloc[:, 1]
    
    # Calculate covariance and variance
    covariance = returns_aligned.cov(market_aligned)
    market_variance = market_aligned.var()
    
    return covariance / market_variance if market_variance > 0 else 0

def calculate_correlation_matrix(returns_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate correlation matrix for multiple assets."""
    if not returns_data:
        return pd.DataFrame()
    
    # Create DataFrame from returns
    df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    return correlation_matrix

def calculate_portfolio_metrics(weights: List[float], returns_data: Dict[str, pd.Series]) -> Dict:
    """Calculate portfolio metrics given weights and returns."""
    if not weights or not returns_data:
        return {}
    
    # Create DataFrame from returns
    df = pd.DataFrame(returns_data)
    
    # Calculate portfolio returns
    portfolio_returns = (df * weights).sum(axis=1)
    
    # Calculate metrics
    metrics = {
        'total_return': (1 + portfolio_returns).prod() - 1,
        'annualized_return': portfolio_returns.mean() * 252,
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
        'max_drawdown': calculate_max_drawdown((1 + portfolio_returns).cumprod()),
        'var_95': portfolio_returns.quantile(0.05),
        'cvar_95': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
    }
    
    return metrics 