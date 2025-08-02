"""
Performance metrics module for calculating trading performance indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Class for calculating various trading performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily returns from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of daily returns
        """
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of log returns
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year))
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() * periods_per_year) / (downside_returns.std() * np.sqrt(periods_per_year))
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            prices: Series of prices with datetime index
            
        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        if len(prices) == 0:
            return 0.0, None, None
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find the peak before the maximum drawdown
        peak_idx = running_max.loc[:max_dd_idx].idxmax()
        
        return max_drawdown, peak_idx, max_dd_idx
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series, 
                             periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Series of returns
            prices: Series of prices
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown(prices)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 5%)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 5%)
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate win rate from trade list.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Win rate as percentage
        """
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Profit factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_average_trade(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average trade statistics.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Dictionary with average trade metrics
        """
        if not trades:
            return {
                'avg_trade': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        pnls = [trade.get('pnl', 0) for trade in trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        return {
            'avg_trade': np.mean(pnls),
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'largest_win': max(pnls) if pnls else 0.0,
            'largest_loss': min(pnls) if pnls else 0.0
        }
    
    def calculate_recovery_factor(self, total_pnl: float, max_drawdown: float) -> float:
        """
        Calculate recovery factor (total PnL / max drawdown).
        
        Args:
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown
            
        Returns:
            Recovery factor
        """
        if max_drawdown == 0:
            return 0.0
        
        return total_pnl / abs(max_drawdown)
    
    def calculate_risk_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            prices: Series of prices
            
        Returns:
            Dictionary with risk metrics
        """
        if len(returns) == 0:
            return {}
        
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(prices)
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'var_95': self.calculate_var(returns, 0.05),
            'cvar_95': self.calculate_cvar(returns, 0.05),
            'max_drawdown': max_dd,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_skew': returns[returns > 0].skew() if len(returns[returns > 0]) > 0 else 0,
            'negative_skew': returns[returns < 0].skew() if len(returns[returns < 0]) > 0 else 0
        }
        
        return metrics
    
    def calculate_performance_metrics(self, returns: pd.Series, prices: pd.Series, 
                                    trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of returns
            prices: Series of prices
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with all performance metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic return metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annual_return = returns.mean() * 252 * 100
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(returns, prices)
        
        # Ratio metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns, prices)
        
        # Trade metrics
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        avg_trade_metrics = self.calculate_average_trade(trades)
        
        # Recovery factor
        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        recovery_factor = self.calculate_recovery_factor(total_pnl, risk_metrics.get('max_drawdown', 0))
        
        return {
            'returns': {
                'total_return_pct': total_return,
                'annual_return_pct': annual_return,
                'daily_return_mean': returns.mean(),
                'daily_return_std': returns.std()
            },
            'risk': risk_metrics,
            'ratios': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'recovery_factor': recovery_factor
            },
            'trades': {
                'total_trades': len(trades),
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                **avg_trade_metrics
            },
            'summary': {
                'total_pnl': total_pnl,
                'best_month': self._get_best_month(returns),
                'worst_month': self._get_worst_month(returns),
                'consecutive_wins': self._get_consecutive_wins(trades),
                'consecutive_losses': self._get_consecutive_losses(trades)
            }
        }
    
    def _get_best_month(self, returns: pd.Series) -> float:
        """Get best monthly return."""
        if len(returns) == 0:
            return 0.0
        
        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return 0.0
        
        monthly_returns = returns.resample('M').sum()
        return monthly_returns.max() * 100
    
    def _get_worst_month(self, returns: pd.Series) -> float:
        """Get worst monthly return."""
        if len(returns) == 0:
            return 0.0
        
        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return 0.0
        
        monthly_returns = returns.resample('M').sum()
        return monthly_returns.min() * 100
    
    def _get_consecutive_wins(self, trades: List[Dict[str, Any]]) -> int:
        """Get maximum consecutive wins."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _get_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Get maximum consecutive losses."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Formatted report string
        """
        if not metrics:
            return "No performance data available."
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Returns
        returns = metrics.get('returns', {})
        report.append(f"Total Return: {returns.get('total_return_pct', 0):.2f}%")
        report.append(f"Annual Return: {returns.get('annual_return_pct', 0):.2f}%")
        report.append(f"Daily Return (Mean): {returns.get('daily_return_mean', 0):.4f}")
        report.append(f"Daily Return (Std): {returns.get('daily_return_std', 0):.4f}")
        
        # Risk metrics
        risk = metrics.get('risk', {})
        report.append(f"\nRisk Metrics:")
        report.append(f"Volatility (Annual): {risk.get('volatility', 0):.2f}%")
        report.append(f"Max Drawdown: {risk.get('max_drawdown', 0):.2f}%")
        report.append(f"VaR (95%): {risk.get('var_95', 0):.4f}")
        report.append(f"CVaR (95%): {risk.get('cvar_95', 0):.4f}")
        
        # Ratios
        ratios = metrics.get('ratios', {})
        report.append(f"\nRisk-Adjusted Returns:")
        report.append(f"Sharpe Ratio: {ratios.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {ratios.get('sortino_ratio', 0):.2f}")
        report.append(f"Calmar Ratio: {ratios.get('calmar_ratio', 0):.2f}")
        report.append(f"Recovery Factor: {ratios.get('recovery_factor', 0):.2f}")
        
        # Trade metrics
        trades = metrics.get('trades', {})
        report.append(f"\nTrade Statistics:")
        report.append(f"Total Trades: {trades.get('total_trades', 0)}")
        report.append(f"Win Rate: {trades.get('win_rate_pct', 0):.1f}%")
        report.append(f"Profit Factor: {trades.get('profit_factor', 0):.2f}")
        report.append(f"Average Trade: ₹{trades.get('avg_trade', 0):.2f}")
        report.append(f"Average Win: ₹{trades.get('avg_win', 0):.2f}")
        report.append(f"Average Loss: ₹{trades.get('avg_loss', 0):.2f}")
        
        # Summary
        summary = metrics.get('summary', {})
        report.append(f"\nSummary:")
        report.append(f"Total PnL: ₹{summary.get('total_pnl', 0):.2f}")
        report.append(f"Best Month: {summary.get('best_month', 0):.2f}%")
        report.append(f"Worst Month: {summary.get('worst_month', 0):.2f}%")
        report.append(f"Max Consecutive Wins: {summary.get('consecutive_wins', 0)}")
        report.append(f"Max Consecutive Losses: {summary.get('consecutive_losses', 0)}")
        
        report.append("=" * 60)
        
        return "\n".join(report) 