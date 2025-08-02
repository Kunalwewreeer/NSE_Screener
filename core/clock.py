"""
Unified time handler for backtest and live trading synchronization.
"""
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, List
from utils.logger import get_logger

logger = get_logger(__name__)


class Clock:
    """
    Unified time handler for managing trading time in backtest and live modes.
    """
    
    def __init__(self, mode: str = "live", start_time: Optional[datetime] = None, 
                 end_time: Optional[datetime] = None, speed: float = 1.0):
        """
        Initialize the clock.
        
        Args:
            mode: 'live' or 'backtest'
            start_time: Start time for backtest
            end_time: End time for backtest
            speed: Speed multiplier for backtest (1.0 = real time)
        """
        self.mode = mode
        self.speed = speed
        self.current_time = start_time or datetime.now()
        self.start_time = start_time
        self.end_time = end_time
        
        # Market hours (IST: 9:15 AM to 3:30 PM, Monday to Friday)
        self.market_start = "09:15"
        self.market_end = "15:30"
        
        # Callbacks for time events
        self.on_market_open_callbacks: List[Callable] = []
        self.on_market_close_callbacks: List[Callable] = []
        self.on_time_update_callbacks: List[Callable] = []
        
        logger.info(f"Initialized {mode} clock with speed {speed}x")
    
    def add_market_open_callback(self, callback: Callable) -> None:
        """Add callback for market open event."""
        self.on_market_open_callbacks.append(callback)
    
    def add_market_close_callback(self, callback: Callable) -> None:
        """Add callback for market close event."""
        self.on_market_close_callbacks.append(callback)
    
    def add_time_update_callback(self, callback: Callable) -> None:
        """Add callback for time update event."""
        self.on_time_update_callbacks.append(callback)
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is open at given time.
        
        Args:
            check_time: Time to check (defaults to current time)
            
        Returns:
            True if market is open
        """
        if check_time is None:
            check_time = self.current_time
        
        # Check if it's a weekday
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours
        time_str = check_time.strftime("%H:%M")
        return self.market_start <= time_str <= self.market_end
    
    def is_market_open_today(self) -> bool:
        """Check if market is open today."""
        return self.current_time.weekday() < 5
    
    def get_market_open_time(self, date: Optional[datetime] = None) -> datetime:
        """Get market open time for a given date."""
        if date is None:
            date = self.current_time
        
        return date.replace(hour=9, minute=15, second=0, microsecond=0)
    
    def get_market_close_time(self, date: Optional[datetime] = None) -> datetime:
        """Get market close time for a given date."""
        if date is None:
            date = self.current_time
        
        return date.replace(hour=15, minute=30, second=0, microsecond=0)
    
    def get_next_market_open(self) -> datetime:
        """Get next market open time."""
        now = self.current_time
        
        # If it's weekend, move to next Monday
        if now.weekday() >= 5:
            days_ahead = 7 - now.weekday()
            next_monday = now + timedelta(days=days_ahead)
            return self.get_market_open_time(next_monday)
        
        # If market is closed today, return next day's open
        if not self.is_market_open():
            if now.time() > datetime.strptime(self.market_end, "%H:%M").time():
                # Market closed for today, return tomorrow's open
                tomorrow = now + timedelta(days=1)
                return self.get_market_open_time(tomorrow)
            else:
                # Market hasn't opened yet today
                return self.get_market_open_time(now)
        
        return self.get_market_open_time(now)
    
    def get_time_until_market_open(self) -> timedelta:
        """Get time until next market open."""
        next_open = self.get_next_market_open()
        return next_open - self.current_time
    
    def advance_time(self, delta: timedelta) -> None:
        """
        Advance time by given delta (for backtest mode).
        
        Args:
            delta: Time delta to advance
        """
        if self.mode != "backtest":
            logger.warning("advance_time() only works in backtest mode")
            return
        
        self.current_time += delta
        
        # Check for market open/close events
        self._check_market_events()
        
        # Call time update callbacks
        for callback in self.on_time_update_callbacks:
            try:
                callback(self.current_time)
            except Exception as e:
                logger.error(f"Error in time update callback: {e}")
    
    def _check_market_events(self) -> None:
        """Check for market open/close events and trigger callbacks."""
        time_str = self.current_time.strftime("%H:%M")
        
        # Check for market open
        if time_str == self.market_start:
            logger.info("Market opened")
            for callback in self.on_market_open_callbacks:
                try:
                    callback(self.current_time)
                except Exception as e:
                    logger.error(f"Error in market open callback: {e}")
        
        # Check for market close
        if time_str == self.market_end:
            logger.info("Market closed")
            for callback in self.on_market_close_callbacks:
                try:
                    callback(self.current_time)
                except Exception as e:
                    logger.error(f"Error in market close callback: {e}")
    
    def sleep(self, seconds: float) -> None:
        """
        Sleep for given seconds (respects speed multiplier in backtest).
        
        Args:
            seconds: Seconds to sleep
        """
        if self.mode == "backtest":
            # In backtest, advance time instead of sleeping
            self.advance_time(timedelta(seconds=seconds))
        else:
            # In live mode, sleep normally
            time.sleep(seconds / self.speed)
    
    def wait_until_market_open(self) -> None:
        """Wait until market opens."""
        if self.is_market_open():
            logger.info("Market is already open")
            return
        
        time_until_open = self.get_time_until_market_open()
        logger.info(f"Waiting {time_until_open} until market opens")
        
        if self.mode == "backtest":
            self.advance_time(time_until_open)
        else:
            time.sleep(time_until_open.total_seconds())
    
    def wait_until_market_close(self) -> None:
        """Wait until market closes."""
        if not self.is_market_open():
            logger.info("Market is already closed")
            return
        
        close_time = self.get_market_close_time()
        time_until_close = close_time - self.current_time
        
        if time_until_close.total_seconds() <= 0:
            logger.info("Market is already closed")
            return
        
        logger.info(f"Waiting {time_until_close} until market closes")
        
        if self.mode == "backtest":
            self.advance_time(time_until_close)
        else:
            time.sleep(time_until_close.total_seconds())
    
    def get_current_time(self) -> datetime:
        """Get current time."""
        return self.current_time
    
    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        if self.start_time is None:
            return timedelta(0)
        return self.current_time - self.start_time
    
    def get_remaining_time(self) -> Optional[timedelta]:
        """Get remaining time until end (for backtest)."""
        if self.end_time is None:
            return None
        return self.end_time - self.current_time
    
    def is_finished(self) -> bool:
        """Check if backtest is finished."""
        if self.mode != "backtest" or self.end_time is None:
            return False
        return self.current_time >= self.end_time
    
    def get_market_status(self) -> str:
        """Get current market status."""
        if not self.is_market_open_today():
            return "weekend"
        elif not self.is_market_open():
            time_str = self.current_time.strftime("%H:%M")
            if time_str < self.market_start:
                return "pre_market"
            else:
                return "after_market"
        else:
            return "open"
    
    def __str__(self) -> str:
        """String representation of the clock."""
        status = self.get_market_status()
        return f"Clock({self.mode}, {self.current_time}, {status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the clock."""
        return f"Clock(mode='{self.mode}', current_time={self.current_time}, speed={self.speed})" 