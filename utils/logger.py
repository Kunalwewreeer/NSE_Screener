"""
Unified logging system for the trading platform.
"""
import os
import sys
from datetime import datetime
from loguru import logger
import yaml
from pathlib import Path


def setup_logger(config_path: str = "config/config.yaml"):
    """
    Setup the logging system based on configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default logging settings.")
        config = {
            'logging': {
                'level': 'INFO',
                'file': 'logs/trading.log',
                'max_size': '10MB',
                'backup_count': 5
            }
        }
    
    # Remove default logger
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_file = config['logging']['file']
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add console logger
    logger.add(
        sys.stdout,
        level=config['logging']['level'],
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger with rotation
    logger.add(
        log_file,
        level=config['logging']['level'],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=config['logging']['max_file_size'],
        retention=config['logging']['backup_count'],
        compression="zip"
    )
    
    logger.info("Logging system initialized")
    return logger


def get_logger(name: str = None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (usually module name)
    
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger when module is imported
setup_logger() 