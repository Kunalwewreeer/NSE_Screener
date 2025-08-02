"""
Utility functions for file operations including YAML, JSON, and CSV handling.
"""
import os
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Loaded YAML config from {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the YAML file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, indent=2)
        logger.debug(f"Saved YAML config to {file_path}")
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")
        raise


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary containing the JSON data
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        logger.debug(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        DataFrame containing the CSV data
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.debug(f"Loaded CSV from {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
        **kwargs: Additional arguments to pass to df.to_csv
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, **kwargs)
        logger.debug(f"Saved CSV to {file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {e}")
        raise


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        raise


def list_files(directory_path: str, pattern: str = "*") -> List[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory_path: Path to the directory
        pattern: File pattern to match (e.g., "*.csv", "*.json")
    
    Returns:
        List of file paths
    """
    try:
        path = Path(directory_path)
        files = list(path.glob(pattern))
        file_paths = [str(f) for f in files if f.is_file()]
        logger.debug(f"Found {len(file_paths)} files in {directory_path} matching {pattern}")
        return file_paths
    except Exception as e:
        logger.error(f"Error listing files in {directory_path}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in bytes
    """
    try:
        size = os.path.getsize(file_path)
        logger.debug(f"File {file_path} size: {size} bytes")
        return size
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        raise 