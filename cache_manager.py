"""
Cache Management Utility for Live Deployment
Handles cache refresh and data completeness validation
"""

import os
import shutil
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

class CacheManager:
    """Manages cache operations for live deployment."""
    
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    
    def clear_cache(self) -> bool:
        """Clear all cached data."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                return True
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        info = {
            'cache_dir_exists': os.path.exists(self.cache_dir),
            'cache_files': [],
            'total_files': 0,
            'total_size_mb': 0
        }
        
        if info['cache_dir_exists']:
            files = os.listdir(self.cache_dir)
            info['cache_files'] = files
            info['total_files'] = len(files)
            
            total_size = 0
            for file in files:
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info
    
    def validate_data_completeness(self, df: pd.DataFrame, min_data_points: int = 30) -> Dict:
        """Validate data completeness for analysis."""
        if df is None or df.empty:
            return {
                'is_complete': False,
                'data_points': 0,
                'completeness_ratio': 0.0,
                'message': 'No data available'
            }
        
        data_points = len(df)
        completeness_ratio = data_points / min_data_points if min_data_points > 0 else 1.0
        
        return {
            'is_complete': data_points >= min_data_points,
            'data_points': data_points,
            'completeness_ratio': completeness_ratio,
            'message': f'{data_points} points (minimum {min_data_points} recommended)'
        }

def add_cache_management_to_sidebar():
    """Add cache management controls to sidebar."""
    st.sidebar.subheader("🗄️ Cache Management")
    
    cache_manager = CacheManager()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        refresh_cache = st.button("🔄 Refresh Cache", type="secondary", help="Force refresh all cached data")
    with col2:
        clear_cache = st.button("🗑️ Clear Cache", type="secondary", help="Clear all cached data")
    
    if refresh_cache:
        st.sidebar.success("✅ Cache refresh initiated! Data will be re-downloaded from API.")
        st.session_state['refresh_cache'] = True
    elif clear_cache:
        if cache_manager.clear_cache():
            st.sidebar.success("✅ Cache cleared! Next run will download fresh data.")
        else:
            st.sidebar.error("❌ Failed to clear cache")
    
    # Show cache status
    cache_info = cache_manager.get_cache_info()
    if cache_info['cache_dir_exists']:
        st.sidebar.info(f"📁 Cache Status: {cache_info['total_files']} files ({cache_info['total_size_mb']:.1f} MB)")
    else:
        st.sidebar.info("📁 Cache Status: No cache directory found")
    
    return cache_manager

def add_data_completeness_settings():
    """Add data completeness settings to sidebar."""
    st.sidebar.subheader("📊 Data Completeness")
    
    min_data_points = st.sidebar.slider(
        "Min Data Points Required",
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of data points required for analysis"
    )
    
    allow_incomplete_data = st.sidebar.checkbox(
        "Allow Incomplete Data",
        value=True,
        help="Allow screening with incomplete data (useful for live trading)"
    )
    
    if allow_incomplete_data:
        st.sidebar.info("✅ Will proceed with available data even if incomplete")
    else:
        st.sidebar.warning("⚠️ Will skip stocks with incomplete data")
    
    return min_data_points, allow_incomplete_data

def validate_screening_data(screened_df: pd.DataFrame, min_data_points: int, allow_incomplete_data: bool):
    """Validate screening data and provide user feedback."""
    if screened_df.empty:
        st.error("❌ No stocks found with sufficient data. Try:")
        st.error("• Reducing 'Min Data Points Required'")
        st.error("• Enabling 'Allow Incomplete Data'")
        st.error("• Refreshing cache to get latest data")
        st.error("• Using a different date (yesterday recommended)")
        return False
    
    # Check data completeness
    if 'data_completeness' in screened_df.columns:
        avg_completeness = screened_df['data_completeness'].mean()
        if avg_completeness < 0.8:
            st.warning(f"⚠️ Average data completeness is {avg_completeness:.1%}. Consider refreshing cache.")
    
    return True 