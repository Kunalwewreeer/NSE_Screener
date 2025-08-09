#!/usr/bin/env python3
"""
Test Cache Management Functionality
"""

import os
import sys
from cache_manager import CacheManager

def test_cache_management():
    """Test the cache management functionality."""
    
    print("🧪 Testing Cache Management...")
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Test cache info
    print("\n📁 Cache Information:")
    cache_info = cache_manager.get_cache_info()
    print(f"Cache directory exists: {cache_info['cache_dir_exists']}")
    print(f"Total files: {cache_info['total_files']}")
    print(f"Total size: {cache_info['total_size_mb']:.1f} MB")
    
    # Test cache clearing
    print("\n🗑️ Testing cache clearing...")
    if cache_manager.clear_cache():
        print("✅ Cache cleared successfully")
    else:
        print("❌ Failed to clear cache")
    
    # Test cache info after clearing
    print("\n📁 Cache Information (after clearing):")
    cache_info_after = cache_manager.get_cache_info()
    print(f"Cache directory exists: {cache_info_after['cache_dir_exists']}")
    print(f"Total files: {cache_info_after['total_files']}")
    print(f"Total size: {cache_info_after['total_size_mb']:.1f} MB")
    
    print("\n✅ Cache management test completed!")

if __name__ == "__main__":
    test_cache_management() 