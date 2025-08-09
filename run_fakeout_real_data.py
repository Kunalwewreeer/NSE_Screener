#!/usr/bin/env python3
"""
Runner script for Fakeout Detector with Real Data Integration
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app for real data analysis."""
    print("🚀 Launching Fakeout Detector - Real Data Integration")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed successfully")
    
    # Check if core modules exist
    try:
        from core.data_handler import DataHandler
        print("✅ Data handler found")
    except ImportError:
        print("⚠️  Warning: Data handler not found. Make sure core modules are available.")
    
    # Launch the app
    print("\n🌐 Starting Streamlit server for real data analysis...")
    print("📱 The app will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8502")
    print("\n📊 Features:")
    print("   - Real data integration with your data handler")
    print("   - Multiple symbols analysis")
    print("   - Configurable detection parameters")
    print("   - Interactive visualizations")
    print("   - Signal analysis and filtering")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run streamlit on a different port to avoid conflicts
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "fakeout_real_data_app.py",
        "--server.port", "8502",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main() 