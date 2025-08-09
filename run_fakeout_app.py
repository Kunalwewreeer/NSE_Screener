#!/usr/bin/env python3
"""
Runner script for the Fakeout Detector Streamlit App
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🚀 Launching Fakeout Detector Streamlit App...")
    print("=" * 50)
    
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
    
    # Launch the app
    print("\n🌐 Starting Streamlit server...")
    print("📱 The app will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "fakeout_streamlit_app.py",
        "--server.port", "8502",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main() 