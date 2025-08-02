#!/usr/bin/env python3
"""
Launch the Stock Screener Dashboard
"""

import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_screener import create_screening_dashboard

if __name__ == "__main__":
    create_screening_dashboard()