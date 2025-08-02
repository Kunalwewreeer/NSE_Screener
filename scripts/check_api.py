#!/usr/bin/env python3
"""
Script to check Zerodha API response and get proper symbol list.
This will help verify API connectivity and get accurate instrument data.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from token_manager import get_kite_instance, is_token_valid, get_login_url
from utils.logger import get_logger

logger = get_logger(__name__)

def check_api_connectivity():
    """Check if we can connect to Zerodha API."""
    print("=" * 60)
    print("CHECKING ZERODHA API CONNECTIVITY")
    print("=" * 60)
    
    try:
        # Check if token is valid
        if not is_token_valid():
            print("❌ Token is not valid or expired!")
            print(f"🔗 Get login URL: {get_login_url()}")
            print("Please generate a new token using: python3 token_manager.py")
            return False
        
        print("✅ Token is valid!")
        
        # Get kite instance
        kite = get_kite_instance()
        
        # Test basic API call
        profile = kite.profile()
        print(f"✅ API connection successful!")
        print(f"👤 User: {profile.get('user_name', 'N/A')}")
        print(f"📧 Email: {profile.get('email', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def get_instruments_info():
    """Get detailed information about available instruments."""
    print("\n" + "=" * 60)
    print("FETCHING INSTRUMENTS INFORMATION")
    print("=" * 60)
    
    try:
        kite = get_kite_instance()
        
        # Get instruments for different exchanges
        exchanges = ["NSE", "BSE", "NFO", "CDS"]
        
        for exchange in exchanges:
            print(f"\n📊 {exchange} Exchange:")
            try:
                instruments = kite.instruments(exchange=exchange)
                df = pd.DataFrame(instruments)
                
                print(f"   Total instruments: {len(df)}")
                
                if not df.empty:
                    # Show unique instrument types
                    if 'instrument_type' in df.columns:
                        types = df['instrument_type'].value_counts()
                        print(f"   Instrument types: {dict(types.head())}")
                    
                    # Show unique segments
                    if 'segment' in df.columns:
                        segments = df['segment'].value_counts()
                        print(f"   Segments: {dict(segments.head())}")
                    
                    # For NSE, show some sample stocks
                    if exchange == "NSE":
                        equity_stocks = df[df['instrument_type'] == 'EQ'].head(10)
                        print(f"   Sample equity stocks:")
                        for _, stock in equity_stocks.iterrows():
                            print(f"     {stock['tradingsymbol']} ({stock['name']}) - Token: {stock['instrument_token']}")
                
            except Exception as e:
                print(f"   ❌ Error fetching {exchange} instruments: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching instruments: {e}")
        return False

def get_nifty50_instruments():
    """Get Nifty 50 instruments with proper tokens."""
    print("\n" + "=" * 60)
    print("FETCHING NIFTY 50 INSTRUMENTS")
    print("=" * 60)
    
    try:
        kite = get_kite_instance()
        
        # Get NSE instruments
        instruments = kite.instruments(exchange="NSE")
        df = pd.DataFrame(instruments)
        
        # Filter for equity instruments
        equity_df = df[df['instrument_type'] == 'EQ'].copy()
        
        # Nifty 50 stock names (without .NS suffix)
        nifty50_names = [
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL",
            "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
            "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
            "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
            "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
            "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
            "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO",
            "UPL", "WIPRO"
        ]
        
        # Find Nifty 50 stocks in the instruments
        nifty50_instruments = []
        found_symbols = []
        
        for name in nifty50_names:
            matches = equity_df[equity_df['tradingsymbol'] == name]
            if not matches.empty:
                instrument = matches.iloc[0]
                nifty50_instruments.append({
                    'symbol': f"{name}.NS",
                    'tradingsymbol': name,
                    'name': instrument['name'],
                    'instrument_token': instrument['instrument_token'],
                    'lot_size': instrument['lot_size'],
                    'tick_size': instrument['tick_size']
                })
                found_symbols.append(name)
            else:
                print(f"⚠️  Not found: {name}")
        
        print(f"✅ Found {len(nifty50_instruments)} Nifty 50 stocks")
        print(f"📋 Found symbols: {found_symbols}")
        
        # Create DataFrame and save to CSV
        nifty50_df = pd.DataFrame(nifty50_instruments)
        csv_path = "nifty50_instruments.csv"
        nifty50_df.to_csv(csv_path, index=False)
        print(f"💾 Saved to: {csv_path}")
        
        # Display sample
        print(f"\n📊 Sample Nifty 50 instruments:")
        print(nifty50_df.head(10).to_string(index=False))
        
        return nifty50_df
        
    except Exception as e:
        print(f"❌ Error fetching Nifty 50 instruments: {e}")
        return pd.DataFrame()

def test_historical_data_fetching():
    """Test historical data fetching for a few symbols."""
    print("\n" + "=" * 60)
    print("TESTING HISTORICAL DATA FETCHING")
    print("=" * 60)
    
    try:
        kite = get_kite_instance()
        
        # Test symbols (using instrument tokens)
        test_symbols = [
            {"name": "RELIANCE", "token": 2885},  # RELIANCE token
            {"name": "TCS", "token": 11536},      # TCS token
            {"name": "INFY", "token": 1594}       # INFY token
        ]
        
        start_date = "2024-01-01"
        end_date = "2024-01-05"
        
        for symbol_info in test_symbols:
            print(f"\n📈 Testing {symbol_info['name']} (Token: {symbol_info['token']}):")
            
            try:
                # Fetch historical data
                candles = kite.historical_data(
                    instrument_token=symbol_info['token'],
                    from_date=start_date,
                    to_date=end_date,
                    interval="day"
                )
                
                if candles:
                    df = pd.DataFrame(candles)
                    print(f"   ✅ Success! Fetched {len(df)} records")
                    print(f"   📊 Data shape: {df.shape}")
                    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
                    print(f"   💰 Price range: ₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}")
                    print(f"   📈 Sample data:")
                    print(df.head(3).to_string(index=False))
                else:
                    print(f"   ⚠️  No data returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing historical data: {e}")
        return False

def create_improved_data_handler():
    """Create an improved data handler with proper API references."""
    print("\n" + "=" * 60)
    print("CREATING IMPROVED DATA HANDLER")
    print("=" * 60)
    
    # Load Nifty 50 instruments
    try:
        nifty50_df = pd.read_csv("nifty50_instruments.csv")
        print(f"✅ Loaded {len(nifty50_df)} Nifty 50 instruments")
        
        # Create symbol to token mapping
        symbol_to_token = dict(zip(nifty50_df['symbol'], nifty50_df['instrument_token']))
        
        print(f"📋 Symbol to token mapping created")
        print(f"🔍 Sample mappings:")
        for i, (symbol, token) in enumerate(list(symbol_to_token.items())[:5]):
            print(f"   {symbol} -> {token}")
        
        return symbol_to_token
        
    except Exception as e:
        print(f"❌ Error creating improved data handler: {e}")
        return {}

def main():
    """Main function to run all checks."""
    print("🚀 ZERODHA API DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check API connectivity
    if not check_api_connectivity():
        print("\n❌ Cannot proceed without valid API connection")
        return
    
    # Get instruments info
    get_instruments_info()
    
    # Get Nifty 50 instruments
    nifty50_df = get_nifty50_instruments()
    
    # Test historical data fetching
    test_historical_data_fetching()
    
    # Create improved data handler
    symbol_to_token = create_improved_data_handler()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ API connectivity: Working")
    print(f"✅ Nifty 50 instruments: {len(nifty50_df)} found")
    print(f"✅ Symbol mappings: {len(symbol_to_token)} created")
    print("\n🎯 Next steps:")
    print("1. Use the generated nifty50_instruments.csv for accurate symbol mapping")
    print("2. Update data_handler.py to use proper instrument tokens")
    print("3. Test with real symbols from the instrument list")

if __name__ == "__main__":
    main() 