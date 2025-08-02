#!/usr/bin/env python3
"""
Token manager for Zerodha Kite API with automatic token refresh.
Handles API key, secret, and access token management.
"""

import os
import json
import time
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

class TokenManager:
    """Manages Zerodha Kite API tokens with automatic refresh."""
    
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.token_expiry = None
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API_KEY and API_SECRET must be set in environment variables")
        
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Set access token if available
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            logger.info("Access token loaded from environment")
    
    def get_login_url(self):
        """Get login URL for manual token generation."""
        return self.kite.login_url()
    
    def set_access_token(self, access_token):
        """Set access token manually."""
        self.access_token = access_token
        self.kite.set_access_token(access_token)
        logger.info("Access token set manually")
    
    def refresh_token(self):
        """Refresh the access token using refresh token."""
        try:
            # Get refresh token from environment
            refresh_token = os.getenv("REFRESH_TOKEN")
            if not refresh_token:
                logger.error("REFRESH_TOKEN not found in environment variables")
                return False
            
            # Generate new access token
            data = self.kite.renew_access_token(refresh_token=refresh_token)
            new_access_token = data["access_token"]
            
            # Update token
            self.access_token = new_access_token
            self.kite.set_access_token(new_access_token)
            
            # Update environment variable
            os.environ["ACCESS_TOKEN"] = new_access_token
            
            logger.info("Access token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return False
    
    def is_token_valid(self):
        """Check if current access token is valid."""
        try:
            # Try to get user profile (lightweight API call)
            profile = self.kite.profile()
            return True
        except Exception as e:
            if "token" in str(e).lower() or "expired" in str(e).lower():
                return False
            # Other errors might not be token-related
            return True
    
    def ensure_valid_token(self):
        """Ensure we have a valid access token, refresh if needed."""
        if not self.access_token:
            logger.error("No access token available. Please generate one manually.")
            return False
        
        if not self.is_token_valid():
            logger.info("Token appears to be expired, attempting refresh...")
            if not self.refresh_token():
                logger.error("Failed to refresh token. Please generate a new one manually.")
                return False
        
        return True
    
    def get_kite_instance(self):
        """Get Kite instance with valid token."""
        if self.ensure_valid_token():
            return self.kite
        else:
            raise Exception("Unable to get valid Kite instance. Please check your tokens.")

# Global token manager instance
_token_manager = None

def get_kite_instance():
    """Get a Kite instance with automatic token management."""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager()
    
    return _token_manager.get_kite_instance()

def refresh_token():
    """Manually refresh the access token."""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager()
    
    return _token_manager.refresh_token()

def get_login_url():
    """Get login URL for manual token generation."""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager()
    
    return _token_manager.get_login_url()

def set_access_token(access_token):
    """Set access token manually."""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager()
    
    _token_manager.set_access_token(access_token)

def is_token_valid():
    """Check if current token is valid."""
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager()
    
    return _token_manager.is_token_valid()

# Utility functions for token generation
def generate_tokens():
    """Interactive token generation."""
    print("=" * 60)
    print("ZERODHA KITE TOKEN GENERATION")
    print("=" * 60)
    
    # Get login URL
    login_url = get_login_url()
    print(f"\n1. Visit this URL to login: {login_url}")
    print("\n2. After login, you'll be redirected to a URL like:")
    print("   https://your-app.com/?action=login&status=success&request_token=XXXXX")
    print("\n3. Copy the 'request_token' from the URL")
    
    # Get request token from user
    request_token = input("\nEnter the request_token: ").strip()
    
    if not request_token:
        print("No request token provided. Exiting.")
        return
    
    try:
        # Generate session
        global _token_manager
        if _token_manager is None:
            _token_manager = TokenManager()
        
        data = _token_manager.kite.generate_session(request_token, api_secret=_token_manager.api_secret)
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        
        print("\n" + "=" * 60)
        print("TOKENS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Access Token: {access_token}")
        print(f"Refresh Token: {refresh_token}")
        print("\nAdd these to your .env file:")
        print(f"ACCESS_TOKEN={access_token}")
        print(f"REFRESH_TOKEN={refresh_token}")
        
        # Update environment
        os.environ["ACCESS_TOKEN"] = access_token
        os.environ["REFRESH_TOKEN"] = refresh_token
        
        # Set in token manager
        _token_manager.set_access_token(access_token)
        
        print("\nTokens have been set in the current session!")
        
    except Exception as e:
        print(f"Error generating tokens: {e}")
        print("Please check your API credentials and try again.")

if __name__ == "__main__":
    # Interactive token generation
    generate_tokens() 