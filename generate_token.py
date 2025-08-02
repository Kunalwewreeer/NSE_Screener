# src/generate_token.py
import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()
kite = KiteConnect(api_key=os.getenv("API_KEY"))
data = kite.generate_session("Uw7N9Gd3F6yPtodsoPG63IKD11FNhVdy", api_secret=os.getenv("API_SECRET"))

print("Access Token:", data["access_token"])
