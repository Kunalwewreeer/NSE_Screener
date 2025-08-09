# src/generate_token.py
import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()
kite = KiteConnect(api_key=os.getenv("API_KEY"))
data = kite.generate_session("P7XWaNZQcbMC9yzYqVBq6kxg11tQHmmO", api_secret=os.getenv("API_SECRET"))

print("Access Token:", data["access_token"])
