"""
Unified data handler for loading historical and live market data.
Supports Zerodha Kite API, CSV files, and in-memory data feeds.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import pickle
import hashlib
from typing import Dict, List, Optional, Union
from kiteconnect import KiteConnect
from utils.logger import get_logger
from utils.file_utils import load_yaml
from token_manager import get_kite_instance

logger = get_logger(__name__)

# Get kite instance with automatic token management
kite = get_kite_instance()

_instruments_cache = None

def get_cached_instruments(refresh=False):
    """Get cached instruments from the generated CSV file."""
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'nifty50_instruments.csv')
    if not refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path).to_dict('records')
    
    # If CSV doesn't exist, fetch from API
    try:
        df = pd.DataFrame(kite.instruments("NSE"))
        # Filter for Nifty 50 stocks
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
        
        equity_df = df[df['instrument_type'] == 'EQ'].copy()
        nifty50_instruments = []
        
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
        
        nifty50_df = pd.DataFrame(nifty50_instruments)
        nifty50_df.to_csv(cache_path, index=False)
        return nifty50_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []

def get_instrument_token(symbol):
    """Get instrument token for a given symbol using proper API reference."""
    global _instruments_cache
    if _instruments_cache is None:
        _instruments_cache = get_cached_instruments()
    
    # Try exact match first
    for inst in _instruments_cache:
        if inst['symbol'] == symbol:
            return inst['instrument_token']
    
    # Try without .NS suffix
    symbol_clean = symbol.replace('.NS', '')
    for inst in _instruments_cache:
        if inst['tradingsymbol'] == symbol_clean:
            return inst['instrument_token']
    
    # If not found in cache, try direct API call
    try:
        instruments = kite.instruments(exchange="NSE")
        for inst in instruments:
            if inst['tradingsymbol'] == symbol_clean and inst['instrument_type'] == 'EQ':
                return inst['instrument_token']
    except Exception as e:
        logger.error(f"Error fetching instrument token from API: {e}")
    
    raise ValueError(f"Instrument token not found for symbol: {symbol}")

def get_cache_key(symbols, start, end, interval):
    symbols_str = ','.join(sorted(symbols))
    key_str = f"{symbols_str}_{start}_{end}_{interval}"
    return hashlib.md5(key_str.encode()).hexdigest()

def fetch_single_symbol_data(symbol, start_dt, end_dt, interval, max_days):
    try:
        token = get_instrument_token(symbol)
        dfs = []
        chunk_start = start_dt
        
        # Adjust chunk size based on interval for better API handling
        if interval == "minute":
            # For 1-minute data, use smaller chunks to avoid API limits
            chunk_days = min(max_days, 1)  # 1 day chunks for minute data
        elif interval == "5minute":
            chunk_days = min(max_days, 3)  # 3 day chunks for 5-minute data
        else:
            chunk_days = max_days  # Default for daily data
        
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)
            try:
                candles = kite.historical_data(
                    token,
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    interval
                )
            except Exception as api_error:
                logger.error(f"API error for {symbol}: {api_error}")
                if "Insufficient permission" in str(api_error) or "token" in str(api_error).lower():
                    logger.error(f"Token error detected. Please manually refresh your token.")
                return symbol, None
            
            if candles:
                df = pd.DataFrame(candles)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                dfs.append(df)
            
            chunk_start = chunk_end
            
            # Add small delay for minute data to avoid rate limiting
            if interval in ["minute", "5minute"]:
                import time
                time.sleep(0.1)
        
        if dfs:
            combined_df = pd.concat(dfs).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in combined_df.columns for col in required_cols):
                combined_df = combined_df[required_cols]
                logger.info(f"Fetched {len(combined_df)} records for {symbol} ({interval} interval)")
                return symbol, combined_df
            else:
                logger.warning(f"{symbol} missing required columns. Available: {combined_df.columns.tolist()}")
                return symbol, None
        return symbol, None
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return symbol, None

def fetch_data(symbols, start, end, interval="5minute", refresh_cache=False):
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = get_cache_key(symbols, start, end, interval)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    if not refresh_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded data from cache: {len(cached_data)} symbols")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    if interval == "minute":
        max_days = 60
    elif interval == "5minute":
        max_days = 100
    elif interval == "day":
        max_days = 2000
    else:
        max_days = 100
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    all_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_symbol_data, symbol, start_dt, end_dt, interval, max_days): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, data = future.result()
            if data is not None:
                all_data[symbol] = data
    if all_data:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_data, f)
            logger.info(f"Cached data for {len(all_data)} symbols")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    if not all_data:
        raise ValueError("No valid OHLCV data found for any symbol. Check API response and symbol list.")
    if len(symbols) == 1:
        return all_data[symbols[0]]
    else:
        return all_data



class DataHandler:
    """
    Unified interface for loading historical or live market data.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data handler.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml(config_path)
        self.kite = kite
        
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        from_date: str,
        to_date: str,
        interval: str = "day",
        refresh_cache: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch historical OHLCV data for one or more symbols using robust, cached, parallelized fetching.
        Args:
            symbols: Symbol or list of symbols (e.g., 'RELIANCE.NS')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            interval: Data interval (minute, 5minute, day, etc.)
            refresh_cache: If True, force re-download from API
        Returns:
            DataFrame (single symbol) or dict of DataFrames (multiple symbols)
        Raises:
            ValueError if data cannot be fetched
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        return fetch_data(symbols, from_date, to_date, interval, refresh_cache=refresh_cache)
    
    def get_live_data(self, instrument_tokens: List[Union[int, str]]) -> Dict[str, Dict]:
        """
        Get live market data for given instruments.
        
        Args:
            instrument_tokens: List of instrument tokens
            
        Returns:
            Dictionary with live data for each instrument
        """
        if not self.kite:
            logger.error("Kite connection not initialized")
            return {}
            
        try:
            # Get live data
            live_data = self.kite.ltp(instrument_tokens)
            
            logger.debug(f"Fetched live data for {len(live_data)} instruments")
            return live_data
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return {}
    
    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """
        Get list of instruments from Zerodha.
        
        Args:
            exchange: Exchange name (NSE, BSE, etc.)
            
        Returns:
            DataFrame with instrument details
        """
        if not self.kite:
            logger.error("Kite connection not initialized")
            return pd.DataFrame()
            
        try:
            instruments = self.kite.instruments(exchange=exchange)
            df = pd.DataFrame(instruments)
            
            logger.info(f"Loaded {len(df)} instruments from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching instruments from {exchange}: {e}")
            return pd.DataFrame()
    
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with market data
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            
            # Ensure date column is properly formatted
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            logger.info(f"Loaded data from CSV: {file_path}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the CSV file
            **kwargs: Additional arguments for df.to_csv
        """
        try:
            df.to_csv(file_path, **kwargs)
            logger.info(f"Saved data to CSV: {file_path}")
        except Exception as e:
            logger.error(f"Error saving CSV file {file_path}: {e}")
    
    def get_nifty50_stocks(self) -> List[str]:
        """
        Get list of Nifty 50 stocks with proper NSE suffixes.
        
        Returns:
            List of Nifty 50 stock symbols with .NS suffix
        """
        try:
            # Load from the generated CSV file
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'nifty50_instruments.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                return df['symbol'].tolist()
            
            # Fallback to hardcoded list if CSV doesn't exist
            logger.warning("nifty50_instruments.csv not found, using hardcoded list")
            return [
                "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
                "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS",
                "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
                "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
                "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
                "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
                "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
                "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
                "UPL.NS", "WIPRO.NS"
            ]
        except Exception as e:
            logger.error(f"Error loading Nifty 50 stocks: {e}")
            return []
    
    def get_nifty100_stocks(self) -> List[str]:
        """Get list of Nifty 100 stocks."""
        nifty50 = self.get_nifty50_stocks()
        additional_50 = [
            "AMBUJACEM.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "BEL.NS", "BIOCON.NS",
            "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS",
            "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DLF.NS",
            "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HDFCAMC.NS", "HINDPETRO.NS",
            "ICICIPRULI.NS", "IDEA.NS", "INDIGO.NS", "IOC.NS", "IRCTC.NS",
            "JINDALSTEL.NS", "JUBLFOOD.NS", "LICHSGFIN.NS", "LTIM.NS", "LUPIN.NS",
            "MARICO.NS", "MFSL.NS", "MGL.NS", "MOTHERSUMI.NS", "MRF.NS",
            "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NMDC.NS", "OFSS.NS",
            "OIL.NS", "PAGEIND.NS", "PEL.NS", "PETRONET.NS", "PFC.NS",
            "PIDILITIND.NS", "PNB.NS", "RAMCOCEM.NS", "SAIL.NS", "SHREECEM.NS",
            "SIEMENS.NS", "SRF.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS",
            "TVSMOTOR.NS", "VEDL.NS", "VOLTAS.NS"
        ]
        return nifty50 + additional_50
    
    def get_nifty500_stocks(self) -> List[str]:
        """Get list of Nifty 500 stocks (subset - major ones)."""
        nifty100 = self.get_nifty100_stocks()
        additional_400 = [
            "3MINDIA.NS", "ABB.NS", "ACC.NS", "AUROPHARMA.NS", "ALKEM.NS",
            "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "ATUL.NS", "BALKRISIND.NS",
            "BATAINDIA.NS", "BHARATFORG.NS", "BHEL.NS", "BSOFT.NS", "CEATLTD.NS",
            "CHAMBLFERT.NS", "COFORGE.NS", "CROMPTON.NS", "CUB.NS", "DELTACORP.NS",
            "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS",
            "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GRANULES.NS", "GRAPHITE.NS",
            "GUJGASLTD.NS", "HINDZINC.NS", "HONAUT.NS", "IBREALEST.NS", "IDFCFIRSTB.NS",
            "IEX.NS", "INDIANB.NS", "INDUSTOWER.NS", "INTELLECT.NS", "ISEC.NS",
            "JKCEMENT.NS", "JSWENERGY.NS", "KAJARIACER.NS", "KPITTECH.NS", "LALPATHLAB.NS",
            "LAURUSLABS.NS", "MANAPPURAM.NS", "MAXHEALTH.NS", "METROPOLIS.NS", "MPHASIS.NS",
            "NAVINFLUOR.NS", "OBEROIRLTY.NS", "PERSISTENT.NS", "PIIND.NS", "POLYCAB.NS",
            "RBLBANK.NS", "RECLTD.NS", "RELAXO.NS", "SANOFI.NS", "SCHAEFFLER.NS",
            "SONACOMS.NS", "STAR.NS", "SUNDARMFIN.NS", "SYNGENE.NS", "TATACHEM.NS",
            "TATACOMM.NS", "TATAELXSI.NS", "TATAPOWER.NS", "THERMAX.NS", "TIINDIA.NS",
            "TORRENTPHAR.NS", "TRIDENT.NS", "UBL.NS", "UNIONBANK.NS", "VGUARD.NS",
            "VINATIORGA.NS", "WABCOINDIA.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS",
            "ZYDUSLIFE.NS", "AAVAS.NS", "ABCAPITAL.NS", "ABFRL.NS", "ADANIGREEN.NS",
            "ADANIPOWER.NS", "AEGISCHEM.NS", "AIAENG.NS", "AJANTPHARMA.NS", "AKZOINDIA.NS",
            "AMARAJABAT.NS", "ARVINDFASN.NS", "ASAHIINDIA.NS", "ASTERDM.NS", "AUBANK.NS",
            "AVANTI.NS", "BALMLAWRIE.NS", "BALRAMCHIN.NS", "BANKBARODA.NS", "BANKINDIA.NS",
            "BIRLACORPN.NS", "BLISSGVS.NS", "BLUEDART.NS", "CANFINHOME.NS", "CAPLIPOINT.NS",
            "CARBORUNIV.NS", "CASTROLIND.NS", "CCL.NS", "CENTRALBK.NS", "CENTURYTEX.NS",
            "CESC.NS", "COROMANDEL.NS", "DHANI.NS", "DISHTV.NS", "EIDPARRY.NS",
            "EIHOTEL.NS", "ELGIEQUIP.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ENGINERSIN.NS",
            "EQUITAS.NS", "FINCABLES.NS", "FINPIPE.NS", "FIRSTSOURCE.NS", "GDL.NS",
            "GEPIL.NS", "GICRE.NS", "GILLETTE.NS", "GLAXO.NS", "GODREJIND.NS",
            "GODREJPROP.NS", "GREAVESCOT.NS", "GRINDWELL.NS", "GSFC.NS", "GSPL.NS",
            "GUJALKALI.NS", "GULFOILLUB.NS", "HAL.NS", "HFCL.NS", "HINDCOPPER.NS",
            "HSCL.NS", "HUDCO.NS", "IBULHSGFIN.NS", "ICICIGI.NS", "IDBI.NS",
            "IDFC.NS", "IFBIND.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS",
            "INDIAMART.NS", "INOXLEISUR.NS", "IPCALAB.NS", "JBCHEPHARM.NS", "JKLAKSHMI.NS",
            "JKPAPER.NS", "JMFINANCIL.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", "KALPATPOWR.NS",
            "KANSAINER.NS", "KEI.NS", "KNRCON.NS", "KRBL.NS", "L&TFH.NS",
            "M&MFIN.NS", "MAHABANK.NS", "MAHINDCIE.NS", "MCX.NS", "MIDHANI.NS",
            "MRPL.NS", "NBCC.NS", "NCC.NS", "NETWORK18.NS", "NHPC.NS",
            "NLCINDIA.NS", "ORIENTBANK.NS", "PFIZER.NS", "PRAJIND.NS", "PRESTIGE.NS",
            "PRINCEPIPE.NS", "PVR.NS", "QUESS.NS", "RADICO.NS", "RAIN.NS",
            "RAJESHEXPO.NS", "RALLIS.NS", "RCF.NS", "REDINGTON.NS", "REPCOHOME.NS",
            "RNAM.NS", "ROUTE.NS", "RPOWER.NS", "SBICARD.NS", "SIS.NS",
            "SKFINDIA.NS", "SRTRANSFIN.NS", "STARCEMENT.NS", "SUNDRMFAST.NS", "SUNTV.NS",
            "SUPRAJIT.NS", "SUPREMEIND.NS", "SUZLON.NS", "THYROCARE.NS", "UJJIVAN.NS"
        ]
        return nifty100 + additional_400
    
    def get_stocks_by_universe(self, universe_type: str) -> List[str]:
        """Get stocks by universe type."""
        if universe_type == "nifty50":
            return self.get_nifty50_stocks()
        elif universe_type == "nifty100":
            return self.get_nifty100_stocks()
        elif universe_type == "nifty500":
            return self.get_nifty500_stocks()
        else:
            return self.get_nifty50_stocks()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        if df.empty:
            return df
            
        try:
            # Check if required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns for technical indicators: {missing_columns}")
                logger.debug(f"Available columns: {list(df.columns)}")
                return df
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volatility
            df['daily_return'] = df['close'].pct_change()
            df['volatility'] = df['daily_return'].rolling(window=20).std()
            
            logger.debug("Calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame shape: {df.shape}")
            return df
    
    def get_market_status(self) -> Dict[str, str]:
        """
        Get current market status.
        
        Returns:
            Dictionary with market status information
        """
        if not self.kite:
            return {"status": "disconnected", "message": "Kite connection not initialized"}
            
        try:
            # Get current time
            now = datetime.now()
            
            # Simple market hours check (9:15 AM to 3:30 PM IST, Monday to Friday)
            is_weekday = now.weekday() < 5
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if not is_weekday:
                return {"status": "closed", "message": "Market closed (weekend)"}
            elif now < market_start:
                return {"status": "pre_market", "message": "Market opens at 9:15 AM"}
            elif now > market_end:
                return {"status": "closed", "message": "Market closed for the day"}
            else:
                return {"status": "open", "message": "Market is open"}
                
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {"status": "error", "message": str(e)} 