import yfinance as yf
import pandas as pd
import os
import toml
from tqdm import tqdm
from datetime import datetime
import time
import random
import argparse

# ==========================================
# Configuration
# ==========================================

# Parse Args
parser = argparse.ArgumentParser(description="Download commodity data based on TOML config.")
parser.add_argument("--config", help="Path to configuration file (default: download.toml)")
args, _ = parser.parse_known_args()

# Load Config
if args.config:
    CONFIG_PATH = args.config
else:
    CONFIG_PATH = "download.toml"
    # Handle running from root or bt/
    if not os.path.exists(CONFIG_PATH):
        # Try looking in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG_PATH = os.path.join(script_dir, "download.toml")

try:
    config = toml.load(CONFIG_PATH)
    print(f"ℹ️  Loaded configuration from: {CONFIG_PATH}")
except Exception as e:
    print(f"Error loading config from {CONFIG_PATH}: {e}")
    # Fallback defaults if config fails, though we should probably exit
    config = {}

download_conf = config.get("download", {})
OUTPUT_DIR = download_conf.get("target_dir", "commodity_data")
START_DATE = download_conf.get("start_date", "2012-01-01")
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Proxy settings
PROXY_HTTP = download_conf.get("proxy_http")
PROXY_HTTPS = download_conf.get("proxy_https")

if PROXY_HTTP:
    os.environ["HTTP_PROXY"] = PROXY_HTTP
if PROXY_HTTPS:
    os.environ["HTTPS_PROXY"] = PROXY_HTTPS

# Tickers
COMMODITIES = config.get("commodities", {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Iron_Ore": "TI=F",
})

def download_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"⬇️ Downloading data from {START_DATE} to {END_DATE}...")
    if PROXY_HTTP:
        print(f"   🌐 Using Proxy: {PROXY_HTTP}")

    # Answer to volume question:
    # Volume in commodity futures (e.g., GC=F) typically represents the number of contracts traded.
    # For ETFs/Indices, it represents the number of shares/units traded.
    # It is NOT the total value (Price * Volume).

    pbar = tqdm(COMMODITIES.items(), desc="Downloading")
    for name, ticker in pbar:
        pbar.set_description(f"Processing {name}")
        try:
            # Random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))

            # Download using Ticker object
            dat = yf.Ticker(ticker)
            df_new = dat.history(start=START_DATE, end=END_DATE)
            
            if df_new.empty:
                tqdm.write(f"   ⚠️ No data found for {name} ({ticker})!")
                continue
                
            # Flatten MultiIndex columns if present
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            # Reset index to make Date a column
            df_new = df_new.reset_index()
            
            # Ensure columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df_new.columns:
                    df_new[col] = 0.0
            
            # Backtrader expects OpenInterest
            df_new['OpenInterest'] = 0 
            
            df_new = df_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']]
            
            # Normalize Date for merging (UTC)
            df_new['Date'] = pd.to_datetime(df_new['Date'], utc=True)
            
            # Save/Merge
            filename = os.path.join(OUTPUT_DIR, f"{name.lower()}.csv")
            
            if os.path.exists(filename):
                try:
                    df_old = pd.read_csv(filename)
                    # Normalize old dates too
                    df_old['Date'] = pd.to_datetime(df_old['Date'], utc=True)
                    
                    # Combine: append new data, drop duplicates based on Date, keep last (newest)
                    df_combined = pd.concat([df_old, df_new])
                    df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
                    df_combined = df_combined.sort_values(by='Date')
                    
                    final_df = df_combined
                except Exception as e:
                    tqdm.write(f"   ⚠️ Error reading existing file {filename}, overwriting: {e}")
                    final_df = df_new
            else:
                final_df = df_new

            final_df.to_csv(filename, index=False)
            # tqdm.write(f"   ✅ Saved to {filename} ({len(final_df)} rows)")
            
        except Exception as e:
            tqdm.write(f"   ❌ Error downloading {name}: {e}")

if __name__ == "__main__":
    download_data()
