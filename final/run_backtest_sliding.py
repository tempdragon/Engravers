import pandas as pd
import numpy as np
import yfinance as yf
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from datetime import timedelta

# ================= Configuration Defaults =================
DEFAULT_MODEL_PATH = "/home/dragon/AI/llama-3-8B-4bit-finance" 
DEFAULT_NEWS_FILE = "gold_news_10years.csv"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
DEFAULT_START_DATE = "2025-09-01" # Q4 Backtest
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_DOWNLOAD_END_DATE = "2026-01-10"
DEFAULT_WINDOW_SIZE = 3 

# Global variables to be populated by args
MODEL_PATH = DEFAULT_MODEL_PATH
NEWS_FILE = DEFAULT_NEWS_FILE
CACHE_FILE = DEFAULT_CACHE_FILE
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
DOWNLOAD_END_DATE = DEFAULT_DOWNLOAD_END_DATE
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
ENABLE_DOWNLOAD = False

# ================= 1. Helper Functions (Legacy + Enhanced) =================
def compute_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])
    df['Percent_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # KDJ
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

def get_next_trading_day(date, valid_dates):
    d = pd.Timestamp(date).normalize()
    idx = valid_dates.searchsorted(d)
    if idx >= len(valid_dates):
        return valid_dates[-1]
    return valid_dates[idx]

def prepare_daily_data():
    print("Loading and preparing data...")
    # 1. Load News
    df_news = pd.read_csv(NEWS_FILE)
    df_news = df_news[df_news['Date'] >= "2025-08-01"] # Buffer for window
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    
    # 2. Load Market Data
    print(f"Checking Market Data at: {CACHE_FILE}")
    gold = None
    
    # Try to load existing data
    if os.path.exists(CACHE_FILE):
        try:
            gold = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            if not isinstance(gold.index, pd.DatetimeIndex):
                gold.index = pd.to_datetime(gold.index)
            
            last_date = gold.index.max()
            print(f"Local data ends at: {last_date}")
            
            required_end = pd.to_datetime(DOWNLOAD_END_DATE)
            
            # Check if update is needed
            if last_date < required_end:
                if ENABLE_DOWNLOAD:
                    print(f"Local data stale. Downloading {last_date} -> {DOWNLOAD_END_DATE}...")
                    start_missing = last_date + timedelta(days=1)
                    if start_missing <= required_end:
                        try:
                            new_data = yf.download("GC=F", start=start_missing, end=DOWNLOAD_END_DATE, progress=False)
                            if isinstance(new_data.columns, pd.MultiIndex):
                                new_data.columns = new_data.columns.get_level_values(0)
                            
                            if not new_data.empty:
                                gold = pd.concat([gold, new_data])
                                gold = gold[~gold.index.duplicated(keep='last')]
                                gold.to_csv(CACHE_FILE)
                                print(f"Updated and saved to {CACHE_FILE}")
                        except Exception as e:
                            print(f"Update failed: {e}")
                else:
                    if last_date < pd.to_datetime(END_DATE):
                         raise RuntimeError(f"Local data ends {last_date}, need {END_DATE}. Download disabled.")
                    print("Local data sufficient for backtest.")
        except Exception as e:
            print(f"Error reading cache: {e}")
            gold = None

    # If no data found or read failed, download full
    if gold is None:
        if not ENABLE_DOWNLOAD:
            raise RuntimeError(f"No valid data at {CACHE_FILE} and download disabled.")
        
        print("Downloading full history...")
        gold = yf.download("GC=F", start="2020-01-01", end=DOWNLOAD_END_DATE, progress=False)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
        
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        gold.to_csv(CACHE_FILE)
        print(f"Saved to {CACHE_FILE}")

    # Calculate Indicators
    gold = compute_technical_indicators(gold)
    valid_dates = pd.DatetimeIndex(gold.index).normalize()
    
    # 3. Align News to Trading Days
    df_news['Trading_Date'] = df_news['Date'].apply(lambda x: get_next_trading_day(x, valid_dates))
    
    # 4. Aggregate by Day
    daily_groups = df_news.groupby('Trading_Date')
    daily_records = []
    
    for date, group in daily_groups:
        if date not in gold.index: continue
        
        # News Context
        headlines = group['Headline'].tolist()
        news_text = "\n".join([f"- {h}" for h in headlines])
        
        # Technical Context
        row = gold.loc[date]
        tech_text = (
            f"Price: {row['Close']:.2f} (O: {row['Open']:.2f}, H: {row['High']:.2f}, L: {row['Low']:.2f})\n"
            f"Indicators: RSI={row['RSI']:.1f}, MACD_Hist={row['MACD_Hist']:.2f}, BB_%B={row['Percent_B']:.2f}"
        )
        
        daily_records.append({
            'Date': date,
            'News': news_text,
            'Technicals': tech_text,
            'Full_Row': row
        })
        
    df_daily = pd.DataFrame(daily_records).set_index('Date').sort_index()
    return df_daily, gold

# ================= 2. Backtest Engine =================
def run_backtest():
    # 1. Prepare Data
    df_daily, gold_price = prepare_daily_data()
    
    # Filter for Backtest Period
    test_dates = df_daily[START_DATE:END_DATE].index
    
    # 2. Load Model
    print(f"Loading Model: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 4096, # Increased for sliding window
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    results = []
    
    print(f"Running Sliding Window Backtest (Window={WINDOW_SIZE})...")
    
    # "Long-Term Memory" Buffer
    # We will let the model UPDATE this memory every day if something significant happens.
    long_term_memory = "No significant long-term market drivers identified yet."
    
    for current_date in tqdm(test_dates):
        # Sliding Window Logic
        try:
            curr_idx = df_daily.index.get_loc(current_date)
        except KeyError:
            continue
            
        start_idx = max(0, curr_idx - WINDOW_SIZE + 1)
        window_slice = df_daily.iloc[start_idx : curr_idx + 1]
        
        # Construct Prompt with History
        history_text = ""
        for dt, row in window_slice.iterrows():
            date_str = dt.strftime('%Y-%m-%d')
            is_today = (dt == current_date)
            prefix = "TODAY" if is_today else "PAST"
            
            history_text += f"\n[{prefix} - {date_str}]\n"
            history_text += f"Technicals: {row['Technicals']}\n"
            history_text += f"News:\n{row['News']}\n"
            
        prompt = f"""### Instruction:
You are a Macro Quant Strategist. 
1. Analyze the market data history below (Window of {WINDOW_SIZE} days).
2. Consider the 'Long Term Memory' of key drivers.
3. Predict the sentiment score for TODAY ({current_date.strftime('%Y-%m-%d')}) from -5 (Bearish) to +5 (Bullish).
4. Update the 'Long Term Memory' (max 200 chars) with any CRITICAL new structural drivers (e.g. "War started", "Fed cut rates"). If nothing changed, keep it same.

### Long Term Memory (Persistent Context):
{long_term_memory}

### Input (Sliding Window):
{history_text}

### Response:
Score: [Your Score]
Memory_Update: [Brief Summary]"""

        # Inference
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True) # Increased tokens for memory update
        out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse Score & Memory
        try:
            response_part = out_text.split("### Response:")[-1].strip()
            
            # Parse Score
            import re
            score_match = re.search(r"Score:\s*(-?\d+(\.\d+)?)", response_part)
            score = float(score_match.group(1)) if score_match else 0.0
            
            # Parse Memory Update
            memory_match = re.search(r"Memory_Update:(.*)", response_part, re.DOTALL)
            if memory_match:
                new_memory = memory_match.group(1).strip()
                # Safety check: Prevent memory from exploding or becoming empty noise
                if len(new_memory) > 10 and "Score" not in new_memory: 
                    # Truncate to ~200 chars as requested
                    long_term_memory = new_memory[:250] 
        except:
            score = 0.0
            
        results.append({
            'Date': current_date,
            'Score': score,
            'Memory_State': long_term_memory
        })

    # 3. Calculate Returns (Legacy Framework)
    df_res = pd.DataFrame(results).set_index('Date')
    
    # Calculate Strategy Returns
    # Merge with next day return (T+1)
    gold_returns = gold_price['Close'].pct_change().shift(-1)
    
    strategy_df = df_res.join(gold_returns.rename("Gold_Ret"))
    
    # Signal Logic
    strategy_df['Position'] = np.where(strategy_df['Score'] > 0, 1, -1)
    strategy_df['Position'] = np.where(strategy_df['Score'] == 0, 0, strategy_df['Position'])
    
    strategy_df['Strategy_Ret'] = strategy_df['Position'] * strategy_df['Gold_Ret']
    
    # Cumulative
    strategy_df['Cum_Gold'] = (1 + strategy_df['Gold_Ret'].fillna(0)).cumprod()
    strategy_df['Cum_Strategy'] = (1 + strategy_df['Strategy_Ret'].fillna(0)).cumprod()
    
    # ================= Performance Metrics =================
    # Annualize factor (assuming daily data, 252 trading days)
    ann_factor = 252
    
    # 1. Sharpe Ratio (Risk-Free Rate assumed 0 for simplicity)
    # Strategy Sharpe
    strat_mean = strategy_df['Strategy_Ret'].mean() * ann_factor
    strat_std = strategy_df['Strategy_Ret'].std() * np.sqrt(ann_factor)
    strat_sharpe = strat_mean / strat_std if strat_std != 0 else 0
    
    # Gold Sharpe
    gold_mean = strategy_df['Gold_Ret'].mean() * ann_factor
    gold_std = strategy_df['Gold_Ret'].std() * np.sqrt(ann_factor)
    gold_sharpe = gold_mean / gold_std if gold_std != 0 else 0
    
    # 2. Alpha (Jensen's Alpha) & Beta
    # Regression: Strategy_Ret = Alpha + Beta * Gold_Ret
    from scipy import stats
    # Filter out NaNs for regression
    clean_df = strategy_df[['Strategy_Ret', 'Gold_Ret']].dropna()
    if len(clean_df) > 10:
        beta, alpha, r_value, p_value, std_err = stats.linregress(clean_df['Gold_Ret'], clean_df['Strategy_Ret'])
        # Alpha is daily, annualize it
        alpha_ann = alpha * ann_factor
    else:
        alpha_ann = 0.0
        beta = 0.0

    print("\n" + "="*40)
    print(f"📊 Performance Metrics")
    print("="*40)
    print(f"Strategy Sharpe: {strat_sharpe:.2f}")
    print(f"Gold (B&H) Sharpe: {gold_sharpe:.2f}")
    print(f"Alpha (Annualized): {alpha_ann:.2%}")
    print(f"Beta: {beta:.2f}")
    print("="*40 + "\n")

    # Save & Plot
    strategy_df.to_csv("q4_sliding_window_results.csv")
    print("✅ Results saved to q4_sliding_window_results.csv")
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df.index, strategy_df['Cum_Strategy'], label='Sliding Window AI', color='purple', linewidth=2)
    plt.plot(strategy_df.index, strategy_df['Cum_Gold'], label='Gold Benchmark (Buy & Hold)', color='gray', linestyle='--')
    
    # Add Text Box with Metrics to Plot
    metrics_text = (
        f"Sharpe (AI): {strat_sharpe:.2f}\n"
        f"Sharpe (Gold): {gold_sharpe:.2f}\n"
        f"Alpha: {alpha_ann:.2%}"
    )
    plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"Sliding Window Backtest (Window={WINDOW_SIZE})\nGold vs AI Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("q4_sliding_window_chart.png")

    print("✅ Chart saved to q4_sliding_window_chart.png")
    
    final_ret = strategy_df['Cum_Strategy'].iloc[-1] - 1
    print(f"🚀 Final Strategy Return: {final_ret*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sliding Window Backtest")
    parser.add_argument("--enable-download", action="store_true", help="Enable downloading market data via yfinance")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model directory")
    parser.add_argument("--news-file", type=str, default=DEFAULT_NEWS_FILE, help="Path to the news CSV file")
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE, help="Path to the gold price cache CSV")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--download-end-date", type=str, default=DEFAULT_DOWNLOAD_END_DATE, help="End date for data downloading")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Sliding window size in days")

    args = parser.parse_args()
    
    # Update globals
    ENABLE_DOWNLOAD = args.enable_download
    MODEL_PATH = args.model_path
    NEWS_FILE = args.news_file
    CACHE_FILE = args.cache_file
    START_DATE = args.start_date
    END_DATE = args.end_date
    DOWNLOAD_END_DATE = args.download_end_date
    WINDOW_SIZE = args.window_size
        
    run_backtest()
