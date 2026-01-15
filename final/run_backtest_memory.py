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
import re

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= Configuration Defaults =================
DEFAULT_MODEL_PATH = "final/llama3_gold_deterministic_checkpoint"
DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
DEFAULT_NEWS_FILE = "final/gold_news_10years.csv"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
DEFAULT_START_DATE = "2025-09-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_DOWNLOAD_END_DATE = "2026-01-10"

DEFAULT_OUTPUT_CSV = "q4_memory_results.csv"
DEFAULT_OUTPUT_CHART = "q4_memory_chart.png"

# ================= 1. Helper Functions (Copied) =================
def compute_technical_indicators(df):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["SMA_20"] + (2 * df["Std_Dev"])
    df["Lower_Band"] = df["SMA_20"] - (2 * df["Std_Dev"])
    df["Percent_B"] = (df["Close"] - df["Lower_Band"]) / (df["Upper_Band"] - df["Lower_Band"])

    low_min = df["Low"].rolling(window=9).min()
    high_max = df["High"].rolling(window=9).max()
    df["RSV"] = (df["Close"] - low_min) / (high_max - low_min) * 100
    df["K"] = df["RSV"].ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df

def get_next_trading_day(date, valid_dates):
    d = pd.Timestamp(date).normalize()
    idx = valid_dates.searchsorted(d)
    if idx >= len(valid_dates):
        return valid_dates[-1]
    return valid_dates[idx]

def prepare_daily_data(news_file, cache_file, start_date, end_date):
    print("Loading and preparing data...")
    df_news = pd.read_csv(news_file)
    df_news["Date"] = pd.to_datetime(df_news["Date"])
    
    # Load Market Data
    if os.path.exists(cache_file):
        gold = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        # Fallback simple download
        gold = yf.download("GC=F", start="2020-01-01", end=end_date, progress=False)
        if isinstance(gold.columns, pd.MultiIndex): gold.columns = gold.columns.get_level_values(0)
        gold.to_csv(cache_file)

    gold = compute_technical_indicators(gold)
    if gold.index.tz is not None: gold.index = gold.index.tz_localize(None)
    valid_dates = pd.DatetimeIndex(gold.index).normalize()

    df_news["Trading_Date"] = df_news["Date"].apply(lambda x: get_next_trading_day(x, valid_dates))

    daily_records = []
    daily_groups = df_news.groupby("Trading_Date")
    
    for date, group in daily_groups:
        if date not in gold.index: continue
        headlines = group["Headline"].tolist()
        row = gold.loc[date]
        
        # Format Tech String for Model
        tech_str = (
            f"Open: {row['Open']:.2f}, High: {row['High']:.2f}, Low: {row['Low']:.2f}, Close: {row['Close']:.2f}, Volume: {int(row['Volume'])}\n"
            f"RSI: {row['RSI']:.1f}, MACD: {row['MACD']:.2f}, Signal: {row['Signal_Line']:.2f}, Hist: {row['MACD_Hist']:.2f}\n"
            f"KDJ_K: {row['K']:.1f}, KDJ_D: {row['D']:.1f}, KDJ_J: {row['J']:.1f}\n"
            f"BB_Upper: {row['Upper_Band']:.2f}, BB_Lower: {row['Lower_Band']:.2f}, %B: {row['Percent_B']:.2f}"
        )
        
        daily_records.append({
            "Date": date, 
            "Headlines": headlines, # Keep as list for summary
            "Technicals_Str": tech_str,
            "Full_Row": row
        })

    return pd.DataFrame(daily_records).set_index("Date").sort_index(), gold

# ================= 2. Memory & Backtest Logic =================
def run_memory_backtest(args):
    # 1. Data
    df_daily, gold_price = prepare_daily_data(args.news_file, args.cache_file, args.start_date, args.end_date)
    test_dates = df_daily[args.start_date:args.end_date].index

    # 2. Load Model
    # We load the FINE-TUNED model first. 
    # Unsloth allows disabling adapters to treat it as Base Model.
    print(f"Loading Model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    
    # MEMORY BUFFER: List of {"headline": str, "date": Timestamp}
    memory_buffer = [] 
    
    print(f"🚀 Running Memory-Augmented Backtest ({len(test_dates)} days)...")

    for current_date in tqdm(test_dates):
        try:
            row = df_daily.loc[current_date]
        except KeyError: continue

        # --- A. PREPARE INPUT (With Memory Injection) ---
        
        # 1. Prune Memory (Remove > 14 days)
        valid_memory = []
        cutoff_date = current_date - timedelta(days=14)
        for mem in memory_buffer:
            if mem['date'] >= cutoff_date:
                valid_memory.append(mem)
        memory_buffer = valid_memory # Update buffer

        # 2. Inject Memory into News
        current_headlines = row['Headlines']
        
        # Create "Synthetic" headlines from memory
        memory_headlines = [f"[HISTORY {m['date'].strftime('%Y-%m-%d')}] {m['headline']}" for m in valid_memory]
        
        # Combine
        combined_headlines = current_headlines + memory_headlines
        news_text = "\n".join([f"- {h}" for h in combined_headlines])
        
        input_text = f"Date: {current_date.strftime('%Y-%m-%d')}\n\n[Technical Indicators]\n{row['Technicals_Str']}\n\n[News Headlines]\n{news_text}"

        # --- B. INFERENCE (Scoring) ---
        # Ensure Adapters are ENABLED for scoring
        # model.enable_adapters() # Unsloth usually keeps them enabled by default
        
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Macro Quant Strategist specializing in Gold (XAU/USD). Analyze the given news and technical indicators for the day. Determine the Daily Sentiment Score (-5 to +5).

### Input:
{input_text}

### Response:
"""
        # Generate Score
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Deterministic generation
        outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, temperature=0.1)
        out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse Score
        response_part = out_text.split("### Response:")[-1].strip()
        score = 0.0
        # Aggressive Regex
        match = re.search(r"(?:Score:\s*)?([+\-]?\d+(\.\d+)?)", response_part)
        if match:
            score = float(match.group(1))
        
        results.append({"Date": current_date, "Score": score, "Memory_Size": len(valid_memory)})

        # --- C. UPDATE MEMORY (Summarization) ---
        # Disable Adapters to use Base Model logic (General Summarization)
        # Note: Unsloth uses 'with model.disable_adapter():' context manager usually, 
        # or we can try generating without specific prompt format.
        # However, to be safe and explicit:
        
        # Construct Summary Prompt (Base Llama 3 Style)
        # We ask it to identify ONE major event to remember.
        summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial news filter. Identify the single most important event from today's news that will impact Gold prices for the next 2 weeks.
If nothing is major, reply "None".
Format: "EVENT SUMMARY" (Max 15 words).<|eot_id|><|start_header_id|>user<|end_header_id|>

News:
{chr(10).join(current_headlines)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        with torch.no_grad():
            # Temporarily disable adapters if possible, or just trust the base capability
            # Unsloth models are merged? No, they use LoRA.
            # To use base model, we can try:
            with model.disable_adapter():
                inputs_sum = tokenizer([summary_prompt], return_tensors="pt").to("cuda")
                outputs_sum = model.generate(**inputs_sum, max_new_tokens=48, temperature=0.1, use_cache=True)
                sum_text = tokenizer.batch_decode(outputs_sum, skip_special_tokens=True)[0]
                
                # Extract response
                raw_summary = sum_text.split("assistant")[-1].strip()
                
                if "None" not in raw_summary and len(raw_summary) > 5:
                    # Clean up
                    clean_summary = raw_summary.replace('"', '').split('\n')[0]
                    # Add to memory
                    memory_buffer.append({"headline": clean_summary, "date": current_date})
                    # print(f"  [Memory Added] {clean_summary}")

    # ================= 3. Analysis & Plotting =================
    df_res = pd.DataFrame(results).set_index("Date")
    
    # Returns
    gold_returns = gold_price["Close"].pct_change().shift(-1)
    strategy_df = df_res.join(gold_returns.rename("Gold_Ret"))
    
    # Signal
    strategy_df["Position"] = np.where(strategy_df["Score"] > 0, 1, -1)
    strategy_df["Position"] = np.where(strategy_df["Score"] == 0, 0, strategy_df["Position"])
    
    strategy_df["Strategy_Ret"] = strategy_df["Position"] * strategy_df["Gold_Ret"]
    strategy_df["Cum_Strategy"] = (1 + strategy_df["Strategy_Ret"].fillna(0)).cumprod()
    strategy_df["Cum_Gold"] = (1 + strategy_df["Gold_Ret"].fillna(0)).cumprod()
    
    # Metrics
    final_ret = strategy_df["Cum_Strategy"].iloc[-1] - 1
    
    print("\n" + "="*40)
    print(f"🚀 Memory Strategy Final Return: {final_ret*100:.2f}%")
    print("="*40)
    
    strategy_df.to_csv(args.output_csv)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df.index, strategy_df["Cum_Strategy"], label="Memory-Augmented AI", color="blue", linewidth=2)
    plt.plot(strategy_df.index, strategy_df["Cum_Gold"], label="Gold Benchmark", color="gray", linestyle="--")
    plt.title("Gold Strategy with 14-Day Memory Injection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output_chart)
    print(f"✅ Chart saved to {args.output_chart}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--news-file", type=str, default=DEFAULT_NEWS_FILE)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-chart", type=str, default=DEFAULT_OUTPUT_CHART)
    args = parser.parse_args()
    
    run_memory_backtest(args)
