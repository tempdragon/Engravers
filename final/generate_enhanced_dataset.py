import pandas as pd
import numpy as np
import yfinance as yf
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import json
import os
import argparse
from datetime import timedelta

# ================= Configuration =================
# Previous model path
MODEL_PATH = "llama3_gold_quant_checkpoint" 
# Output file
OUTPUT_FILE = "gold_llm_enhanced_train.jsonl"
# Input files
NEWS_FILE = "gold_news_10years.csv"
SCORED_FILE = "gold_training_data_scored.csv" # Using this for Ground Truth labels
CACHE_FILE = "commodity_data/gold.csv"

# Date Range (Aligned with previous notebook)
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
DOWNLOAD_END_DATE = "2026-01-10" # Constant for data fetching limit
ENABLE_DOWNLOAD = False # Set to True to enable yfinance downloading

# ================= 1. Helper Functions for Tech Indicators =================
def compute_technical_indicators(df):
    """
    Computes RSI, MACD, KDJ, Volume stats for Gold (GC=F).
    """
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Standard 12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # Bollinger Bands (20, 2)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])
    
    # Bollinger Band Width & %B
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']
    df['Percent_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])

    # KDJ (Stochastic Oscillator) - Simple approximation
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

# ================= 2. Data Preparation =================
def prepare_data():
    print("Loading data...")
    # Load raw news and scores
    df_news = pd.read_csv(NEWS_FILE)
    df_scores = pd.read_csv(SCORED_FILE) # Contains 'Date', 'Headline', 'Score'
    
    # 2.3 Fetch Technical Data First (to determine valid Trading Days)
    print("Fetching Market Data (Gold)...")
    
    gold = None
    if os.path.exists(CACHE_FILE):
        print(f"Loading Market Data from: {CACHE_FILE}")
        gold = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if not isinstance(gold.index, pd.DatetimeIndex):
            gold.index = pd.to_datetime(gold.index)
        
        last_date = gold.index.max()
        required_end = pd.to_datetime(DOWNLOAD_END_DATE)
        
        print(f"Local data ends at: {last_date}")
        
        if last_date < required_end:
                 if ENABLE_DOWNLOAD:
                    print(f"Local data insufficient. Downloading missing range: {last_date} to {DOWNLOAD_END_DATE}...")
                    try:
                        start_missing = last_date + timedelta(days=1)
                        if start_missing <= required_end:
                            new_data = yf.download("GC=F", start=start_missing, end=DOWNLOAD_END_DATE, progress=False)
                            
                            if isinstance(new_data.columns, pd.MultiIndex):
                                new_data.columns = new_data.columns.get_level_values(0)
                            
                            if not new_data.empty:
                                gold = pd.concat([gold, new_data])
                                gold = gold[~gold.index.duplicated(keep='last')]
                                print(f"Updated data with {len(new_data)} new rows.")
                                # Save
                                gold.to_csv(CACHE_FILE)
                    except Exception as e:
                        print(f"Warning: Failed to update data: {e}")
                 else:
                     if last_date < pd.to_datetime(END_DATE):
                         raise RuntimeError(f"Local data insufficient (Ends {last_date}). Downloading disabled.")
                     print("Local data is sufficient for generation range.")
    
    if gold is None:
         if not ENABLE_DOWNLOAD:
             raise RuntimeError(f"No local data at {CACHE_FILE} and downloading disabled.")
         print("Downloading Market Data...")
         gold = yf.download("GC=F", start=START_DATE, end=DOWNLOAD_END_DATE, progress=False)
         if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
         os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
         gold.to_csv(CACHE_FILE)
    
    # Calculate Indicators
    gold = compute_technical_indicators(gold)
    
    # Create an index of valid trading dates
    valid_trading_dates = pd.DatetimeIndex(gold.index).normalize()
    
    # 2.1 Flexible "Next Trading Day" Logic
    print("Aligning news to valid Trading Days...")
    df_scores['Date'] = pd.to_datetime(df_scores['Date'])

    def get_next_trading_day(date):
        # Normalize date to midnight
        d = pd.Timestamp(date).normalize()
        # Find the first valid date >= d
        # Using searchsorted to find the insertion point
        idx = valid_trading_dates.searchsorted(d)
        
        # If the date is past the last trading day, use the last one (or ignore)
        if idx >= len(valid_trading_dates):
            return valid_trading_dates[-1]
            
        return valid_trading_dates[idx]

    df_scores['Trading_Date'] = df_scores['Date'].apply(get_next_trading_day)
    
    # 2.2 Aggregate News by Trading Date
    print("Aggregating news by Trading Day...")
    # Function to filter 0s and calculate mean (same as 5.1.4.ipynb)
    def non_zero_mean(x):
        filtered = x[x != 0]
        if len(filtered) == 0:
            return 0.0
        return filtered.mean()

    # Aggregate headlines and compute daily score
    daily_groups = df_scores.groupby('Trading_Date')
    
    daily_data = []
    for date, group in daily_groups:
        # Combine headlines
        headlines = group['Headline'].tolist()
        combined_news = ""
        for i, h in enumerate(headlines, 1):
            combined_news += f"{i}. {h}\n"
        
        # Calculate Ground Truth Score
        true_score = non_zero_mean(group['Score'])
        
        daily_data.append({
            'Date': date,
            'News_Text': combined_news,
            'True_Score': true_score
        })
    
    df_daily = pd.DataFrame(daily_data)
    df_daily.set_index('Date', inplace=True)

    # 2.3 Fetch Technical Data (Already done above)
    # gold = yf.download("GC=F", start=START_DATE, end=END_DATE, progress=False)
    # ...
    
    # Merge
    print("Merging News and Technicals...")
    # We join on index (Date)
    df_final = df_daily.join(gold, how='inner') # Only keep days with both News + Trading
    
    return df_final

# ================= 3. Model Generation & Reflection =================
def generate_enhanced_dataset():
    # Load Data
    df = prepare_data()
    
    print(f"Loading previous model from {MODEL_PATH}...")
    # Using Unsloth for fast inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    
    print("Starting generation loop...")
    # Define Prompt Template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Macro Quant Strategist specializing in Gold (XAU/USD).
Analyze the given news and technical indicators for the day.
Determine the Daily Sentiment Score (-5 to +5).
If your analysis contradicts the market reality, provide a reflection.

### Input:
{}

### Response:
"""

    for date, row in tqdm(df.iterrows(), total=len(df)):
        # Construct Input String
        tech_str = (
            f"Open: {row['Open']:.2f}, High: {row['High']:.2f}, Low: {row['Low']:.2f}, Close: {row['Close']:.2f}, Volume: {int(row['Volume'])}\n"
            f"RSI: {row['RSI']:.2f}, MACD: {row['MACD']:.2f}, Signal: {row['Signal_Line']:.2f}, Hist: {row['MACD_Hist']:.2f}\n"
            f"KDJ_K: {row['K']:.2f}, KDJ_D: {row['D']:.2f}, KDJ_J: {row['J']:.2f}\n"
            f"BB_Upper: {row['Upper_Band']:.2f}, BB_Lower: {row['Lower_Band']:.2f}, %B: {row['Percent_B']:.2f}"
        )
        
        input_text = f"Date: {date.strftime('%Y-%m-%d')}\n\n[Technical Indicators]\n{tech_str}\n\n[News Headlines]\n{row['News_Text']}"
        
        # 1. Run Inference to see what the *current* model thinks
        prompt = alpaca_prompt.format(input_text)
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        # Generate just the score first (simple extraction)
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract predicted score (naive parsing)
        # Assuming model outputs "Score: X"
        try:
            pred_part = pred_text.split("### Response:")[-1].strip()
            # Look for number
            import re
            match = re.search(r"Score:\s*(-?\d+(\.\d+)?)", pred_part)
            if match:
                pred_score = float(match.group(1))
            else:
                pred_score = 0 # Fallback
        except:
            pred_score = 0

        true_score = row['True_Score']
        
        # 2. Logic: Result too different?
        diff = abs(pred_score - true_score)
        
        final_response = f"Score: {true_score:.2f}"
        
        if diff > 2.0: # Threshold for "Too Different"
            # Generate Reflection
            # We explicitly use the reflection prompt to contextualize the error in the training data
            reflection_note = (
                f"\n\n[Reflection]\n"
                f"Model Prediction: {pred_score}\n"
                f"Actual Market Score: {true_score:.2f}\n"
                f"Reasoning: The initial prediction ({pred_score}) diverged significantly from the market reality ({true_score:.2f}). "
                f"Technical context (RSI: {row['RSI']:.2f}, BB %B: {row['Percent_B']:.2f}) combined with the news flow suggests "
                f"a stronger directional move than anticipated. Adjusting weight on key news drivers."
            )
            final_response += reflection_note
        
        # 3. Create JSONL Entry
        # The 'output' we want the model to learn is the CORRECT score + the Reflection (if needed)
        # So next time it learns to reason correctly.
        
        entry = {
            "instruction": "You are a Macro Quant Strategist specializing in Gold (XAU/USD). Analyze the given news and technical indicators for the day. Provide a Daily Sentiment Score (-5 to +5) and technical rationale.",
            "input": input_text,
            "output": final_response
        }
        results.append(entry)

    # Save
    print(f"Saving {len(results)} rows to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Enhanced Dataset")
    parser.add_argument("--enable-download", action="store_true", help="Enable downloading market data via yfinance")
    args = parser.parse_args()
    
    if args.enable_download:
        ENABLE_DOWNLOAD = True

    generate_enhanced_dataset()
