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

# ================= Configuration Defaults =================
DEFAULT_MODEL_PATH = "/home/dragon/AI/llama-3-8B-4bit-finance"
DEFAULT_OUTPUT_FILE = "final/gold_llm_enhanced_train.jsonl"
DEFAULT_NEWS_FILE = "final/gold_news_10years.csv"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_DOWNLOAD_END_DATE = "2026-01-10"

# Global variables to be populated by args
MODEL_PATH = DEFAULT_MODEL_PATH
OUTPUT_FILE = DEFAULT_OUTPUT_FILE
NEWS_FILE = DEFAULT_NEWS_FILE
CACHE_FILE = DEFAULT_CACHE_FILE
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
DOWNLOAD_END_DATE = DEFAULT_DOWNLOAD_END_DATE
ENABLE_DOWNLOAD = False


# ================= 1. Helper Functions for Tech Indicators =================
def compute_technical_indicators(df):
    """
    Computes RSI, MACD, KDJ, Volume stats for Gold (GC=F).
    """
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (Standard 12, 26, 9)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    # Bollinger Bands (20, 2)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["SMA_20"] + (2 * df["Std_Dev"])
    df["Lower_Band"] = df["SMA_20"] - (2 * df["Std_Dev"])

    # Bollinger Band Width & %B
    df["BB_Width"] = (df["Upper_Band"] - df["Lower_Band"]) / df["SMA_20"]
    df["Percent_B"] = (df["Close"] - df["Lower_Band"]) / (
        df["Upper_Band"] - df["Lower_Band"]
    )

    # KDJ (Stochastic Oscillator) - Simple approximation
    low_min = df["Low"].rolling(window=9).min()
    high_max = df["High"].rolling(window=9).max()
    df["RSV"] = (df["Close"] - low_min) / (high_max - low_min) * 100
    df["K"] = df["RSV"].ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    return df


# ================= 2. Data Preparation =================
def prepare_data():
    print("Loading data...")
    # Load raw news
    if not os.path.exists(NEWS_FILE):
        raise FileNotFoundError(f"News file not found: {NEWS_FILE}")
        
    df_news = pd.read_csv(NEWS_FILE)
    
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
                print(
                    f"Local data insufficient. Downloading missing range: {last_date} to {DOWNLOAD_END_DATE}..."
                )
                try:
                    start_missing = last_date + timedelta(days=1)
                    if start_missing <= required_end:
                        new_data = yf.download(
                            "GC=F",
                            start=start_missing,
                            end=DOWNLOAD_END_DATE,
                            progress=False,
                        )

                        if isinstance(new_data.columns, pd.MultiIndex):
                            new_data.columns = new_data.columns.get_level_values(0)

                        if not new_data.empty:
                            gold = pd.concat([gold, new_data])
                            gold = gold[~gold.index.duplicated(keep="last")]
                            print(f"Updated data with {len(new_data)} new rows.")
                            # Save
                            gold.to_csv(CACHE_FILE)
                except Exception as e:
                    print(f"Warning: Failed to update data: {e}")
            else:
                if last_date < pd.to_datetime(END_DATE):
                    raise RuntimeError(
                        f"Local data insufficient (Ends {last_date}). Downloading disabled."
                    )
                print("Local data is sufficient for generation range.")

    if gold is None:
        if not ENABLE_DOWNLOAD:
            raise RuntimeError(
                f"No local data at {CACHE_FILE} and downloading disabled."
            )
        print("Downloading Market Data...")
        gold = yf.download(
            "GC=F", start=START_DATE, end=DOWNLOAD_END_DATE, progress=False
        )
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        gold.to_csv(CACHE_FILE)

    # Calculate Indicators
    gold = compute_technical_indicators(gold)
    
    # Calculate Ground Truth Labels from Price Returns (Look-ahead)
    # We use T+1 return to determine what the "correct" sentiment for today (T) should have been.
    # Logic: If price rises tomorrow, today's sentiment regarding future prospect is Bullish.
    gold['Next_Ret'] = gold['Close'].pct_change().shift(-1)
    
    # Drop the last row where Next_Ret is NaN (we can't train on the last day without future data)
    gold = gold.dropna(subset=['Next_Ret'])
    
    # Map Return to Score (-5 to +5)
    # Assumption: 1% move is a "strong" move (Score ~5)
    # Factor = 500 (0.01 * 500 = 5)
    gold['Calculated_Score'] = np.clip(gold['Next_Ret'] * 500, -5, 5)

    # Ensure index is timezone-naive for merging
    if gold.index.tz is not None:
        gold.index = gold.index.tz_localize(None)

    # Create an index of valid trading dates
    valid_trading_dates = pd.DatetimeIndex(gold.index).normalize()

    # 2.1 Flexible "Next Trading Day" Logic
    print("Aligning news to valid Trading Days...")
    df_news["Date"] = pd.to_datetime(df_news["Date"])

    def get_next_trading_day(date):
        # Normalize date to midnight
        d = pd.Timestamp(date).normalize()
        # Find the first valid date >= d
        idx = valid_trading_dates.searchsorted(d)

        # If the date is past the last trading day, use the last one (or ignore)
        if idx >= len(valid_trading_dates):
            return valid_trading_dates[-1]

        return valid_trading_dates[idx]

    df_news["Trading_Date"] = df_news["Date"].apply(get_next_trading_day)

    # 2.2 Aggregate News by Trading Date
    print("Aggregating news by Trading Day...")

    # Aggregate headlines
    daily_groups = df_news.groupby("Trading_Date")

    daily_data = []
    for date, group in daily_groups:
        # Combine headlines
        headlines = group["Headline"].tolist()
        combined_news = ""
        for i, h in enumerate(headlines, 1):
            combined_news += f"{i}. {h}\n"

        daily_data.append(
            {"Date": date, "News_Text": combined_news}
        )

    df_daily = pd.DataFrame(daily_data)
    df_daily.set_index("Date", inplace=True)
    
    # Merge News with Technicals (and Calculated Score)
    print("Merging News and Technicals...")
    df_final = df_daily.join(
        gold, how="inner"
    )  # Only keep days with both News + Trading
    
    return df_final


# ================= 3. Model Generation & Reflection =================
def generate_enhanced_dataset():
    # Load Data
    df = prepare_data()

    print(f"Loading previous model from {MODEL_PATH}...")
    # Using Unsloth for fast inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
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
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate just the score first (simple extraction)
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
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
                pred_score = 0  # Fallback
        except:
            pred_score = 0

        true_score = row["Calculated_Score"]

        # 2. Logic: Result too different?
        diff = abs(pred_score - true_score)

        final_response = f"Score: {true_score:.2f}"

        # Construct technical context string for the rationale
        # Simple trend logic for the text
        trend = "bullish" if true_score > 1 else "bearish" if true_score < -1 else "neutral"
        tech_context = (
            f"RSI ({row['RSI']:.1f}) and BB %B ({row['Percent_B']:.2f})"
        )

        if diff > 2.0:  # Threshold for "Too Different"
            # Generate Reflection (Correction)
            # We explicitly use the reflection prompt to contextualize the error in the training data
            reflection_note = (
                f"\n\n[Reflection]\n"
                f"Model Prediction: {pred_score}\n"
                f"Actual Market Score: {true_score:.2f}\n"
                f"Reasoning: The initial prediction ({pred_score}) diverged significantly from the market reality ({true_score:.2f}). "
                f"Technical context: {tech_context} combined with the news flow suggests "
                f"a different directional move than anticipated. Adjusting weight on key news drivers."
            )
            final_response += reflection_note
        else:
            # Generate Standard Rationale (Confirmation)
            # The model was close enough, so we reinforce the link between indicators/news and the score
            rationale_note = (
                f"\n\n[Rationale]\n"
                f"The market sentiment ({true_score:.2f}) is supported by the technical structure. "
                f"{tech_context} indicates a {trend} momentum that aligns with the impact of the provided news headlines."
            )
            final_response += rationale_note

        # 3. Create JSONL Entry
        # The 'output' we want the model to learn is the CORRECT score + the Reflection (if needed)
        # So next time it learns to reason correctly.

        entry = {
            "instruction": "You are a Macro Quant Strategist specializing in Gold (XAU/USD). Analyze the given news and technical indicators for the day. Provide a Daily Sentiment Score (-5 to +5) and technical rationale.",
            "input": input_text,
            "output": final_response,
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
    parser.add_argument(
        "--enable-download",
        action="store_true",
        help="Enable downloading market data via yfinance",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--news-file", type=str, default=DEFAULT_NEWS_FILE, help="Path to news CSV file"
    )
    # Scored file argument removed as we calculate scores dynamically
    parser.add_argument(
        "--cache-file",
        type=str,
        default=DEFAULT_CACHE_FILE,
        help="Path to gold price cache CSV",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--download-end-date",
        type=str,
        default=DEFAULT_DOWNLOAD_END_DATE,
        help="End date for data downloading",
    )

    args = parser.parse_args()

    # Update globals
    ENABLE_DOWNLOAD = args.enable_download
    MODEL_PATH = args.model_path
    OUTPUT_FILE = args.output_file
    NEWS_FILE = args.news_file
    # SCORED_FILE removed
    CACHE_FILE = args.cache_file
    START_DATE = args.start_date
    END_DATE = args.end_date
    DOWNLOAD_END_DATE = args.download_end_date

    generate_enhanced_dataset()
