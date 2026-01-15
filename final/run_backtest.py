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

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= Configuration Defaults =================
DEFAULT_MODEL_PATH = "/home/dragon/AI/llama-3-8B-4bit-finance"
DEFAULT_NEWS_FILE = "final/gold_news_10years.csv"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
DEFAULT_START_DATE = "2025-09-01"  # Q4 Backtest
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_DOWNLOAD_END_DATE = "2026-01-10"
DEFAULT_WINDOW_SIZE = 3
DEFAULT_REASONING_MODE = "step-by-step"  # "step-by-step" or "thinking"

DEFAULT_OUTPUT_CSV = "q4_sliding_window_results_new.csv"
DEFAULT_OUTPUT_CHART = "q4_sliding_window_chart_new.png"
DEFAULT_ORIGINAL_STRATEGY_CSV = "final/q4_strategy_daily.csv"
DEFAULT_SIMPLE_MODE = False

# Global variables to be populated by args
MODEL_PATH = DEFAULT_MODEL_PATH
NEWS_FILE = DEFAULT_NEWS_FILE
CACHE_FILE = DEFAULT_CACHE_FILE
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
DOWNLOAD_END_DATE = DEFAULT_DOWNLOAD_END_DATE
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
REASONING_MODE = DEFAULT_REASONING_MODE
ENABLE_DOWNLOAD = False
OUTPUT_CSV = DEFAULT_OUTPUT_CSV
OUTPUT_CHART = DEFAULT_OUTPUT_CHART
COMPARE_CSV = None  # Path to another result CSV for comparison
ORIGINAL_STRATEGY_CSV = DEFAULT_ORIGINAL_STRATEGY_CSV
SIMPLE_MODE = DEFAULT_SIMPLE_MODE


# ================= 1. Helper Functions (Legacy + Enhanced) =================
def compute_technical_indicators(df):
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    # Bollinger Bands
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["SMA_20"] + (2 * df["Std_Dev"])
    df["Lower_Band"] = df["SMA_20"] - (2 * df["Std_Dev"])
    df["Percent_B"] = (df["Close"] - df["Lower_Band"]) / (
        df["Upper_Band"] - df["Lower_Band"]
    )

    # KDJ
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


def prepare_daily_data():
    print("Loading and preparing data...")
    # 1. Load News
    df_news = pd.read_csv(NEWS_FILE)
    df_news = df_news[df_news["Date"] >= "2025-08-01"]  # Buffer for window
    df_news["Date"] = pd.to_datetime(df_news["Date"])

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
                    print(
                        f"Local data stale. Downloading {last_date} -> {DOWNLOAD_END_DATE}..."
                    )
                    start_missing = last_date + timedelta(days=1)
                    if start_missing <= required_end:
                        try:
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
                                gold.to_csv(CACHE_FILE)
                                print(f"Updated and saved to {CACHE_FILE}")
                        except Exception as e:
                            print(f"Update failed: {e}")
                else:
                    if last_date < pd.to_datetime(END_DATE):
                        raise RuntimeError(
                            f"Local data ends {last_date}, need {END_DATE}. Download disabled."
                        )
                    print("Local data sufficient for backtest.")
        except Exception as e:
            print(f"Error reading cache: {e}")
            gold = None

    # If no data found or read failed, download full
    if gold is None:
        if not ENABLE_DOWNLOAD:
            raise RuntimeError(f"No valid data at {CACHE_FILE} and download disabled.")

        print("Downloading full history...")
        gold = yf.download(
            "GC=F", start="2020-01-01", end=DOWNLOAD_END_DATE, progress=False
        )
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)

        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        gold.to_csv(CACHE_FILE)
        print(f"Saved to {CACHE_FILE}")

    # Calculate Indicators
    gold = compute_technical_indicators(gold)
    
    # Ensure index is timezone-naive for merging
    if gold.index.tz is not None:
        gold.index = gold.index.tz_localize(None)

    valid_dates = pd.DatetimeIndex(gold.index).normalize()

    # 3. Align News to Trading Days
    df_news["Trading_Date"] = df_news["Date"].apply(
        lambda x: get_next_trading_day(x, valid_dates)
    )

    # 4. Aggregate by Day
    daily_groups = df_news.groupby("Trading_Date")
    daily_records = []

    for date, group in daily_groups:
        if date not in gold.index:
            continue

        # News Context
        headlines = group["Headline"].tolist()
        news_text = "\n".join([f"- {h}" for h in headlines])

        # Technical Context
        row = gold.loc[date]
        tech_text = (
            f"Price: {row['Close']:.2f} (O: {row['Open']:.2f}, H: {row['High']:.2f}, L: {row['Low']:.2f})\n"
            f"Indicators: RSI={row['RSI']:.1f}, MACD_Hist={row['MACD_Hist']:.2f}, BB_%B={row['Percent_B']:.2f}"
        )

        daily_records.append(
            {"Date": date, "News": news_text, "Technicals": tech_text, "Full_Row": row}
        )

    df_daily = pd.DataFrame(daily_records).set_index("Date").sort_index()
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
        model_name=MODEL_PATH,
        max_seq_length=4096,  # Increased for sliding window
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
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

        if SIMPLE_MODE:
            # Deterministic Model Logic (No Window, No Memory, Simple Prompt)
            row = df_daily.iloc[curr_idx]
            
            # Reconstruct the exact training input format
            # Training format was: "Date: ...\n\n[Technical Indicators]\n...\n\n[News Headlines]\n..."
            # Note: df_daily['News'] is already formatted as text, but check format.
            # In prepare_daily_data, news_text = "\n".join([f"- {h}" for h in headlines])
            # Technicals = "Price: ... \nIndicators: ..."
            
            # We need to reconstruct the EXACT technical string format used in training if possible,
            # or at least providing the same info.
            # The training script used detailed tech string.
            # Here we have 'Technicals' and 'News' in df_daily.
            
            # Let's try to match the training input as closely as possible.
            # Training:
            # tech_str = f"Open: {row['Open']:.2f}, ... RSI: {row['RSI']:.2f}..."
            
            full_row = row['Full_Row']
            tech_str_training = (
                f"Open: {full_row['Open']:.2f}, High: {full_row['High']:.2f}, Low: {full_row['Low']:.2f}, Close: {full_row['Close']:.2f}, Volume: {int(full_row['Volume'])}\n"
                f"RSI: {full_row['RSI']:.2f}, MACD: {full_row['MACD']:.2f}, Signal: {full_row['Signal_Line']:.2f}, Hist: {full_row['MACD_Hist']:.2f}\n"
                f"KDJ_K: {full_row['K']:.2f}, KDJ_D: {full_row['D']:.2f}, KDJ_J: {full_row['J']:.2f}\n"
                f"BB_Upper: {full_row['Upper_Band']:.2f}, BB_Lower: {full_row['Lower_Band']:.2f}, %B: {full_row['Percent_B']:.2f}"
            )
            
            input_text = f"Date: {current_date.strftime('%Y-%m-%d')}\n\n[Technical Indicators]\n{tech_str_training}\n\n[News Headlines]\n{row['News']}"

            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Macro Quant Strategist specializing in Gold (XAU/USD). Analyze the given news and technical indicators for the day. Determine the Daily Sentiment Score (-5 to +5).

### Input:
{input_text}

### Response:
"""
        else:
            # Normal Sliding Window Logic
            start_idx = max(0, curr_idx - WINDOW_SIZE + 1)
            window_slice = df_daily.iloc[start_idx : curr_idx + 1]

            # Construct Prompt with History
            history_text = ""
            for dt, row in window_slice.iterrows():
                date_str = dt.strftime("%Y-%m-%d")
                is_today = dt == current_date
                prefix = "TODAY" if is_today else "PAST"

                history_text += f"\n[{prefix} - {date_str}]\n"
                history_text += f"Technicals: {row['Technicals']}\n"
                history_text += f"News:\n{row['News']}\n"

            if REASONING_MODE == "thinking":
                prompt = f"""### Instruction:
You are a Macro Quant Strategist.
1. Analyze the market data history below (Window of {WINDOW_SIZE} days).
2. Consider the 'Long Term Memory' of key drivers.
3. Engage in a deep thinking process about the market dynamics.
4. Predict the sentiment score for TODAY ({current_date.strftime('%Y-%m-%d')}) from -5 (Bearish) to +5 (Bullish).
5. Update the 'Long Term Memory' (max 200 chars).

### Long Term Memory (Persistent Context):
{long_term_memory}

### Input (Sliding Window):
{history_text}

### Response:
<think>
[Your internal monologue and analysis goes here]
</think>
Score: [Your Score]
Memory_Update: [Brief Summary]"""

            else:  # step-by-step (default)
                prompt = f"""### Instruction:
You are a Macro Quant Strategist.
1. Analyze the market data history below (Window of {WINDOW_SIZE} days).
2. Consider the 'Long Term Memory' of key drivers.
3. Provide step-by-step reasoning for your prediction.
4. Predict the sentiment score for TODAY ({current_date.strftime('%Y-%m-%d')}) from -5 (Bearish) to +5 (Bullish).
5. Update the 'Long Term Memory' (max 200 chars).

### Long Term Memory (Persistent Context):
{long_term_memory}

### Input (Sliding Window):
{history_text}

### Response:
Reasoning:
1. [Step 1]
2. [Step 2]
...
Score: [Your Score]
Memory_Update: [Brief Summary]"""

        # Inference with Retry Mechanism
        max_retries = 3
        score = 0.0
        
        # Adjust generation params for simple mode
        gen_temp = 0.1 if SIMPLE_MODE else 0.6
        gen_tokens = 32 if SIMPLE_MODE else 256

        for attempt in range(max_retries):
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, max_new_tokens=gen_tokens, use_cache=True, temperature=gen_temp
            )
            out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse Score & Memory
            try:
                response_part = out_text.split("### Response:")[-1].strip()

                # Parse Score - Adaptive search
                import re
                
                # 1. Try finding "Score: X"
                score_matches = list(re.finditer(r"Score:\s*([+\-]?\d+(\.\d+)?)", response_part))
                if score_matches:
                    score = float(score_matches[-1].group(1))
                    found = True
                else:
                    # 2. If Simple Mode, try finding JUST a number at the start or end
                    if SIMPLE_MODE:
                        # Match a number that is the whole string or separated by newlines
                        bare_match = re.search(r"^([+\-]?\d+(\.\d+)?)$", response_part.strip())
                        if bare_match:
                            score = float(bare_match.group(1))
                            found = True
                        else:
                            found = False
                    else:
                        found = False

                if found:
                    if not SIMPLE_MODE:
                        # Parse Memory Update (only if score found)
                        memory_match = re.search(r"Memory_Update:(.*)", response_part, re.DOTALL)
                        if memory_match:
                            new_memory = memory_match.group(1).strip()
                            # Safety check
                            if len(new_memory) > 10 and "Score" not in new_memory:
                                long_term_memory = new_memory[:250]
                            
                    # Success, break loop
                    break
                else:
                    print(f"  [Warning] No score found (Attempt {attempt+1}/{max_retries}). Retrying...")
                    print(f"  [Debug] Raw LLM Output:\n{out_text}\n" + "-"*20)
                    score = 0.0 # Reset for retry
                    
            except Exception as e:
                print(f"  [Error] Parsing failed: {e} (Attempt {attempt+1}/{max_retries}). Retrying...")
                print(f"  [Debug] Raw LLM Output:\n{out_text}\n" + "-"*20)
                score = 0.0

        results.append(
            {"Date": current_date, "Score": score, "Memory_State": long_term_memory}
        )

    # 3. Calculate Returns (Legacy Framework)
    df_res = pd.DataFrame(results).set_index("Date")

    # Calculate Strategy Returns
    # Merge with next day return (T+1)
    gold_returns = gold_price["Close"].pct_change().shift(-1)

    strategy_df = df_res.join(gold_returns.rename("Gold_Ret"))

    # Signal Logic
    strategy_df["Position"] = np.where(strategy_df["Score"] > 0, 1, -1)
    strategy_df["Position"] = np.where(
        strategy_df["Score"] == 0, 0, strategy_df["Position"]
    )

    strategy_df["Strategy_Ret"] = strategy_df["Position"] * strategy_df["Gold_Ret"]

    # Cumulative
    strategy_df["Cum_Gold"] = (1 + strategy_df["Gold_Ret"].fillna(0)).cumprod()
    strategy_df["Cum_Strategy"] = (1 + strategy_df["Strategy_Ret"].fillna(0)).cumprod()

    # Calculate Drawdown for Strategy
    cum_max = strategy_df["Cum_Strategy"].cummax()
    drawdown = (strategy_df["Cum_Strategy"] - cum_max) / cum_max
    max_drawdown = drawdown.min()

    # Final Return
    final_ret = strategy_df["Cum_Strategy"].iloc[-1] - 1

    # ================= Performance Metrics =================
    # Annualize factor (assuming daily data, 252 trading days)
    ann_factor = 252

    # 1. Sharpe Ratio (Risk-Free Rate assumed 0 for simplicity)
    # Strategy Sharpe
    strat_mean = strategy_df["Strategy_Ret"].mean() * ann_factor
    strat_std = strategy_df["Strategy_Ret"].std() * np.sqrt(ann_factor)
    strat_sharpe = strat_mean / strat_std if strat_std != 0 else 0

    # Gold Sharpe
    gold_mean = strategy_df["Gold_Ret"].mean() * ann_factor
    gold_std = strategy_df["Gold_Ret"].std() * np.sqrt(ann_factor)
    gold_sharpe = gold_mean / gold_std if gold_std != 0 else 0

    # 2. Alpha (Jensen's Alpha) & Beta
    # Regression: Strategy_Ret = Alpha + Beta * Gold_Ret
    from scipy import stats

    # Filter out NaNs for regression
    clean_df = strategy_df[["Strategy_Ret", "Gold_Ret"]].dropna()
    if len(clean_df) > 10:
        beta, alpha, r_value, p_value, std_err = stats.linregress(
            clean_df["Gold_Ret"], clean_df["Strategy_Ret"]
        )
        # Alpha is daily, annualize it
        alpha_ann = alpha * ann_factor
    else:
        alpha_ann = 0.0
        beta = 0.0

    print("\n" + "=" * 40)
    print(f"📊 Performance Metrics")
    print("=" * 40)
    print(f"Strategy Sharpe: {strat_sharpe:.2f}")
    print(f"Gold (B&H) Sharpe: {gold_sharpe:.2f}")
    print(f"Alpha (Annualized): {alpha_ann:.2%}")
    print(f"Beta: {beta:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print("=" * 40 + "\n")

    # Save & Plot
    strategy_df.to_csv(OUTPUT_CSV)
    print(f"✅ Results saved to {OUTPUT_CSV}")

    plt.figure(figsize=(12, 6))
    
    # Plot Main Strategy
    plt.plot(
        strategy_df.index,
        strategy_df["Cum_Strategy"],
        label=f"Sliding Window AI (Sharpe: {strat_sharpe:.2f}, MDD: {max_drawdown:.2%})",
        color="purple",
        linewidth=2,
    )
    
    extra_metrics = ""
    
    # Plot Original Notebook Strategy (if available)
    if ORIGINAL_STRATEGY_CSV and os.path.exists(ORIGINAL_STRATEGY_CSV):
        try:
            print(f"Loading original strategy data from: {ORIGINAL_STRATEGY_CSV}")
            orig_df = pd.read_csv(ORIGINAL_STRATEGY_CSV)
            # Ensure Date parsing
            if "Date" in orig_df.columns:
                orig_df["Date"] = pd.to_datetime(orig_df["Date"])
                orig_df.set_index("Date", inplace=True)
            
            # Align indices
            common_idx = strategy_df.index.intersection(orig_df.index)
            if not common_idx.empty:
                orig_aligned = orig_df.loc[common_idx]
                
                # Check for needed columns
                if "Cumulative_Strategy" in orig_aligned.columns:
                    o_cum = orig_aligned["Cumulative_Strategy"]
                    o_ret = orig_aligned["Strategy_Return"] if "Strategy_Return" in orig_aligned.columns else o_cum.pct_change()
                    
                    o_mean = o_ret.mean() * ann_factor
                    o_std = o_ret.std() * np.sqrt(ann_factor)
                    o_sharpe = o_mean / o_std if o_std != 0 else 0
                    
                    o_max = o_cum.cummax()
                    o_dd = (o_cum - o_max) / o_max
                    o_mdd = o_dd.min()
                    
                    o_final = o_cum.iloc[-1] - 1

                    print("-" * 20)
                    print(f"📉 Original Strategy (5.1.4)")
                    print(f"Sharpe: {o_sharpe:.2f}")
                    print(f"Max Drawdown: {o_mdd:.2%}")
                    print(f"Final Return: {o_final*100:.2f}%")
                    print("-" * 20)

                    plt.plot(
                        orig_aligned.index,
                        o_cum,
                        label=f"Original Notebook Strategy (Sharpe: {o_sharpe:.2f}, MDD: {o_mdd:.2%})",
                        color="green",
                        linestyle=":",
                        linewidth=1.5,
                    )
                    
                    # Update Metrics Text
                    extra_metrics += f"\nOriginal Ret: {o_final*100:.2f}%"
        except Exception as e:
            print(f"Warning: Failed to load original strategy CSV: {e}")

    # Plot Comparison Strategy if provided
    if COMPARE_CSV and os.path.exists(COMPARE_CSV):
        try:
            print(f"Loading comparison data from: {COMPARE_CSV}")
            comp_df = pd.read_csv(COMPARE_CSV, index_col=0, parse_dates=True)
            # Ensure indices align for fair comparison
            common_idx = strategy_df.index.intersection(comp_df.index)
            if not common_idx.empty:
                comp_aligned = comp_df.loc[common_idx]
                
                # Calculate Metrics for Comparison
                # Assuming 'Strategy_Ret' or 'Cum_Strategy' exists. Prefer recalculating from returns to be safe.
                if "Strategy_Ret" in comp_aligned.columns:
                    c_ret = comp_aligned["Strategy_Ret"]
                    c_cum = (1 + c_ret.fillna(0)).cumprod()
                elif "Cum_Strategy" in comp_aligned.columns:
                     c_cum = comp_aligned["Cum_Strategy"]
                     c_ret = c_cum.pct_change()
                else:
                    c_cum = None

                if c_cum is not None:
                    c_mean = c_ret.mean() * ann_factor
                    c_std = c_ret.std() * np.sqrt(ann_factor)
                    c_sharpe = c_mean / c_std if c_std != 0 else 0
                    
                    c_max = c_cum.cummax()
                    c_dd = (c_cum - c_max) / c_max
                    c_mdd = c_dd.min()
                    
                    plt.plot(
                        comp_aligned.index,
                        c_cum,
                        label=f"Comparison (Sharpe: {c_sharpe:.2f}, MDD: {c_mdd:.2%})",
                        color="orange",
                        linestyle="-.",
                        linewidth=2,
                    )
        except Exception as e:
            print(f"Warning: Failed to load comparison CSV: {e}")

    plt.plot(
        strategy_df.index,
        strategy_df["Cum_Gold"],
        label="Gold Benchmark (Buy & Hold)",
        color="gray",
        linestyle="--",
    )

    # Add Text Box with Metrics to Plot
    metrics_text = (
        f"Sharpe (AI): {strat_sharpe:.2f}\n"
        f"Alpha: {alpha_ann:.2%}\n"
        f"Max DD: {max_drawdown:.2%}\n"
        f"Final Ret: {final_ret*100:.2f}%"
        f"{extra_metrics}"
    )
    plt.text(
        0.02,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title(f"Sliding Window Backtest (Window={WINDOW_SIZE})\nGold vs AI Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(OUTPUT_CHART)

    print(f"✅ Chart saved to {OUTPUT_CHART}")
    print(f"🚀 Final Strategy Return: {final_ret*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sliding Window Backtest")
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
        "--news-file",
        type=str,
        default=DEFAULT_NEWS_FILE,
        help="Path to the news CSV file",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=DEFAULT_CACHE_FILE,
        help="Path to the gold price cache CSV",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-end-date",
        type=str,
        default=DEFAULT_DOWNLOAD_END_DATE,
        help="End date for data downloading",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Sliding window size in days",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_REASONING_MODE,
        choices=["thinking", "step-by-step"],
        help="Reasoning mode: 'thinking' (internal monologue) or 'step-by-step' (structured reasoning)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default=DEFAULT_OUTPUT_CHART,
        help="Path to save results Chart",
    )
    parser.add_argument(
        "--original-csv",
        type=str,
        default=DEFAULT_ORIGINAL_STRATEGY_CSV,
        help="Path to the original notebook output CSV (q4_strategy_daily.csv)",
    )
    parser.add_argument(
        "--compare-csv",
        type=str,
        default=None,
        help="Path to another result CSV to plot for comparison",
    )

    parser.add_argument(
        "--simple-mode",
        action="store_true",
        help="Enable simple mode for deterministic models (Single day input, no reasoning, no memory).",
    )

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
    REASONING_MODE = args.mode
    OUTPUT_CSV = args.output_csv
    OUTPUT_CHART = args.output_chart
    COMPARE_CSV = args.compare_csv
    ORIGINAL_STRATEGY_CSV = args.original_csv
    SIMPLE_MODE = args.simple_mode

    run_backtest()
