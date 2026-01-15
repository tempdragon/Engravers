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
import json
from scipy import stats

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= Configuration Defaults =================
DEFAULT_MODEL_PATH = "llama3_gold_quant_checkpoint" 
DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
DEFAULT_NEWS_FILE = "final/gold_llm_test.jsonl"
DEFAULT_CACHE_FILE = "commodity_data/gold.csv"
# DEFAULTS Set to None to auto-detect from JSONL
DEFAULT_START_DATE = None 
DEFAULT_END_DATE = None

DEFAULT_OUTPUT_CSV = "q4_multistrat_results.csv"
DEFAULT_OUTPUT_CHART = "q4_multistrat_chart.png"
DEFAULT_ORIGINAL_STRATEGY_CSV = "final/q4_strategy_daily.csv"

# ================= 1. Helper Functions =================
def get_next_trading_day(date, valid_dates):
    d = pd.Timestamp(date).normalize()
    idx = valid_dates.searchsorted(d)
    if idx >= len(valid_dates):
        return valid_dates[-1]
    return valid_dates[idx]

def prepare_daily_data(news_file, cache_file):
    print(f"Loading and preparing data from {news_file}...")
    
    if news_file.endswith(".jsonl"):
        records = []
        with open(news_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    inp = item.get("input", "")
                    m_date = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", inp)
                    m_news = re.search(r"News:\s*(.*)", inp, re.DOTALL)
                    if m_date and m_news:
                        records.append({
                            "Date": m_date.group(1),
                            "Headline": m_news.group(1).strip()
                        })
                except Exception as e:
                    continue
        df_news = pd.DataFrame(records)
    else:
        df_news = pd.read_csv(news_file)

    df_news["Date"] = pd.to_datetime(df_news["Date"])
    
    # Auto-detect range for market data download
    min_date = df_news["Date"].min()
    max_date = df_news["Date"].max()
    print(f"News Data Range: {min_date.date()} to {max_date.date()}")
    
    # Download buffer
    download_start = (min_date - timedelta(days=30)).strftime('%Y-%m-%d')
    download_end = (max_date + timedelta(days=10)).strftime('%Y-%m-%d')
    
    if os.path.exists(cache_file):
        gold = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Check if cache covers range
        if gold.index.max() < max_date:
            print("Cache stale, updating...")
            try:
                gold = yf.download("GC=F", start="2020-01-01", end=download_end, progress=False)
                if isinstance(gold.columns, pd.MultiIndex): gold.columns = gold.columns.get_level_values(0)
                gold.to_csv(cache_file)
            except: pass
    else:
        gold = yf.download("GC=F", start="2020-01-01", end=download_end, progress=False)
        if isinstance(gold.columns, pd.MultiIndex): gold.columns = gold.columns.get_level_values(0)
        gold.to_csv(cache_file)

    if gold.index.tz is not None: gold.index = gold.index.tz_localize(None)
    valid_dates = pd.DatetimeIndex(gold.index).normalize()

    df_news["Trading_Date"] = df_news["Date"].apply(lambda x: get_next_trading_day(x, valid_dates))

    daily_records = []
    daily_groups = df_news.groupby("Trading_Date")
    
    for date, group in daily_groups:
        if date not in gold.index: continue
        headlines = group["Headline"].tolist()
        row = gold.loc[date]
        daily_records.append({"Date": date, "Headlines": headlines, "Full_Row": row})

    return pd.DataFrame(daily_records).set_index("Date").sort_index(), gold

def get_llm_score(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, temperature=0.05)
    out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_part = out_text.split("### Response:")[-1].strip()
    match = re.search(r"(?:Score:\s*)?([+\-]?\d+(\.\d+)?)", response_part)
    if match:
        return float(match.group(1))
    return 0.0

# ================= 2. Multi-Strategy Logic =================
def run_multistrat_backtest(args):
    # 1. Setup
    df_daily, gold_price = prepare_daily_data(args.news_file, args.cache_file)
    
    # Filter by Args ONLY if provided, else use full data
    if args.start_date:
        df_daily = df_daily[df_daily.index >= args.start_date]
    if args.end_date:
        df_daily = df_daily[df_daily.index <= args.end_date]
        
    test_dates = df_daily.index
    print(f"Backtest execution range: {test_dates.min()} to {test_dates.max()} ({len(test_dates)} trading days)")

    # 2. Model
    print(f"Loading Model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model, max_seq_length = 4096, dtype = None, load_in_4bit = True,
    )
    try:
        model.load_adapter(args.model_path)
    except:
        model.load_adapter(os.path.abspath(args.model_path))
    FastLanguageModel.for_inference(model)

    # 3. Memory Buffers
    memory_buffer_A = [] # Standard Summarization
    memory_buffer_D = [] # AI Filtered

    results = []

    print(f"🚀 Running Multi-Strategy Backtest (Scores A, B, D)...")
    
    for current_date in tqdm(test_dates):
        try:
            row = df_daily.loc[current_date]
        except KeyError: continue
        
        current_headlines = row['Headlines']
        
        # --- PREPARE MEMORIES (Weighted 14d) ---
        cutoff_date = current_date - timedelta(days=14)
        
        # Memory A (Standard)
        valid_mem_A = [m for m in memory_buffer_A if m['date'] >= cutoff_date]
        memory_buffer_A = valid_mem_A
        mem_strs_A = [f"[HISTORY {m['date'].strftime('%Y-%m-%d')} | W:{max(0.0, 1.0-(current_date-m['date']).days/14.0):.2f}] {m['headline']}" for m in valid_mem_A]
        
        # Memory D (AI Filtered)
        valid_mem_D = [m for m in memory_buffer_D if m['date'] >= cutoff_date]
        memory_buffer_D = valid_mem_D
        mem_strs_D = [f"[AI-MEM {m['date'].strftime('%Y-%m-%d')} | W:{max(0.0, 1.0-(current_date-m['date']).days/14.0):.2f}] {m['headline']}" for m in valid_mem_D]

        # --- GENERATE BASE SIGNALS (3 LLM Calls) ---
        
        # Signal A: Standard News + Memory A
        input_A = f"Date: {current_date.strftime('%Y-%m-%d')}\nNews:\n" + "\n".join([f"- {h}" for h in current_headlines + mem_strs_A])
        score_A = get_llm_score(model, tokenizer, f"### Instruction:\nAnalyze Gold news. Sentiment Score -5 to +5.\n### Input:\n{input_A}\n### Response:\n")
        
        # Signal B: News Only (No Memory)
        input_B = f"Date: {current_date.strftime('%Y-%m-%d')}\nNews:\n" + "\n".join([f"- {h}" for h in current_headlines])
        score_B = get_llm_score(model, tokenizer, f"### Instruction:\nAnalyze Gold news. Sentiment Score -5 to +5.\n### Input:\n{input_B}\n### Response:\n")
        
        # Signal D: News + AI Filtered Memory D
        input_D = f"Date: {current_date.strftime('%Y-%m-%d')}\nNews:\n" + "\n".join([f"- {h}" for h in current_headlines + mem_strs_D])
        score_D = get_llm_score(model, tokenizer, f"### Instruction:\nAnalyze Gold news. Sentiment Score -5 to +5.\n### Input:\n{input_D}\n### Response:\n")

        # --- UPDATE MEMORIES ---
        with torch.no_grad():
            with model.disable_adapters():
                # Update A: Standard Summary
                sum_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nIdentify single most important Gold event (Max 15 words) or 'None'.<|eot_id|><|start_header_id|>user<|end_header_id|>\nNews:\n{chr(10).join(current_headlines)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                s_out = model.generate(**tokenizer([sum_prompt], return_tensors="pt").to("cuda"), max_new_tokens=48, temperature=0.1, use_cache=True)
                raw_sum = tokenizer.batch_decode(s_out, skip_special_tokens=True)[0].split("assistant")[-1].strip()
                if "None" not in raw_sum and len(raw_sum) > 5:
                    memory_buffer_A.append({"headline": raw_sum.replace('"','').split('\n')[0], "date": current_date})
                
                # Update D: AI Filtered Summary
                # 1. Filter Check
                filter_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nIs there ANY event in today's news that will significantly impact Gold prices for the next 2 weeks? Reply ONLY 'YES' or 'NO'.<|eot_id|><|start_header_id|>user<|end_header_id|>\nNews:\n{chr(10).join(current_headlines)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                f_out = model.generate(**tokenizer([filter_prompt], return_tensors="pt").to("cuda"), max_new_tokens=5, temperature=0.05, use_cache=True)
                is_imp = "YES" in tokenizer.batch_decode(f_out, skip_special_tokens=True)[0].split("assistant")[-1].strip().upper()
                
                if is_imp:
                    if "None" not in raw_sum and len(raw_sum) > 5:
                         memory_buffer_D.append({"headline": raw_sum.replace('"','').split('\n')[0], "date": current_date})
                    else:
                        s_out_d = model.generate(**tokenizer([sum_prompt], return_tensors="pt").to("cuda"), max_new_tokens=48, temperature=0.1, use_cache=True)
                        raw_sum_d = tokenizer.batch_decode(s_out_d, skip_special_tokens=True)[0].split("assistant")[-1].strip()
                        if "None" not in raw_sum_d:
                             memory_buffer_D.append({"headline": raw_sum_d.replace('"','').split('\n')[0], "date": current_date})

        # --- STRATEGIES ---
        strats = {'Date': current_date, 'Score_A': score_A, 'Score_B': score_B, 'Score_D': score_D}
        
        # A Strategies (Standard Mem)
        strats['S01_A_Base'] = 1 if score_A > 0 else (-1 if score_A < 0 else 0)
        strats['S02_A_HighConf'] = 1 if score_A > 2 else (-1 if score_A < -2 else 0)
        strats['S03_A_Scaled'] = score_A / 5.0

        # B Strategies (No Mem)
        strats['S04_B_Base'] = 1 if score_B > 0 else (-1 if score_B < 0 else 0)
        strats['S05_B_HighConf'] = 1 if score_B > 2 else (-1 if score_B < -2 else 0)

        # D Strategies (AI Filtered Mem)
        strats['S06_D_Base'] = 1 if score_D > 0 else (-1 if score_D < 0 else 0)
        strats['S07_D_HighConf'] = 1 if score_D > 2 else (-1 if score_D < -2 else 0)
        strats['S08_D_Scaled'] = score_D / 5.0

        # Combinations
        strats['S09_APlusB'] = 1 if (score_A + score_B) > 0 else (-1 if (score_A + score_B) < 0 else 0)
        strats['S10_DPlusB'] = 1 if (score_D + score_B) > 0 else (-1 if (score_D + score_B) < 0 else 0)
        strats['S11_All_Consensus'] = 1 if (score_A > 0 and score_B > 0 and score_D > 0) else (-1 if (score_A < 0 and score_B < 0 and score_D < 0) else 0)
        
        # Agreement
        strats['S12_A_Agree_B'] = 1 if (score_A > 0 and score_B > 0) else (-1 if (score_A < 0 and score_B < 0) else 0)
        strats['S13_D_Agree_B'] = 1 if (score_D > 0 and score_B > 0) else (-1 if (score_D < 0 and score_B < 0) else 0)

        # Divergence
        if score_D > 2 and score_B < 1: strats['S14_D_Alpha'] = 1
        elif score_D < -2 and score_B > -1: strats['S14_D_Alpha'] = -1
        else: strats['S14_D_Alpha'] = 0

        # Scaled Avg
        strats['S15_Avg_ABD'] = (score_A + score_B + score_D) / 15.0

        results.append(strats)

    # ================= 3. Analysis =================
    df_res = pd.DataFrame(results).set_index("Date")
    gold_returns = gold_price["Close"].pct_change().shift(-1)
    
    summary_metrics = []
    plt.figure(figsize=(15, 8))
    
    # Gold
    gold_cum = (1 + gold_returns.fillna(0)).cumprod()
    final_gold = gold_cum.iloc[-1] - 1
    plt.plot(gold_cum.index, gold_cum, label=f"Gold B&H ({final_gold:.1%})", color="black", linestyle="--", linewidth=2, alpha=0.5)
    
    print("\n" + "="*95)
    print(f"{('STRATEGY'):<25} | {'RET':<8} | {'SHARPE':<6} | {'MDD':<7} | {'ALPHA':<7}")
    print("-" * 95)
    print(f"{('Gold Benchmark'):<25} | {final_gold*100:6.1f}% | {'N/A':<6} | {'N/A':<7} | {'0.0%':<7}")

    for strat in [c for c in df_res.columns if c.startswith('S')]:
        s_pos = df_res[strat]
        aligned_gold = gold_returns.loc[s_pos.index]
        
        strat_ret = s_pos * aligned_gold
        strat_cum = (1 + strat_ret.fillna(0)).cumprod()
        
        final_ret = strat_cum.iloc[-1] - 1
        ann_mean = strat_ret.mean() * 252
        ann_std = strat_ret.std() * np.sqrt(252)
        sharpe = ann_mean / ann_std if ann_std != 0 else 0
        mdd = ((strat_cum - strat_cum.cummax()) / strat_cum.cummax()).min()
        
        clean_df = pd.DataFrame({"Strat": strat_ret, "Gold": aligned_gold}).dropna()
        if len(clean_df) > 10:
            beta, alpha, r, p, err = stats.linregress(clean_df["Gold"], clean_df["Strat"])
            alpha_ann = alpha * 252
        else:
            alpha_ann = 0.0

        summary_metrics.append({
            "Name": strat, "Return": final_ret, "Sharpe": sharpe, 
            "MDD": mdd, "Alpha": alpha_ann, "Series": strat_cum
        })

    summary_metrics.sort(key=lambda x: x['Sharpe'], reverse=True)
    
    for m in summary_metrics[:5]:
        print(f"{m['Name']:<25} | {m['Return']*100:6.1f}% | {m['Sharpe']:6.2f} | {m['MDD']:6.2%} | {m['Alpha']:6.2%}")
        plt.plot(m['Series'].index, m['Series'], label=f"{m['Name']} (Sh: {m['Sharpe']:.2f})")

    if len(summary_metrics) > 5:
        print("-" * 95)
        print("... Remaining Strategies ...")
        for m in summary_metrics[5:]:
            print(f"{m['Name']:<25} | {m['Return']*100:6.1f}% | {m['Sharpe']:6.2f} | {m['MDD']:6.2%} | {m['Alpha']:6.2%}")

    if args.original_csv and os.path.exists(args.original_csv):
        try:
            orig_df = pd.read_csv(args.original_csv)
            if "Date" in orig_df.columns:
                orig_df["Date"] = pd.to_datetime(orig_df["Date"])
                orig_df.set_index("Date", inplace=True)
            common = gold_cum.index.intersection(orig_df.index)
            if not common.empty:
                o_cum = orig_df.loc[common, "Cumulative_Strategy"]
                o_final = o_cum.iloc[-1] - 1
                
                o_ret = o_cum.pct_change().dropna()
                g_ret_aligned = gold_returns.loc[o_ret.index].dropna()
                c_idx = o_ret.index.intersection(g_ret_aligned.index)
                if len(c_idx) > 10:
                    b_o, a_o, r, p, e = stats.linregress(g_ret_aligned.loc[c_idx], o_ret.loc[c_idx])
                    alpha_o = a_o * 252
                else: alpha_o = 0.0

                plt.plot(common, o_cum, label=f"Original 5.1.4 ({o_final:.1%})", color="green", linestyle=":", linewidth=2)
                print("-" * 95)
                print(f"{('Original 5.1.4'):<25} | {o_final*100:6.1f}% | {'N/A':<6} | {'N/A':<7} | {alpha_o:6.2%}")
        except: pass

    plt.title("Top Strategies vs Gold vs Legacy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output_chart)
    df_res.to_csv(args.output_csv)
    print("="*95)
    print(f"✅ Results saved to {args.output_csv}")
    print(f"✅ Chart saved to {args.output_chart}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--news-file", type=str, default=DEFAULT_NEWS_FILE)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-chart", type=str, default=DEFAULT_OUTPUT_CHART)
    parser.add_argument("--original-csv", type=str, default=DEFAULT_ORIGINAL_STRATEGY_CSV)
    args = parser.parse_args()
    run_multistrat_backtest(args)