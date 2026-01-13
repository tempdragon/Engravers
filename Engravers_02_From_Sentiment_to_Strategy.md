---
Title: From Sentiment to Strategy: Engineering a "Stateful" AI Macro Strategist for Gold Futures (by Group "Engravers")
Date: 2026-01-13
Category: Reflective Report
Tags: GenAI, Macro Trading, Llama-3, Financial Engineering
---

By Group "Engravers"

In our previous post, we discussed building a robust data pipeline. In this phase, we moved from simple *sentiment analysis* to building a **"Stateful" AI Macro Strategist** capable of complex reasoning. This report details the evolution of our methodology, comparing our initial "Phase 1" approach with the enhanced "Phase 2" architecture designed for high-fidelity trading.

## 1. The Paradigm Shift: From "Reader" to "Strategist"

Our initial experiments (Phase 1) treated financial news processing as a classification task: *Read Headline -> Predict Bullish/Bearish*. While effective for measuring sentiment, real-world trading requires more. A human strategist doesn't just read a headline in isolation; they:
1.  **Synthesize** multiple news events from the entire day.
2.  **Cross-reference** news with market data (Price action, Technicals).
3.  **Remember** the structural narrative (e.g., "We are in a rate-cut cycle") even if today's news is silent on it.

We re-engineered our pipeline to mimic this cognitive process.

## 2. Methodology Evolution: A Comparative Analysis

### 2.1 Data Context: From Single-Shot to Multi-Modal Aggregation

| Feature | Phase 1 (Baseline) | Phase 2 (Enhanced) |
| :--- | :--- | :--- |
| **Input Scope** | Single Headline per sample. | **Daily Aggregation:** All headlines from the trading day combined into one prompt. |
| **Market Data** | None (Text only). | **Multi-Modal:** Integrated OHLCV + Technical Indicators (**RSI, MACD, Bollinger Bands, KDJ**). |
| **Philosophy** | "Is this sentence positive?" | "Given this news *and* this overbought RSI, what is the trade?" |

**Why this matters:** A "War Warning" headline might be bullish for Gold in a vacuum. But if Gold is already up 5% and RSI is at 85 (Overbought), a human strategist might fade the move. Phase 2 enables the model to learn these nuanced, conditional dependencies.

### 2.2 Temporal Logic: Handling the "Time Flow"

**The Weekend Problem:**
Markets sleep, but news doesn't. In Phase 1, we identified the issue of Saturday news. In Phase 2, we implemented a robust **Trading Calendar Alignment**:
*   A "Next Valid Trading Day" algorithm automatically shifts weekend and holiday news to the next open market session (e.g., Saturday news -> Monday price action).
*   This ensures strict causal consistency: the model is trained only on information actually available before the trade.

**The "Memory" Upgrade (Backtesting):**
Our most significant architectural change is the introduction of **Stateful Memory** in the backtest engine:
*   **Sliding Window:** Instead of seeing one day at a time, the model sees a rolling window of the past **3 days**. This allows it to detect *trend evolution* (e.g., "Rumors on Mon" -> "Confirmation on Wed").
*   **Long-Term Persistent Memory:** We injected a mutable "Memory State" (approx. 200 tokens) that persists across the entire simulation. The model explicitly updates this memory state (e.g., *“Driver Update: Fed pivoted to dovish”*), allowing it to maintain a strategic narrative across weeks, mimicking a human trader's retention of macro themes.

### 2.3 The "Reflective" Training Loop

To improve the quality of our fine-tuning data, we moved beyond static labels to a **Self-Reflective Data Generation** process:
1.  **Teacher-Student Loop:** We use the base model to predict scores on the training set.
2.  **Error Detection:** If the model's prediction diverges significantly from the Ground Truth (Market Return), the script appends a **Reflection Block** to the training sample.
    *   *Example:* "Model predicted Bearish, but Market rallied +2%. **Reflection:** The RSI was oversold, overriding the bearish news."
3.  **Outcome:** The fine-tuned model (Student) learns not just the rule, but the *exception* logic, effectively "learning from its past mistakes" before they happen in live trading.

## 3. Preliminary Experiments & Results

*(PLACEHOLDER: Insert Sharpe Ratio and Alpha comparison chart here once backtest completes)*

*(PLACEHOLDER: Insert Sliding Window Backtest Cumulative Return Plot here)*

*(PLACEHOLDER: Insert "Long Term Memory" evolution examples from logs here)*

## 4. Tech Stack & Implementation Details

We optimized our stack for high-performance deployment:
*   **Training Hardware:** **RTX 5090 (24GB)**. The massive VRAM allows for full parameter fine-tuning or high-rank LoRA, ensuring the model captures deep macro relationships.
*   **Inference Hardware:** **Tesla V100 (16GB)**. The optimized inference scripts (`generate_enhanced_dataset.py`, `run_backtest_sliding.py`) fit comfortably within 16GB VRAM using 4-bit quantization.
*   **Libraries:** `Unsloth` for 2x faster training, `Pandas/YFinance` for data engineering, and `Matplotlib` for visualizing the Alpha/Sharpe metrics.

## 4. Conclusion

By evolving from a stateless headline classifier to a stateful, multi-modal strategist, we aim to bridge the gap between NLP and quantitative trading. The "Engravers" Phase 2 architecture doesn't just read the news; it contextualizes it within the market's technical reality and remembers the story as it unfolds.

*Next steps include deploying this model for paper trading and further refining the memory update mechanism with Reinforcement Learning (RLHF).*

*(PLACEHOLDER: Add discussion on specific trade examples where Memory State corrected a wrong signal)*
