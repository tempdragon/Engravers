# Backtesting Module

This directory contains scripts for downloading financial data and running backtests using Backtrader and LLM-based strategies.

## Contents

*   `download_commodities.py`: Script to download historical commodity and index data from Yahoo Finance.
*   `backtest_main.py`: Main script to run backtests using `backtrader` and `llama-cpp-python`.
*   `download.toml`: Configuration file for the downloader (tickers, proxy, dates).
*   `backtest.toml`: Configuration file for the backtester (dates, cash, model).
*   `requirements.txt`: Python dependencies for this module.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download Data

Configure your tickers and settings in `download.toml`, then run:

```bash
python3 download_commodities.py
# Or specify a custom config
python3 download_commodities.py --config my_download.toml
```

This will download CSV files to the `commodity_data/` directory (or whatever is configured).

### 2. Run Backtest

Configure your backtest settings in `backtest.toml`, then run:

```bash
python3 backtest_main.py --ticker gold
# Or specify a custom config
python3 backtest_main.py --ticker gold --config my_backtest.toml
```

Arguments:
*   `--ticker`: The commodity/index name (must match a downloaded CSV filename, e.g., `gold`, `silver`).
*   `--config`: Path to the TOML configuration file (default: `backtest.toml`).
