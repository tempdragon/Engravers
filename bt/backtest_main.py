import backtrader as bt
import datetime
import pandas as pd
import json
import os
import argparse
import toml
import importlib
from llama_cpp import Llama
from news_feed import NewsFeed

# ==================================================================================
# 1. LLM Client Wrapper
# ==================================================================================

class LlamaCppClient:
    def __init__(self, model_path, model_type="standard", context_window=2048):
        self.model_type = model_type
        print(f"🤖 Loading LLM from {model_path} (Type: {model_type})...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_window,
                n_gpu_layers=-1, # Offload all layers to GPU if available
                verbose=False
            )
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.llm = None

    def get_decision(self, context_text):
        if not self.llm:
            return "HOLD"

        # Construct Prompt based on model type
        if self.model_type == "reasoning":
            # For reasoning models (e.g., DeepSeek-R1), allow free thought
            prompt = f"""
            Analyze the following financial data and news for the commodity.
            
            Context:
            {context_text}
            
            Think step-by-step about the market trend, sentiment, and risks.
            Finally, provide a trading decision: BUY, SELL, or HOLD.
            
            Output your final decision in the last paragraph clearly.
            """
        else:
            # Standard mode
            prompt = f"""
            You are a financial trading assistant. Analyze the data below and decide whether to BUY, SELL, or HOLD.
            
            Context:
            {context_text}
            
            Provide a short analysis. Then, on a new line, output exactly one word: BUY, SELL, or HOLD.
            """

        # Generate
        output = self.llm(
            prompt,
            max_tokens=256,
            stop=["User:", "\n\n\n"],
            echo=False
        )
        
        response_text = output['choices'][0]['text'].strip()
        # print(f"📝 LLM Thought:\n{response_text}\n{'-'*20}")
        
        # Parse Decision (Search for keywords)
        # We look for the last occurrence of the keywords to capture the final conclusion
        response_upper = response_text.upper()
        
        # Simple keyword scoring or last-found priority
        decisions = {"BUY": -1, "SELL": -1, "HOLD": -1}
        
        for d in decisions:
            decisions[d] = response_upper.rfind(d)
            
        # Find the keyword that appears last in the text
        best_decision = max(decisions, key=decisions.get)
        
        if decisions[best_decision] == -1:
            # No keyword found
            print("⚠️ LLM did not output a clear decision. Defaulting to HOLD.")
            return "HOLD"
            
        return best_decision

class HybridClient:
    """
    Stub for a hybrid client that combines LLM features with traditional logic.
    """
    def __init__(self):
        pass
        
    def get_decision(self, context):
        # Placeholder logic
        return "HOLD"

# ==================================================================================
# 2. Backtrader Strategy
# ==================================================================================

class LLMStrategy(bt.Strategy):
    params = (
        ('llm_client', None),
        ('news_feed', None), # NewsFeed object
        ('news_func_name', 'get_news'), # Name of the function to call on news_feed
        ('ticker', 'Unknown'),
        ('period', 5), # Lookback period for price context
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datavolume = self.datas[0].volume
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def next(self):
        # Skip if order is pending
        if self.order:
            return

        # 1. Prepare Context
        # Get last N days of price data
        if len(self) < self.params.period:
            return

        price_history = []
        for i in range(-self.params.period, 0):
            price_history.append(f"Day {i}: Close={self.dataclose[i]:.2f}, Vol={self.datavolume[i]}")
        
        current_price_str = f"Today Open: {self.dataopen[0]:.2f}"
        
        # Get News
        current_date_str = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        
        news_items = []
        if self.params.news_feed:
            # Dynamic call to the specified function
            try:
                news_func = getattr(self.params.news_feed, self.params.news_func_name)
                news_items = news_func(current_date_str)
            except AttributeError:
                print(f"⚠️ News function '{self.params.news_func_name}' not found in NewsFeed.")
                news_items = []
        
        news_str = "\n".join(news_items) if news_items else "No significant news."

        context = f"""
        Ticker: {self.params.ticker}
        Recent Price History:
        {chr(10).join(price_history)}
        {current_price_str}
        
        News for {current_date_str}:
        {news_str}
        """

        # 2. Get Decision from LLM
        if self.params.llm_client:
            decision = self.params.llm_client.get_decision(context)
        else:
            decision = "HOLD" # Mock mode

        # 3. Execute
        self.log(f"LLM Decision: {decision}")
        
        if decision == "BUY" and not self.position:
            self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
            self.order = self.buy()
            
        elif decision == "SELL" and self.position:
            self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
            self.order = self.sell()
            
        # Note: Simple logic - Buy if no position, Sell to close. 
        # Could be extended to Shorting or resizing.

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

# ==================================================================================
# 3. Main Runner
# ==================================================================================

def run_backtest(ticker="gold", config_path="backtest.toml"):
    # Load Config
    if not os.path.exists(config_path):
        # Try looking in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "backtest.toml")

    try:
        config = toml.load(config_path)
        print(f"ℹ️  Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
        config = {}

    bt_conf = config.get("backtest", {})
    llm_conf = config.get("llm", {})
    strategy_conf = config.get("strategy", {})

    # Parse dates
    start_date_str = bt_conf.get("start_date", "2020-01-01")
    end_date_str = bt_conf.get("end_date", "2024-01-01")
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    
    initial_cash = bt_conf.get("initial_cash", 100000.0)
    model_path = llm_conf.get("model_path", "model.gguf")
    model_type = llm_conf.get("model_type", "standard")
    
    # News Config
    news_file = strategy_conf.get("news_file", "news.csv")
    news_func_name = strategy_conf.get("news_func", "get_news")

    # A. Setup Cerebro
    cerebro = bt.Cerebro()
    
    # B. Load Data
    csv_file = f"commodity_data/{ticker.lower()}.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Data file '{csv_file}' not found.")
        print("   Please run 'python3 bt/download_commodities.py' first.")
        return

    print(f"📈 Loading data for {ticker} from {csv_file}...")
    
    # GenericCSVData configuration matching our download script format
    # Date, Open, High, Low, Close, Volume, OpenInterest
    data = bt.feeds.GenericCSVData(
        dataname=csv_file,
        dtformat='%Y-%m-%d',
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=6,
        header=0, # Expect header row
        fromdate=start_date,
        todate=end_date,
        plot=False
    )
    cerebro.adddata(data)

    # C. Load News Data
    print(f"📰 Initializing NewsFeed from {news_file}...")
    news_feed = NewsFeed(file_path=news_file)

    # D. Initialize LLM Client
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model file '{model_path}' not found.")
        print("   Running in MOCK mode (no LLM).")
        client = None # Or implement a MockClient
    else:
        client = LlamaCppClient(model_path=model_path, model_type=model_type)

    # E. Add Strategy
    cerebro.addstrategy(
        LLMStrategy, 
        llm_client=client, 
        news_feed=news_feed, 
        news_func_name=news_func_name,
        ticker=ticker
    )

    # F. Run
    cerebro.broker.setcash(initial_cash)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM Backtest')
    parser.add_argument('--ticker', type=str, default='gold', help='Ticker/Commodity name (e.g., gold, silver)')
    parser.add_argument('--config', type=str, default='backtest.toml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    run_backtest(ticker=args.ticker, config_path=args.config)
