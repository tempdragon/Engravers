import pandas as pd
import os

class NewsFeed:
    def __init__(self, file_path=None):
        self.news_data = {}
        if file_path and os.path.exists(file_path):
            self.load_csv(file_path)
        else:
            print(f"⚠️ News file not found: {file_path}. Using empty news feed.")

    def load_csv(self, file_path):
        """
        Expects a CSV with columns: 'Date', 'Headline' (or similar)
        """
        try:
            df = pd.read_csv(file_path)
            # Ensure Date is string YYYY-MM-DD
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                
                # Group by Date and aggregate headlines
                # Assuming column 'Headline' or 'Title' exists
                text_col = None
                for col in ['Headline', 'Title', 'News', 'Content']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col:
                    grouped = df.groupby('Date')[text_col].apply(list).to_dict()
                    self.news_data = grouped
                    print(f"📰 Loaded news for {len(self.news_data)} days from {file_path}")
                else:
                    print(f"⚠️ No suitable text column found in {file_path}")
        except Exception as e:
            print(f"❌ Error loading news CSV: {e}")

    def get_news(self, date_str):
        """
        Returns a list of headlines for the given date string (YYYY-MM-DD).
        """
        return self.news_data.get(date_str, [])

    def get_mock_news(self, date_str):
        """
        Example of a dynamic function that could be called via getattr
        """
        return [f"Mock news for {date_str}"]
