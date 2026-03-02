import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

class ENTSOEFullDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.api_key = os.environ.get("ENTSOE_TOKEN_KEY")
        if not self.api_key:
            raise ValueError("ENTSOE_TOKEN_KEY not found in environment variables.")
        self.client = EntsoePandasClient(api_key=self.api_key)

    def download_prices(self, country_code, start_year, end_year):
        print(f"\\n--- Downloading Prices for {country_code} ---")
        all_data = []
        for year in range(start_year, end_year + 1):
            start = pd.Timestamp(f"{year}-01-01", tz='UTC')
            end = pd.Timestamp(f"{year+1}-01-01", tz='UTC') if year < end_year else pd.Timestamp(datetime.now(), tz='UTC')
            try:
                print(f"  Fetching prices {year}...")
                df = self.client.query_day_ahead_prices(country_code, start=start, end=end)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  Error {year}: {e}")
            time.sleep(1) # rate limit
            
        if all_data:
            final_df = pd.concat(all_data)
            output_path = os.path.join(self.data_dir, f"entsoe_prices_{country_code}.csv")
            final_df.to_csv(output_path, index_label='utc_timestamp', header=['price_eur_mwh'])
            print(f"Prices saved to {output_path}")

    def download_generation(self, country_code, start_year, end_year):
        print(f"\\n--- Downloading Generation for {country_code} ---")
        all_data = []
        for year in range(start_year, end_year + 1):
            start = pd.Timestamp(f"{year}-01-01", tz='UTC')
            end = pd.Timestamp(f"{year+1}-01-01", tz='UTC') if year < end_year else pd.Timestamp(datetime.now(), tz='UTC')
            try:
                print(f"  Fetching generation {year}...")
                df = self.client.query_generation(country_code, start=start, end=end)
                if df is not None and not df.empty:
                    # Flatten multi-index columns if they exist
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    all_data.append(df)
            except Exception as e:
                print(f"  Error {year}: {e}")
            time.sleep(2) # generation queries are heavier
            
        if all_data:
            final_df = pd.concat(all_data)
            output_path = os.path.join(self.data_dir, f"entsoe_generation_{country_code}.csv")
            final_df.to_csv(output_path, index_label='utc_timestamp')
            print(f"Generation saved to {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "europe")
    downloader = ENTSOEFullDownloader(data_dir=RAW_DIR)
    
    countries = ['ES', 'FR', 'DE', 'GR', 'IT', 'PT', 'NL', 'BE']
    start_year = 2015
    end_year = datetime.now().year
    
    for country in countries:
        downloader.download_prices(country, start_year, end_year)
        downloader.download_generation(country, start_year, end_year)
        
    print("\\nENTSOE full download complete!")
