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

    def _download_yearly_data(self, country_code, start_year, end_year, label, query_fn, sleep_seconds):
        all_data = []
        for year in range(start_year, end_year + 1):
            start = pd.Timestamp(f"{year}-01-01", tz='UTC')
            end = pd.Timestamp(f"{year+1}-01-01", tz='UTC') if year < end_year else pd.Timestamp(datetime.now(), tz='UTC')
            try:
                print(f"  Fetching {label} {year}...")
                df = query_fn(country_code, start=start, end=end)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  Error {year}: {e}")
            time.sleep(sleep_seconds)
        return all_data

    def _build_output_path(self, category, filename):
        output_dir = os.path.join(self.data_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def download_demand(self, country_code, start_year, end_year):
        print(f"\\n--- Downloading Demand for {country_code} ---")
        all_data = self._download_yearly_data(
            country_code,
            start_year,
            end_year,
            "demand",
            self.client.query_load,
            1,
        )

        if all_data:
            final_df = pd.concat(all_data).resample('1h').mean()
            output_path = self._build_output_path("demand", f"entsoe_demand_{country_code}.csv")
            if isinstance(final_df, pd.Series):
                final_df.name = 'demand'
            final_df.to_csv(output_path, index_label='utc_timestamp', header=['demand'])
            print(f"Demand saved to {output_path}")

    def download_generation(self, country_code, start_year, end_year):
        print(f"\\n--- Downloading Generation for {country_code} ---")
        all_data = self._download_yearly_data(
            country_code,
            start_year,
            end_year,
            "generation",
            self.client.query_generation,
            2,
        )

        if all_data:
            final_df = pd.concat(all_data)
            if isinstance(final_df.columns, pd.MultiIndex):
                final_df.columns = ['_'.join(col).strip() for col in final_df.columns.values]
            output_path = self._build_output_path("generation", f"entsoe_generation_{country_code}.csv")
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
        downloader.download_demand(country, start_year, end_year)
        downloader.download_generation(country, start_year, end_year)
        
    print("\\nENTSOE full download complete!")
