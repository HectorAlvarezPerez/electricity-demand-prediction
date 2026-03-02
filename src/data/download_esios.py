import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
import time

load_dotenv()

class ESIOSCataloniaDownloader:
    BASE_URL = "https://api.esios.ree.es/indicators"
    
    # Selection of interesting indicators for Catalonia (geo_id = 10)
    # The API filter ?geo_ids[]=10 will be used to limit geographic scope
    INDICATORS = {
        "demand_real": 10343,        # Real demand
        "demand_scheduled": 10342,   # Scheduled demand
        "generation_solar": 10328,   # Solar PV Real Generation (may be national scaled but let's see if CAT exists)
        "generation_wind": 10324,    # Wind Onshore
        "generation_hydro": 10332,   # Hydro
        "generation_nuclear": 10331, # Nuclear
        "generation_ccgt": 10330,    # Combined Cycle
        "spot_price_spain": 600,     # Spot market price (National, affects CAT identical)
        "emissions_co2": 10350       # CO2 Emissions (gCO2eq/kWh)
    }

    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.token = os.environ.get("REE_TOKEN") or os.environ.get("ESIOS_TOKEN_KEY")
        if not self.token:
            raise ValueError("ESIOS token must be provided in .env (REE_TOKEN or ESIOS_TOKEN_KEY)")
        self.headers = {
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "Content-Type": "application/json",
            "x-api-key": self.token
        }

    def fetch_indicator(self, indicator_name, indicator_id, start_date, end_date):
        url = f"{self.BASE_URL}/{indicator_id}"
        
        # geo_ids=9 and geo_limit=ccaa is Catalonia
        params = {
            "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_trunc": "hour",
            "geo_limit": "ccaa",
            "geo_ids": 9
        }
        
        print(f"  Fetching {indicator_name} ({indicator_id}) for {start_date.year}-{start_date.month:02d}...")
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code != 200:
            print(f"    Failed. Status: {response.status_code}")
            return pd.DataFrame()
            
        data = response.json()
        if 'indicator' not in data or 'values' not in data['indicator']:
            return pd.DataFrame()
            
        values = data['indicator']['values']
        if not values:
            return pd.DataFrame()
            
        df = pd.DataFrame(values)
        cols_to_keep = ['datetime', 'value', 'geo_id', 'geo_name']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        
        # In case the ?geo_ids filter wasn't strictly respected by API, filter manually here if it exists
        if 'geo_id' in df.columns and 9 in df['geo_id'].unique():
             df = df[df['geo_id'] == 9]
             
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime')
        return df

    def download_all(self, start_year=2015):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now()
        
        for name, ind_id in self.INDICATORS.items():
            print(f"\\n--- Downloading {name} (Indicator {ind_id}) ---")
            all_data = []
            current_date = start_date
            
            while current_date < end_date:
                # Fetch month by month
                next_date = current_date + relativedelta(months=1)
                if next_date > end_date:
                    next_date = end_date
                    
                df_chunk = self.fetch_indicator(name, ind_id, current_date, next_date)
                if not df_chunk.empty:
                    df_chunk = df_chunk.rename(columns={'value': name})
                    # Drop geo columns to make merging easier later
                    df_chunk = df_chunk[[name]]
                    all_data.append(df_chunk)
                    
                current_date = next_date
                time.sleep(0.5) # ESIOS rate limiting
                
            if all_data:
                df_indicator = pd.concat(all_data)
                df_indicator = df_indicator[~df_indicator.index.duplicated(keep='first')]
                
                output_path = os.path.join(self.data_dir, f"esios_{name}.csv")
                df_indicator.to_csv(output_path)
                print(f"-> Saved {len(df_indicator)} hours to {output_path}")
            else:
                print(f"-> No detailed Catalonia data found for {name}.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "catalonia")
    downloader = ESIOSCataloniaDownloader(data_dir=RAW_DIR)
    downloader.download_all(start_year=2015)
