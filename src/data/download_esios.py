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
    DEMAND_URL = "https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real"
    GEO_LIMIT = "ccaa"
    GEO_ID = 9
    DEMAND_SERIES = {
        "demanda real": "demand_real",
        "demanda programada": "demand_scheduled",
    }
    
    # Selection of interesting indicators for Catalonia.
    INDICATORS = {
        "generation_solar": 10328,   # Solar PV Real Generation (may be national scaled but let's see if CAT exists)
        "generation_wind": 10324,    # Wind Onshore
        "generation_hydro": 10332,   # Hydro
        "generation_nuclear": 10331, # Nuclear
        "generation_ccgt": 10330,    # Combined Cycle
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

    def _build_output_path(self, indicator_name):
        category = indicator_name.split("_", 1)[0]
        output_dir = os.path.join(self.data_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"esios_{indicator_name}.csv")

    def _format_redata_datetime(self, value):
        return value.strftime("%Y-%m-%dT%H:%M")

    def fetch_demand(self, start_date, end_date):
        params = {
            "start_date": self._format_redata_datetime(start_date),
            "end_date": self._format_redata_datetime(end_date),
            "time_trunc": "hour",
            "geo_trunc": "electric_system",
            "geo_limit": self.GEO_LIMIT,
            "geo_ids": self.GEO_ID,
        }

        print(f"  Fetching demand for {start_date.year}-{start_date.month:02d}...")
        response = requests.get(self.DEMAND_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"    Failed. Status: {response.status_code}")
            if response.text:
                print(f"    Detail: {response.text[:300]}")
            return {}

        data = response.json()
        series = {}
        for item in data.get("included", []):
            attributes = item.get("attributes", {})
            title = (attributes.get("title") or "").strip().lower()
            target_name = self.DEMAND_SERIES.get(title)
            values = attributes.get("values") or []
            if not target_name or not values:
                continue

            df = pd.DataFrame(values)
            if "datetime" not in df.columns or "value" not in df.columns:
                continue

            df = df[["datetime", "value"]].dropna()
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            series[target_name] = df.set_index("datetime").rename(columns={"value": target_name})

        return series

    def fetch_indicator(self, indicator_name, indicator_id, start_date, end_date):
        url = f"{self.BASE_URL}/{indicator_id}"
        
        params = {
            "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_trunc": "hour",
            "geo_limit": self.GEO_LIMIT,
            "geo_ids": self.GEO_ID
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
        
        if 'geo_id' in df.columns:
            df = df[df['geo_id'] == self.GEO_ID]
            if df.empty:
                return pd.DataFrame()
             
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime')
        return df

    def download_demand(self, start_year=2015):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now()
        all_data = {name: [] for name in self.DEMAND_SERIES.values()}
        current_date = start_date

        print("\n--- Downloading demand from REData ---")
        while current_date < end_date:
            next_date = current_date + relativedelta(months=1)
            if next_date > end_date:
                next_date = end_date
            else:
                next_date = next_date - relativedelta(minutes=1)

            chunk_series = self.fetch_demand(current_date, next_date)
            for name, df_chunk in chunk_series.items():
                if not df_chunk.empty:
                    all_data[name].append(df_chunk)

            current_date = next_date + relativedelta(minutes=1)
            time.sleep(0.5)

        for name, chunks in all_data.items():
            if not chunks:
                print(f"-> No hourly Catalonia demand found for {name}.")
                continue

            df_demand = pd.concat(chunks)
            df_demand = df_demand[~df_demand.index.duplicated(keep='first')]
            output_path = self._build_output_path(name)
            df_demand.to_csv(output_path)
            print(f"-> Saved {len(df_demand)} rows to {output_path}")

    def download_all(self, start_year=2015):
        self.download_demand(start_year=start_year)

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
                
                output_path = self._build_output_path(name)
                df_indicator.to_csv(output_path)
                print(f"-> Saved {len(df_indicator)} hours to {output_path}")
            else:
                print(f"-> No detailed Catalonia data found for {name}.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "catalonia")
    downloader = ESIOSCataloniaDownloader(data_dir=RAW_DIR)
    downloader.download_all(start_year=2026)
