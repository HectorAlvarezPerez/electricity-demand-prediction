"""
Preprocessing pipeline: construeix df_model a partir dels CSVs raw d'ENTSO-E,
afegeix features temporals, normalitza i guarda els splits com a parquet.

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CODE = "ES"
SOURCE_CODES = ["BE", "DE", "FR", "GR", "IT", "NL", "PT"]
ALL_CODES = [TARGET_CODE, *SOURCE_CODES]

TRAIN_START = pd.Timestamp("2015-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rename_series(code: str) -> str:
    """Retorna el nom de columna normalitzat: target_es o source_XX."""
    code = code.lower()
    return "target_es" if code == "es" else f"source_{code}"


def _load_country_demand(demand_dir: Path, code: str) -> pd.DataFrame:
    """Carrega un CSV de demanda d'un país i retorna un DataFrame indexat per UTC."""
    path = demand_dir / f"entsoe_demand_{code}.csv"
    df = pd.read_csv(path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    col = rename_series(code)
    df = df.rename(columns={"demand": col})
    df = df[["utc_timestamp", col]].drop_duplicates(subset=["utc_timestamp"])
    return df.set_index("utc_timestamp").sort_index()


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------
# Mapping of countries to their respective timezones
COUNTRY_TIMEZONES = {
    "ES": "Europe/Madrid",
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "GR": "Europe/Athens",
    "IT": "Europe/Rome",
    "NL": "Europe/Amsterdam",
    "PT": "Europe/Lisbon"
}

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix features temporals cícliques i is_weekend al DataFrame
    per a cadascun dels països, respectant la seva zona horària local (DST inclòs).

    Espera que el DataFrame tingui un DatetimeIndex (amb timezone UTC).
    Retorna una còpia sense modificar l'original.
    """
    df = df.copy()
    idx = df.index
    
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    for code, tz_name in COUNTRY_TIMEZONES.items():
        # Convert index to local timezone to capture daylight saving times properly
        idx_local = idx.tz_convert(tz_name)
        
        prefix = f"{code.lower()}_"
        
        df[f"{prefix}hour_sin"] = np.sin(2 * np.pi * idx_local.hour / 24)
        df[f"{prefix}hour_cos"] = np.cos(2 * np.pi * idx_local.hour / 24)
        df[f"{prefix}dow_sin"] = np.sin(2 * np.pi * idx_local.dayofweek / 7)
        df[f"{prefix}dow_cos"] = np.cos(2 * np.pi * idx_local.dayofweek / 7)
        df[f"{prefix}month_sin"] = np.sin(2 * np.pi * (idx_local.month - 1) / 12)
        df[f"{prefix}month_cos"] = np.cos(2 * np.pi * (idx_local.month - 1) / 12)
        df[f"{prefix}is_weekend"] = (idx_local.dayofweek >= 5).astype(int)

    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_data(
    df: pd.DataFrame,
    method: str = "standard",
    params: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Normalitza el DataFrame columna a columna.

    Args:
        df: DataFrame a normalitzar.
        method: 'standard' (z-score) o 'minmax' (escalar a [0, 1]).
        params: Si es proporciona, utilitza aquests paràmetres (per val/test).
                Si és None, calcula els paràmetres a partir de df (train).

    Returns:
        (df_normalitzat, params) — params es un dict amb les estadístiques
        necessàries per desnormalitzar o per aplicar la mateixa transformació
        a altres splits.
    """
    df = df.copy()

    if method == "standard":
        if params is None:
            params = {
                "method": "standard",
                "mean": df.mean(),
                "std": df.std().replace(0, 1),  # evita divisió per zero
            }
        df = (df - params["mean"]) / params["std"]

    elif method == "minmax":
        if params is None:
            params = {
                "method": "minmax",
                "min": df.min(),
                "max": df.max(),
            }
        denom = (params["max"] - params["min"]).replace(0, 1)
        df = (df - params["min"]) / denom

    else:
        raise ValueError(f"Mètode de normalització desconegut: {method!r}")

    return df, params


def denormalize_data(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Desnormalitza el DataFrame usant els paràmetres guardats."""
    df = df.copy()
    if params["method"] == "standard":
        df = df * params["std"] + params["mean"]
    elif params["method"] == "minmax":
        denom = params["max"] - params["min"]
        df = df * denom + params["min"]
    return df


# ---------------------------------------------------------------------------
# Build df_model
# ---------------------------------------------------------------------------
def build_df_model(demand_dir: Path) -> pd.DataFrame:
    """Carrega tots els CSVs, alinea temporalment, interpola huecos curts
    i afegeix features temporals.

    Retorna un DataFrame net, sense NaN, amb DatetimeIndex UTC.
    """
    # Carregar series individuals
    demand_frames = {
        code: _load_country_demand(demand_dir, code) for code in ALL_CODES
    }

    # Rang d'overlap comú a tots els països
    overlap_start = max(f.index.min() for f in demand_frames.values())
    overlap_end = min(f.index.max() for f in demand_frames.values())

    # Timeline horària d'ES com a referència
    es_timeline = pd.date_range(
        start=overlap_start, end=overlap_end, freq="h", tz="UTC"
    )

    # Alinear i interpolar (màxim 3h consecutives)
    aligned_frames = []
    for code in ALL_CODES:
        frame = demand_frames[code].loc[overlap_start:overlap_end].copy()
        frame = frame.reindex(es_timeline)
        frame = frame.interpolate(method="time", limit=3, limit_area="inside")
        aligned_frames.append(frame)

    df = pd.concat(aligned_frames, axis=1).dropna().copy()
    df.index.name = "utc_timestamp"

    # Features temporals
    df = add_temporal_features(df)

    # Asserts de sanitat
    assert not df.index.duplicated().any(), "Índex amb duplicats!"
    assert df.index.tz is not None, "L'índex no té timezone!"
    assert df.isna().sum().sum() == 0, "Encara queden NaN!"
    assert "target_es" in df.columns, "Falta columna target_es!"
    assert {rename_series(c) for c in SOURCE_CODES}.issubset(df.columns), \
        "Falten columnes de source!"

    return df


# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------
def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split temporal: train 2015-2022, val 2023, test 2024+."""
    return {
        "train": df.loc[TRAIN_START:TRAIN_END].copy(),
        "val": df.loc[VAL_START:VAL_END].copy(),
        "test": df.loc[TEST_START:].copy(),
    }


# ---------------------------------------------------------------------------
# Main: genera parquets
# ---------------------------------------------------------------------------
def main() -> None:
    root = Path(__file__).resolve().parents[2]
    demand_dir = root / "data" / "raw" / "europe" / "demand"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building df_model from raw CSVs...")
    df = build_df_model(demand_dir)
    print(f"  Shape: {df.shape}")
    print(f"  Range: {df.index.min()} -> {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    splits = split_by_time(df)
    for name, split_df in splits.items():
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path)
        print(f"  Saved {name}: {len(split_df):,} rows -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
