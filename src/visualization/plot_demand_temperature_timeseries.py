"""
Plot one-year normalized demand and temperature time series for selected countries.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_artifact_dirs

DATA_DIR = PROCESSED_DATA_DIR
COUNTRIES = ["ES", "FR", "GR"]
COUNTRY_LABELS = {
    "ES": "Espanya (target)",
    "FR": "França",
    "GR": "Grècia",
}
START = "2024-01-01"
END = "2025-01-01"
ROLLING_HOURS = 24 * 7


def load_all_splits() -> pd.DataFrame:
    parts = [pd.read_parquet(DATA_DIR / f"{split}.parquet") for split in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def main() -> None:
    ensure_artifact_dirs()

    df = load_all_splits()
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df[(df["utc_timestamp"] >= START) & (df["utc_timestamp"] < END)].copy()

    fig, axes = plt.subplots(len(COUNTRIES), 1, figsize=(13, 8), sharex=True)

    for ax, code in zip(axes, COUNTRIES):
        country = df[df["country_code"] == code].sort_values("utc_timestamp").copy()
        country["demand_z"] = zscore(country["demand"]).rolling(ROLLING_HOURS, center=True, min_periods=24).mean()
        country["temp_z"] = zscore(country["temperature_2m"]).rolling(
            ROLLING_HOURS, center=True, min_periods=24
        ).mean()

        ax.plot(
            country["utc_timestamp"],
            country["demand_z"],
            color="#08519c",
            linewidth=1.9,
            label="Demanda normalitzada",
        )
        ax.plot(
            country["utc_timestamp"],
            country["temp_z"],
            color="#cb181d",
            linewidth=1.6,
            label="Temperatura normalitzada",
        )

        corr = float(country["demand_z"].corr(country["temp_z"]))
        if code == "ES":
            ax.set_facecolor("#fff8e7")
        ax.set_title(f"{COUNTRY_LABELS[code]}  |  corr(z) = {corr:.3f}", fontsize=11)
        ax.grid(alpha=0.18)
        ax.set_ylabel("z-score", fontsize=10)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Any 2024", fontsize=10)
    fig.autofmt_xdate(rotation=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Demanda i temperatura normalitzades en una finestra d'un any", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.10)

    out = FIGURES_DIR / "demand_temperature_timeseries_2024.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out}")
    for code in COUNTRIES:
        country = df[df["country_code"] == code].sort_values("utc_timestamp").copy()
        country["demand_z"] = zscore(country["demand"]).rolling(ROLLING_HOURS, center=True, min_periods=24).mean()
        country["temp_z"] = zscore(country["temperature_2m"]).rolling(
            ROLLING_HOURS, center=True, min_periods=24
        ).mean()
        print(f"{code}: corr(z)={country['demand_z'].corr(country['temp_z']):.4f}")


if __name__ == "__main__":
    main()
