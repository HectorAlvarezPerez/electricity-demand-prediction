import numpy as np
import pandas as pd

from src.data.preprocess_renewables import attach_target_hour_weather
from src.data.renewables import (
    HOURLY_EXTERNAL_COLUMNS,
    aggregate_country_generation_hourly,
    add_hour_ahead_targets,
    add_lag_features,
    matching_generation_columns,
)


def test_generation_matching_excludes_consumption_columns():
    columns = [
        "Solar_Actual Aggregated",
        "Solar_Actual Consumption",
        "Wind Onshore",
    ]

    assert matching_generation_columns(columns, "Solar") == ["Solar_Actual Aggregated"]
    assert matching_generation_columns(columns, "Wind Onshore") == ["Wind Onshore"]


def test_country_generation_hourly_converts_interval_mw_to_mwh_and_excludes_pumped_storage():
    df = pd.DataFrame(
        {
            "utc_timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                ],
                utc=True,
            ),
            "Solar_Actual Aggregated": [10.0, 20.0, 30.0],
            "Wind Onshore_Actual Aggregated": [1.0, 2.0, 3.0],
            "Hydro Water Reservoir_Actual Aggregated": [4.0, 5.0, 6.0],
            "Hydro Pumped Storage_Actual Aggregated": [100.0, 100.0, 100.0],
            "Hydro Pumped Storage_Actual Consumption": [50.0, 50.0, 50.0],
            "Fossil Gas_Actual Aggregated": [7.0, 8.0, 9.0],
        }
    )

    hourly = aggregate_country_generation_hourly(df, "ES")

    assert len(hourly) == 3
    row = hourly.iloc[1]
    assert row["utc_timestamp"] == pd.Timestamp("2024-01-01T01:00:00Z")
    assert row["solar_mwh"] == 20.0
    assert row["wind_mwh"] == 2.0
    assert row["hydro_mwh"] == 5.0
    assert row["renewable_total_mwh"] == 27.0
    assert row["total_generation_mwh"] == 135.0
    assert np.isclose(row["renewable_share"], 27.0 / 135.0)


def test_hour_ahead_targets_and_lags_do_not_use_future_values():
    df = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-01-01 00:00:00", periods=200, freq="h", tz="UTC"),
            "country_code": ["ES"] * 200,
            "solar_mwh": np.arange(200, dtype=float),
            "wind_mwh": np.arange(100, 300, dtype=float),
            "hydro_mwh": np.arange(200, 400, dtype=float),
            "renewable_total_mwh": np.arange(300, 500, dtype=float),
        }
    )

    with_lags = add_lag_features(df)
    with_targets = add_hour_ahead_targets(with_lags)

    row = with_targets.iloc[170]
    assert row["lag_1_solar_mwh"] == 169.0
    assert row["lag_24_solar_mwh"] == 146.0
    assert row["lag_168_solar_mwh"] == 2.0
    assert row["roll24_mean_solar_mwh"] == np.mean(np.arange(146, 170, dtype=float))
    assert row["y_solar_mwh"] == 171.0
    assert row["target_timestamp"] == pd.Timestamp("2024-01-08T03:00:00Z")


def test_external_weather_join_uses_target_hour(tmp_path):
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    weather = pd.DataFrame(
        {
            "utc_timestamp": pd.to_datetime(["2024-01-01T01:00:00Z"]),
            **{col: [float(i)] for i, col in enumerate(HOURLY_EXTERNAL_COLUMNS, start=1)},
        }
    )
    weather.to_csv(weather_dir / "weather_ES.csv", index=False)
    df = pd.DataFrame(
        {
            "utc_timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "country_code": ["ES"],
        }
    )

    joined = attach_target_hour_weather(df, weather_dir)

    assert joined.loc[0, "temperature_2m"] == float(HOURLY_EXTERNAL_COLUMNS.index("temperature_2m") + 1)
