import numpy as np
import pandas as pd

from src.data.renewables import (
    DAILY_EXTERNAL_COLUMNS,
    aggregate_country_generation_daily,
    add_day_ahead_targets,
    add_lag_features,
    matching_generation_columns,
)
from src.data.preprocess_renewables import attach_target_day_weather


def test_generation_matching_excludes_consumption_columns():
    columns = [
        "Solar_Actual Aggregated",
        "Solar_Actual Consumption",
        "Wind Onshore",
    ]

    assert matching_generation_columns(columns, "Solar") == ["Solar_Actual Aggregated"]
    assert matching_generation_columns(columns, "Wind Onshore") == ["Wind Onshore"]


def test_country_generation_daily_converts_interval_mw_to_mwh_and_excludes_pumped_storage():
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

    daily = aggregate_country_generation_daily(df, "ES")

    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["solar_mwh"] == 60.0
    assert row["wind_mwh"] == 6.0
    assert row["hydro_mwh"] == 15.0
    assert row["renewable_total_mwh"] == 81.0
    assert row["total_generation_mwh"] == 405.0
    assert np.isclose(row["renewable_share"], 81.0 / 405.0)


def test_day_ahead_targets_and_lags_do_not_use_future_values():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=35, freq="D"),
            "country_code": ["ES"] * 35,
            "solar_mwh": np.arange(35, dtype=float),
            "wind_mwh": np.arange(100, 135, dtype=float),
            "hydro_mwh": np.arange(200, 235, dtype=float),
            "renewable_total_mwh": np.arange(300, 335, dtype=float),
            "renewable_share": np.linspace(0.1, 0.5, 35),
        }
    )

    with_lags = add_lag_features(df)
    with_targets = add_day_ahead_targets(with_lags)

    row = with_targets.iloc[30]
    assert row["lag_1_solar_mwh"] == 29.0
    assert row["lag_7_solar_mwh"] == 23.0
    assert row["roll7_mean_solar_mwh"] == np.mean(np.arange(23, 30, dtype=float))
    assert row["y_solar_mwh"] == 31.0
    assert row["target_date"] == pd.Timestamp("2024-02-01")


def test_external_weather_join_uses_target_day(tmp_path):
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            **{col: [float(i)] for i, col in enumerate(DAILY_EXTERNAL_COLUMNS, start=1)},
        }
    )
    weather.to_csv(weather_dir / "weather_daily_ES.csv", index=False)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "country_code": ["ES"],
        }
    )

    joined = attach_target_day_weather(df, weather_dir)

    assert joined.loc[0, "shortwave_radiation_sum"] == float(
        DAILY_EXTERNAL_COLUMNS.index("shortwave_radiation_sum") + 1
    )
