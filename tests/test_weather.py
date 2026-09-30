"""Offline tests for src/weather.py (no network needed).

Run:  pytest -q tests/test_weather.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import weather  # noqa: E402

EXPORT = ROOT / "data" / "raw" / "openmeteo_export_11.00N76.99E.csv"
P2_DRAFT = ROOT / "data" / "raw" / "daily_orders_p2_draft.csv"


def mock_payload(days=5, start="2026-01-01"):
    """Response shaped exactly like the documented Open-Meteo archive JSON."""
    dates = pd.date_range(start, periods=days, freq="D")
    hours = pd.date_range(start, periods=24 * days, freq="h")
    return {
        "latitude": 11.0, "longitude": 77.0, "elevation": 422.0,
        "utc_offset_seconds": 19800, "timezone": "Asia/Kolkata",
        "daily_units": {"time": "iso8601", "precipitation_sum": "mm"},
        "hourly_units": {"time": "iso8601", "relative_humidity_2m": "%"},
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in dates],
            "precipitation_sum": [0.0, 3.2, 10.5, 0.0, 1.0][:days],
            "rain_sum": [0.0, 3.2, 10.5, 0.0, 1.0][:days],
            "temperature_2m_max": [30.0, 29.0, 27.5, 31.0, 32.0][:days],
            "temperature_2m_min": [21.0, 21.5, 22.0, 20.5, 21.0][:days],
            "temperature_2m_mean": [25.0, 24.5, 24.0, 25.5, 26.0][:days],
            "wind_speed_10m_max": [12.0, 15.0, 18.0, 10.0, 9.0][:days],
            "weather_code": [3, 61, 63, 3, 51][:days],
            "daylight_duration": [41000.0] * days,
        },
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            "relative_humidity_2m": [60 + (i % 24) for i in range(24 * days)],
            "temperature_2m": [25.0] * (24 * days),
            # day 2 (index 1): rain in hours 10, 11, 12 -> precip_hours == 3
            "precipitation": [1.0 if 24 + 10 <= i <= 24 + 12 else 0.0 for i in range(24 * days)],
        },
    }


def test_parse_api_response_schema_and_humidity():
    daily, hourly, meta = weather.parse_api_response(mock_payload())
    assert list(daily.columns) == weather.DAILY_COLUMNS + ["precip_hours", "humidity_mean_pct"]
    assert daily["precip_hours"].tolist() == [0, 3, 0, 0, 0]
    assert len(daily) == 5 and len(hourly) == 120
    # mean of 60..83 is 71.5 for every day
    assert np.allclose(daily["humidity_mean_pct"], 71.5)
    assert meta["timezone"] == "Asia/Kolkata"
    assert meta["days_with_incomplete_hourly_data"] == []


def test_parse_flags_incomplete_hourly_day():
    p = mock_payload()
    for k in p["hourly"]:
        if k != "time":
            p["hourly"][k] = p["hourly"][k][:-3]
    p["hourly"]["time"] = p["hourly"]["time"][:-3]
    _, _, meta = weather.parse_api_response(p)
    assert meta["days_with_incomplete_hourly_data"] == ["2026-01-05"]


def test_build_url_has_timezone_and_variables():
    url = weather.build_url("2025-09-24", "2026-09-24")
    assert "timezone=Asia%2FKolkata" in url
    assert "relative_humidity_2m" in url and "precipitation_sum" in url
    assert "start_date=2025-09-24" in url


def test_validate_catches_gap_and_nulls():
    daily, _, _ = weather.parse_api_response(mock_payload())
    assert weather.validate_daily(daily, "2026-01-01", "2026-01-05") == []
    broken = daily.drop(index=2).copy()
    broken.loc[broken.index[0], "precip_mm"] = np.nan
    problems = weather.validate_daily(broken)
    assert any("missing dates" in p for p in problems)
    assert any("null values in precip_mm" in p for p in problems)


@pytest.mark.skipif(not EXPORT.exists(), reason="export not present")
def test_export_normalises_and_is_clean():
    df, meta = weather.normalise_export(EXPORT)
    assert list(df.columns) == weather.DAILY_COLUMNS
    assert len(df) == 366
    assert df["date"].min() == pd.Timestamp("2025-09-24") and df["date"].max() == pd.Timestamp("2026-09-24")
    assert weather.validate_daily(df, "2025-09-24", "2026-09-24") == []
    assert meta["elevation_m"] == 422.0


@pytest.mark.skipif(not (EXPORT.exists() and P2_DRAFT.exists()), reason="files not present")
def test_export_matches_weather_columns_used_by_p2_generator():
    df, _ = weather.normalise_export(EXPORT)
    p2 = pd.read_csv(P2_DRAFT, parse_dates=["date"])
    m = df.merge(p2, on="date", suffixes=("", "_p2"))
    assert len(m) == 366
    for col in ["precip_mm", "temp_max_c", "temp_min_c", "wind_max_kmh", "temp_mean_c"]:
        assert np.allclose(m[col], m[col + "_p2"]), col


def test_fetch_writes_cache_and_second_call_is_cache_hit(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return mock_payload()

    monkeypatch.setattr(weather, "http_get_json", fake_get)
    df = weather.fetch("2026-01-01", "2026-01-05", out_dir=tmp_path)
    assert len(df) == 5 and calls["n"] == 1
    assert (tmp_path / "weather_hourly.csv").exists()
    meta = json.loads((tmp_path / "weather_meta.json").read_text())
    assert meta["validation_problems"] == [] and "request_url" in meta
    weather.fetch("2026-01-01", "2026-01-05", out_dir=tmp_path)
    assert calls["n"] == 1  # served from cache
    weather.fetch("2026-01-01", "2026-01-05", out_dir=tmp_path, force=True)
    assert calls["n"] == 2
