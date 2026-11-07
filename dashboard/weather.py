"""Fetch weather needed for current and future dashboard forecasts."""
from __future__ import annotations

import sys
import urllib.parse
from datetime import date
from pathlib import Path

import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEATHER_CSV = ROOT / "data" / "raw" / "weather.csv"


def _weather_module():
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import weather

    return weather


def local_today() -> pd.Timestamp:
    """Return today's normalized date in the project's weather timezone."""
    timezone = _weather_module().TIMEZONE
    return pd.Timestamp.now(tz=timezone).normalize().tz_localize(None)


def _api_daily(url: str) -> pd.DataFrame:
    weather = _weather_module()
    payload = weather.http_get_json(url, timeout=60)
    daily, _, _ = weather.parse_api_response(payload)
    columns = ["date", "precip_mm", "rain_mm", "temp_max_c", "humidity_mean_pct"]
    missing = set(columns) - set(daily.columns)
    if missing:
        raise ValueError(f"Open-Meteo response is missing dashboard weather fields: {sorted(missing)}")
    result = daily[columns].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["is_rainy"] = (result["precip_mm"] >= 2.5).astype(int)
    result["heavy_rain"] = (result["precip_mm"] >= 15.0).astype(int)
    result["is_hot"] = (result["temp_max_c"] >= 33.0).astype(int)
    result["is_festival"] = 0
    return result


def fetch_weather_extension(
    last_order_date: str | pd.Timestamp,
    weather_csv: str | Path = WEATHER_CSV,
    today: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Get contiguous bridge weather and a 16-day forecast from Open-Meteo."""
    weather_api = _weather_module()
    last_order_date = pd.Timestamp(last_order_date).normalize()
    if today is None:
        today = local_today()
    else:
        today = pd.Timestamp(today).normalize()

    cached = pd.read_csv(weather_csv, parse_dates=["date"])
    cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
    cached = cached[["date", "precip_mm", "rain_mm", "temp_max_c", "humidity_mean_pct"]]
    params = {
        "latitude": weather_api.LATITUDE,
        "longitude": weather_api.LONGITUDE,
        "timezone": weather_api.TIMEZONE,
        "daily": ",".join(weather_api.DAILY_VARS),
        "hourly": ",".join(weather_api.HOURLY_VARS),
    }

    archive_start = max(last_order_date + pd.Timedelta(days=1), cached["date"].max() + pd.Timedelta(days=1))
    archive_end = today - pd.Timedelta(days=7)
    sources = [cached.loc[cached["date"] > last_order_date]]
    if archive_start <= archive_end:
        archive_params = {
            **params,
            "start_date": archive_start.date().isoformat(),
            "end_date": archive_end.date().isoformat(),
        }
        archive_url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            + urllib.parse.urlencode(archive_params, safe=",")
        )
        sources.append(_api_daily(archive_url))

    forecast_params = {**params, "forecast_days": 16, "past_days": 6}
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode(forecast_params, safe=",")
    )
    sources.append(_api_daily(forecast_url))

    extension = (
        pd.concat(sources, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    extension = extension.loc[extension["date"] > last_order_date].reset_index(drop=True)
    extension["is_rainy"] = (extension["precip_mm"] >= 2.5).astype(int)
    extension["heavy_rain"] = (extension["precip_mm"] >= 15.0).astype(int)
    extension["is_hot"] = (extension["temp_max_c"] >= 33.0).astype(int)
    holiday_calendar = holidays.country_holidays(
        "IN",
        subdiv="TN",
        years=range(last_order_date.year, extension["date"].max().year + 1),
    )
    extension["is_festival"] = extension["date"].dt.date.map(
        lambda day: day in holiday_calendar
    ).astype(int)
    expected_dates = pd.date_range(last_order_date + pd.Timedelta(days=1), extension["date"].max(), freq="D")
    if not extension["date"].equals(pd.Series(expected_dates, name="date")):
        missing = expected_dates.difference(pd.DatetimeIndex(extension["date"]))
        raise ValueError(f"Weather extension has missing dates: {', '.join(day.date().isoformat() for day in missing[:10])}")
    if extension[["precip_mm", "temp_max_c", "humidity_mean_pct"]].isna().any().any():
        raise ValueError("Open-Meteo returned missing weather values needed for forecasting.")
    return extension
