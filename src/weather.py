"""Open-Meteo weather fetcher for Coimbatore (P1 / T02).

Pulls real historical weather from the Open-Meteo Historical Weather (Archive) API
and writes two cached CSVs plus a metadata file:

    data/raw/weather.csv          one row per local day (the table every other module joins on)
    data/raw/weather_hourly.csv   hourly humidity / temperature / precipitation (for hourly EDA)
    data/raw/weather_meta.json    exact request, retrieval time, grid point returned, units

Usage
-----
    python src/weather.py                                   # default window, uses cache if present
    python src/weather.py --start 2024-09-24 --end 2026-09-24 --force
    python src/weather.py --from-export data/raw/openmeteo_export_11.00N76.99E.csv

The API is queried with timezone=Asia/Kolkata so that a "day" is an Indian calendar day.
Two daily columns are derived from the hourly data rather than requested directly: humidity_mean_pct
(mean of the 24 hourly relative_humidity_2m values) and precip_hours (number of hours with rain > 0 mm,
i.e. how long it rained). This keeps the result independent of which daily variables the API offers.

Only the Python standard library and pandas are required.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Location. Open-Meteo snaps the request to its nearest grid cell; the grid point actually
# used is stored in weather_meta.json. Keep these identical across the whole team.
LATITUDE = 11.00
LONGITUDE = 76.99
TIMEZONE = "Asia/Kolkata"

DEFAULT_START = "2023-09-24"
DEFAULT_END = "2026-09-24"

DAILY_VARS = [
    "precipitation_sum",
    "rain_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "weather_code",
    "daylight_duration",
]
HOURLY_VARS = ["relative_humidity_2m", "temperature_2m", "precipitation"]

# API variable name -> project column name (same names/order as the order generator expects)
DAILY_RENAME = {
    "time": "date",
    "precipitation_sum": "precip_mm",
    "rain_sum": "rain_mm",
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "wind_speed_10m_max": "wind_max_kmh",
    "weather_code": "weather_code",
    "temperature_2m_mean": "temp_mean_c",
    "daylight_duration": "daylight_s",
}
# Column names used in the CSV downloaded by hand from the open-meteo.com web form
EXPORT_RENAME = {
    "time": "date",
    "precipitation_sum (mm)": "precip_mm",
    "rain_sum (mm)": "rain_mm",
    "temperature_2m_max (°C)": "temp_max_c",
    "temperature_2m_min (°C)": "temp_min_c",
    "wind_speed_10m_max (km/h)": "wind_max_kmh",
    "weather_code (wmo code)": "weather_code",
    "temperature_2m_mean (°C)": "temp_mean_c",
    "daylight_duration (s)": "daylight_s",
}
DAILY_COLUMNS = [
    "date", "precip_mm", "rain_mm", "temp_max_c", "temp_min_c", "wind_max_kmh",
    "weather_code", "temp_mean_c", "daylight_s",
]
HOURLY_RENAME = {
    "time": "datetime",
    "relative_humidity_2m": "humidity_pct",
    "temperature_2m": "temp_c",
    "precipitation": "precip_mm",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# --------------------------------------------------------------------------- request
def build_url(start: str, end: str, lat: float = LATITUDE, lon: float = LONGITUDE) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    return f"{ARCHIVE_URL}?{urllib.parse.urlencode(params, safe=',')}"


def http_get_json(url: str, retries: int = 4, timeout: int = 60) -> dict:
    """GET with exponential back-off. Open-Meteo returns HTTP 400 with a JSON 'reason' on bad input."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 400:  # request is wrong; retrying will not help
                raise RuntimeError(f"Open-Meteo rejected the request: {body}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------- parsing
def parse_api_response(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Turn the Open-Meteo JSON into (daily_df, hourly_df, meta)."""
    if "daily" not in payload or "hourly" not in payload:
        raise ValueError("Response has no 'daily'/'hourly' block: " + json.dumps(payload)[:300])

    daily = pd.DataFrame(payload["daily"]).rename(columns=DAILY_RENAME)
    hourly = pd.DataFrame(payload["hourly"]).rename(columns=HOURLY_RENAME)

    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    daily["date"] = pd.to_datetime(daily["date"])

    # daily humidity = mean of the 24 hourly values; precip_hours = hours of the day with measurable rain (> 0 mm)
    agg = (
        hourly.assign(date=hourly["datetime"].dt.normalize())
        .groupby("date")
        .agg(humidity_mean_pct=("humidity_pct", "mean"),
             precip_hours=("precip_mm", lambda x: int((x > 0).sum())),
             n_hours=("humidity_pct", "count"))
        .reset_index()
    )
    daily = daily.merge(agg, on="date", how="left")
    daily["humidity_mean_pct"] = daily["humidity_mean_pct"].round(1)
    incomplete = daily.loc[daily["n_hours"] != 24, "date"]
    daily = daily.drop(columns="n_hours")

    daily = daily[DAILY_COLUMNS + ["precip_hours", "humidity_mean_pct"]]
    meta = {
        "source": "Open-Meteo Historical Weather API",
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "daily_units": payload.get("daily_units"),
        "hourly_units": payload.get("hourly_units"),
        "days_with_incomplete_hourly_data": [d.strftime("%Y-%m-%d") for d in incomplete],
    }
    return daily, hourly, meta


def normalise_export(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a CSV downloaded manually from open-meteo.com into the project schema.

    The export has a 2-row location header, a blank line, then the daily table.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    header = dict(zip(lines[0].split(","), lines[1].split(",")))
    start = next(i for i, ln in enumerate(lines) if ln.startswith("time,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df = df.rename(columns=EXPORT_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df[DAILY_COLUMNS]
    meta = {
        "source": "Open-Meteo web export (manual download)",
        "file": Path(path).name,
        "grid_latitude": float(header["latitude"]),
        "grid_longitude": float(header["longitude"]),
        "elevation_m": float(header["elevation"]),
        "timezone": header["timezone"],
        "utc_offset_seconds": int(header["utc_offset_seconds"]),
        "note": "Daily values are aggregated over GMT days, not Indian days; humidity not included. "
                "Run this script without --from-export to refresh from the API.",
    }
    return df, meta


# --------------------------------------------------------------------------- validation
def validate_daily(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> list[str]:
    """Return a list of problems (empty list = clean)."""
    problems: list[str] = []
    d = pd.to_datetime(df["date"])
    if d.duplicated().any():
        problems.append(f"{int(d.duplicated().sum())} duplicate dates")
    full = pd.date_range(d.min(), d.max(), freq="D")
    missing = full.difference(d)
    if len(missing):
        problems.append(f"{len(missing)} missing dates, first: {missing[0].date()}")
    if start and d.min() != pd.Timestamp(start):
        problems.append(f"starts {d.min().date()}, expected {start}")
    if end and d.max() != pd.Timestamp(end):
        problems.append(f"ends {d.max().date()}, expected {end}")
    nulls = df.isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{n} null values in {col}")
    if (df["precip_mm"] < 0).any():
        problems.append("negative precipitation")
    if (df["temp_max_c"] < df["temp_min_c"]).any():
        problems.append("temp_max below temp_min on some days")
    return problems


# --------------------------------------------------------------------------- main
def fetch(start: str, end: str, force: bool = False, out_dir: Path = RAW_DIR,
          lat: float = LATITUDE, lon: float = LONGITUDE) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "weather.csv"
    hourly_path = out_dir / "weather_hourly.csv"
    meta_path = out_dir / "weather_meta.json"

    if daily_path.exists() and meta_path.exists() and not force:
        cached = json.loads(meta_path.read_text())
        if cached.get("start_date") == start and cached.get("end_date") == end \
                and cached.get("source", "").startswith("Open-Meteo Historical"):
            print(f"Cache hit: {daily_path} (use --force to refetch)")
            return pd.read_csv(daily_path, parse_dates=["date"])

    url = build_url(start, end, lat, lon)
    print("GET", url)
    daily, hourly, meta = parse_api_response(http_get_json(url))
    meta.update(request_url=url, start_date=start, end_date=end,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                rows_daily=len(daily), rows_hourly=len(hourly))

    problems = validate_daily(daily, start, end)
    meta["validation_problems"] = problems
    if problems:
        print("WARNING - validation problems:", *problems, sep="\n  - ")

    daily.to_csv(daily_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {daily_path} ({len(daily)} days), {hourly_path} ({len(hourly)} hours)")
    return daily


def from_export(path: Path, out_dir: Path = RAW_DIR) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = normalise_export(path)
    problems = validate_daily(df)
    meta.update(start_date=str(df["date"].min().date()), end_date=str(df["date"].max().date()),
                rows_daily=len(df), validation_problems=problems)
    df.to_csv(out_dir / "weather.csv", index=False)
    (out_dir / "weather_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {out_dir / 'weather.csv'} ({len(df)} days) from export; problems: {problems or 'none'}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--lat", type=float, default=LATITUDE)
    ap.add_argument("--lon", type=float, default=LONGITUDE)
    ap.add_argument("--force", action="store_true", help="ignore the cache and refetch")
    ap.add_argument("--from-export", type=Path, help="normalise a manually downloaded Open-Meteo CSV instead of calling the API")
    args = ap.parse_args(argv)
    if args.from_export:
        from_export(args.from_export)
    else:
        fetch(args.start, args.end, args.force, lat=args.lat, lon=args.lon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
