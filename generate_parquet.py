"""
Generate daily parquet files for NIFTY50 from yfinance.
- Without args: backfills ALL missing trading days from 2026-08-01 to today
- --today: only generates/updates today's parquet
"""
import argparse
import os
import sys

import pandas as pd
import pytz
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from main import compute_indicators, COLS

DATA_DIR   = "data"
SYMBOL     = "NIFTY50"
YF_TICKER  = "^NSEI"
ETF_TICKER = "NIFTYBEES.NS"
IST        = pytz.timezone("Asia/Kolkata")
START_DATE = "2026-08-01"


def _parquet_path(date: str) -> str:
    return os.path.join(DATA_DIR, f"{SYMBOL}_{date}.parquet")


def fetch_yf_data(start: str, end: str) -> pd.DataFrame:
    """Fetch 1-min OHLCV from yfinance with ETF volume merge."""
    df = yf.download(YF_TICKER, start=start, end=end, interval="1m",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    # merge ETF volume
    etf = yf.download(ETF_TICKER, start=start, end=end, interval="1m",
                      progress=False, auto_adjust=True)
    if etf is not None and not etf.empty:
        if isinstance(etf.columns, pd.MultiIndex):
            etf.columns = [c[0].lower() for c in etf.columns]
        else:
            etf.columns = [c.lower() for c in etf.columns]
        etf = etf.reset_index()
        etf.rename(columns={etf.columns[0]: "datetime"}, inplace=True)
        etf["datetime"] = pd.to_datetime(etf["datetime"]).dt.floor("min")
        df["datetime"]  = pd.to_datetime(df["datetime"]).dt.floor("min")
        merged = df[["datetime"]].merge(
            etf[["datetime", "volume"]].rename(columns={"volume": "etf_vol"}),
            on="datetime", how="left"
        )
        df["volume"] = merged["etf_vol"].ffill().fillna(0).astype(int).values

    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df


def _prev_day_pivots(date: str) -> dict:
    """Get pivot levels from the previous available parquet file."""
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"{SYMBOL}_") and f.endswith(".parquet")
    )
    for fname in reversed(files):
        date_str = fname[len(SYMBOL) + 1:-len(".parquet")]
        if date_str < date:
            try:
                prev = pd.read_parquet(os.path.join(DATA_DIR, fname))
                ph = float(prev["high"].max())
                pl = float(prev["low"].min())
                pc = float(prev["close"].iloc[-1])
                p  = (ph + pl + pc) / 3
                return {
                    "pivot":    round(p, 2),
                    "pivot_r1": round(2*p - pl, 2),
                    "pivot_r2": round(p + (ph - pl), 2),
                    "pivot_r3": round(ph + 2*(p - pl), 2),
                    "pivot_s1": round(2*p - ph, 2),
                    "pivot_s2": round(p - (ph - pl), 2),
                    "pivot_s3": round(pl - 2*(ph - p), 2),
                }
            except Exception:
                continue
    return {}


def _signal(row) -> str:
    try:
        req = ["close", "ema_20", "rsi_14", "macd", "macd_signal", "adx"]
        if any(pd.isna(row.get(c)) for c in req):
            return "HOLD"
        if row["close"] > row["ema_20"] and row["rsi_14"] > 55 and row["macd"] > row["macd_signal"] and row["adx"] > 20:
            return "BUY"
        if row["close"] < row["ema_20"] and row["rsi_14"] < 45 and row["macd"] < row["macd_signal"] and row["adx"] > 20:
            return "SELL"
    except Exception:
        pass
    return "HOLD"


def build_parquet(date: str, day_df: pd.DataFrame):
    """Compute indicators and save parquet for a single day."""
    day_df = day_df.copy()
    day_df["stock_name"] = SYMBOL

    # convert datetime to IST string
    dt = pd.to_datetime(day_df["datetime"])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    dt = dt.dt.tz_convert(IST)
    day_df["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M")

    # filter 09:15-15:30 IST weekdays
    dt_parsed = pd.to_datetime(day_df["datetime"])
    day_df = day_df[
        (dt_parsed.dt.weekday < 5) &
        (dt_parsed.dt.strftime("%H:%M") >= "09:15") &
        (dt_parsed.dt.strftime("%H:%M") <= "15:30")
    ]
    if day_df.empty:
        print(f"  No market-hours data for {date}, skipping.")
        return

    day_df = compute_indicators(day_df)

    # pivot points
    pivots = _prev_day_pivots(date)
    for col, val in pivots.items():
        day_df[col] = val

    day_df["signal"]     = day_df.apply(_signal, axis=1)
    day_df["updated_at"] = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")

    for col in COLS:
        if col not in day_df.columns:
            day_df[col] = None

    out = _parquet_path(date)
    day_df[COLS].to_parquet(out, index=False, engine="pyarrow")
    print(f"  Saved {len(day_df)} rows -> {out}  |  close: {day_df['close'].iloc[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", action="store_true", help="Only process today")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    today = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")

    if args.today:
        dates_to_run = [today]
    else:
        # all business days from START_DATE to today
        all_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(START_DATE, today)]
        # only missing ones
        existing = {
            f[len(SYMBOL)+1:-len(".parquet")]
            for f in os.listdir(DATA_DIR)
            if f.startswith(f"{SYMBOL}_") and f.endswith(".parquet")
        }
        dates_to_run = [d for d in all_dates if d not in existing]
        if not dates_to_run:
            print("All parquet files already exist. Nothing to do.")
            return

    print(f"Dates to process: {dates_to_run}")

    # yfinance 1-min data is only available for the last 7 days.
    # Split dates into 7-calendar-day chunks and fetch each separately.
    def _chunks(lst, days=6):
        """Yield (start, end) pairs covering at most `days` calendar days."""
        if not lst:
            return
        chunk_start = pd.Timestamp(lst[0])
        chunk = []
        for d in lst:
            ts = pd.Timestamp(d)
            if (ts - chunk_start).days > days:
                yield chunk
                chunk_start = ts
                chunk = []
            chunk.append(d)
        if chunk:
            yield chunk

    all_raw_frames = []
    for chunk in _chunks(dates_to_run):
        start = (pd.Timestamp(chunk[0])  - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (pd.Timestamp(chunk[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Fetching yfinance {start} -> {end} ...")
        raw = fetch_yf_data(start, end)
        if not raw.empty:
            all_raw_frames.append(raw)

    if not all_raw_frames:
        print("No data fetched from yfinance.")
        return

    raw = pd.concat(all_raw_frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if raw["datetime"].dt.tz is None:
        raw["datetime"] = raw["datetime"].dt.tz_localize("UTC")
    raw["datetime"] = raw["datetime"].dt.tz_convert(IST)
    raw["date_str"] = raw["datetime"].dt.strftime("%Y-%m-%d")
    raw = raw.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    for date in dates_to_run:
        print(f"Processing {date}...")
        day_df = raw[raw["date_str"] == date].drop(columns=["date_str"]).copy()
        day_df["datetime"] = day_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        if day_df.empty:
            print(f"  No yfinance data for {date} (beyond 7-day limit), skipping.")
            continue
        build_parquet(date, day_df)


if __name__ == "__main__":
    main()
