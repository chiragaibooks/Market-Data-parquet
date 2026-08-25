"""
Generate daily parquet files for NIFTY50 (2026-08-17 onwards).
For each date: uses real parquet data if available, else simulated candles.
Re-running updates the parquet with the latest rows.
"""
import numpy as np
import pandas as pd
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from main import compute_indicators, COLS, DATA_DIR

IST_OFFSET = "Asia/Kolkata"
SYMBOL = "NIFTY50"
DATES = [
    d.strftime("%Y-%m-%d")
    for d in pd.date_range("2026-08-17", "2026-12-31", freq="B")
]


def _parquet_path(date: str) -> str:
    return os.path.join(DATA_DIR, f"{SYMBOL}_{date}.parquet")


def get_last_close() -> float:
    """Return the most recent closing price from existing parquet files."""
    if not os.path.isdir(DATA_DIR):
        return 24300.0
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"{SYMBOL}_") and f.endswith(".parquet")
    )
    for fname in reversed(files):
        try:
            df = pd.read_parquet(os.path.join(DATA_DIR, fname))
            if not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception:
            continue
    return 24300.0


def _read_prev_day_parquet(today: str) -> pd.DataFrame | None:
    """Return the parquet DataFrame for the most recent trading day before today."""
    if not os.path.isdir(DATA_DIR):
        return None
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"{SYMBOL}_") and f.endswith(".parquet")
    )
    for fname in reversed(files):
        date_str = fname[len(SYMBOL) + 1:-len(".parquet")]
        if date_str < today:
            try:
                return pd.read_parquet(os.path.join(DATA_DIR, fname))
            except Exception:
                continue
    return None


def generate_candles(date: str, start_close: float) -> pd.DataFrame:
    times = pd.date_range(f"{date} 09:15", f"{date} 15:30", freq="1min", tz=IST_OFFSET)
    n = len(times)

    np.random.seed(hash(date) % (2**31))
    returns = np.random.normal(0.00005, 0.0008, n)
    closes = start_close * np.cumprod(1 + returns)

    noise = np.abs(np.random.normal(0, 0.0004, n))
    highs  = closes * (1 + noise)
    lows   = closes * (1 - noise)
    opens  = np.roll(closes, 1)
    opens[0] = start_close

    # keep high >= max(open, close), low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows,  np.minimum(opens, closes))

    volumes = np.random.randint(5000, 80000, n).astype(float)
    volumes[:5]   *= 1.8
    volumes[-10:] *= 1.5

    df = pd.DataFrame({
        "datetime": times.strftime("%Y-%m-%d %H:%M"),
        "open":     np.round(opens,   2),
        "high":     np.round(highs,   2),
        "low":      np.round(lows,    2),
        "close":    np.round(closes,  2),
        "volume":   np.round(volumes, 0),
    })
    return df


def fetch_parquet_candles(date: str) -> pd.DataFrame:
    """Return existing candles from the parquet file for this date, if present."""
    path = _parquet_path(date)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume"])
        return df
    except Exception as e:
        print(f"Parquet read skipped for {date}: {e}")
        return pd.DataFrame()


def build_day(date: str, start_close: float) -> pd.DataFrame:
    existing_df = fetch_parquet_candles(date)
    if not existing_df.empty:
        print(f"  Using {len(existing_df)} existing rows from parquet")
        df = existing_df.copy()
    else:
        df = generate_candles(date, start_close)

    df["stock_name"] = SYMBOL
    df = compute_indicators(df)

    # pivot points from previous day's parquet
    try:
        prev = _read_prev_day_parquet(date)
        if prev is not None and not prev.empty:
            ph = float(prev["high"].max())
            pl = float(prev["low"].min())
            pc = float(prev["close"].iloc[-1])
            p  = (ph + pl + pc) / 3
            df["pivot"]    = round(p, 2)
            df["pivot_r1"] = round(2 * p - pl, 2)
            df["pivot_r2"] = round(p + (ph - pl), 2)
            df["pivot_r3"] = round(ph + 2 * (p - pl), 2)
            df["pivot_s1"] = round(2 * p - ph, 2)
            df["pivot_s2"] = round(p - (ph - pl), 2)
            df["pivot_s3"] = round(pl - 2 * (ph - p), 2)
    except Exception as e:
        print(f"Pivot calc skipped: {e}")

    df["signal"]     = df.apply(_signal, axis=1)
    df["updated_at"] = pd.Timestamp.now(tz=IST_OFFSET).strftime("%Y-%m-%d %H:%M:%S IST")

    for col in COLS:
        if col not in df.columns:
            df[col] = None

    return df[COLS]


def _signal(row) -> str:
    try:
        req = ["close", "ema_20", "rsi_14", "macd", "macd_signal", "adx"]
        if any(pd.isna(row.get(c)) for c in req):
            return "HOLD"
        if (row["close"] > row["ema_20"] and row["rsi_14"] > 55
                and row["macd"] > row["macd_signal"] and row["adx"] > 20):
            return "BUY"
        if (row["close"] < row["ema_20"] and row["rsi_14"] < 45
                and row["macd"] < row["macd_signal"] and row["adx"] > 20):
            return "SELL"
    except Exception:
        pass
    return "HOLD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", action="store_true", help="Only process today's date")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    last_close = get_last_close()
    print(f"Starting close: {last_close:.2f}")

    today_str = pd.Timestamp.now(tz=IST_OFFSET).strftime("%Y-%m-%d")
    dates_to_run = [today_str] if args.today else DATES

    for date in dates_to_run:
        print(f"Generating {date}...")
        df = build_day(date, last_close)
        last_close = float(df["close"].iloc[-1])  # chain days

        out_path = _parquet_path(date)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"  {'Updated' if os.path.exists(out_path) else 'Saved'} {len(df)} rows -> {out_path}")
        print(f"  Close range: {df['close'].min():.2f} – {df['close'].max():.2f}")
        print(f"  Signals: {df['signal'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
