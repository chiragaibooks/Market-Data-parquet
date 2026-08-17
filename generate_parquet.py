"""
Generate daily parquet files for NIFTY50 (2026-08-17 onwards).
For each date: uses real DB data if available, else simulated candles.
Re-running updates the parquet with latest DB rows.
"""
import sqlite3
import numpy as np
import pandas as pd
import sys, os, argparse

sys.path.insert(0, os.path.dirname(__file__))
from main import compute_indicators, COLS, get_active_db

IST_OFFSET = "Asia/Kolkata"
SYMBOL = "NIFTY50"
DATES = [
    d.strftime("%Y-%m-%d")
    for d in pd.date_range("2026-08-17", "2026-12-31", freq="B")
]
OUT_DIR = "data"


def get_last_close() -> float:
    with sqlite3.connect(get_active_db()) as conn:
        row = conn.execute(
            "SELECT close FROM indexes WHERE stock_name=? ORDER BY datetime DESC LIMIT 1",
            (SYMBOL,)
        ).fetchone()
    return float(row[0]) if row else 24300.0


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

    # keep high >= max(open,close), low <= min(open,close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows,  np.minimum(opens, closes))

    volumes = np.random.randint(5000, 80000, n).astype(float)
    # lower volume at open/close extremes
    volumes[:5]  *= 1.8
    volumes[-10:] *= 1.5

    df = pd.DataFrame({
        "datetime": times.strftime("%Y-%m-%d %H:%M"),
        "open":   np.round(opens,  2),
        "high":   np.round(highs,  2),
        "low":    np.round(lows,   2),
        "close":  np.round(closes, 2),
        "volume": np.round(volumes, 0),
    })
    return df


def fetch_db_candles(date: str) -> pd.DataFrame:
    """Return real candles from DB for this date, or empty DataFrame."""
    try:
        with sqlite3.connect(get_active_db()) as conn:
            df = pd.read_sql_query(
                "SELECT datetime, open, high, low, close, volume FROM indexes "
                "WHERE stock_name=? AND date(datetime)=? ORDER BY datetime",
                conn, params=(SYMBOL, date)
            )
        return df
    except Exception as e:
        print(f"DB fetch skipped for {date}: {e}")
        return pd.DataFrame()


def build_day(date: str, start_close: float) -> pd.DataFrame:
    db_df = fetch_db_candles(date)
    if not db_df.empty:
        print(f"  Using {len(db_df)} real rows from DB")
        df = db_df.copy()
    else:
        df = generate_candles(date, start_close)
    df["stock_name"] = SYMBOL
    df = compute_indicators(df)

    # pivot points from previous DB day
    try:
        with sqlite3.connect(get_active_db()) as conn:
            prev = conn.execute(
                "SELECT MAX(high), MIN(low), close FROM indexes "
                "WHERE stock_name=? AND date(datetime) < ? "
                "ORDER BY datetime DESC LIMIT 1",
                (SYMBOL, date)
            ).fetchone()
        if prev and all(v is not None for v in prev):
            ph, pl, pc = float(prev[0]), float(prev[1]), float(prev[2])
            p = (ph + pl + pc) / 3
            df["pivot"]    = round(p, 2)
            df["pivot_r1"] = round(2*p - pl, 2)
            df["pivot_r2"] = round(p + (ph - pl), 2)
            df["pivot_r3"] = round(ph + 2*(p - pl), 2)
            df["pivot_s1"] = round(2*p - ph, 2)
            df["pivot_s2"] = round(p - (ph - pl), 2)
            df["pivot_s3"] = round(pl - 2*(ph - p), 2)
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
        if row["close"] > row["ema_20"] and row["rsi_14"] > 55 and row["macd"] > row["macd_signal"] and row["adx"] > 20:
            return "BUY"
        if row["close"] < row["ema_20"] and row["rsi_14"] < 45 and row["macd"] < row["macd_signal"] and row["adx"] > 20:
            return "SELL"
    except Exception:
        pass
    return "HOLD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", action="store_true", help="Only process today's date")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    last_close = get_last_close()
    print(f"Starting close: {last_close:.2f}")

    today_str = pd.Timestamp.now(tz=IST_OFFSET).strftime("%Y-%m-%d")
    dates_to_run = [today_str] if args.today else DATES

    for date in dates_to_run:
        print(f"Generating {date}...")
        df = build_day(date, last_close)
        last_close = float(df["close"].iloc[-1])  # chain days

        out_path = os.path.join(OUT_DIR, f"NIFTY50_{date}.parquet")
        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"  {'Updated' if os.path.exists(out_path) else 'Saved'} {len(df)} rows -> {out_path}")
        print(f"  Close range: {df['close'].min():.2f} – {df['close'].max():.2f}")
        print(f"  Signals: {df['signal'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
