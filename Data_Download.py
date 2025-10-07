# Data.py — one-time downloader: fixed 5-year snapshot ending now
# Saves: data/Binance_BTCUSDT_1h.csv

import os
from datetime import datetime, timedelta
import pandas as pd
import ccxt
import time

OUT_PATH = "data/Binance_BTCUSDT_1h.csv"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 1000  # max per request

def fetch_last_5_years_binance_1h(symbol=SYMBOL, timeframe=TIMEFRAME):
    ex = ccxt.binance({"enableRateLimit": True})
    now_ms = ex.milliseconds()
    start_ms = now_ms - int(5 * 365 * 24 * 60 * 60 * 1000)  # ~5 years
    all_rows = []
    since = start_ms

    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=LIMIT)
        if not candles:
            break
        all_rows.extend(candles)
        last_open = candles[-1][0]
        next_since = last_open + 1
        if next_since <= since:  # no progress, stop
            break
        since = next_since
        time.sleep(0.25)  # be nice to the API

    if not all_rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume"])

    df = pd.DataFrame(all_rows, columns=["open_time","open","high","low","close","volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    return df.set_index("open_time").sort_index()

def main():
    os.makedirs("data", exist_ok=True)
    df = fetch_last_5_years_binance_1h()
    if df.empty:
        raise RuntimeError("Download returned no rows. Check your internet or try again.")
    out = df.reset_index()
    out.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    # Record the exact range you captured (for reproducibility in your report)
    with open("DATA_RANGE.txt", "w", encoding="utf-8") as f:
        f.write(f"Rows: {len(out)}\n")
        f.write(f"Start: {out['open_time'].iloc[0]}\n")
        f.write(f"End:   {out['open_time'].iloc[-1]}\n")

if __name__ == "__main__":
    main()
