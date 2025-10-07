from __future__ import annotations
from typing import Tuple, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

ESSENTIAL_COL_ALIASES = {
    "timestamp": ["timestamp", "date", "time", "datetime", "open_time", "open time"],
    "open":      ["open", "o"],
    "high":      ["high", "h"],
    "low":       ["low", "l"],
    "close":     ["close", "c", "price"],
    "volume":    ["volume", "v", "vol"],
}

def _coalesce_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def _detect_skiprows(csv_path: Path) -> int:
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            if first.startswith(("http://","https://")) or first.lower().startswith(("url","#","//")):
                return 1
    except Exception:
        pass
    return 0

def _parse_timestamp(s: pd.Series) -> pd.Series:

    # If mostly numeric, treat as epoch
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() >= len(s) * 0.8:
        # Heuristic: > 1e12 → milliseconds; else seconds
        med = s_num.dropna().median()
        unit = "ms" if med and med > 1e12 else "s"
        dt = pd.to_datetime(s_num, unit=unit, utc=True)
    else:
        dt = pd.to_datetime(s, utc=True, errors="coerce")

    # Remove timezone correctly for a Series
    return dt.dt.tz_localize(None)

def _find_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for std, aliases in ESSENTIAL_COL_ALIASES.items():
        col = _coalesce_col(df, aliases)
        if col is None and std == "volume":
            continue
        if col is None:
            raise ValueError(f"Missing required column for '{std}'. Have: {list(df.columns)}")
        mapping[col] = std
    out = df.rename(columns=mapping)
    out["timestamp"] = _parse_timestamp(out["timestamp"])
    for c in ["open","high","low","close","volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return out.set_index("timestamp")

def _split_chronologically(df: pd.DataFrame, ratios: Tuple[float,float,float]):
    tr, te, va = ratios
    n = len(df)
    i_tr_end = int(round(n*tr))
    i_te_end = i_tr_end + int(round(n*te))
    train = df.iloc[:i_tr_end].copy()
    test  = df.iloc[i_tr_end:i_te_end].copy()
    val   = df.iloc[i_te_end:].copy()
    return train, test, val

def load_and_preprocess_data(csv_path: str | Path, split_ratios=(0.60,0.20,0.20)):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    df = pd.read_csv(csv_path, skiprows=_detect_skiprows(csv_path))
    df = _find_ohlcv(df)
    for c in ["open","high","low","close"]:
        if c in df.columns:
            df = df[df[c] > 0]
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    return _split_chronologically(df, split_ratios)

if __name__ == "__main__":
    from config import DATA_DIR, CSV_FILE, SPLIT_RATIOS
    tr, te, va = load_and_preprocess_data(DATA_DIR / CSV_FILE, SPLIT_RATIOS)
    print({"train":len(tr),"test":len(te),"validation":len(va)})
    if len(te)>0 and len(va)>0:
        assert tr.index.max() < te.index.min()
        assert te.index.max() < va.index.min()
    print("Chronological split OK ✅")
