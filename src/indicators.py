from __future__ import annotations
import pandas as pd
import numpy as np

def _ensure_float(df, cols=("open","high","low","close","volume")):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out

# ---------- RSI ----------
def compute_rsi(df: pd.DataFrame, period=14, price_col="close", col_name="rsi", inplace=False):
    if price_col not in df.columns:
        raise ValueError(f"Missing {price_col}")
    out = df if inplace else df.copy()
    price = out[price_col].astype(float)
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out[col_name] = rsi
    return out

# ---------- Stochastic ----------
def compute_stochastic(df: pd.DataFrame, k_period=14, d_period=3, smooth_k=3,
                       high_col="high", low_col="low", close_col="close",
                       k_name="stoch_k", d_name="stoch_d", inplace=False):
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise ValueError(f"Missing {c}")
    out = df if inplace else df.copy()
    out = _ensure_float(out)
    lowest_low = out[low_col].rolling(k_period, min_periods=k_period).min()
    highest_high = out[high_col].rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    raw_k = 100.0 * (out[close_col] - lowest_low) / denom
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    out[k_name] = k
    out[d_name] = d
    return out

# ---------- Bollinger Bands ----------
def compute_bollinger_bands(df: pd.DataFrame, period=20, num_std=2.0, price_col="close",
                            mid_name="bb_mid", upper_name="bb_upper", lower_name="bb_lower",
                            pct_name="bb_pct", width_name="bb_width", inplace=False):
    if price_col not in df.columns:
        raise ValueError("Missing close")
    out = df if inplace else df.copy()
    out = _ensure_float(out)
    mid = out[price_col].rolling(period, min_periods=period).mean()
    std = out[price_col].rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    denom = (upper - lower).replace(0.0, np.nan)
    pct = (out[price_col] - lower) / denom
    # normalized bandwidth: wider = more potential edge
    width = (upper - lower) / mid.replace(0.0, np.nan)
    out[mid_name], out[upper_name], out[lower_name], out[pct_name], out[width_name] = mid, upper, lower, pct, width
    return out

# ---------- MACD (regime) ----------
def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9,
                 price_col="close",
                 macd_col="macd", sig_col="macd_signal", hist_col="macd_hist",
                 inplace=False):
    if price_col not in df.columns:
        raise ValueError("Missing close")
    out = df if inplace else df.copy()
    price = out[price_col].astype(float)
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    out[macd_col], out[sig_col], out[hist_col] = macd, signal_line, hist
    return out

# ---------- SMA (regime) ----------
def compute_sma(df: pd.DataFrame, period=200, price_col="close", col_name="sma_regime", inplace=False):
    if price_col not in df.columns:
        raise ValueError("Missing close")
    out = df if inplace else df.copy()
    out[col_name] = out[price_col].astype(float).rolling(period, min_periods=period).mean()
    return out

# ---------- All-in-one ----------
def add_indicators(df: pd.DataFrame,
                   rsi_period=14,
                   stoch_k_period=14, stoch_d_period=3, stoch_smooth_k=3,
                   bb_period=20, bb_num_std=2.0,
                   macd_fast=12, macd_slow=26, macd_signal=9,
                   sma_period=200,
                   inplace=False):
    out = df if inplace else df.copy()
    out = compute_rsi(out, period=rsi_period, inplace=True)
    out = compute_stochastic(out, k_period=stoch_k_period, d_period=stoch_d_period,
                             smooth_k=stoch_smooth_k, inplace=True)
    out = compute_bollinger_bands(out, period=bb_period, num_std=bb_num_std, inplace=True)
    out = compute_macd(out, fast=macd_fast, slow=macd_slow, signal=macd_signal, inplace=True)
    out = compute_sma(out, period=sma_period, inplace=True)
    return out
