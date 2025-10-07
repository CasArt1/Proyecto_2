from __future__ import annotations
import pandas as pd
import numpy as np

def _to_int(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(int)

# ----- votes: fixed and percentile versions -----
def rsi_votes_fixed(df, rsi_col="rsi", buy_below=30.0, sell_above=70.0):
    r = df[rsi_col].astype(float)
    return _to_int(r < buy_below), _to_int(r > sell_above)

def rsi_votes_percentile(df, rsi_col="rsi", window=720, low_q=0.10, high_q=0.90):
    r = df[rsi_col].astype(float)
    low_thr  = r.rolling(window, min_periods=window).quantile(low_q)
    high_thr = r.rolling(window, min_periods=window).quantile(high_q)
    return _to_int(r < low_thr), _to_int(r > high_thr)

def stochastic_votes_fixed(df, k_col="stoch_k", d_col="stoch_d",
                           buy_below=20.0, sell_above=80.0, use_cross=False):
    k, d = df[k_col].astype(float), df[d_col].astype(float)
    if not use_cross:
        return _to_int(k < buy_below), _to_int(k > sell_above)
    pk, pd = k.shift(1), d.shift(1)
    cross_up = (pk <= pd) & (k > d)
    cross_dn = (pk >= pd) & (k < d)
    return _to_int((k < buy_below) & cross_up), _to_int((k > sell_above) & cross_dn)

def stochastic_votes_percentile(df, k_col="stoch_k", window=720, low_q=0.10, high_q=0.90):
    k = df[k_col].astype(float)
    low_thr  = k.rolling(window, min_periods=window).quantile(low_q)
    high_thr = k.rolling(window, min_periods=window).quantile(high_q)
    return _to_int(k < low_thr), _to_int(k > high_thr)

def bollinger_votes_fixed(df, pct_col="bb_pct", buy_below=0.0, sell_above=1.0):
    p = df[pct_col].astype(float)
    return _to_int(p < buy_below), _to_int(p > sell_above)

def bollinger_votes_percentile(df, pct_col="bb_pct", window=720, low_q=0.10, high_q=0.90):
    p = df[pct_col].astype(float)
    low_thr  = p.rolling(window, min_periods=window).quantile(low_q)
    high_thr = p.rolling(window, min_periods=window).quantile(high_q)
    return _to_int(p < low_thr), _to_int(p > high_thr)

def generate_signals(df: pd.DataFrame,
                     rsi_buy_below=30.0, rsi_sell_above=70.0,
                     stoch_buy_below=20.0, stoch_sell_above=80.0, stoch_use_cross=False,
                     bb_buy_below=0.0, bb_sell_above=1.0,
                     signal_col="signal",
                     confirm_bars: int = 1,
                     # filters
                     macd_filter: bool = False, macd_col: str = "macd",
                     use_sma_filter: bool = True, sma_col: str = "sma_regime",
                     bb_width_min: float = 0.0, bb_width_col: str = "bb_width",
                     # adaptive thresholds
                     use_percentiles: bool = True,
                     roll_window: int = 720,
                     rsi_low_q: float = 0.10, rsi_high_q: float = 0.90,
                     stoch_low_q: float = 0.10, stoch_high_q: float = 0.90,
                     bb_low_q: float = 0.10, bb_high_q: float = 0.90):
    """
    2-of-3 MR votes with persistence and optional regime/width filters.
    """
    out = df.copy()

    # --- votes ---
    if use_percentiles:
        r_b, r_s = rsi_votes_percentile(out, "rsi", window=roll_window, low_q=rsi_low_q, high_q=rsi_high_q)
        s_b, s_s = stochastic_votes_percentile(out, "stoch_k", window=roll_window, low_q=stoch_low_q, high_q=stoch_high_q)
        b_b, b_s = bollinger_votes_percentile(out, "bb_pct", window=roll_window, low_q=bb_low_q, high_q=bb_high_q)
    else:
        r_b, r_s = rsi_votes_fixed(out, "rsi", buy_below=rsi_buy_below, sell_above=rsi_sell_above)
        s_b, s_s = stochastic_votes_fixed(out, "stoch_k", "stoch_d",
                                          buy_below=stoch_buy_below, sell_above=stoch_sell_above,
                                          use_cross=stoch_use_cross)
        b_b, b_s = bollinger_votes_fixed(out, "bb_pct", buy_below=bb_buy_below, sell_above=bb_sell_above)

    out["buy_count"]  = r_b + s_b + b_b
    out["sell_count"] = r_s + s_s + b_s

    buy2  = (out["buy_count"]  >= 2).astype(int)
    sell2 = (out["sell_count"] >= 2).astype(int)

    if confirm_bars > 1:
        buy2  = (buy2.rolling(confirm_bars, min_periods=confirm_bars).sum()  == confirm_bars)
        sell2 = (sell2.rolling(confirm_bars, min_periods=confirm_bars).sum() == confirm_bars)
    else:
        buy2  = buy2.astype(bool)
        sell2 = sell2.astype(bool)

    sig = np.where(buy2 & ~sell2, 1, np.where(sell2 & ~buy2, -1, 0)).astype(int)

    # --- regime & bandwidth filters ---
    if macd_filter and macd_col in out.columns:
        macd = out[macd_col].astype(float)
        sig = np.where((sig == 1) & (macd <= 0), 0, sig)
        sig = np.where((sig == -1) & (macd >= 0), 0, sig)

    if use_sma_filter and sma_col in out.columns:
        sma = out[sma_col].astype(float)
        close = out["close"].astype(float)
        allow_long  = close > sma
        allow_short = close < sma
        sig = np.where((sig == 1) & (~allow_long), 0, sig)
        sig = np.where((sig == -1) & (~allow_short), 0, sig)

    if bb_width_min > 0 and bb_width_col in out.columns:
        wide_enough = out[bb_width_col].astype(float) >= bb_width_min
        sig = np.where(wide_enough, sig, 0)

    out[signal_col] = sig.astype(int)
    return out
