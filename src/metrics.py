from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365

def _returns(eq: pd.Series) -> pd.Series:
    return eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0.0, np.nan)
    return float(dd.min()) if len(dd) else 0.0

def sharpe_ratio(eq: pd.Series, rf_annual=0.0) -> float:
    r = _returns(eq)
    if r.empty: return 0.0
    rf_per = (1 + rf_annual) ** (1 / HOURS_PER_YEAR) - 1
    ex = r - rf_per
    mu = ex.mean()
    sigma = ex.std(ddof=0)
    if sigma == 0 or np.isnan(sigma): return 0.0
    return float((mu / sigma) * np.sqrt(HOURS_PER_YEAR))

def sortino_ratio(eq: pd.Series, rf_annual=0.0) -> float:
    r = _returns(eq)
    if r.empty: return 0.0
    rf_per = (1 + rf_annual) ** (1 / HOURS_PER_YEAR) - 1
    ex = r - rf_per
    dn = ex[ex < 0]
    sigma_dn = dn.std(ddof=0)
    if sigma_dn == 0 or np.isnan(sigma_dn): return 0.0
    return float((ex.mean() / sigma_dn) * np.sqrt(HOURS_PER_YEAR))

def annual_return(eq: pd.Series) -> float:
    if eq.empty: return 0.0
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    n = len(eq)
    if n <= 1: return float(total)
    per = (1 + total) ** (1 / n) - 1
    return float((1 + per) ** HOURS_PER_YEAR - 1)

def calmar_ratio(eq: pd.Series) -> float:
    ar = annual_return(eq)
    mdd = max_drawdown(eq)
    denom = abs(mdd) if mdd < 0 else np.nan
    if not denom or np.isnan(denom) or denom == 0.0: return 0.0
    return float(ar / denom)

def win_rate_from_trades(trades) -> float:
    if not trades: return 0.0
    wins = tot = 0
    state = "none"; open_eq = None; prev_eq = None
    for t in trades:
        if prev_eq is None: prev_eq = t.equity_after
        if state == "none" and t.reason.startswith("open_"):
            state = "long" if t.reason.endswith("long") else "short"
            open_eq = prev_eq
        elif state != "none" and any(k in t.reason for k in ["close", "tp", "stop", "flip"]):
            pnl = t.equity_after - (open_eq if open_eq is not None else t.equity_after)
            if pnl > 0: wins += 1
            tot += 1
            state, open_eq = "none", None
        prev_eq = t.equity_after
    return float(wins / tot) if tot > 0 else 0.0

def trade_count(trades) -> int:
    if not trades: return 0
    cnt = 0
    state = "none"
    for t in trades:
        if state == "none" and t.reason.startswith("open_"):
            state = "open"
        elif state == "open" and any(k in t.reason for k in ["close", "tp", "stop", "flip"]):
            cnt += 1
            state = "none"
    return cnt

def compute_all_metrics(hist_df: pd.DataFrame, trade_log) -> Dict[str, float]:
    if hist_df is None or hist_df.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "trades": 0}
    eq = hist_df["equity"].astype(float)
    return {
        "sharpe": sharpe_ratio(eq),
        "sortino": sortino_ratio(eq),
        "calmar": calmar_ratio(eq),
        "max_drawdown": float(max_drawdown(eq)),
        "win_rate": win_rate_from_trades(trade_log),
        "trades": trade_count(trade_log),
    }
