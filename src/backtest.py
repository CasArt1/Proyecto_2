from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from .portfolio import Portfolio, PortfolioConfig

@dataclass
class BacktestConfig:
    # signals
    signal_col: str = "signal"
    signal_shift: int = 1  # act next bar open
    # exits
    long_sl_pct: float = 0.02
    long_tp_pct: float = 0.04
    short_sl_pct: float = 0.015
    short_tp_pct: float = 0.03
    # risk controls
    max_drawdown_stop: float = 0.45
    close_on_circuit_break: bool = True
    # sizing / fees
    initial_cash: float = 10_000.0
    fee_rate: float = 0.00125
    max_long_pct: float = 0.95
    max_short_pct: float = 0.50
    # churn/volatility
    min_hold_bars: int = 1
    cooldown_bars: int = 0
    vol_window: int = 48
    vol_max: float = 0.02
    # --- Session filter (UTC) ---
    session_start: int = 0          # hour 0..23
    session_length: int = 24        # 1..24
    trade_weekends: bool = True     # if False, only Mon–Fri

def _within_session(ts: pd.Timestamp, start: int, length: int, weekends: bool) -> bool:
    h = int(ts.hour)
    in_hours = ((h - start) % 24) < length
    if weekends:
        return in_hours
    # Monday=0 ... Sunday=6
    return in_hours and (ts.weekday() < 5)

def backtest(df: pd.DataFrame, cfg: Optional[BacktestConfig]=None) -> Tuple[Portfolio, pd.DataFrame]:
    cfg = cfg or BacktestConfig()
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c}")
    if cfg.signal_col not in df.columns:
        raise ValueError(f"Missing {cfg.signal_col}")

    price_df = df.copy()
    price_df["exec_signal"] = price_df[cfg.signal_col].shift(cfg.signal_shift).fillna(0).astype(int)

    # realized volatility (entry filter)
    ret = price_df["close"].pct_change()
    price_df["rv"] = ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=0)

    p = Portfolio(PortfolioConfig(cfg.initial_cash, cfg.fee_rate, cfg.max_long_pct, cfg.max_short_pct))
    circuit = False
    peak = p.get_portfolio_value(price_df["close"].iloc[0])

    # churn controls
    bars_since_entry = None
    cooldown_left = 0

    for ts, row in price_df.iterrows():
        o,h,l,c = map(float, (row["open"],row["high"],row["low"],row["close"]))
        sig = int(row["exec_signal"])

        # mark equity & circuit breaker
        p.mark_to_market(c, ts)
        eq = p.get_portfolio_value(c)
        if eq > peak: peak = eq
        mdd = (eq - peak) / peak if peak > 0 else 0.0
        if mdd <= -cfg.max_drawdown_stop and not circuit:
            circuit = True
            if cfg.close_on_circuit_break:
                p.close_position(c, ts, "circuit_breaker")
                cooldown_left = cfg.cooldown_bars
                bars_since_entry = None

        if bars_since_entry is not None:
            bars_since_entry += 1
        if cooldown_left > 0:
            cooldown_left -= 1

        # intrabar exits: long
        if p.position_type == "long" and p.entry_price is not None and p.qty > 0:
            sl = p.entry_price * (1.0 - cfg.long_sl_pct)
            tp = p.entry_price * (1.0 + cfg.long_tp_pct)
            if l <= sl:
                p.close_position(sl, ts, "stop_long"); cooldown_left = cfg.cooldown_bars; bars_since_entry = None
            elif h >= tp:
                p.close_position(tp, ts, "tp_long");   cooldown_left = cfg.cooldown_bars; bars_since_entry = None

        # intrabar exits: short
        elif p.position_type == "short" and p.entry_price is not None and p.qty < 0:
            sl = p.entry_price * (1.0 + cfg.short_sl_pct)
            tp = p.entry_price * (1.0 - cfg.short_tp_pct)
            if h >= sl:
                p.close_position(sl, ts, "stop_short"); cooldown_left = cfg.cooldown_bars; bars_since_entry = None
            elif l <= tp:
                p.close_position(tp, ts, "tp_short");   cooldown_left = cfg.cooldown_bars; bars_since_entry = None

        # ENTRY FILTERS
        rv = price_df.at[ts, "rv"]
        allow_entry = (cooldown_left == 0) and (np.isnan(rv) or (rv <= cfg.vol_max))
        allow_entry &= _within_session(ts, cfg.session_start, cfg.session_length, cfg.trade_weekends)

        if not circuit and allow_entry:
            if sig == 1:
                if p.position_type == "short":
                    if bars_since_entry is None or bars_since_entry >= cfg.min_hold_bars:
                        p.close_position(c, ts, "flip_to_long"); cooldown_left = cfg.cooldown_bars; bars_since_entry = None
                if p.position_type == "none":
                    if p.open_long(o, ts): bars_since_entry = 0

            elif sig == -1:
                if p.position_type == "long":
                    if bars_since_entry is None or bars_since_entry >= cfg.min_hold_bars:
                        p.close_position(c, ts, "flip_to_short"); cooldown_left = cfg.cooldown_bars; bars_since_entry = None
                if p.position_type == "none":
                    if p.open_short(o, ts): bars_since_entry = 0

    hist = pd.DataFrame(p.history)
    if not hist.empty:
        hist.set_index("timestamp", inplace=True)
        hist["equity_peak"] = hist["equity"].cummax()
        hist["drawdown"] = (hist["equity"] - hist["equity_peak"]) / hist["equity_peak"].replace(0.0, np.nan)
        hist["drawdown"] = hist["drawdown"].fillna(0.0)
    return p, hist
