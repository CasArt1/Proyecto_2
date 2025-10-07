# src/optimizer.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np
import optuna

from .indicators import add_indicators
from .signals import generate_signals
from .backtest import backtest, BacktestConfig
from .metrics import compute_all_metrics, annual_return
from config import OPTIMIZER

def _features_and_signals(df, p):
    df = add_indicators(
        df,
        rsi_period=p["rsi_period"],
        stoch_k_period=p["stoch_k_period"],
        stoch_d_period=p["stoch_d_period"],
        stoch_smooth_k=p["stoch_smooth_k"],
        bb_period=p["bb_period"],
        bb_num_std=p["bb_num_std"],
        macd_fast=p["macd_fast"],
        macd_slow=p["macd_slow"],
        macd_signal=p["macd_signal"],
        sma_period=p["sma_period"],
    )
    df = generate_signals(
        df,
        rsi_buy_below=p["rsi_buy_below"], rsi_sell_above=p["rsi_sell_above"],
        stoch_buy_below=p["stoch_buy_below"], stoch_sell_above=p["stoch_sell_above"],
        stoch_use_cross=False,
        bb_buy_below=p["bb_buy_below"], bb_sell_above=p["bb_sell_above"],
        confirm_bars=p["confirm_bars"],
        macd_filter=p["use_macd_filter"],
        use_sma_filter=p["use_sma_filter"],
        bb_width_min=p["bb_width_min"],
        use_percentiles=p["use_percentiles"], roll_window=p["roll_window"],
        rsi_low_q=p["rsi_low_q"], rsi_high_q=p["rsi_high_q"],
        stoch_low_q=p["stoch_low_q"], stoch_high_q=p["stoch_high_q"],
        bb_low_q=p["bb_low_q"], bb_high_q=p["bb_high_q"],
    )
    return df

def _bt_cfg_from(base: BacktestConfig, p: dict) -> BacktestConfig:
    # Build with only the fields we KNOW exist, then try set optional ones if present.
    cfg = BacktestConfig(
        signal_shift=1,
        long_sl_pct=p.get("long_sl_pct", base.long_sl_pct),
        long_tp_pct=p.get("long_tp_pct", base.long_tp_pct),
        short_sl_pct=p.get("short_sl_pct", base.short_sl_pct),
        short_tp_pct=p.get("short_tp_pct", base.short_tp_pct),
        max_drawdown_stop=base.max_drawdown_stop,
        close_on_circuit_break=base.close_on_circuit_break,
        initial_cash=base.initial_cash,
        fee_rate=base.fee_rate,
        max_long_pct=base.max_long_pct,
        max_short_pct=p.get("max_short_pct", base.max_short_pct),
    )
    # Optional knobs: set only if BacktestConfig actually has them
    for k in ["min_hold_bars","cooldown_bars","vol_window","vol_max",
              "session_start","session_length","trade_weekends"]:
        if hasattr(cfg, k) and (k in p):
            setattr(cfg, k, p[k])
    return cfg

def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    rsi_period = trial.suggest_int("rsi_period", 5, 20)
    st_k       = trial.suggest_int("stoch_k_period", 5, 20)
    st_d       = trial.suggest_int("stoch_d_period", 2, 5)
    st_sm      = trial.suggest_int("stoch_smooth_k", 1, 3)
    bb_period  = trial.suggest_int("bb_period", 10, 40)
    bb_std     = trial.suggest_float("bb_num_std", 1.6, 2.6)

    macd_fast   = trial.suggest_int("macd_fast", 8, 16)
    macd_slow   = trial.suggest_int("macd_slow", 20, 32)
    macd_signal = trial.suggest_int("macd_signal", 5, 10)
    sma_period  = trial.suggest_int("sma_period", 120, 300)
    use_macd_f  = trial.suggest_categorical("use_macd_filter", [False, False, True])
    use_sma_f   = trial.suggest_categorical("use_sma_filter",  [False, False, True])
    bb_width_min= trial.suggest_float("bb_width_min", 0.00, 0.012)

    use_pcts   = trial.suggest_categorical("use_percentiles", [True, False])
    rsi_buy    = trial.suggest_float("rsi_buy_below", 20.0, 40.0)
    rsi_sell   = trial.suggest_float("rsi_sell_above", 60.0, 80.0)
    st_buy     = trial.suggest_float("stoch_buy_below", 10.0, 35.0)
    st_sel     = trial.suggest_float("stoch_sell_above", 65.0, 90.0)
    bb_low_fix = trial.suggest_float("bb_buy_below", -0.02, 0.02)
    bb_hi_fix  = trial.suggest_float("bb_sell_above", 0.98, 1.02)

    roll_win   = trial.suggest_int("roll_window", 120, 720)
    rsi_low_q  = trial.suggest_float("rsi_low_q",   0.15, 0.35)
    rsi_high_q = trial.suggest_float("rsi_high_q",  0.65, 0.85)
    st_low_q   = trial.suggest_float("stoch_low_q", 0.15, 0.35)
    st_high_q  = trial.suggest_float("stoch_high_q",0.65, 0.85)
    bb_low_q   = trial.suggest_float("bb_low_q",    0.15, 0.35)
    bb_high_q  = trial.suggest_float("bb_high_q",   0.65, 0.85)

    vol_win      = trial.suggest_int("vol_window", 12, 48)
    vol_max      = trial.suggest_float("vol_max", 0.02, 0.06)
    min_hold     = trial.suggest_int("min_hold_bars", 1, 2)
    cooldown     = trial.suggest_int("cooldown_bars", 0, 1)

    session_start  = trial.suggest_int("session_start", 0, 23)
    session_length = trial.suggest_int("session_length", 18, 24)
    trade_weekends = trial.suggest_categorical("trade_weekends", [True])

    long_sl   = trial.suggest_float("long_sl_pct", 0.0075, 0.02)
    long_tp   = trial.suggest_float("long_tp_pct", 0.0125, 0.05)
    short_sl  = trial.suggest_float("short_sl_pct", 0.0075, 0.02)
    short_tp  = trial.suggest_float("short_tp_pct", 0.0125, 0.05)
    max_short_pct = trial.suggest_float("max_short_pct", 0.00, 0.30)

    return dict(
        rsi_period=rsi_period,
        stoch_k_period=st_k, stoch_d_period=st_d, stoch_smooth_k=st_sm,
        bb_period=bb_period, bb_num_std=bb_std,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
        sma_period=sma_period,
        use_macd_filter=use_macd_f, use_sma_filter=use_sma_f,
        use_percentiles=use_pcts,
        rsi_buy_below=rsi_buy, rsi_sell_above=rsi_sell,
        stoch_buy_below=st_buy, stoch_sell_above=st_sel,
        bb_buy_below=bb_low_fix, bb_sell_above=bb_hi_fix,
        roll_window=roll_win,
        rsi_low_q=rsi_low_q, rsi_high_q=rsi_high_q,
        stoch_low_q=st_low_q, stoch_high_q=st_high_q,
        bb_low_q=bb_low_q, bb_high_q=bb_high_q,
        confirm_bars=1,
        vol_window=vol_win, vol_max=vol_max,
        min_hold_bars=min_hold, cooldown_bars=cooldown,
        session_start=session_start, session_length=session_length, trade_weekends=trade_weekends,
        bb_width_min=bb_width_min,
        long_sl_pct=long_sl, long_tp_pct=long_tp,
        short_sl_pct=short_sl, short_tp_pct=short_tp,
        max_short_pct=max_short_pct,
    )

def _split_folds(n: int, k: int) -> List[slice]:
    base, rem = n // k, n % k
    i, folds = 0, []
    for j in range(k):
        step = base + (1 if j < rem else 0)
        folds.append(slice(i, i + step))
        i += step
    return folds

def _objective_single(train_df, base_bt_cfg: BacktestConfig):
    MIN_TRADES = OPTIMIZER.get("min_trades", 60)
    MAX_TRADES = OPTIMIZER.get("max_trades", 4000)
    def objective(trial: optuna.Trial) -> float:
        P = _suggest_params(trial)
        df = _features_and_signals(train_df.copy(), P)
        bt_cfg = _bt_cfg_from(base_bt_cfg, P)
        p, hist = backtest(df, bt_cfg)
        M = compute_all_metrics(hist, p.trade_log)
        calmar = M.get("calmar", np.nan)
        trades = M.get("trades", 0)
        if hist is None or hist.empty or np.isnan(calmar) or trades < MIN_TRADES or trades > MAX_TRADES:
            return -1e9
        cagr = annual_return(hist["equity"].astype(float))
        if cagr <= 0:
            return -1e9
        return float(calmar)
    return objective

def _objective_cv(train_df, base_bt_cfg: BacktestConfig):
    K = OPTIMIZER.get("cv_folds", 3)
    MIN_TR = OPTIMIZER.get("min_trades_fold", 40)
    MAX_TR = OPTIMIZER.get("max_trades_fold", 4000)
    def objective(trial: optuna.Trial) -> float:
        P = _suggest_params(trial)
        n = len(train_df)
        if n < 600:
            return -1e9
        calmars = []
        for sl in _split_folds(n, K):
            sub = train_df.iloc[sl]
            if len(sub) < 500:
                return -1e9
            df = _features_and_signals(sub.copy(), P)
            bt_cfg = _bt_cfg_from(base_bt_cfg, P)
            p, hist = backtest(df, bt_cfg)
            M = compute_all_metrics(hist, p.trade_log)
            calmar = M.get("calmar", np.nan)
            trades = M.get("trades", 0)
            if hist is None or hist.empty or np.isnan(calmar) or trades < MIN_TR or trades > MAX_TR:
                return -1e9
            cagr = annual_return(hist["equity"].astype(float))
            if cagr <= 0:
                return -1e9
            calmars.append(float(calmar))
        if not calmars:
            return -1e9
        return float(np.median(calmars))
    return objective

def run_optimization(train_df, base_bt_cfg: BacktestConfig, n_trials=200, seed=42) -> Tuple[Dict[str, Any], float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    obj = _objective_cv(train_df, base_bt_cfg) if OPTIMIZER.get("use_cv", False) else _objective_single(train_df, base_bt_cfg)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study.best_params, float(study.best_value)

