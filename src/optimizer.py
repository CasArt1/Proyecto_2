from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np
import optuna

from .indicators import add_indicators
from .signals import generate_signals
from .backtest import backtest, BacktestConfig
from .metrics import compute_all_metrics, annual_return

# ---- Walk-forward CV across TRAIN ----
N_FOLDS = 3
MIN_TRADES_FOLD = 30        # lower floor so we don't choke the search
MAX_TRADES_FOLD = 5000

# penalties
PENALTY_LOW_TRADES = 1.5    # subtract up to ~1.5 Calmar from a fold if trades << MIN
PENALTY_NEG_CAGR   = 1.0    # subtract if fold CAGR <= 0

def _split_folds(n: int, k: int) -> List[slice]:
    base = n // k
    rem = n % k
    idx = 0
    folds = []
    for i in range(k):
        step = base + (1 if i < rem else 0)
        folds.append(slice(idx, idx + step))
        idx += step
    return folds

def _bt_cfg_from(base: BacktestConfig, p: dict) -> BacktestConfig:
    return BacktestConfig(
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
        min_hold_bars=p.get("min_hold_bars", 1),
        cooldown_bars=p.get("cooldown_bars", 0),
        vol_window=p.get("vol_window", 48),
        vol_max=p.get("vol_max", 0.02),
        session_start=p.get("session_start", 0),
        session_length=p.get("session_length", 24),
        trade_weekends=p.get("trade_weekends", True),
    )

def _prep_df(df, p):
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

def _objective_factory(train_df, base_bt_cfg: BacktestConfig):
    def objective(trial: optuna.Trial) -> float:
        # --- Indicator params (bias toward more signals) ---
        rsi_period = trial.suggest_int("rsi_period", 5, 20)
        st_k       = trial.suggest_int("stoch_k_period", 5, 20)
        st_d       = trial.suggest_int("stoch_d_period", 2, 5)
        st_sm      = trial.suggest_int("stoch_smooth_k", 1, 3)
        bb_period  = trial.suggest_int("bb_period", 10, 40)
        bb_std     = trial.suggest_float("bb_num_std", 1.6, 2.6)

        # Regime features
        macd_fast   = trial.suggest_int("macd_fast", 8, 16)
        macd_slow   = trial.suggest_int("macd_slow", 20, 32)
        macd_signal = trial.suggest_int("macd_signal", 5, 10)
        sma_period  = trial.suggest_int("sma_period", 120, 300)

        # Filters (bias OFF so we don’t kill trades)
        use_macd_f  = trial.suggest_categorical("use_macd_filter", [False, False, True])
        use_sma_f   = trial.suggest_categorical("use_sma_filter",  [False, False, True])
        bb_width_min= trial.suggest_float("bb_width_min", 0.00, 0.010)

        # Threshold style
        use_pcts   = trial.suggest_categorical("use_percentiles", [True, False])
        # Fixed thresholds (when use_percentiles=False)
        rsi_buy    = trial.suggest_float("rsi_buy_below", 20.0, 40.0)
        rsi_sell   = trial.suggest_float("rsi_sell_above", 60.0, 80.0)
        st_buy     = trial.suggest_float("stoch_buy_below", 10.0, 35.0)
        st_sel     = trial.suggest_float("stoch_sell_above", 65.0, 90.0)
        bb_low_fix = trial.suggest_float("bb_buy_below", -0.02, 0.02)
        bb_hi_fix  = trial.suggest_float("bb_sell_above", 0.98, 1.02)
        # Percentiles (shorter window -> more signals)
        roll_win   = trial.suggest_int("roll_window", 120, 480)
        rsi_low_q  = trial.suggest_float("rsi_low_q",   0.15, 0.35)
        rsi_high_q = trial.suggest_float("rsi_high_q",  0.65, 0.85)
        st_low_q   = trial.suggest_float("stoch_low_q", 0.15, 0.35)
        st_high_q  = trial.suggest_float("stoch_high_q",0.65, 0.85)
        bb_low_q   = trial.suggest_float("bb_low_q",    0.15, 0.35)
        bb_high_q  = trial.suggest_float("bb_high_q",   0.65, 0.85)

        # Churn/vol (looser)
        confirm_bars = 1
        vol_win      = trial.suggest_int("vol_window", 12, 48)
        vol_max      = trial.suggest_float("vol_max", 0.02, 0.08)
        min_hold     = trial.suggest_int("min_hold_bars", 1, 2)
        cooldown     = trial.suggest_int("cooldown_bars", 0, 1)

        # Session (very loose)
        session_start  = trial.suggest_int("session_start", 0, 23)
        session_length = trial.suggest_int("session_length", 20, 24)
        trade_weekends = trial.suggest_categorical("trade_weekends", [True])

        # Risk & sizing
        long_sl   = trial.suggest_float("long_sl_pct", 0.0075, 0.02)
        long_tp   = trial.suggest_float("long_tp_pct", 0.0125, 0.05)
        short_sl  = trial.suggest_float("short_sl_pct", 0.0075, 0.02)
        short_tp  = trial.suggest_float("short_tp_pct", 0.0125, 0.05)
        max_short_pct = trial.suggest_float("max_short_pct", 0.00, 0.30)   # allow long-only

        P = dict(
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
            confirm_bars=confirm_bars,
            vol_window=vol_win, vol_max=vol_max,
            min_hold_bars=min_hold, cooldown_bars=cooldown,
            session_start=session_start, session_length=session_length, trade_weekends=trade_weekends,
            bb_width_min=bb_width_min,
            long_sl_pct=long_sl, long_tp_pct=long_tp,
            short_sl_pct=short_sl, short_tp_pct=short_tp,
            max_short_pct=max_short_pct,
        )

        # ---- Evaluate on CV folds ----
        n = len(train_df)
        folds = _split_folds(n, N_FOLDS)
        fold_scores = []
        cagr_list = []

        for sl in folds:
            sub = train_df.iloc[sl]
            if len(sub) < 500:
                return -1e9

            dfp = _prep_df(sub.copy(), P)
            btc = _bt_cfg_from(base_bt_cfg, P)
            p, hist = backtest(dfp, btc)
            m = compute_all_metrics(hist, p.trade_log)
            trades = m.get("trades", 0)
            calmar = m.get("calmar", np.nan)

            if hist is None or hist.empty or np.isnan(calmar) or trades > MAX_TRADES_FOLD:
                # harsh penalty, but not -inf so optimizer can still learn
                fold_scores.append(-3.0)
                cagr_list.append(-1.0)
                continue

            # penalties
            penalty = 0.0
            if trades < MIN_TRADES_FOLD:
                penalty += PENALTY_LOW_TRADES * (1.0 - trades / max(1, MIN_TRADES_FOLD))

            cagr = annual_return(hist["equity"].astype(float))
            cagr_list.append(cagr)
            if cagr <= 0:
                penalty += PENALTY_NEG_CAGR

            fold_scores.append(float(calmar) - penalty)

        if not fold_scores:
            return -1e9

        median_score = float(np.median(fold_scores))
        # Require median CAGR > 0 across folds (so the system isn't systematically losing)
        if np.median(cagr_list) <= 0:
            median_score -= 1.0  # soft penalty, not hard reject

        return median_score
    return objective

def run_optimization(train_df, base_bt_cfg: BacktestConfig, n_trials=200, seed=42) -> Tuple[Dict[str, Any], float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective_factory(train_df, base_bt_cfg), n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study.best_params, float(study.best_value)
