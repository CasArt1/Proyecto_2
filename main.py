from pathlib import Path
import json
import pandas as pd

from config import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, CSV_FILE, SPLIT_RATIOS,
    INDICATORS, SIGNALS, PORTFOLIO, BACKTEST, OPTIMIZER
)
from src.data_loader import load_and_preprocess_data
from src.indicators import add_indicators
from src.signals import generate_signals
from src.backtest import backtest, BacktestConfig
from src.optimizer import run_optimization
from src.metrics import compute_all_metrics
from src.visualization import (
    plot_equity_curve, plot_cumulative_returns, plot_drawdown
)

# ----- label outputs based on the optimizer toggle -----
RUN_LABEL = "cv" if OPTIMIZER.get("use_cv", False) else "no_cv"


def _prepare_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add indicators + signals using *optimized* params (with robust fallbacks)."""
    out = add_indicators(
        df,
        rsi_period=params.get("rsi_period", INDICATORS["rsi_period"]),
        stoch_k_period=params.get("stoch_k_period", INDICATORS["stoch_k_period"]),
        stoch_d_period=params.get("stoch_d_period", INDICATORS["stoch_d_period"]),
        stoch_smooth_k=params.get("stoch_smooth_k", INDICATORS["stoch_smooth_k"]),
        bb_period=params.get("bb_period", INDICATORS["bb_period"]),
        bb_num_std=params.get("bb_num_std", INDICATORS["bb_num_std"]),
        macd_fast=params.get("macd_fast", INDICATORS.get("macd_fast", 12)),
        macd_slow=params.get("macd_slow", INDICATORS.get("macd_slow", 26)),
        macd_signal=params.get("macd_signal", INDICATORS.get("macd_signal", 9)),
        sma_period=params.get("sma_period", INDICATORS.get("sma_period", 200)),
    )

    out = generate_signals(
        out,
        # fixed thresholds (used when use_percentiles=False)
        rsi_buy_below=params.get("rsi_buy_below", SIGNALS["rsi_buy_below"]),
        rsi_sell_above=params.get("rsi_sell_above", SIGNALS["rsi_sell_above"]),
        stoch_buy_below=params.get("stoch_buy_below", SIGNALS["stoch_buy_below"]),
        stoch_sell_above=params.get("stoch_sell_above", SIGNALS["stoch_sell_above"]),
        stoch_use_cross=params.get("stoch_use_cross", SIGNALS["stoch_use_cross"]),
        bb_buy_below=params.get("bb_buy_below", SIGNALS["bb_buy_below"]),
        bb_sell_above=params.get("bb_sell_above", SIGNALS["bb_sell_above"]),
        confirm_bars=params.get("confirm_bars", SIGNALS["confirm_bars"]),

        # filters
        macd_filter=params.get("use_macd_filter", SIGNALS.get("use_macd_filter", False)),
        use_sma_filter=params.get("use_sma_filter", SIGNALS.get("use_sma_filter", True)),
        bb_width_min=params.get("bb_width_min", SIGNALS.get("bb_width_min", 0.0)),

        # adaptive thresholds (enable with use_percentiles=True)
        use_percentiles=params.get("use_percentiles", SIGNALS.get("use_percentiles", False)),
        roll_window=params.get("roll_window", SIGNALS.get("roll_window", 720)),
        rsi_low_q=params.get("rsi_low_q", SIGNALS.get("rsi_low_q", 0.10)),
        rsi_high_q=params.get("rsi_high_q", SIGNALS.get("rsi_high_q", 0.90)),
        stoch_low_q=params.get("stoch_low_q", SIGNALS.get("stoch_low_q", 0.10)),
        stoch_high_q=params.get("stoch_high_q", SIGNALS.get("stoch_high_q", 0.90)),
        bb_low_q=params.get("bb_low_q", SIGNALS.get("bb_low_q", 0.10)),
        bb_high_q=params.get("bb_high_q", SIGNALS.get("bb_high_q", 0.90)),
    )
    return out


def _run_split(name: str, df: pd.DataFrame, bt_cfg: BacktestConfig, fig_dir: Path):
    p, hist = backtest(df, bt_cfg)
    metrics = compute_all_metrics(hist, p.trade_log)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Equity: your visualization.py uses (hist, trades, out_dir)
    try:
        plot_equity_curve(hist, None, fig_dir)   # <= pass trades=None
    except TypeError:
        # fallback if you later standardize to (hist, out_dir)
        plot_equity_curve(hist, fig_dir)

    # These usually take (hist, out_dir). If yours differs, mirror the try/fallback pattern.
    plot_cumulative_returns(hist, fig_dir)
    plot_drawdown(hist, fig_dir)
    # (heatmap omitted)

    return metrics




def main():
    csv_path = DATA_DIR / CSV_FILE
    train, test, val = load_and_preprocess_data(csv_path, SPLIT_RATIOS)

    # base config used for optimizer; session/vol knobs kept here if your BacktestConfig supports them
    base_bt_cfg = BacktestConfig(
        signal_shift=BACKTEST["signal_shift"],
        long_sl_pct=BACKTEST["long_sl_pct"], long_tp_pct=BACKTEST["long_tp_pct"],
        short_sl_pct=BACKTEST["short_sl_pct"], short_tp_pct=BACKTEST["short_tp_pct"],
        max_drawdown_stop=BACKTEST["max_drawdown_stop"],
        close_on_circuit_break=BACKTEST["close_on_circuit_break"],
        initial_cash=PORTFOLIO["initial_cash"],
        fee_rate=PORTFOLIO["fee_rate"],
        max_long_pct=PORTFOLIO["max_long_pct"],
        max_short_pct=PORTFOLIO["max_short_pct"],
        # Keep these only if your BacktestConfig supports them:
        min_hold_bars=1, cooldown_bars=0, vol_window=48, vol_max=0.02,
        session_start=0, session_length=24, trade_weekends=True,
    )

    # === Optimize on TRAIN ===
    best_params, best_val = run_optimization(
        train, base_bt_cfg, OPTIMIZER["n_trials"], OPTIMIZER["seed"]
    )
    params = {**INDICATORS, **SIGNALS, **best_params}

    # Prepare all splits with the SAME signal logic the optimizer saw
    train_df = _prepare_df(train.copy(), params)
    test_df  = _prepare_df(test.copy(),  params)
    val_df   = _prepare_df(val.copy(),   params)

    # Backtest settings must ALSO mirror optimizer (including session args if present)
    bt_cfg = BacktestConfig(
        signal_shift=1,
        long_sl_pct=params.get("long_sl_pct", base_bt_cfg.long_sl_pct),
        long_tp_pct=params.get("long_tp_pct", base_bt_cfg.long_tp_pct),
        short_sl_pct=params.get("short_sl_pct", base_bt_cfg.short_sl_pct),
        short_tp_pct=params.get("short_tp_pct", base_bt_cfg.short_tp_pct),
        max_drawdown_stop=base_bt_cfg.max_drawdown_stop,
        close_on_circuit_break=base_bt_cfg.close_on_circuit_break,
        initial_cash=base_bt_cfg.initial_cash,
        fee_rate=base_bt_cfg.fee_rate,
        max_long_pct=base_bt_cfg.max_long_pct,
        max_short_pct=params.get("max_short_pct", base_bt_cfg.max_short_pct),
        # optional extras (only if BacktestConfig supports them)
        min_hold_bars=params.get("min_hold_bars", getattr(base_bt_cfg, "min_hold_bars", 1)),
        cooldown_bars=params.get("cooldown_bars", getattr(base_bt_cfg, "cooldown_bars", 0)),
        vol_window=params.get("vol_window", getattr(base_bt_cfg, "vol_window", 48)),
        vol_max=params.get("vol_max", getattr(base_bt_cfg, "vol_max", 0.02)),
        session_start=params.get("session_start", getattr(base_bt_cfg, "session_start", 0)),
        session_length=params.get("session_length", getattr(base_bt_cfg, "session_length", 24)),
        trade_weekends=params.get("trade_weekends", getattr(base_bt_cfg, "trade_weekends", True)),
    )

    results = {}
    # labeled figures directory: results/figures/<RUN_LABEL>/<split>/
    base_fig_dir = FIGURES_DIR / RUN_LABEL
    for name, d in [("train", train_df), ("test", test_df), ("validation", val_df)]:
        results[name] = _run_split(name, d, bt_cfg, base_fig_dir / name)

    # labeled summary path: results/summary_<RUN_LABEL>.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"summary_{RUN_LABEL}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"best_params": best_params, "best_calmar_train": best_val, "metrics": results},
            f, indent=2
        )

    print("=== Best params ===")
    print(json.dumps(best_params, indent=2))
    print("\n=== Metrics ===")
    print(json.dumps(results, indent=2))
    print(f"\nSummary -> {summary_path}")
    print("Figures ->", base_fig_dir)


if __name__ == "__main__":
    main()

