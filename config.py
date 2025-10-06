from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
CSV_FILE     = "Binance_BTCUSDT_1h.csv"

SPLIT_RATIOS = (0.60, 0.20, 0.20)

INDICATORS = {
    "rsi_period": 14,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_smooth_k": 3,
    "bb_period": 20,
    "bb_num_std": 2.0,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "sma_period": 200,
}

SIGNALS = {
    "rsi_buy_below": 30.0, "rsi_sell_above": 70.0,
    "stoch_buy_below": 20.0, "stoch_sell_above": 80.0,
    "stoch_use_cross": False,
    "bb_buy_below": 0.0, "bb_sell_above": 1.0,
    "use_macd_filter": False,
    "use_sma_filter": True,
    "bb_width_min": 0.0,
    "use_percentiles": True,
    "roll_window": 720,
    "rsi_low_q": 0.10, "rsi_high_q": 0.90,
    "stoch_low_q": 0.10, "stoch_high_q": 0.90,
    "bb_low_q": 0.10, "bb_high_q": 0.90,
    "confirm_bars": 1,
}

PORTFOLIO = {
    "initial_cash": 10_000.0,
    "fee_rate": 0.00125,
    "max_long_pct": 0.95,
    "max_short_pct": 0.20,
}

BACKTEST = {
    "signal_shift": 1,
    "long_sl_pct": 0.02,
    "long_tp_pct": 0.04,
    "short_sl_pct": 0.015,
    "short_tp_pct": 0.03,
    "max_drawdown_stop": 0.45,
    "close_on_circuit_break": True,
}

OPTIMIZER = {
    "n_trials": 100,
    "seed": 42,
}
