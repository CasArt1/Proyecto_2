# Proyecto_2



002 INTRODUCTION TO TRADING EXECUTIVE REPORT
ITESO UNIVERSIDAD JESUITA DE GUADALAJARA
HÉCTOR SEBASTIÁN CASTAÑEDA ARTEAGA
MICROSTRUCTURE & TRADING SYSTEMS
MTRO. LUIS FELIPE GÓMEZ ESTRADA
DOMINGO 5 DE OCTUBRE DEL 2025






002 INTRODUCTION TO TRADING EXECUTIVE REPORT
Basic Information
Dataset: Hourly BTCUSDT for the past 5 years 
Objective Function: Maximize Calmar Ratio
Dataset Split: 60% Train, 20% Test, 20% Validation

General Guidelines
Transaction fees 0.125%
No leverage
Long + Short positions
Signal Confirmation (2 out of 3 indicators agree)
Walk-forward analysis to avoid overfitting
Performance metrics: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Win Rate
Charts & Tables: Portfolio value through time, Monthly + Quarterly + Annually returns table
Instructions
Implement a multi-indicator technical trading strategy with signal confirmation (3 indicators)
Build a realistic backtesting environment accounting for transaction costs, long, short positions without leverage.
Optimize strategy hyper-parameters (stop loss, take profit, indicators, n shares, etc.) using systematic methods like Grid Search, Random Search or Bayesian optimization.
Analyze strategy performance using industry-standard metrics


Detailed description of the strategy and rationale

Mean-Reversion with 2-of-3 Confirmation
This project tests a mean-reversion hypothesis on Bitcoin hourly candles. When short-term price gets stretched relative to its recent behavior, it tends to snap back toward its local average. To isolate stretch, we use three aligned indicators—RSI, Stochastic Oscillator, and Bollinger Bands—and only trade when at least two out of three agree.. This 2-of-3 rule is meant to suppress false positives from any single indicator and avoid mixing trend-following with mean-reversion signals, which can create contradictory instructions.

Relative Strength Index (RSI)
RSI gauges the speed and magnitude of recent gains vs. losses over a fixed lookback. It oscillates between 0 and 100: classic mean-reversion interpretation is oversold below ~30 and overbought above ~70. Extremely low RSI often coincides with short-term capitulation; extremely high RSI with short-term euphoria. Both can precede reversion toward the recent mean—especially on intraday BTC where micro reversals are frequent.

Stochastic Oscillator
Stochastics compares the close to the recent high-low range. It is also bounded 0–100 and highlights range extremes. When price closes near the bottom or top of its recent range, short-term exhaustion is more likely. Stochastics captures that positional extremity even if absolute momentum is moderate.

Bollinger Bands (BB)
BB adapt to volatility, plotting dynamic envelopes around a moving average. We read band touches/penetrations as statistical extremes and sometimes use bandwidth as a volatility filter. A close near/through the lower band often signals a short-term underpricing vs. its recent average, and vice versa at the upper band. Because BB expand/contract with volatility, they keep the definition of “extreme” adaptive to current market conditions.






Risk Framing

Trade management. 
Fixed stop-loss / take-profit bands scaled to the timeframe (e.g., 2% SL and 4% TP for longs; slightly tighter on shorts), per-trade fees (0.125% in/out), and a circuit breaker if peak-to-trough drawdown exceeds ~45% to avoid catastrophic spiral during hostile regimes.

Objective & tuning. 
We optimize hyperparameters with Bayesian optimization (Optuna) using Calmar Ratio (annualized return / max drawdown) on the training split only, with optional walk-forward CV and trade-count guards to discourage overfitting to a handful of lucky trades. Sharpe, Sortino, Max Drawdown, and Win Rate are reported for diagnostic context.

Data analysis and preprocessing
A fixed, non-rolling five-year snapshot was used to ensure that all backtests are perfectly reproducible across runs. Missing hours were neither resampled nor forward-filled; when the exchange record omitted a bar, it was retained as a gap. Indicator warm-up periods were handled in a standard manner: RSI, Stochastic, and Bollinger Bands were computed first, and the initial rows containing undefined values were then trimmed so that the backtester operated only on fully defined features. Finally, the dataset was partitioned chronologically into 60% training, 20% testing, and 20% validation subsets.

Methodology and implementation

This study was designed as a disciplined hypothesis test of a mean-reversion strategy on BTC/USDT hourly data. The full pipeline was implemented in modular Python so that each stage—data handling, feature construction, signal generation, portfolio simulation, optimization, evaluation, and reporting—can be inspected and reproduced. The codebase is organized into clearly separated modules, making  it straightforward to rerun experiments with different settings without touching core logic.

Data ingestion relies on a fixed CSV snapshot of Binance BTC/USDT 1-hour bars. Timestamps are parsed to UTC and set as the index; prices and volumes are coerced to numeric; basic integrity checks (e.g., high ≥ low) are applied; and indicator warm-up rows are dropped after features are computed. The dataset is split chronologically into 60% training, 20% test, and 20% validation partitions to prevent information leakage. 
Feature engineering is intentionally minimalist and scale-free. Three mean-reversion indicators are computed: RSI, the Stochastic Oscillator, and Bollinger Bands. To reduce regime-sensitivity, the system can switch from fixed thresholds to rolling percentile thresholds over a configurable window, allowing “oversold/overbought” levels to adapt to the recent distribution. Signals are produced by a 2-of-3 confirmation rule: a long (short) setup is admitted only when at least two indicators simultaneously flag an extreme on the same side. A one-bar execution delay (signal_shift=1) is enforced to avoid look-ahead bias.
Portfolio simulation is handled by a vector-aware yet bar-by-bar backtester that executes next-bar entries and exits, applies a 0.125% transaction fee on both entry and exit, and respects dynamic position-sizing caps (up to 95% of cash for longs and a lower cap for shorts). Risk controls include symmetric stop-loss/take-profit bands (tighter on shorts), a configurable minimum holding period and cooldown to reduce churn, and a circuit breaker that halts trading if peak-to-trough drawdown exceeds 45%. The backtester records a complete trade log and an equity time series from which performance statistics are computed.

Optimization is performed with Optuna’s TPE sampler on the training split only, with Calmar Ratio as the objective. The search space covers indicator periods, band width, fixed and percentile thresholds, volatility and session filters, holding/cooldown rules, stop/target distances, and short-exposure caps. To discourage degenerate solutions, the objective imposes guards and returns a strong penalty when constraints are violated. Two optimization regimes are supported: a single-split mode (No-CV) and a walk-forward cross-validation mode (CV) that maximizes the median Calmar across folds, thereby emphasizing robustness over curve-fit performance. After the best hyperparameters are found, the strategy is retraced on train, test, and validation with the same rules the optimizer saw.
Evaluation emphasizes drawdown-aware and risk-adjusted metrics. Alongside Calmar , the pipeline reports Sharpe, Sortino, maximum drawdown, win rate, and trade count for each split. Deterministic report artifacts are written to disk: a JSON summary containing the best parameters, objective value, and metrics tables, and a set of figures per split. The experiment runner can produce two labeled result trees—no_cv and cv—so that both optimization regimes can be compared side-by-side in the final report.

Artificial Intelligence was used as an engineering aid throughout implementation. A large language model assisted with code scaffolding, modular API design, and targeted debugging, as well as with drafting and tightening documentation sections. Model-generated suggestions were treated as proposals: all logic was reviewed, integrated selectively, and validated empirically with unit-style checks and full reruns. Parameter decisions, risk constraints, and evaluation choices were ultimately made by me, with AI support improving development speed and helping surface edge cases and clearer exposition in this write-up.
Taken together, this methodology delivers a transparent, end-to-end framework that is faithful to time-series best practices (chronological splits, no look-ahead, fixed data snapshot), rigorous about optimization leakage (train-only tuning with optional CV), explicit about costs and risk, and fully reproducible. 
Results and performance analysis

We evaluate the strategy on three chronological splits—Train (60%), Test (20%), and Validation (20%)—with Calmar Ratio as the primary objective and Sharpe, Sortino, Max Drawdown, Win Rate, and trade count as supporting diagnostics. In the current run, the No-CV and CV experiments converged to the same tuned parameters and produced the same split-level metrics, indicating that cross-validation did not uncover a materially different (or more robust) configuration for the BTC/USDT 1-hour setting. 

Calmar

Ratio is negative across all splits, with drawdowns clustering around the circuit-breaker threshold (~-45%). Specifically, Train Calmar = -0.325, Test = -0.885, and Validation = -0.812. Sharpe and Sortino are also negative on Train and Validation and strongly negative on Test, consistent with equity curves that trend downward after fees. Corresponding maximum drawdowns are approximately -45.40% (Train), -45.27% (Test), and -45.16% (Validation).
Hit rate and turnover
The system trades frequently and with modest win rates: Train win rate ~31.7% (622 trades), Test ~40.3% (720 trades), and Validation ~42.8% (636 trades). This profile—high churn with sub-50% hit rates—makes the strategy highly sensitive to transaction costs on the 1-hour timeframe; the 0.125% fee each way contributes materially to the negative risk-adjusted outcomes.
Parameter tendencies

The best configuration reflects a mean-reversion posture with adaptive thresholds and very limited short exposure. Indicatively: RSI(10), Stochastic K/D/Smooth = 20/4/2, Bollinger(14, ~1.76σ), adaptive percentile thresholds over a roll window ~591 bars, and a BB width floor ~0.00365 to avoid trading in ultra-tight conditions. Risk/execution settings cluster around long SL/TP ≈ 1.61%/2.90%, short SL/TP ≈ 0.90%/3.11%, a volatility cap (window 14, vol_max ≈ 5.80%), min hold = 2 bars, cooldown = 1 bar, and session filter (start 07:00, length 18h, weekends on). Notably, the optimizer set max_short_pct ≈ 1.03%, effectively suppressing short exposure, which matches the empirical underperformance of short mean-reversion on this series.

Cross-validation vs. single split. 
Both No-CV and CV experiments report the same best parameters and metrics in this run. Practically, this suggests either the search space’s robust optima are insensitive to fold boundaries on this dataset; or both regimes converged to a similarly over-active configuration whose edge is insufficient after costs. In either case, the negative Calmar on Test and Validation indicates that the strategy does not generalize on BTC at 1-hour resolution.

























#Check results or google drive


Performance metrics tables CV
Best Parameters

Performance 

Performance metrics tables Non CV
Best Parameters

Performance

Performance metrics graphs CV Validation





Comments
The Cumulative Returns plot shows that the strategy initially generated small profits but then experienced a steady downward trend, ending with a significant overall loss. The Drawdown chart highlights prolonged periods where the portfolio failed to recover from its previous peaks, with losses deepening to over 40%, signaling poor risk management or unfavorable market conditions. Finally, the Portfolio Equity graph mirrors these results, showing the account’s value peaking early near 11,000 USDT before continuously dropping to around 6,500 USDT, reflecting sustained underperformance and capital erosion throughout the period.
Risk analysis and limitations

The combination of frequent entries, modest edge per trade, and non-trivial fees explains the results. Even with 2-of-3 confirmation and adaptive thresholds, short-term reversals on BTC often fail to follow through enough to cover costs before stops or cooldowns cut the trade. The optimizer’s suppression of short exposure further implies that short mean-reversion is particularly fragile here. Together with persistent ~-45% drawdowns, these findings make a strong case that this particular design is not viable under the stated constraints.
To improve Calmar, I believe we could lower turnover / higher selectivity (wider extremes, larger min_hold/cooldown, and a stricter bb_width_min),  use a trend-aware gate that disables MR in persistent trends, and/or use a higher timeframe where noise and fee drag are reduced. 

Conclusions
I think the model, should be considered as a learning opportunity rather than a final product. I tried my best at mixing different indicators, but through research found that it is best to keep them in a single family to avoid fake signals of entry and exits. At the end, I finished with a negative Calmar but did have a positive one when tweaking the model. I found out that for very few trades, the model actually did yield a positive Calmar. However, I found this strange and found out that the model was too restrictive with the trading parameters. 
Furthermore, I did everything from clean data handling, appropriate splits, transparent costs, drawdown-aware metrics, and reproducible optimization and applied a strategy that, in this context, does not pass a robustness bar. Under the parameters set, hourly BTC mean-reversion with simple thresholding and majority confirmation is not competitive. I have doubts about whether the CV actually did its work, as I applied it only hyperparameters. The selected parameter set was then held fixed and evaluated on the test and validation windows. We did not run a rolling walk-forward optimization, which I leave as future work.
