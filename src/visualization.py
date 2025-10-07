from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def plot_equity_curve(hist_df: pd.DataFrame, trade_log, out_dir: Path):
    if hist_df is None or hist_df.empty: return None
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hist_df.index, hist_df["equity"], label="Equity")
    if trade_log:
        buys_x = [t.timestamp for t in trade_log if t.side=="BUY"]
        buys_y = [t.equity_after for t in trade_log if t.side=="BUY"]
        sells_x = [t.timestamp for t in trade_log if t.side=="SELL"]
        sells_y = [t.equity_after for t in trade_log if t.side=="SELL"]
        ax.scatter(buys_x, buys_y, marker="^", s=30, label="BUY")
        ax.scatter(sells_x, sells_y, marker="v", s=30, label="SELL")
    ax.set_title("Portfolio Equity"); ax.set_xlabel("Time"); ax.set_ylabel("USDT"); ax.legend()
    out = out_dir / "equity_curve.png"; fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); return out

def plot_cumulative_returns(hist_df: pd.DataFrame, out_dir: Path):
    if hist_df is None or hist_df.empty: return None
    _ensure_dir(out_dir)
    eq = hist_df["equity"].astype(float); ret = eq.pct_change().fillna(0.0)
    cum = (1.0 + ret).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(cum.index, cum, label="Cumulative Return"); ax.legend(); ax.set_title("Cumulative Returns")
    out = out_dir / "cumulative_returns.png"; fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); return out

def plot_drawdown(hist_df: pd.DataFrame, out_dir: Path):
    if hist_df is None or hist_df.empty: return None
    _ensure_dir(out_dir)
    dd = hist_df["drawdown"].astype(float)
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.fill_between(dd.index, dd, 0, step="pre"); ax.set_title("Drawdown")
    out = out_dir / "drawdown.png"; fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); return out

def plot_monthly_returns_heatmap(hist_df: pd.DataFrame, out_dir: Path):
    if hist_df is None or hist_df.empty: return None
    _ensure_dir(out_dir)
    eq = hist_df["equity"].astype(float); ret = eq.pct_change().dropna()
    daily = (1.0 + ret).resample("D").apply(lambda x: (1.0 + x).prod() - 1.0)
    monthly = (1.0 + daily).resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
    df = monthly.to_frame(name="ret"); df["Year"] = df.index.year; df["Month"] = df.index.month
    pivot = df.pivot_table(index="Year", columns="Month", values="ret", aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(10,4))
    c = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, shading="nearest")
    ax.set_title("Monthly Returns Heatmap"); fig.colorbar(c, ax=ax, label="Return")
    out = out_dir / "monthly_returns_heatmap.png"; fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig); return out