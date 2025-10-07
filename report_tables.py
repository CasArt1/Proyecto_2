# report_tables.py
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results"
OUT  = RES / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def load_summary(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    # flatten into a table
    rows = []
    for split, d in metrics.items():
        rows.append({
            "split": split,
            "sharpe": d.get("sharpe"),
            "sortino": d.get("sortino"),
            "calmar": d.get("calmar"),
            "max_drawdown": d.get("max_drawdown"),
            "win_rate": d.get("win_rate"),
            "trades": d.get("trades"),
        })
    df = pd.DataFrame(rows).set_index("split").loc[["train","test","validation"]]
    params = data.get("best_params", {})
    return df, params

def maybe_process(label: str, filename: str):
    path = RES / filename
    if not path.exists():
        print(f"[{label}] not found:", path)
        return
    df, params = load_summary(path)

    # Table 3/4: performance by split
    perf_csv = OUT / f"performance_{label}.csv"
    df.to_csv(perf_csv, float_format="%.6f")
    print(f"[{label}] performance table -> {perf_csv}")

    # Table 5: best parameters
    params_csv = OUT / f"best_params_{label}.csv"
    pd.Series(params, name="value").to_frame().to_csv(params_csv)
    print(f"[{label}] best-params table -> {params_csv}")

if __name__ == "__main__":
    maybe_process("no_cv", "summary_no_cv.json")  # Table 3 + params
    maybe_process("cv",    "summary_cv.json")     # Table 4 + params
    print("Done.")
