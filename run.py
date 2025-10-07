# run.py
from pathlib import Path
import importlib, json, sys

def run_once(use_cv: bool):
    import config
    config.OPTIMIZER["use_cv"] = use_cv

    # reload main so it picks up the toggle
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    else:
        import main  # noqa: F401
    main_mod = sys.modules["main"]
    main_mod.main()

    label = "cv" if use_cv else "no_cv"
    summary = Path("results") / f"summary_{label}.json"
    figs = Path("results") / "figures" / label / "validation"

    print(f"\n[{label}] Summary -> {summary.resolve()}")
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            print(f"[{label}] best_calmar_train: {data.get('best_calmar_train')}")
        except Exception as e:
            print(f"[{label}] Could not read summary JSON: {e}")
    print(f"[{label}] Figures -> {figs.resolve()}")
    if figs.exists():
        for p in sorted(figs.glob("*.png")):
            print("  -", p.name)

if __name__ == "__main__":
    # Run No-CV then CV
    run_once(False)
    run_once(True)
