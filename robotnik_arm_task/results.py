"""Results recording: one CSV per model in results/<model>.csv, plus one combined file of all models.

train.py fills the training and offline-test columns, compare.py --test-set the sim_* columns (PLAN.md, "Recording
results"). Each script only updates its own columns, so they never overwrite each other's numbers.

Usage (from the robotnik_arm_task folder):
    pixi run python results.py     # stack every results/<model>.csv into results_combined.csv: one row per model,
                                   # plus one mean +- std row per model trained with several seeds (<name>_s0, _s1, ...)
"""
import csv
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Outside results/, so it is never read back as one of the models
COMBINED_FILE = Path(__file__).resolve().parent / "results_combined.csv"
COLUMNS = ["model", "train_episodes", "epochs", "batch", "lr", "hidden", "layers", "best_val_loss", "best_epoch",
           "train_time_s", "parameters", "inference_ms", "test_wheel_rmse", "test_torque_rmse",
           "sim_success_rate", "sim_final_err_mean", "sim_final_err_median", "sim_time_to_reach"]
SIM_COLUMNS = [c for c in COLUMNS if c.startswith("sim_")]


def read_results(model):
    """Saved values of one model as a dict of strings ({} if it has no results file yet)."""
    path = RESULTS_DIR / f"{model}.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return next(csv.DictReader(f), {})


def update_results(model, **values):
    """Set the given columns in results/<model>.csv and keep the others; None empties a column ("not measured").

    Columns outside COLUMNS (model-specific settings) are written after the standard ones.
    """
    row = read_results(model)
    row["model"] = model
    for key, value in values.items():
        if value is None:
            row[key] = ""
        elif isinstance(value, (float, np.floating)):
            row[key] = f"{float(value):.6g}"
        else:
            row[key] = str(value)
    fields = COLUMNS + [k for k in row if k not in COLUMNS]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{model}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def combine(value_strings):
    """One cell of a seed-group row: the value if all seeds agree, else mean +- std; empty if none measured."""
    try:
        values = [float(v) for v in value_strings if v not in ("", None)]
    except ValueError:   # a text setting: its distinct values
        return "/".join(sorted(set(value_strings)))
    if not values:
        return ""
    text = f"{values[0]:.6g}" if len(set(values)) == 1 else f"{np.mean(values):.4g} +- {np.std(values):.2g}"
    # Seeds without this measurement yet are shown, so a mean over fewer seeds isn't taken for the full one
    return text if len(values) == len(value_strings) else f"{text} ({len(values)}/{len(value_strings)})"


def main():
    rows = [read_results(p.stem) for p in sorted(RESULTS_DIR.glob("*.csv"))]
    if not rows:
        print(f"No results in {RESULTS_DIR}")
        return
    # Standard columns first, then every model-specific one; a model without a column gets an empty cell
    fields = COLUMNS + sorted({k for r in rows for k in r if k not in COLUMNS})
    # Seeds of one model differ only by a trailing _s<N>: one extra row per such model, after the single-seed rows
    groups = {}
    for r in rows:
        if re.search(r"_s\d+$", r["model"]):
            groups.setdefault(re.sub(r"_s\d+$", "", r["model"]), []).append(r)
    group_rows = [{"model": f"{name} (mean +- std, {len(members)} seeds)",
                   **{k: combine([m.get(k, "") for m in members]) for k in fields[1:]}}
                  for name, members in groups.items()]
    with open(COMBINED_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows + group_rows)
    print(f"{len(rows)} models and {len(group_rows)} seed groups written to {COMBINED_FILE}")


if __name__ == "__main__":
    main()
