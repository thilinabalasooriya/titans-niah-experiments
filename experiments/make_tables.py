"""Builds presentation tables (LaTeX + plain text) from Experiment 1/3's CSV/JSON
outputs. Pure post-processing -- no GPU needed, safe to run any time after (or during)
the experiments.

Run as: PYTHONPATH=. python3 experiments/make_tables.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from experiments.common import pipeline as pl

TABLES_DIR = pl.RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def make_exp1_table() -> None:
    path = pl.RESULTS_DIR / "niah_summary.csv"
    if not path.exists():
        print(f"[skip] {path} does not exist yet")
        return
    df = pd.read_csv(path)
    agg = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        agg[row["model"]][int(row["length"])].append(float(row["greedy_exact_match_mean"]))
    lengths = sorted({l for m in agg.values() for l in m})
    preferred = [m for m in ("A", "B8", "C", "D") if m in agg]
    models = preferred or sorted(agg)
    table = pd.DataFrame(
        {length: [round(sum(agg[m][length]) / len(agg[m][length]), 3) if agg[m].get(length) else float("nan")
                   for m in models] for length in lengths},
        index=models,
    )
    table.index.name = "model"
    _write(table, "exp1_table", caption="Experiment 1: greedy exact-match by model and "
           "haystack length (mean over 5 depths; own-stage checkpoint per length, "
           "length=1024 checkpoint for 2048/4096).")


def make_exp3_table() -> None:
    path = pl.RESULTS_DIR / "causal_ablation_results.json"
    if not path.exists():
        print(f"[skip] {path} does not exist yet")
        return
    with open(path) as f:
        result = json.load(f)
    by_length = result["by_length"]
    conditions = ["B_normal", "B_frozen_writes", "B_zeroed_reads"]
    lengths = sorted(int(l) for l in by_length)
    table = pd.DataFrame(
        {length: [round(by_length[str(length)][cond]["greedy"]["exact_match"], 3)
                   if cond in by_length[str(length)] else float("nan") for cond in conditions]
         for length in lengths},
        index=conditions,
    )
    table.index.name = "condition"
    _write(table, "exp3_table", caption="Experiment 3: greedy exact-match by condition "
           "and length, on the same B8 checkpoint (causal memory ablation).")


def _write(table: pd.DataFrame, name: str, caption: str) -> None:
    txt_path = TABLES_DIR / f"{name}.txt"
    tex_path = TABLES_DIR / f"{name}.tex"
    with open(txt_path, "w") as f:
        f.write(table.to_string())
    latex = table.to_latex(caption=caption, label=f"tab:{name}", float_format="%.3f", na_rep="--")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  wrote {txt_path} and {tex_path}")
    print(table.to_string())
    print()


if __name__ == "__main__":
    make_exp1_table()
    make_exp3_table()
