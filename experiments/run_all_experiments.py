"""Runs all three experiments in sequence. Each is wrapped in its own try/except so a
failure in one does not prevent the others from running. Every curriculum stage is
idempotent and independently resumable -- re-run any script, or this one, after an
interruption and it picks up where it left off.

Run as: PYTHONPATH=. python3 experiments/run_all_experiments.py
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pipeline as pl


def main():
    pl.ensure_val_sets()
    for name, module_name in [
        ("Experiment 1 (NIAH length/depth)", "experiments.exp1_niah_length_depth"),
        ("Experiment 2 (chunk-size ablation)", "experiments.exp2_chunk_ablation"),
        ("Experiment 3 (causal test-time memorization)", "experiments.exp3_causal_memory"),
    ]:
        print(f"\n{'='*30} STARTING: {name} {'='*30}", flush=True)
        try:
            importlib.import_module(module_name).main()
            print(f"{'='*30} FINISHED: {name} {'='*30}", flush=True)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}", flush=True)
            traceback.print_exc()
            print(f"{'='*30} FAILED: {name} (continuing) {'='*30}", flush=True)
    print("\nALL EXPERIMENTS ATTEMPTED. See results/ for outputs.", flush=True)


if __name__ == "__main__":
    main()
