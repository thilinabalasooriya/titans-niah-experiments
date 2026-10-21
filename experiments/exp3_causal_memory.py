"""Experiment 3: causal test-time memorization, using model B8's own-stage
length=1024 checkpoint (from Experiment 1/2 if already trained, else trained fresh
here -- this script is runnable standalone).

Run as: PYTHONPATH=. python3 experiments/exp3_causal_memory.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.harness import build_model_for_spec
from experiments.common import pipeline as pl
from experiments.common import causal

MODEL_TAG = "B8"


def main():
    run_cfg = load_config(pl.DEFAULT_CONFIG)
    pl.ensure_val_sets()

    letter, overrides = pl.MODEL_SPECS[MODEL_TAG]
    print(f"\n{'#'*25} {MODEL_TAG} curriculum (for Experiment 3) {'#'*25}", flush=True)
    ckpt_by_length = pl.run_curriculum(
        model_tag=MODEL_TAG, letter=letter, model_overrides=overrides, run_cfg=run_cfg,
    )
    b8_ckpt = ckpt_by_length[1024]
    sha = pl.sha256_of_file(b8_ckpt)
    print(f"\nUsing checkpoint {b8_ckpt} (sha256={sha[:16]}...)", flush=True)

    model = build_model_for_spec(letter, run_cfg, overrides).to("cuda")
    ckpt = torch.load(b8_ckpt, map_location="cuda")
    model.load_state_dict(ckpt["model"], assign=True)
    model.eval()

    result = causal.run_experiment3(
        model_b=model, checkpoint_sha256=sha, load_val_set_grid_fn=pl.load_grid_cell,
        lengths=(101, 1024, 2048), far_distance_n=60, device="cuda",
    )
    with open(pl.RESULTS_DIR / "causal_ablation_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(pl.RESULTS_DIR / "counterfactual_results.json", "w") as f:
        json.dump(result["counterfactual_needle_swap"], f, indent=2, default=str)
    print(f"\n=== Experiment 3 done. Wrote causal_ablation_results.json, "
          f"counterfactual_results.json ===", flush=True)


if __name__ == "__main__":
    main()
