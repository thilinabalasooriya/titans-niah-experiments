"""Experiment 1: NIAH length x depth scaling for models A, B8, C, D.

Trains each model through the 101->256->512->1024 length curriculum (equal step
budget per stage) and evaluates each length using THAT length's own end-of-stage
checkpoint -- never a later-stage checkpoint, so there is no curriculum-forgetting
artifact in these numbers. Lengths 2048/4096 (no training stage of their own) are
evaluated with the length=1024 checkpoint, an explicit generalization test.

Run as: PYTHONPATH=. python3 experiments/exp1_niah_length_depth.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.harness import build_model_for_spec
from experiments.common import pipeline as pl

MODELS = ["A", "B8", "C", "D"]


def main():
    run_cfg = load_config(pl.DEFAULT_CONFIG)
    pl.ensure_val_sets()

    all_per_example, all_summary = [], []
    for model_tag in MODELS:
        letter, overrides = pl.MODEL_SPECS[model_tag]
        print(f"\n{'#'*25} MODEL {model_tag} {'#'*25}", flush=True)
        ckpt_by_length = pl.run_curriculum(
            model_tag=model_tag, letter=letter, model_overrides=overrides, run_cfg=run_cfg,
        )
        stage_map = pl.checkpoint_stage_map(ckpt_by_length)

        print(f"\n--- {model_tag}: own-stage-checkpoint NIAH eval ---", flush=True)
        for length in pl.GRID_LENGTHS:
            stage = stage_map[length]
            ckpt_path = pl.CKPT_DIR / f"{model_tag}_{stage}.pt"
            model = build_model_for_spec(letter, run_cfg, overrides).to("cuda")
            ckpt = torch.load(ckpt_path, map_location="cuda")
            model.load_state_dict(ckpt["model"], assign=True)
            model.eval()
            result = pl.evaluate_model_on_grid(
                model, model_tag, [length], pl.GRID_DEPTHS, "cuda",
                checkpoint_stage_by_length={length: stage},
            )
            all_per_example.extend(result["per_example"])
            all_summary.extend(result["summary"])
            del model
            torch.cuda.empty_cache()

    pl.write_csv(all_per_example, pl.RESULTS_DIR / "niah_per_example.csv")
    pl.write_csv(all_summary, pl.RESULTS_DIR / "niah_summary.csv")
    print("\n=== Experiment 1 done. ===", flush=True)


if __name__ == "__main__":
    main()
