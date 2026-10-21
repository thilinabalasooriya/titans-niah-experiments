"""Experiment 2: neural-memory chunk-size ablation, same own-stage-checkpoint
protocol as Experiment 1. Trains B at chunk sizes {4, 8, 16, 32}, each through an
independent curriculum (fresh init per chunk size). Reuses B8's checkpoints from
Experiment 1 automatically if they already exist (run_stage's own idempotency check).

Run as: PYTHONPATH=. python3 experiments/exp2_chunk_ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.harness import build_model_for_spec
from experiments.common import pipeline as pl

CHUNK_MODELS = ["B4", "B8", "B16", "B32"]


def main():
    run_cfg = load_config(pl.DEFAULT_CONFIG)
    pl.ensure_val_sets()

    all_per_example, all_summary = [], []
    for model_tag in CHUNK_MODELS:
        letter, overrides = pl.MODEL_SPECS[model_tag]
        print(f"\n{'#'*25} CHUNK MODEL {model_tag} {'#'*25}", flush=True)
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

    pl.write_csv(all_per_example, pl.RESULTS_DIR / "chunk_size_per_example.csv")
    pl.write_csv(all_summary, pl.RESULTS_DIR / "chunk_size_results.csv")
    print("\n=== Experiment 2 done. ===", flush=True)


if __name__ == "__main__":
    main()
