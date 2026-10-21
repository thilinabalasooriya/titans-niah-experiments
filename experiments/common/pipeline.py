"""Shared pipeline: NIAH grid construction, microbatched evaluation, and a length
curriculum trainer with per-stage checkpointing.

Eval protocol: each curriculum length in {101,256,512,1024} is evaluated using the
checkpoint saved at the END of training on THAT length specifically -- never a
later-stage checkpoint, so there is no curriculum-forgetting artifact in the numbers.
Lengths 2048 and 4096 have no training stage of their own (the curriculum stops at
1024); they are evaluated with the length=1024 checkpoint, an explicit
zero-further-training generalization test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.config import RunConfig, TrainConfig, load_config
from src.train import _make_scheduler, masked_answer_logits
from src.models import count_params
from src import data as data_mod
from src.harness import build_model_for_spec, eval_fixed, eval_greedy, save_checkpoint, grad_global_norm
from .stats import bootstrap_ci

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
CKPT_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
VAL_DIR = RESULTS_DIR / "val_sets"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RUNS_CSV = RESULTS_DIR / "runs.csv"
for d in (CKPT_DIR, LOGS_DIR, VAL_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUNS_CSV_FIELDS = [
    "run_name", "letter", "model_overrides", "precision", "seed", "schedule", "lr",
    "warmup_steps", "steps_completed", "steps_requested", "length", "train_time_s",
    "final_train_loss", "val_loss", "val_digit_acc", "val_exact_match",
    "val_greedy_exact_match", "grad_norm_last", "oom_events", "effective_batch_size",
    "param_count", "parent_run_name", "notes",
]

DEFAULT_CONFIG = "configs/default.yaml"

MODEL_SPECS = {
    "A": ("A", {}),
    "B4": ("B", {"neural_memory_chunk_size": 4}),
    "B8": ("B", {"neural_memory_chunk_size": 8}),
    "B16": ("B", {"neural_memory_chunk_size": 16}),
    "B32": ("B", {"neural_memory_chunk_size": 32}),
    "C": ("C", {}),
    "D": ("D", {}),
}

CURRICULUM_LENGTHS = (101, 256, 512, 1024)
STEPS_PER_STAGE = 2000
BATCH_SIZE = 16
LR = 1e-4

GRID_LENGTHS = (101, 256, 512, 1024, 2048, 4096)
GRID_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_PER_CELL = {101: 60, 256: 60, 512: 60, 1024: 60, 2048: 80, 4096: 80}
N_QUICK_PER_DEPTH = 6

GRID_SEED_BASE = 81_000_000
QUICK_SEED_BASE = 82_000_000

_NEEDLE_RE = re.compile(r"the passkey is (\d{5})\.")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_stage_map(ckpt_by_length: dict) -> dict:
    """{101: 'len101', ..., 1024: 'len1024', 2048: 'len1024', 4096: 'len1024'}."""
    m = {length: f"len{length}" for length in CURRICULUM_LENGTHS}
    m[2048] = "len1024"
    m[4096] = "len1024"
    return m


# ---------------------------------------------------------------------------
# NIAH grid construction (measured distance, not nominal depth*length)
# ---------------------------------------------------------------------------

def measure_exact_distance(decoded_row: str, num_digits: int = 5) -> dict:
    m = _NEEDLE_RE.search(decoded_row)
    if not m:
        raise ValueError(f"could not locate needle sentence in decoded row: {decoded_row[:80]!r}...")
    needle_start, needle_end = m.span()
    qp_idx = decoded_row.rfind(data_mod.QUERY_PREFIX)
    if qp_idx == -1:
        raise ValueError("could not locate QUERY_PREFIX in decoded row")
    return {
        "needle_start_char": needle_start, "needle_end_char": needle_end,
        "query_start_char": qp_idx, "distance_needle_to_query": qp_idx - needle_end,
        "needle_digits": m.group(1),
    }


def build_cell(length: int, depth: float, n: int, seed: int, num_digits: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    batch = data_mod.make_eval_batch(length, depth, n, rng, num_digits=num_digits)
    distances = []
    for i in range(n):
        decoded = data_mod.decode_bytes(batch["input_ids"][i])
        distances.append(measure_exact_distance(decoded, num_digits)["distance_needle_to_query"])
    batch["measured_distance"] = torch.tensor(distances, dtype=torch.long)
    batch["depth_nominal"] = torch.full((n,), depth)
    batch["length_nominal"] = torch.full((n,), length, dtype=torch.long)
    return batch


def build_grid(regenerate: bool = False) -> dict:
    manifest = {"lengths": GRID_LENGTHS, "depths": GRID_DEPTHS, "cells": []}
    for length in GRID_LENGTHS:
        n = N_PER_CELL[length]
        per_depth = []
        for di, depth in enumerate(GRID_DEPTHS):
            seed = GRID_SEED_BASE + length * 100 + di
            path = VAL_DIR / f"len{length}_depth{depth}.pt"
            if path.exists() and not regenerate:
                batch = torch.load(path)
                if batch["input_ids"].shape[0] != n:
                    raise ValueError(
                        f"{path} has {batch['input_ids'].shape[0]} examples, expected {n} "
                        f"-- stale file. Delete it or call build_grid(regenerate=True)."
                    )
            else:
                batch = build_cell(length, depth, n, seed)
                torch.save(batch, path)
            per_depth.append(batch)
            dist = batch["measured_distance"]
            manifest["cells"].append({
                "length": length, "depth": depth, "n": batch["input_ids"].shape[0], "seed": seed,
                "distance_min": int(dist.min()), "distance_max": int(dist.max()),
                "distance_mean": float(dist.float().mean()),
            })
            print(f"  [grid] len={length} depth={depth}: n={batch['input_ids'].shape[0]} "
                  f"distance_range=[{int(dist.min())},{int(dist.max())}]", flush=True)
        merged = {k: torch.cat([b[k] for b in per_depth], dim=0) for k in per_depth[0]}
        torch.save(merged, VAL_DIR / f"len{length}_all.pt")
    with open(VAL_DIR / "grid_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def build_quick_sets(regenerate: bool = False) -> None:
    expected_n = N_QUICK_PER_DEPTH * len(GRID_DEPTHS)
    for length in CURRICULUM_LENGTHS:
        path = VAL_DIR / f"len{length}_quick.pt"
        if path.exists() and not regenerate:
            existing_n = torch.load(path)["input_ids"].shape[0]
            if existing_n != expected_n:
                raise ValueError(f"{path} has {existing_n} examples, expected {expected_n} -- stale file.")
            continue
        batches = []
        for di, depth in enumerate(GRID_DEPTHS):
            seed = QUICK_SEED_BASE + length * 100 + di
            batches.append(build_cell(length, depth, N_QUICK_PER_DEPTH, seed))
        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        torch.save(merged, path)
        print(f"  [quick] wrote {path}: {merged['input_ids'].shape[0]} examples", flush=True)


def ensure_val_sets() -> None:
    build_quick_sets(regenerate=False)
    build_grid(regenerate=False)


def load_grid_cell(length: int, depth) -> dict:
    if depth == "all":
        return torch.load(VAL_DIR / f"len{length}_all.pt")
    return torch.load(VAL_DIR / f"len{length}_depth{depth}.pt")


def load_quick_set(length: int) -> dict:
    return torch.load(VAL_DIR / f"len{length}_quick.pt")


# ---------------------------------------------------------------------------
# Evaluation (microbatched, with lightweight OOM backoff)
# ---------------------------------------------------------------------------

def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


@torch.no_grad()
def eval_per_example(model, batch: dict, device: str, num_digits: int = 5, micro_batch: int = 16) -> dict:
    was_training = model.training
    model.eval()
    n = batch["input_ids"].shape[0]
    tf_exact, tf_correct, g_exact, g_correct = [], [], [], []
    start = 0
    cur_mb = micro_batch
    while start < n:
        sl = slice(start, start + cur_mb)
        try:
            ii = batch["input_ids"][sl].to(device)
            am = batch["answer_mask"][sl].to(device)
            tg = batch["targets"][sl].to(device)
            logits = model(ii, batch["pad_mask"][sl].to(device))
            sel = masked_answer_logits(logits, am, num_digits)
            tf_pred = sel.argmax(-1)
            prefix = ii[:, :-num_digits].clone()
            g_preds = []
            for _ in range(num_digits):
                step_logits = model(prefix, None)
                nxt = step_logits[:, -1].argmax(-1)
                g_preds.append(nxt)
                prefix = torch.cat([prefix, nxt[:, None]], dim=1)
            g_pred = torch.stack(g_preds, dim=1)
        except RuntimeError as e:
            if not _is_oom(e) or cur_mb <= 1:
                raise
            torch.cuda.empty_cache()
            cur_mb = max(1, cur_mb // 2)
            continue
        tf_correct.append((tf_pred == tg).cpu())
        tf_exact.append((tf_pred == tg).all(-1).cpu())
        g_correct.append((g_pred == tg).cpu())
        g_exact.append((g_pred == tg).all(-1).cpu())
        start += cur_mb
    model.train(was_training)
    return {
        "tf_exact": torch.cat(tf_exact).numpy(), "tf_digit_correct": torch.cat(tf_correct).numpy(),
        "g_exact": torch.cat(g_exact).numpy(), "g_digit_correct": torch.cat(g_correct).numpy(),
    }


def evaluate_model_on_grid(model, model_label: str, lengths, depths, device: str,
                            checkpoint_stage_by_length: dict, num_digits: int = 5) -> dict:
    per_example_rows, summary_rows = [], []
    for length in lengths:
        stage = checkpoint_stage_by_length.get(length, "?")
        for depth in depths:
            batch = load_grid_cell(length, depth)
            res = eval_per_example(model, batch, device, num_digits=num_digits)
            n = len(res["tf_exact"])
            dist = batch["measured_distance"].numpy()
            for i in range(n):
                per_example_rows.append({
                    "model": model_label, "length": length, "depth": depth,
                    "checkpoint_stage": stage, "measured_distance": int(dist[i]), "example_idx": i,
                    "tf_exact_match": int(res["tf_exact"][i]), "tf_digit_acc": float(res["tf_digit_correct"][i].mean()),
                    "greedy_exact_match": int(res["g_exact"][i]), "greedy_digit_acc": float(res["g_digit_correct"][i].mean()),
                })
            tf_ci = bootstrap_ci(res["tf_exact"])
            g_ci = bootstrap_ci(res["g_exact"])
            summary_rows.append({
                "model": model_label, "length": length, "depth": depth, "checkpoint_stage": stage, "n": n,
                "measured_distance_mean": float(dist.mean()),
                "tf_exact_match_mean": tf_ci["mean"], "tf_exact_match_ci_lo": tf_ci["lo"], "tf_exact_match_ci_hi": tf_ci["hi"],
                "greedy_exact_match_mean": g_ci["mean"], "greedy_exact_match_ci_lo": g_ci["lo"], "greedy_exact_match_ci_hi": g_ci["hi"],
                "tf_digit_acc_mean": float(res["tf_digit_correct"].mean()),
                "greedy_digit_acc_mean": float(res["g_digit_correct"].mean()),
            })
            print(f"  [{model_label}] len={length} depth={depth} (ckpt={stage}): n={n} "
                  f"greedy_EM={g_ci['mean']:.3f} [{g_ci['lo']:.3f},{g_ci['hi']:.3f}] "
                  f"greedy_digit_acc={res['g_digit_correct'].mean():.3f}", flush=True)
    return {"per_example": per_example_rows, "summary": summary_rows}


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [warn] no rows to write for {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Training: one stage, and a full curriculum with per-stage checkpoints.
# ---------------------------------------------------------------------------

def append_run_row(row: dict) -> None:
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUNS_CSV_FIELDS})


def run_stage(
    *, run_name: str, letter: str, run_cfg: RunConfig, model_overrides: dict,
    lr: float, steps: int, length: int, schedule: str = "constant", warmup_steps: int = 100,
    precision: str = "fp32", batch_size: int = 16, depth_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 0, eval_every: int = 100, checkpoint_every: int = 0, max_minutes: float = 90.0,
    warm_start_checkpoint: Path | None = None, num_digits: int = 5, grad_clip: float = 1.0,
    weight_decay: float = 0.01, extra_notes: str = "", force: bool = False,
) -> dict:
    """Trains ONE curriculum stage. Idempotent: skips if already done unless force=True.
    Saves its own checkpoint at CKPT_DIR/{run_name}.pt -- exactly the checkpoint later
    used to evaluate THIS length with no later-stage forgetting."""
    final_ckpt = CKPT_DIR / f"{run_name}.pt"
    log_path = LOGS_DIR / f"{run_name}.json"
    if not force and final_ckpt.exists() and log_path.exists():
        print(f"  [{run_name}] already done -- skipping (pass force=True to redo)", flush=True)
        with open(log_path) as f:
            return json.load(f)

    device = "cuda"
    torch.manual_seed(seed)
    model = build_model_for_spec(letter, run_cfg, model_overrides).to(device)
    param_count = count_params(model)

    parent_run_name = None
    ckpt = None
    if warm_start_checkpoint is not None:
        if not Path(warm_start_checkpoint).exists():
            raise FileNotFoundError(
                f"{run_name}: warm_start_checkpoint {warm_start_checkpoint} does not exist."
            )
        parent_run_name = Path(warm_start_checkpoint).stem
        ckpt = torch.load(warm_start_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], assign=True)
        print(f"  [{run_name}] warm-started from {warm_start_checkpoint}", flush=True)

    train_cfg = TrainConfig(warmup_steps=warmup_steps, schedule=schedule, weight_decay=weight_decay,
                             grad_clip=grad_clip, bf16=(precision == "bf16"))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_scheduler(opt, train_cfg, steps)
    if ckpt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])

    val_batch = load_quick_set(length)
    rng = np.random.default_rng(seed + 50_000 + (hash(run_name) % 1000))

    step0_val = eval_fixed(model, val_batch, device, num_digits=num_digits)
    print(f"  [{run_name}] step-0 val: loss={step0_val['loss']:.4f} "
          f"digit_acc={step0_val['per_digit_acc']:.3f} EM={step0_val['exact_match']:.4f} "
          f"fresh_init={ckpt is None}", flush=True)

    history = {"config": {
        "run_name": run_name, "letter": letter, "model_overrides": model_overrides, "lr": lr,
        "steps": steps, "length": length, "schedule": schedule, "warmup_steps": warmup_steps,
        "precision": precision, "batch_size": batch_size, "seed": seed, "param_count": param_count,
        "parent_run_name": parent_run_name,
    }, "step0_val": step0_val, "evals": [], "train_losses": []}

    use_bf16 = (precision == "bf16") and torch.cuda.is_bf16_supported()
    grad_norm_last = None
    oom_events = 0
    cur_batch_size = batch_size
    start_wall = time.time()
    model.train()
    step = 0
    while step < steps:
        if (time.time() - start_wall) / 60.0 > max_minutes:
            print(f"  [{run_name}] truncated at step {step} (max_minutes={max_minutes})", flush=True)
            break
        try:
            batch = data_mod.make_batch(cur_batch_size, (length, length), rng, num_digits=num_digits,
                                         depth_range=depth_range)
            ii, pm, am, tg = (batch[k].to(device) for k in ("input_ids", "pad_mask", "answer_mask", "targets"))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(ii, pm)
                sel = masked_answer_logits(logits, am, num_digits)
                loss = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tg.reshape(-1))
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            loss.backward()
            grad_norm_last = grad_global_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            oom_events += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            new_bs = max(2, cur_batch_size // 2)
            print(f"  [{run_name}] OOM at step {step} (batch_size={cur_batch_size}); "
                  f"backing off to batch_size={new_bs}", flush=True)
            if new_bs == cur_batch_size:
                raise
            cur_batch_size = new_bs
            continue
        sched.step()
        history["train_losses"].append({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        if (step % eval_every == 0) or (step == steps - 1):
            val = eval_fixed(model, val_batch, device, num_digits=num_digits)
            val["step"] = step
            history["evals"].append(val)
            print(f"  [{run_name}] step {step:5d} train_loss={loss.item():.4f} "
                  f"val_loss={val['loss']:.4f} val_digit_acc={val['per_digit_acc']:.3f} "
                  f"val_EM={val['exact_match']:.4f} grad_norm={grad_norm_last:.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if checkpoint_every and ((step + 1) % checkpoint_every == 0):
            save_checkpoint(CKPT_DIR / f"{run_name}_step{step+1}.pt", model, opt, sched, step + 1)
        step += 1

    final_step = step
    final_val = eval_fixed(model, val_batch, device, num_digits=num_digits)
    final_greedy = eval_greedy(model, val_batch, device, num_digits=num_digits)
    history["final_val"] = final_val
    history["final_greedy"] = final_greedy

    save_checkpoint(final_ckpt, model, opt, sched, final_step, extra={"run_name": run_name})
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    append_run_row({
        "run_name": run_name, "letter": letter, "model_overrides": json.dumps(model_overrides, default=str),
        "precision": precision, "seed": seed, "schedule": schedule, "lr": lr, "warmup_steps": warmup_steps,
        "steps_completed": final_step, "steps_requested": steps, "length": length,
        "train_time_s": round(time.time() - start_wall, 1),
        "final_train_loss": history["train_losses"][-1]["loss"] if history["train_losses"] else None,
        "val_loss": final_val["loss"], "val_digit_acc": final_val["per_digit_acc"],
        "val_exact_match": final_val["exact_match"], "val_greedy_exact_match": final_greedy["exact_match"],
        "grad_norm_last": grad_norm_last, "oom_events": oom_events, "effective_batch_size": cur_batch_size,
        "param_count": param_count, "parent_run_name": parent_run_name or "", "notes": extra_notes,
    })
    print(f"  [{run_name}] DONE: final_val_EM={final_val['exact_match']:.3f} "
          f"final_greedy_EM={final_greedy['exact_match']:.3f}", flush=True)
    return history


def run_curriculum(
    *, model_tag: str, letter: str, model_overrides: dict, run_cfg: RunConfig,
    lengths=CURRICULUM_LENGTHS, steps_per_stage: int = STEPS_PER_STAGE,
    batch_size: int = BATCH_SIZE, lr: float = LR, seed: int = 0, max_minutes: float = 90.0,
) -> dict:
    """Runs the full length curriculum for one model, EQUAL steps_per_stage at every
    stage. Returns {length: checkpoint_path}."""
    ckpt_by_length: dict[int, Path] = {}
    prev_ckpt = None
    for i, length in enumerate(lengths):
        run_name = f"{model_tag}_len{length}"
        warmup_steps = 100 if prev_ckpt is None else 0
        print(f"\n{'='*20} {model_tag} -- length {length} ({steps_per_stage} steps) {'='*20}", flush=True)
        run_stage(
            run_name=run_name, letter=letter, run_cfg=run_cfg, model_overrides=model_overrides,
            lr=lr, steps=steps_per_stage, length=length, schedule="constant", warmup_steps=warmup_steps,
            precision="fp32" if letter != "D" else "bf16", batch_size=batch_size, seed=seed,
            eval_every=100, max_minutes=max_minutes, warm_start_checkpoint=prev_ckpt,
            extra_notes=f"curriculum, model={model_tag}, stage={i+1}/{len(lengths)}",
        )
        ckpt_path = CKPT_DIR / f"{run_name}.pt"
        ckpt_by_length[length] = ckpt_path
        prev_ckpt = ckpt_path
    return ckpt_by_length
