"""Shared training loop + LR sweep, identical code path for every model (A/B/C/D)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from . import data as data_mod
from .config import DataConfig, RunConfig, TrainConfig
from .models import ModelUnavailable, build_model, log_param_count


def masked_answer_logits(logits: torch.Tensor, answer_mask: torch.Tensor, num_digits: int) -> torch.Tensor:
    """Extracts predicted logits at the answer positions, in per-row order. Relies on
    data.py's collate guarantee that every row has exactly `num_digits` True positions
    in answer_mask, contiguous and identically placed (the second-to-last num_digits
    columns — shifted one left of the digits themselves, so each logit is computed from
    context that does NOT include the digit it predicts; see data.py's `_collate` for
    why unshifted would be a self-attention leak)."""
    b, _l, v = logits.shape
    return logits[answer_mask].view(b, num_digits, v)


@dataclass
class TrainResult:
    model_name: str
    seed: int
    lr: float
    steps: int
    final_train_loss: float
    wall_time_s: float
    param_count: int
    config_tier: str
    truncated: bool = False

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _make_optimizer(model: torch.nn.Module, train_cfg: TrainConfig, lr: float) -> torch.optim.Optimizer:
    if train_cfg.optimizer != "adamw":
        raise ValueError(f"only 'adamw' is supported, got {train_cfg.optimizer!r}")
    return AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.weight_decay)


def _make_scheduler(optimizer: torch.optim.Optimizer, train_cfg: TrainConfig, steps: int):
    warmup = train_cfg.warmup_steps

    if train_cfg.schedule == "cosine":
        def fn(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, steps - warmup)
            return 0.5 * (1 + float(np.cos(np.pi * min(progress, 1.0))))
    elif train_cfg.schedule == "linear":
        def fn(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, steps - warmup)
            return max(0.0, 1.0 - min(progress, 1.0))
    elif train_cfg.schedule == "constant":
        # Phase 4 (2026-10-20): linear warmup to the target LR, then hold constant for
        # the rest of training. Needed so LR-sweep candidates are compared at the SAME
        # effective LR throughout, instead of a cosine/linear decay confounding "lower
        # peak LR" with "different average LR over the run" (CLAUDE_CODE_TITANS_PHASE4_NIAH.md
        # Phase 0/1A/1B explicitly call for constant LR after warmup).
        def fn(step: int) -> float:
            return min(1.0, step / max(1, warmup))
    else:
        raise ValueError(f"unknown schedule {train_cfg.schedule!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def _run_training(
    model: torch.nn.Module,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    lr: float,
    steps: int,
    seed: int,
    device: str,
) -> tuple[float, float, bool]:
    """Returns (final_loss, wall_time_s, truncated)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    model.train()
    optimizer = _make_optimizer(model, train_cfg, lr)
    scheduler = _make_scheduler(optimizer, train_cfg, steps)

    use_bf16 = (
        train_cfg.bf16 and device == "cuda"
        and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    autocast_device = "cuda" if device == "cuda" else "cpu"

    start = time.time()
    final_loss = float("nan")
    truncated = False

    for _step in range(steps):
        elapsed_min = (time.time() - start) / 60.0
        if elapsed_min > train_cfg.max_minutes_per_run:
            truncated = True
            break

        batch = data_mod.make_batch(
            train_cfg.batch_size,
            (data_cfg.train_len_min, data_cfg.train_len_max),
            rng,
            num_digits=data_cfg.passkey_digits,
        )
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        answer_mask = batch["answer_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=use_bf16, dtype=torch.bfloat16):
            logits = model(input_ids, pad_mask)
            sel_logits = masked_answer_logits(logits, answer_mask, data_cfg.passkey_digits)
            loss = F.cross_entropy(sel_logits.reshape(-1, sel_logits.shape[-1]), targets.reshape(-1))

            if data_cfg.aux_lm_loss_weight > 0:
                lm_logits = logits[:, :-1]
                lm_targets = input_ids[:, 1:]
                lm_pad = pad_mask[:, 1:].reshape(-1).float()
                lm_loss = F.cross_entropy(
                    lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1), reduction="none"
                )
                lm_loss = (lm_loss * lm_pad).sum() / lm_pad.sum().clamp(min=1)
                loss = loss + data_cfg.aux_lm_loss_weight * lm_loss

        loss.backward()
        if train_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()

    return final_loss, time.time() - start, truncated


def lr_sweep(
    letter: str,
    run_cfg: RunConfig,
    seed: int,
    device: str,
    num_tokens: int = 256,
) -> tuple[float, list[dict]]:
    """Trains a fresh model per LR candidate for a short sub-run, returns (best_lr,
    all_sweep_results). ALL results are returned/logged, not just the winner, per the
    spec's "log the sweep" requirement."""
    train_cfg = run_cfg.train
    sweep_steps = max(1, int(train_cfg.steps * train_cfg.lr_sweep_step_fraction))
    results = []
    best_lr, best_loss = None, float("inf")

    for lr in train_cfg.lr_sweep:
        model = build_model(letter, run_cfg, num_tokens=num_tokens)
        final_loss, wall_time_s, truncated = _run_training(
            model, run_cfg.data, train_cfg, lr, sweep_steps, seed, device
        )
        results.append({"lr": lr, "steps": sweep_steps, "final_loss": final_loss,
                         "wall_time_s": wall_time_s, "truncated": truncated})
        if final_loss < best_loss:
            best_loss, best_lr = final_loss, lr

    return best_lr, results


def run_model(
    letter: str,
    run_cfg: RunConfig,
    num_tokens: int = 256,
    device: str | None = None,
) -> dict:
    """Single entry point shared by the notebook and any CLI driver: build -> (skip+log
    if ModelUnavailable) -> LR sweep -> per-seed train -> save checkpoint + TrainResult
    JSON. Returns a summary dict; never raises ModelUnavailable (catches it internally)."""
    device = device or run_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    results_dir = Path(run_cfg.results_dir)

    try:
        probe_model = build_model(letter, run_cfg, num_tokens=num_tokens)
    except ModelUnavailable as e:
        with open("FRICTION_LOG.md", "a") as f:
            f.write(f"\n- model {letter} skipped: {e.reason}\n")
        return {"letter": letter, "available": False, "reason": e.reason}

    param_count = log_param_count(f"model_{letter}", probe_model)
    del probe_model  # the real training models are built fresh per seed/lr below

    best_lr, sweep_results = lr_sweep(letter, run_cfg, seed=run_cfg.train.seeds[0], device=device, num_tokens=num_tokens)
    with open(results_dir / "runs" / f"{letter}_lr_sweep_{run_cfg.tier}.json", "w") as f:
        json.dump({"model": letter, "tier": run_cfg.tier, "sweep": sweep_results, "best_lr": best_lr}, f, indent=2)

    train_results = []
    for seed in run_cfg.train.seeds:
        model = build_model(letter, run_cfg, num_tokens=num_tokens)
        final_loss, wall_time_s, truncated = _run_training(
            model, run_cfg.data, run_cfg.train, best_lr, run_cfg.train.steps, seed, device
        )
        result = TrainResult(
            model_name=letter, seed=seed, lr=best_lr, steps=run_cfg.train.steps,
            final_train_loss=final_loss, wall_time_s=wall_time_s, param_count=param_count,
            config_tier=run_cfg.tier, truncated=truncated,
        )
        result.to_json(results_dir / "runs" / f"{letter}_{seed}_{run_cfg.tier}.json")
        ckpt_path = results_dir / "checkpoints" / f"{letter}_{seed}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        train_results.append(result)

    return {
        "letter": letter,
        "available": True,
        "param_count": param_count,
        "best_lr": best_lr,
        "sweep_results": sweep_results,
        "train_results": train_results,
        "device": device,
    }
