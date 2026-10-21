"""Generic training/eval building blocks shared by all three experiment scripts:
model construction, teacher-forced/greedy evaluation (microbatched with OOM backoff),
and checkpoint save/load. No experiment-specific paths or state live here.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RunConfig
from .models import build_model
from .train import masked_answer_logits


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def build_model_for_spec(letter: str, run_cfg: RunConfig, model_overrides: dict, num_tokens: int = 256):
    """letter in {'A','B','C','D'}; model_overrides merged on top of run_cfg.model_overrides[letter]."""
    merged = replace(run_cfg, model_overrides={
        **run_cfg.model_overrides,
        letter: {**run_cfg.model_overrides.get(letter, {}), **model_overrides},
    })
    return build_model(letter, merged, num_tokens=num_tokens)


@torch.no_grad()
def eval_fixed(model, val_batch: dict, device: str, num_digits: int = 5, micro_batch: int = 16) -> dict:
    """Teacher-forced evaluation: loss + per-digit accuracy + exact match."""
    was_training = model.training
    model.eval()
    n = val_batch["input_ids"].shape[0]
    losses, preds, targets = [], [], []
    start = 0
    cur_mb = micro_batch
    while start < n:
        sl = slice(start, start + cur_mb)
        try:
            ii = val_batch["input_ids"][sl].to(device)
            pm = val_batch["pad_mask"][sl].to(device)
            am = val_batch["answer_mask"][sl].to(device)
            tg = val_batch["targets"][sl].to(device)
            logits = model(ii, pm)
            sel = masked_answer_logits(logits, am, num_digits)
            loss = F.cross_entropy(sel.reshape(-1, sel.shape[-1]), tg.reshape(-1), reduction="sum")
        except RuntimeError as e:
            if not _is_oom(e) or cur_mb <= 1:
                raise
            torch.cuda.empty_cache()
            cur_mb = max(1, cur_mb // 2)
            continue
        losses.append(loss.item())
        preds.append(sel.argmax(-1).cpu())
        targets.append(tg.cpu())
        start += cur_mb
    model.train(was_training)
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    return {
        "loss": sum(losses) / (n * num_digits),
        "per_digit_acc": (preds == targets).float().mean().item(),
        "exact_match": (preds == targets).all(-1).float().mean().item(),
        "n": n,
    }


@torch.no_grad()
def eval_greedy(model, val_batch: dict, device: str, num_digits: int = 5, micro_batch: int = 16) -> dict:
    """Autoregressive generation from the query prefix only -- the model never sees the
    true answer digits, even as teacher-forced context. This is the primary metric."""
    was_training = model.training
    model.eval()
    n = val_batch["input_ids"].shape[0]
    all_correct, all_exact = [], []
    start = 0
    cur_mb = micro_batch
    while start < n:
        sl = slice(start, start + cur_mb)
        try:
            ii_full = val_batch["input_ids"][sl].to(device)
            tg = val_batch["targets"][sl].to(device)
            prefix = ii_full[:, :-num_digits].clone()
            preds = []
            for _ in range(num_digits):
                logits = model(prefix, None)
                nxt = logits[:, -1].argmax(-1)
                preds.append(nxt)
                prefix = torch.cat([prefix, nxt[:, None]], dim=1)
            preds = torch.stack(preds, dim=1)
        except RuntimeError as e:
            if not _is_oom(e) or cur_mb <= 1:
                raise
            torch.cuda.empty_cache()
            cur_mb = max(1, cur_mb // 2)
            continue
        all_correct.append((preds == tg).float().cpu())
        all_exact.append((preds == tg).all(-1).float().cpu())
        start += cur_mb
    model.train(was_training)
    correct = torch.cat(all_correct, 0)
    exact = torch.cat(all_exact, 0)
    return {"per_digit_acc": correct.mean().item(), "exact_match": exact.mean().item(), "n": n}


def grad_global_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def save_checkpoint(path: Path, model, opt, sched, step: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: Path, model, opt=None, sched=None, map_location="cuda") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], assign=True)
    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    if sched is not None and ckpt.get("scheduler") is not None:
        sched.load_state_dict(ckpt["scheduler"])
    return ckpt
