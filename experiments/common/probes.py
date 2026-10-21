"""Causal/needle and greedy-vs-teacher-forced verification probes, trimmed from the
original project's `overnight_diagnosis/titans_debug_probes.py` to just the two
functions Experiment 3 actually uses, plus their direct helper dependencies.
"""
from __future__ import annotations

import contextlib
import json
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src import data as data_mod


def _device(device: str) -> torch.device:
    value = torch.device(device)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return value


def _logits(model: nn.Module, input_ids: torch.Tensor, pad_mask=None) -> torch.Tensor:
    output = model(input_ids, pad_mask)
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise TypeError(f"Expected tensor (B,L,V), got {type(output)} / {getattr(output, 'shape', None)}")
    if output.shape[:2] != input_ids.shape:
        raise ValueError(f"Logit shape {output.shape} mismatches input {input_ids.shape}")
    return output.float()


def _answer_logits(logits: torch.Tensor, answer_mask: torch.Tensor, digits: int) -> torch.Tensor:
    b, _, v = logits.shape
    counts = answer_mask.sum(-1)
    if not torch.all(counts == digits):
        raise AssertionError(f"Expected {digits} supervised positions per example, got {counts.tolist()}")
    return logits[answer_mask].reshape(b, digits, v)


def _fixed_batch(*, n: int = 8, length: int = 101, depth: float = 0.5,
                  seed: int = 1234, num_digits: int = 5) -> dict:
    return data_mod.make_eval_batch(length, depth, n, np.random.default_rng(seed), num_digits)


def _tensor_batch(batch: dict, device: str) -> dict:
    return {k: v.to(_device(device)) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@contextlib.contextmanager
def _eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def causal_and_needle_tests(model: nn.Module, device: str = "cuda", *, length: int = 101,
                             depth: float = 0.5, seed: int = 1234, n: int = 4) -> dict:
    """Future-target perturbation, full-vs-prefix, and passkey counterfactual.

    Difference in full-vs-prefix logits is a *warning*, not conclusive evidence of
    leakage, as MAC's positional embeddings/chunking can depend on total length.
    The direct future-token perturbation is the stronger causality test.
    """
    batch = _tensor_batch(_fixed_batch(n=n, length=length, depth=depth, seed=seed), device)
    ids, targets = batch["input_ids"], batch["targets"]
    pos = ids.shape[1] - targets.shape[1] - 1
    with _eval_mode(model), torch.no_grad():
        reference = _logits(model, ids, batch["pad_mask"])[:, pos]
        future = ids.clone()
        future[:, pos + 1:] = (future[:, pos + 1:] - 48 + 5).remainder(10) + 48
        future_output = _logits(model, future, batch["pad_mask"])[:, pos]
        short_ids = ids[:, :pos + 1].contiguous()
        short_output = _logits(model, short_ids, batch["pad_mask"][:, :pos + 1])[:, -1]
        needle = ids.clone()
        for row in range(n):
            found = re.search(rb"the passkey is (\d{5})\.", bytes(ids[row].tolist()))
            if found is None:
                raise RuntimeError(f"Needle not found in example {row}")
            needle_start = found.start(1)
            needle[row, needle_start:needle_start + 5] = \
                (needle[row, needle_start:needle_start + 5] - 48 + 5).remainder(10) + 48
        needle_output = _logits(model, needle, batch["pad_mask"])[:, pos]

    result = dict(first_answer_logit_index=pos,
                  max_future_answer_logit_delta=float((reference - future_output).abs().max().item()),
                  max_prefix_vs_full_logit_delta=float((reference - short_output).abs().max().item()),
                  max_needle_logit_delta=float((reference - needle_output).abs().max().item()),
                  first_digit_pred_before=reference.argmax(-1).tolist(),
                  first_digit_pred_after_needle_change=needle_output.argmax(-1).tolist(),
                  true_first_digit=targets[:, 0].tolist())
    print("CAUSAL / NEEDLE:", json.dumps(result, indent=2))
    return result


def greedy_vs_teacher_forced(model: nn.Module, device: str = "cuda", *, n: int = 16,
                              length: int = 101, depth: float = 0.5,
                              seed: int = 2048) -> dict:
    """Compare actual autoregressive generation with teacher-forced exact match."""
    batch = _tensor_batch(_fixed_batch(n=n, length=length, depth=depth, seed=seed), device)
    ids, mask, target = batch["input_ids"], batch["answer_mask"], batch["targets"]
    digits = target.shape[-1]
    with _eval_mode(model), torch.no_grad():
        all_logits = _logits(model, ids, batch["pad_mask"])
        tf_pred = _answer_logits(all_logits, mask, digits).argmax(-1)
        prefix = ids[:, :-digits].clone()
        autoreg = []
        for _ in range(digits):
            next_digit = _logits(model, prefix, None)[:, -1].argmax(-1)
            autoreg.append(next_digit)
            prefix = torch.cat((prefix, next_digit[:, None]), dim=-1)
        greedy_pred = torch.stack(autoreg, dim=-1)
    out = dict(n=n, teacher_forced_exact_match=float((tf_pred == target).all(-1).float().mean()),
               greedy_exact_match=float((greedy_pred == target).all(-1).float().mean()),
               teacher_forced_per_digit=float((tf_pred == target).float().mean()),
               greedy_per_digit=float((greedy_pred == target).float().mean()),
               examples=[])
    for i in range(min(n, 5)):
        to_text = lambda x: "".join(chr(int(t)) if 32 <= int(t) < 127 else "?" for t in x)
        out["examples"].append(dict(target=to_text(target[i]), teacher=to_text(tf_pred[i]),
                                    greedy=to_text(greedy_pred[i])))
    print("GREEDY vs TF:", json.dumps(out, indent=2))
    return out
