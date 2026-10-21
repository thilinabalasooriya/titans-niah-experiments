"""Experiment 3 -- causal demonstration of test-time memorization on a single trained
B checkpoint. Reuses memory_ablations.py's frozen_memory_writes / zeroed_memory_contribution
context managers (generic, structure-based) and probes.py's causal_and_needle_tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import re

import torch

from src.harness import eval_fixed, eval_greedy
from .memory_ablations import frozen_memory_writes, zeroed_memory_contribution
from .probes import causal_and_needle_tests

try:
    from titans_pytorch import NeuralMemory
except ImportError:
    NeuralMemory = None


def param_hash(model: torch.nn.Module) -> str:
    """Cheap SHA-256 fingerprint of a model's current parameter tensors -- used only to
    verify no outer-loop weight update happened during an inference-only intervention."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _find_neural_memories(model):
    if NeuralMemory is None:
        return []
    return [m for m in model.modules() if isinstance(m, NeuralMemory)]


@contextlib.contextmanager
def instrument_memory_state_trajectory(model):
    """Records, for every NeuralMemory submodule, the per-chunk L2 distance of its
    store_memories trajectory from chunk index 0, WITHOUT altering anything (pure
    recording). Yields a `trajectories` list; read it after the `with` block."""
    mems = _find_neural_memories(model)
    originals = [m.store_memories for m in mems]
    trajectories: list[dict] = []

    def make_recorder(orig_bound, mem_idx):
        def recorder(seq, weights=None, past_state=None, seq_index=0, prev_weights=None,
                     mask=None, return_surprises=True):
            next_updates, next_state, surprises = orig_bound(
                seq, weights=weights, past_state=past_state, seq_index=seq_index,
                prev_weights=prev_weights, mask=mask, return_surprises=True,
            )
            per_chunk_l2 = {}
            for k, v in next_updates.items():
                ref = v[:, :1]
                per_chunk_l2[k] = (v - ref).float().pow(2).sum(dim=tuple(range(2, v.ndim))).sqrt().mean(0).tolist()
            trajectories.append({"mem_idx": mem_idx, "num_chunks": next(iter(next_updates.values())).shape[1],
                                  "per_chunk_l2_from_chunk0": per_chunk_l2})
            return next_updates, next_state, surprises
        return recorder

    try:
        for i, (m, orig) in enumerate(zip(mems, originals)):
            m.store_memories = make_recorder(orig, i)
        yield trajectories
    finally:
        for m, orig in zip(mems, originals):
            m.store_memories = orig


def assert_no_training_sideeffect(model, forward_fn) -> dict:
    """Verification gate: snapshot model.parameters() hash, run forward_fn() under
    torch.no_grad(), confirm the hash is bitwise unchanged afterward."""
    before = param_hash(model)
    with torch.no_grad():
        out = forward_fn()
    after = param_hash(model)
    return {"ok": before == after, "hash_before": before, "hash_after": after, "result": out}


def run_experiment3(
    *, model_b, checkpoint_sha256: str, load_val_set_grid_fn, lengths=(101, 1024, 2048),
    far_distance_n: int = 60, device="cuda", num_digits: int = 5,
) -> dict:
    """model_b: a trained B model, already loaded & in eval mode. load_val_set_grid_fn(
    length, "all") -> batch, the same held-out examples used for Experiment 1."""
    out = {"checkpoint_sha256": checkpoint_sha256, "by_length": {}, "mechanistic_state_test": {},
           "counterfactual_needle_swap": {}, "verification_gates": {}}

    was_training = model_b.training
    model_b.eval()

    conditions = {
        "B_normal": contextlib.nullcontext,
        "B_frozen_writes": lambda: frozen_memory_writes(model_b),
        "B_zeroed_reads": lambda: zeroed_memory_contribution(model_b),
    }

    print(f"[exp3] primary conditions x lengths: {lengths}", flush=True)
    for length in lengths:
        batch = load_val_set_grid_fn(length, "all")
        out["by_length"][length] = {}
        for cond_name, cm_factory in conditions.items():
            print(f"[exp3]   length={length} condition={cond_name} (n={batch['input_ids'].shape[0]})...", flush=True)
            with cm_factory():
                tf = eval_fixed(model_b, batch, device, num_digits=num_digits)
                g = eval_greedy(model_b, batch, device, num_digits=num_digits)
            out["by_length"][length][cond_name] = {"teacher_forced": tf, "greedy": g}
            print(f"[exp3]   length={length} condition={cond_name}: tf_EM={tf['exact_match']:.3f} "
                  f"greedy_EM={g['exact_match']:.3f}", flush=True)

    print("[exp3] mechanistic state test (fast-weight trajectory)...", flush=True)
    probe_batch = load_val_set_grid_fn(1024, "all")
    probe_ids = probe_batch["input_ids"][:4].to(device)
    probe_pad = probe_batch["pad_mask"][:4].to(device)

    def _forward():
        with instrument_memory_state_trajectory(model_b) as traj:
            model_b(probe_ids, probe_pad)
        return traj

    gate_result = assert_no_training_sideeffect(model_b, _forward)
    out["mechanistic_state_test"] = {
        "no_training_sideeffect_ok": gate_result["ok"],
        "param_hash_before": gate_result["hash_before"], "param_hash_after": gate_result["hash_after"],
        "trajectories": gate_result["result"],
    }
    any_nonzero_drift = any(
        any(abs(v) > 1e-8 for v in traj_entry["per_chunk_l2_from_chunk0"][k])
        for traj_entry in gate_result["result"] for k in traj_entry["per_chunk_l2_from_chunk0"]
    )
    out["mechanistic_state_test"]["fast_weights_genuinely_change"] = any_nonzero_drift
    print(f"[exp3]   fast_weights_genuinely_change={any_nonzero_drift} "
          f"no_training_sideeffect_ok={gate_result['ok']}", flush=True)

    def _forward_frozen():
        with frozen_memory_writes(model_b), instrument_memory_state_trajectory(model_b) as traj:
            model_b(probe_ids, probe_pad)
        return traj
    frozen_traj = assert_no_training_sideeffect(model_b, _forward_frozen)["result"]
    frozen_all_zero = all(
        all(abs(v) < 1e-6 for v in traj_entry["per_chunk_l2_from_chunk0"][k])
        for traj_entry in frozen_traj for k in traj_entry["per_chunk_l2_from_chunk0"]
    )
    out["mechanistic_state_test"]["frozen_writes_genuinely_constant"] = frozen_all_zero
    print(f"[exp3]   frozen_writes_genuinely_constant={frozen_all_zero}", flush=True)

    print("[exp3] verification gate: normal->frozen->normal logit restoration...", flush=True)
    small_batch_ids = probe_ids[:2]
    small_pad = probe_pad[:2]
    with torch.no_grad():
        logits_normal_1 = model_b(small_batch_ids, small_pad).clone()
        with frozen_memory_writes(model_b):
            _ = model_b(small_batch_ids, small_pad)
        logits_normal_2 = model_b(small_batch_ids, small_pad).clone()
    restore_max_diff = (logits_normal_1 - logits_normal_2).abs().max().item()
    out["verification_gates"]["normal_frozen_normal_restoration_max_diff"] = restore_max_diff
    out["verification_gates"]["normal_frozen_normal_restoration_ok"] = restore_max_diff < 1e-4
    print(f"[exp3]   restoration_max_diff={restore_max_diff:.3e} ok={restore_max_diff < 1e-4}", flush=True)

    print("[exp3] verification gate: microbatch-size invariance (mb=16 vs mb=4) at length=2048...", flush=True)
    batch2048 = load_val_set_grid_fn(2048, "all")
    tf_mb16 = eval_fixed(model_b, batch2048, device, num_digits=num_digits, micro_batch=16)
    tf_mb4 = eval_fixed(model_b, batch2048, device, num_digits=num_digits, micro_batch=4)
    out["verification_gates"]["microbatch_invariance_em_mb16"] = tf_mb16["exact_match"]
    out["verification_gates"]["microbatch_invariance_em_mb4"] = tf_mb4["exact_match"]
    out["verification_gates"]["microbatch_invariance_ok"] = abs(tf_mb16["exact_match"] - tf_mb4["exact_match"]) < 1e-6
    print(f"[exp3]   EM(mb16)={tf_mb16['exact_match']:.3f} EM(mb4)={tf_mb4['exact_match']:.3f}", flush=True)

    print("[exp3] verification gate: causal/needle battery per condition...", flush=True)
    out["verification_gates"]["causal_needle_by_condition"] = {}
    for cond_name, cm_factory in conditions.items():
        with cm_factory():
            r = causal_and_needle_tests(model_b, device=device, length=101, n=8)
        out["verification_gates"]["causal_needle_by_condition"][cond_name] = r
        print(f"[exp3]   {cond_name}: {r}", flush=True)

    print(f"[exp3] counterfactual needle swap at length=2048, n={far_distance_n}...", flush=True)
    out["counterfactual_needle_swap"] = run_counterfactual_needle_swap(
        model_b, batch2048, device, n=far_distance_n, num_digits=num_digits,
    )
    print("[exp3] done.", flush=True)

    model_b.train(was_training)
    return out


_NEEDLE_BYTES_RE = re.compile(rb"the passkey is (\d{5})\.")


def run_counterfactual_needle_swap(model, batch, device, n: int, num_digits: int = 5,
                                    replacement_offsets=(5, 3, 7)) -> dict:
    """Preserves all filler/query bytes; alters ONLY the five needle digits (by several
    distinct offsets, mod 10); greedily decodes under B_normal and B_frozen_writes;
    reports % matching the CURRENT (new) needle, % matching the stale original, %
    matching neither. Microbatched with OOM backoff."""
    ids_all = batch["input_ids"]
    n = min(n, ids_all.shape[0])
    idx = torch.randperm(ids_all.shape[0])[:n]
    ids = ids_all[idx].clone()

    results = {}
    for offset in replacement_offsets:
        print(f"[exp3]   counterfactual offset={offset} (n={n})...", flush=True)
        modified = ids.clone()
        original_digits, new_digits = [], []
        for row in range(n):
            raw = bytes(ids[row].tolist())
            m = _NEEDLE_BYTES_RE.search(raw)
            if m is None:
                raise RuntimeError(f"needle not found in row {row}")
            start = m.start(1)
            orig = ids[row, start:start + 5].clone()
            new = (orig - 48 + offset).remainder(10) + 48
            modified[row, start:start + 5] = new
            original_digits.append(orig.cpu())
            new_digits.append(new.cpu())
        original_digits = torch.stack(original_digits)
        new_digits = torch.stack(new_digits)

        for cond_name, cm_factory in (("B_normal", contextlib.nullcontext),
                                       ("B_frozen_writes", lambda: frozen_memory_writes(model))):
            all_preds = []
            start = 0
            cur_mb = 16
            body = modified[:, :-num_digits]
            while start < n:
                sl = slice(start, start + cur_mb)
                try:
                    prefix = body[sl].to(device)
                    mb_preds = []
                    with torch.no_grad(), cm_factory():
                        for _ in range(num_digits):
                            logits = model(prefix, None)
                            nxt = logits[:, -1].argmax(-1)
                            mb_preds.append(nxt)
                            prefix = torch.cat([prefix, nxt[:, None]], dim=1)
                    mb_preds = torch.stack(mb_preds, dim=1).cpu()
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or cur_mb <= 1:
                        raise
                    torch.cuda.empty_cache()
                    print(f"[exp3]     OOM at microbatch={cur_mb}, backing off to {max(1, cur_mb // 2)}", flush=True)
                    cur_mb = max(1, cur_mb // 2)
                    continue
                all_preds.append(mb_preds)
                start += cur_mb
            preds = torch.cat(all_preds, dim=0)
            matches_new = (preds == new_digits).all(-1)
            matches_old = (preds == original_digits).all(-1)
            neither = ~(matches_new | matches_old)
            key = f"offset_{offset}__{cond_name}"
            results[key] = {
                "n": n, "pct_matches_new_needle": matches_new.float().mean().item(),
                "pct_matches_stale_needle": matches_old.float().mean().item(),
                "pct_matches_neither": neither.float().mean().item(),
            }
            print(f"[exp3]     {cond_name}: new={results[key]['pct_matches_new_needle']:.3f} "
                  f"stale={results[key]['pct_matches_stale_needle']:.3f} "
                  f"neither={results[key]['pct_matches_neither']:.3f}", flush=True)
    return results
