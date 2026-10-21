"""Two non-destructive, context-managed interventions on a TRAINED B model's NeuralMemory
submodules, for Phase 2E ("does memory do real work?"):

1. `frozen_memory_writes(model)`: disables TEST-TIME writes. Monkeypatches each
   NeuralMemory's `store_memories` bound method so the real call still happens
   (preserving correct shapes/dtypes/state-trajectory structure -- never hand-reimplemented)
   but every chunk-trajectory step is overwritten to equal chunk index 0 (the weights
   as they stood BEFORE any update in this call). Since a fresh call with no carried
   `past_state` starts from `init_weights()` -- itself derived from the memory model's
   OUTER-trained nn.Parameters -- this keeps "all other parameters and initial memory
   weights" exactly as trained, while preventing any further within-sequence (test-time)
   adaptation. The carried-forward state is pinned too, so a later chunk in the same
   call also can't drift away from the frozen weights.

2. `zeroed_memory_contribution(model)`: disables the RETRIEVED signal's contribution to
   the residual stream (multiplies `retrieved` by 0 after the real forward call),
   independent of whether writes are frozen or not -- isolates "does the attention/FF
   backbone alone (no memory signal at all) still solve it" from "does disabling writes
   alone (but still reading SOME memory state) change anything."

Both are context managers that monkeypatch and restore the original bound methods; they
never mutate the model's actual parameters.
"""
import contextlib

import torch
from tensordict import TensorDict
from titans_pytorch import NeuralMemory


def _find_neural_memories(model):
    found = []
    for m in model.modules():
        if isinstance(m, NeuralMemory):
            found.append(m)
    return found


@contextlib.contextmanager
def frozen_memory_writes(model):
    mems = _find_neural_memories(model)
    originals = [m.store_memories for m in mems]

    def make_frozen(orig_bound):
        def frozen(seq, weights=None, past_state=None, seq_index=0, prev_weights=None,
                   mask=None, return_surprises=True):
            next_updates, next_state, surprises = orig_bound(
                seq, weights=weights, past_state=past_state, seq_index=seq_index,
                prev_weights=prev_weights, mask=mask, return_surprises=True,
            )
            frozen_updates = TensorDict({k: v[:, :1].expand_as(v).contiguous()
                                          for k, v in next_updates.items()})
            frozen_last = TensorDict({k: v[:, 0] for k, v in frozen_updates.items()})
            _, last_momentum = next_state.states
            next_state = next_state._replace(weights=frozen_last, states=(frozen_last, last_momentum))
            return frozen_updates, next_state, surprises
        return frozen

    try:
        for m, orig in zip(mems, originals):
            m.store_memories = make_frozen(orig)
        yield
    finally:
        for m, orig in zip(mems, originals):
            m.store_memories = orig


@contextlib.contextmanager
def zeroed_memory_contribution(model):
    mems = _find_neural_memories(model)
    originals = [m.forward for m in mems]

    def make_zeroed(orig_bound):
        def zeroed(*args, **kwargs):
            retrieved, state = orig_bound(*args, **kwargs)
            return retrieved * 0.0, state
        return zeroed

    try:
        for m, orig in zip(mems, originals):
            m.forward = make_zeroed(orig)
        yield
    finally:
        for m, orig in zip(mems, originals):
            m.forward = orig
