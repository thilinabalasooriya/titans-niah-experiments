# Titans NIAH Experiments

Small-scale tests on [Titans](https://arxiv.org/abs/2501.00663) neural long-term memory mechanism
("Memory as Context", via [`lucidrains/titans-pytorch`](https://github.com/lucidrains/titans-pytorch)).

Defines a simple byte-level, single-passkey "needle in a haystack" task (vocab=256, no tokenizer) to test Titans MAC  against three baselines: a hand-written sliding-window
transformer (**A**, no long-range mechanism at all), the same MAC scaffold with memory
disabled (**C**, isolates whether the extra scaffolding — persistent/long-term tokens,
segmented attention, hyper-connections — matters on its own), and GatedDeltaNet
(**D**, a non-Titans linear-attention/delta-rule alternative). 

By Thilina Balasooriya for COMS 6998 CLMM Fall 2026 w/ Prof. Richard Zemel.

## Experiments

1. **NIAH length × depth scaling** (`experiments/exp1_niah_length_depth.py`) — trains
   A, B (chunk size 8), C, D through a 101→256→512→1024 length curriculum and measures
   greedy exact-match retrieval accuracy across lengths up to 4096 and five needle
   depths. Each trained length is evaluated with *that length's own* end-of-training
   checkpoint (not a later-stage one), so the numbers reflect genuine per-length
   capability rather than later-curriculum forgetting. Lengths 2048/4096 have no
   training stage of their own — they are evaluated with the length=1024 checkpoint,
   an explicit zero-further-training generalization test.
2. **Neural-memory chunk-size ablation** (`experiments/exp2_chunk_ablation.py`) —
   same protocol, trains B at memory chunk sizes {4, 8, 16, 32} to separate whether
   chunk size is just an inference-time trade-off knob or something the model needs to
   be trained around.
3. **Causal test-time memorization** (`experiments/exp3_causal_memory.py`) — on a
   single trained B checkpoint, compares normal inference against two interventions
   (frozen memory writes, zeroed memory reads) plus a counterfactual needle-swap test,
   to establish that retrieval is *caused* by the memory mechanism rather than
   incidental to it.

## Setup

```bash
pip install -r requirements.txt
```

Model D (GatedDeltaNet) additionally needs a working Triton/CUDA toolchain at runtime
and is trained under bf16 autocast — its Triton kernel cannot run in true fp32 on an
A100 due to a shared-memory limit (see `experiments/common/pipeline.py`).

## Reproducing

```bash
PYTHONPATH=. python3 experiments/run_all_experiments.py       # all three, in sequence
# or individually (each is independently resumable and idempotent per curriculum stage):
PYTHONPATH=. python3 experiments/exp1_niah_length_depth.py
PYTHONPATH=. python3 experiments/exp2_chunk_ablation.py
PYTHONPATH=. python3 experiments/exp3_causal_memory.py

# presentation tables (LaTeX + plain text), once Experiment 1/3 have produced output:
PYTHONPATH=. python3 experiments/make_tables.py
```

All outputs (checkpoints, logs, the NIAH grid, CSVs, LaTeX tables) are written under
`results/` (gitignored).

This repo was created using Claude Code.
