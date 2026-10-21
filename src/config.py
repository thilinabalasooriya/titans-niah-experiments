"""Config schema shared by all three tiers (flash/smoke/full).

The three YAML files in configs/ differ only in *values*; this module defines the
*shape* once so that switching tiers is a one-line change in the notebook/CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

MODEL_LETTERS = ("A", "B", "C", "D")


@dataclass
class ModelConfig:
    dim: int = 256
    depth: int = 8
    heads: int = 8
    dim_head: int = 32
    ff_mult: int = 4
    segment_len: int = 128
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 16
    # None -> all layers get neural memory (model B); () -> no layers do (model C ablation)
    neural_memory_layers: tuple[int, ...] | None = None
    neural_memory_chunk_size: int | None = None
    neural_memory_mlp_dim: int = 64
    neural_memory_mlp_depth: int = 2
    # Model D (gated_deltanet) only
    fla_hidden_size: int | None = None
    fla_num_heads: int | None = None
    fla_head_dim: int | None = None
    # Model A dense-mask memory guard: switch to block-local attention above this many
    # mask bytes (L*L for a bool mask) to avoid OOM at long eval lengths.
    max_dense_mask_bytes: int = 64 * 1024 * 1024  # 64MB
    # ReZero/Flamingo-style fix for model B's stuck training (DECISIONS.md D17/D18):
    # wraps every NeuralMemory submodule with a learned scalar gate initialized to this
    # value, so the memory branch starts silent and is phased in by gradient descent.
    # None (default) = off; B's own definition stays unchanged unless overridden.
    memory_gate_init: float | None = None
    # MAC's hyper-connections residual-stream count (B/C only). None = library default
    # (4). Overnight-debug ablation knob (2026-10-20): isolates whether hyper-connections
    # specifically blocks C/B_gated's rung-0 training; narrowly scoped, not a general
    # tuning surface.
    num_residual_streams: int | None = None
    # Overnight-debug ablation (2026-10-20): switches MAC's segmented attention to
    # sliding-window attention (upstream's own `sliding_window_attn` flag) to isolate
    # whether segmented attention specifically blocks C's rung-0 training.
    sliding_window_attn: bool | None = None


@dataclass
class DataConfig:
    vocab_size: int = 256
    train_len_min: int = 256
    train_len_max: int = 1024
    window_for_training: int = 128
    eval_lengths: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
    eval_lengths_extended: tuple[int, ...] = (16384,)
    eval_depths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    eval_examples_per_cell: int = 100
    eval_micro_batch: int = 16  # eval examples are chunked into micro-batches of this size (memory safety at long lengths)
    sanity_length: int = 128
    sanity_window: int = 128
    sanity_examples: int = 200
    sanity_min_accuracy: float = 0.98
    passkey_digits: int = 5
    aux_lm_loss_weight: float = 0.0
    seed: int = 0


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    lr_sweep: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    lr_sweep_step_fraction: float = 0.15  # each sweep candidate trains this fraction of `steps`
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    schedule: str = "cosine"  # "cosine" | "linear"
    warmup_steps: int = 0
    grad_clip: float = 1.0
    bf16: bool = True
    seeds: tuple[int, ...] = (0,)
    max_minutes_per_run: int = 15
    eval_every: int | None = None


@dataclass
class RunConfig:
    tier: str = "smoke"
    models: tuple[str, ...] = MODEL_LETTERS
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_base: ModelConfig = field(default_factory=ModelConfig)
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    results_dir: str = "results"
    device: str = "cuda"
    run_extension: bool = True
    run_chunk_sweep: bool = False


def _dataclass_from_dict(cls, data: dict[str, Any] | None):
    data = data or {}
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    # YAML gives lists where we want tuples for the "shape never changes" guarantee.
    coerced = {}
    for k, v in data.items():
        default = cls.__dataclass_fields__[k].default
        if isinstance(v, list):
            v = tuple(v)
        coerced[k] = v
    return cls(**coerced)


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = _dataclass_from_dict(DataConfig, raw.get("data"))
    train_cfg = _dataclass_from_dict(TrainConfig, raw.get("train"))
    model_base = _dataclass_from_dict(ModelConfig, raw.get("model_base"))
    model_overrides = raw.get("model_overrides", {}) or {}

    known_top = {"tier", "models", "data", "train", "model_base", "model_overrides",
                 "results_dir", "device", "run_extension", "run_chunk_sweep"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_top)}")

    return RunConfig(
        tier=raw.get("tier", "smoke"),
        models=tuple(raw.get("models", MODEL_LETTERS)),
        data=data_cfg,
        train=train_cfg,
        model_base=model_base,
        model_overrides=model_overrides,
        results_dir=raw.get("results_dir", "results"),
        device=raw.get("device", "cuda"),
        run_extension=raw.get("run_extension", True),
        run_chunk_sweep=raw.get("run_chunk_sweep", False),
    )


def resolve_model_config(run_cfg: RunConfig, letter: str) -> ModelConfig:
    """model_base with per-letter overrides from model_overrides[letter] applied."""
    if letter not in MODEL_LETTERS:
        raise ValueError(f"Unknown model letter {letter!r}, expected one of {MODEL_LETTERS}")
    overrides = dict(run_cfg.model_overrides.get(letter, {}))
    for k, v in list(overrides.items()):
        if isinstance(v, list):
            overrides[k] = tuple(v)
    return replace(run_cfg.model_base, **overrides)
