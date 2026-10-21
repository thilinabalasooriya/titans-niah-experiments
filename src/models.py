"""Model builders A-D. All byte-level (vocab=256 by default), same width/depth family.

A (window_transformer) is hand-written from scratch, no external deps, buildable
anywhere. B/C (titans_mac / titans_mac_nomem) and D (gated_deltanet) depend on
titans-pytorch / flash-linear-attention respectively; they are wrapped so a missing
package, or an installed version whose API drifted from what Step 1 (verify_apis.py)
expects, raises a caught `ModelUnavailable` instead of crashing the whole run — the
caller (train.py / the notebook) is expected to catch it, log it, and skip that model.
"""
from __future__ import annotations

import inspect
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, RunConfig, resolve_model_config

FRICTION_LOG = "FRICTION_LOG.md"


class ModelUnavailable(Exception):
    """Raised when a model can't be built: missing package, or a required API that
    Step 1's recon expected is missing from whatever version actually installed."""

    def __init__(self, letter: str, reason: str):
        super().__init__(f"model {letter} unavailable: {reason}")
        self.letter = letter
        self.reason = reason


def _log_friction(msg: str, log_path: str = FRICTION_LOG) -> None:
    with open(log_path, "a") as f:
        f.write(f"\n- {msg}\n")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def log_param_count(name: str, model: nn.Module, log_path: str = FRICTION_LOG) -> int:
    n = count_params(model)
    print(f"{name}: {n:,} params")
    with open(log_path, "a") as f:
        f.write(f"\n- param count `{name}`: {n:,}\n")
    return n


# ---------------------------------------------------------------------------
# Shared small building blocks (used by A and D)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class GEGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.proj = nn.Linear(dim, inner * 2)
        self.out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(x * F.gelu(gate))


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, l, _ = x.shape
    return x.view(b, l, heads, -1).transpose(1, 2)  # (B, H, L, Dh)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, l, dh = x.shape
    return x.transpose(1, 2).reshape(b, l, h * dh)


# ---------------------------------------------------------------------------
# Model A: window_transformer (hand-written, no new dependencies)
# ---------------------------------------------------------------------------

class CausalWindowAttention(nn.Module):
    """Sliding-window causal self-attention. Uses a dense (B,1,L,L) boolean mask via
    F.scaled_dot_product_attention when that's cheap enough; otherwise falls back to a
    block-local implementation that never materializes an L x L tensor (needed at long
    eval lengths, e.g. 8K-16K, especially with many examples in a batch)."""

    def __init__(self, dim: int, heads: int, dim_head: int, window: int, max_dense_mask_bytes: int):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.window = window
        self.max_dense_mask_bytes = max_dense_mask_bytes
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self._band_mask_cache: dict[int, torch.Tensor] = {}
        self._block_mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_band_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = length
        if key not in self._band_mask_cache or self._band_mask_cache[key].device != device:
            idx = torch.arange(length, device=device)
            causal = idx[None, :] <= idx[:, None]
            band = (idx[:, None] - idx[None, :]) < self.window
            self._band_mask_cache[key] = (causal & band)[None, None]
        return self._band_mask_cache[key]

    def _get_block_mask(self, window: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        key = (window, num_blocks)
        if key not in self._block_mask_cache or self._block_mask_cache[key].device != device:
            key_pos = torch.arange(2 * window, device=device) - window
            query_pos = torch.arange(window, device=device)
            causal = key_pos[None, :] <= query_pos[:, None]
            band = (query_pos[:, None] - key_pos[None, :]) < window
            base = (causal & band)[None].expand(num_blocks, window, 2 * window).clone()
            base[0, :, :window] = False  # block 0 has no real previous block (fake zero keys)
            self._block_mask_cache[key] = base[None, None]  # (1,1,nb,W,2W)
        return self._block_mask_cache[key]

    def _block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, h, length, dh = q.shape
        w = self.window
        pad_len = (-length) % w
        if pad_len:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        lp = length + pad_len
        nb = lp // w
        q_b = q.view(b, h, nb, w, dh)
        k_b = k.view(b, h, nb, w, dh)
        v_b = v.view(b, h, nb, w, dh)
        k_prev = torch.cat([torch.zeros_like(k_b[:, :, :1]), k_b[:, :, :-1]], dim=2)
        v_prev = torch.cat([torch.zeros_like(v_b[:, :, :1]), v_b[:, :, :-1]], dim=2)
        k_cat = torch.cat([k_prev, k_b], dim=3)  # (B,H,nb,2W,Dh)
        v_cat = torch.cat([v_prev, v_b], dim=3)
        mask = self._get_block_mask(w, nb, q.device)
        out = F.scaled_dot_product_attention(q_b, k_cat, v_cat, attn_mask=mask)
        out = out.reshape(b, h, lp, dh)[:, :, :length]
        return out

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (_split_heads(t, self.heads) for t in qkv)

        dense_bytes = b * length * length
        if dense_bytes <= self.max_dense_mask_bytes:
            mask = self._get_band_mask(length, x.device)
            mask = mask & pad_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            if not bool(pad_mask.all()):
                raise NotImplementedError(
                    "block-local attention fallback does not support left-padded batches "
                    "(only triggered for long, uniform-length eval batches in this project); "
                    "reduce eval batch size or raise max_dense_mask_bytes instead."
                )
            out = self._block_local_attention(q, k, v)

        return self.to_out(_merge_heads(out))


class WindowTransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalWindowAttention(
            cfg.dim, cfg.heads, cfg.dim_head, cfg.segment_len, cfg.max_dense_mask_bytes
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = GEGLU(cfg.dim, cfg.ff_mult)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class WindowTransformerLM(nn.Module):
    def __init__(self, num_tokens: int, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, cfg.dim)
        self.blocks = nn.ModuleList([WindowTransformerBlock(cfg) for _ in range(cfg.depth)])
        self.norm_out = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, num_tokens, bias=False)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pad_mask is None:
            pad_mask = torch.ones_like(input_ids, dtype=torch.bool)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm_out(x)
        return self.head(x)


def build_window_transformer(cfg: ModelConfig, num_tokens: int = 256) -> WindowTransformerLM:
    return WindowTransformerLM(num_tokens, cfg)


# ---------------------------------------------------------------------------
# Models B/C: titans_mac / titans_mac_nomem
# ---------------------------------------------------------------------------

REQUIRED_MAC_ARGS = ("num_tokens", "dim", "depth", "segment_len")

# Set True by verify_apis.py after a live check; models.py re-checks live anyway (belt
# and suspenders) rather than trusting this flag blindly.
MEMORY_DISABLE_MECHANISM_CONFIRMED: bool | None = None


def _filter_kwargs_to_signature(cls, kwargs: dict) -> tuple[dict, dict]:
    sig_params = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in sig_params}
    dropped = {k: v for k, v in kwargs.items() if k not in sig_params}
    return filtered, dropped


class _UniformForwardWrapper(nn.Module):
    """Standardizes every model's call signature to forward(input_ids, pad_mask=None)
    -> logits (B,L,vocab), so train.py/eval.py never special-case a model. titans-pytorch's
    MemoryAsContextTransformer does not accept a key-padding mask argument (unverified
    until Step 1 runs for real against the installed version — see FRICTION_LOG.md), so
    pad_mask is accepted here but not forwarded; this only matters for mixed-length
    TRAINING batches (eval batches are uniform-length, so pad_mask is all-True there and
    dropping it changes nothing)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.inner(input_ids)


def build_titans_mac(cfg: ModelConfig, num_tokens: int = 256, neural_memory_layers=None):
    try:
        from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
    except ImportError as e:
        raise ModelUnavailable("B", f"titans_pytorch not importable: {e}") from e

    sig_params = set(inspect.signature(MemoryAsContextTransformer.__init__).parameters)
    missing_required = [a for a in REQUIRED_MAC_ARGS if a not in sig_params]
    if missing_required:
        raise ModelUnavailable(
            "B", f"MemoryAsContextTransformer is missing required args {missing_required} "
            f"in the installed titans-pytorch version — API has drifted from what this "
            f"project expects (see FRICTION_LOG.md / run verify_apis.py)."
        )

    nm_layers = cfg.neural_memory_layers if neural_memory_layers is None else neural_memory_layers

    # Live API finding (confirmed by reading the installed titans_pytorch source, not
    # just the README — see FRICTION_LOG.md): NeuralMemory's own `dim_head` defaults to
    # the FULL model `dim` (neural_memory.py: `dim_head = default(dim_head, dim)`), and
    # mac_transformer.py always constructs `NeuralMemory(dim=dim, ...)` with the main
    # model dim. So to keep the memory MLP small, dim_head/heads for the memory
    # submodule must be passed via `neural_memory_kwargs`, matching
    # `neural_memory_model`'s own `dim` — otherwise NeuralMemory's internal self-test
    # ("memory model unable to accept a tensor of shape ...") fails at construction time.
    # heads must be >1 whenever dim_head != dim (titans_pytorch/neural_memory.py:
    # `assert not (heads == 1 and dim_head != dim)`) — 4 is a small, arbitrary-but-fixed
    # choice so dim_inner (= dim_head * heads) stays modest relative to the main model.
    nm_kwargs: dict = {"dim_head": cfg.neural_memory_mlp_dim, "heads": 4}
    if cfg.neural_memory_chunk_size is not None:
        # chunk_size is NOT set via neural_memory_kwargs: MAC passes
        # `chunk_size=self.neural_memory_segment_len` to NeuralMemory directly, so a
        # 'chunk_size' key in neural_memory_kwargs would collide (duplicate kwarg
        # TypeError). It's controlled via MAC's own `neural_memory_segment_len` arg.
        pass  # handled below via kwargs["neural_memory_segment_len"]

    kwargs = dict(
        num_tokens=num_tokens, dim=cfg.dim, depth=cfg.depth, segment_len=cfg.segment_len,
        num_persist_mem_tokens=cfg.num_persist_mem_tokens,
        num_longterm_mem_tokens=cfg.num_longterm_mem_tokens,
        dim_head=cfg.dim_head, heads=cfg.heads, ff_mult=cfg.ff_mult,
        neural_memory_model=MemoryMLP(dim=cfg.neural_memory_mlp_dim, depth=cfg.neural_memory_mlp_depth),
        neural_memory_kwargs=nm_kwargs,
    )
    if nm_layers is not None:
        kwargs["neural_memory_layers"] = nm_layers
    if cfg.neural_memory_chunk_size is not None:
        kwargs["neural_memory_segment_len"] = cfg.neural_memory_chunk_size
    if cfg.num_residual_streams is not None:
        kwargs["num_residual_streams"] = cfg.num_residual_streams
    if cfg.sliding_window_attn is not None:
        kwargs["sliding_window_attn"] = cfg.sliding_window_attn

    filtered, dropped = _filter_kwargs_to_signature(MemoryAsContextTransformer, kwargs)
    if dropped:
        _log_friction(
            f"models.py: dropped unsupported kwargs building titans_mac: {sorted(dropped)} "
            f"(cosmetic API drift from what this project expects; continuing with the rest)"
        )
    mac_model = MemoryAsContextTransformer(**filtered)
    if cfg.memory_gate_init is not None:
        _apply_memory_gate(mac_model, cfg.memory_gate_init)
    return _UniformForwardWrapper(mac_model)


class GatedNeuralMemory(nn.Module):
    """ReZero/Flamingo-style fix for model B's stuck training (DECISIONS.md D17/D18):
    wraps a NeuralMemory module, scaling its output ('retrieved') by a learned scalar
    gate before returning — same (retrieved, next_neural_mem_cache) return shape
    NeuralMemory.forward uses, so it drop-in-replaces the `mem` submodule stored at
    index 4 of each layer's inner ModuleList (confirmed via mac_transformer.py:
    `self.layers.append(ModuleList([mem_hyper_conn, init_hyper_conn(), init_hyper_conn(),
    mem_qkv_layer_selector, mem, attn, ff]))`)."""

    def __init__(self, inner: nn.Module, init_gate: float):
        super().__init__()
        self.inner = inner
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, *args, **kwargs):
        retrieved, next_neural_mem_cache = self.inner.forward(*args, **kwargs)
        return retrieved * self.gate, next_neural_mem_cache


def _apply_memory_gate(mac_model: nn.Module, init_gate: float) -> None:
    """Replaces every non-None `mem` submodule (index 4) inside each layer's inner
    ModuleList with a GatedNeuralMemory wrapper, in place."""
    for layer in mac_model.layers:
        mem = layer[4]
        if mem is not None:
            layer[4] = GatedNeuralMemory(mem, init_gate)


def build_titans_mac_nomem(cfg: ModelConfig, num_tokens: int = 256):
    try:
        from titans_pytorch import MemoryAsContextTransformer
    except ImportError as e:
        raise ModelUnavailable("C", f"titans_pytorch not importable: {e}") from e

    sig_params = set(inspect.signature(MemoryAsContextTransformer.__init__).parameters)
    if "neural_memory_layers" not in sig_params:
        raise ModelUnavailable(
            "C", "installed titans-pytorch version does not accept `neural_memory_layers`; "
            "no clean memory-disable mechanism found in this version, skipping ablation C "
            "(see PLAN.md / DECISIONS.md D2)."
        )
    return build_titans_mac(cfg, num_tokens=num_tokens, neural_memory_layers=())


# ---------------------------------------------------------------------------
# Model D: gated_deltanet (optional, only if flash-linear-attention works)
# ---------------------------------------------------------------------------

def build_gated_deltanet(cfg: ModelConfig, num_tokens: int = 256):
    try:
        from fla.layers import GatedDeltaNet
    except ImportError as e:
        raise ModelUnavailable("D", f"flash-linear-attention not importable: {e}") from e

    hidden_size = cfg.fla_hidden_size or cfg.dim
    num_heads = cfg.fla_num_heads or cfg.heads
    head_dim = cfg.fla_head_dim
    ff_mult = cfg.ff_mult
    depth = cfg.depth

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(hidden_size)
            gdn_kwargs = dict(hidden_size=hidden_size, num_heads=num_heads, mode="chunk")
            if head_dim is not None:
                gdn_kwargs["head_dim"] = head_dim
            self.attn = GatedDeltaNet(**gdn_kwargs)
            self.norm2 = RMSNorm(hidden_size)
            self.mlp = GEGLU(hidden_size, ff_mult)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out = self.attn(self.norm1(x))
            if isinstance(attn_out, tuple):  # some fla layers return (out, cache, ...)
                attn_out = attn_out[0]
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x

    class GatedDeltaNetLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(num_tokens, hidden_size)
            self.blocks = nn.ModuleList([_Block() for _ in range(depth)])
            self.norm_out = RMSNorm(hidden_size)
            self.head = nn.Linear(hidden_size, num_tokens, bias=False)

        def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
            x = self.embed(input_ids)
            for block in self.blocks:
                x = block(x)
            x = self.norm_out(x)
            return self.head(x)

    return GatedDeltaNetLM()


def find_gated_deltanet_hidden_size(
    cfg: ModelConfig, target_params: int, num_tokens: int = 256,
    candidates: tuple[int, ...] = (192, 256, 320, 384, 448),
) -> tuple[float, int, int]:
    """Small grid search so model D's param count lands near target_params (typically
    model B's count), respecting num_heads*head_dim ~= 0.75*hidden_size (see FLA source
    docstring / FRICTION_LOG.md). Returns (relative_diff, best_hidden_size, param_count).
    Requires flash-linear-attention importable; raises ModelUnavailable otherwise."""
    best = None
    for hs in candidates:
        trial_cfg = replace(cfg, fla_hidden_size=hs, fla_num_heads=max(1, hs // 64), fla_head_dim=None)
        model = build_gated_deltanet(trial_cfg, num_tokens=num_tokens)
        n = count_params(model)
        diff = abs(n - target_params) / target_params
        if best is None or diff < best[0]:
            best = (diff, hs, n)
    return best


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BUILDERS = {
    "A": build_window_transformer,
    "B": build_titans_mac,
    "C": build_titans_mac_nomem,
    "D": build_gated_deltanet,
}


def build_model(letter: str, run_cfg: RunConfig, num_tokens: int = 256) -> nn.Module:
    """Builds model `letter` from run_cfg. Raises ModelUnavailable (never a bare crash
    from a missing/changed dependency) if it can't be built; callers should catch it,
    log it, and continue with the remaining models."""
    if letter not in _BUILDERS:
        raise ValueError(f"Unknown model letter {letter!r}")
    cfg = resolve_model_config(run_cfg, letter)
    return _BUILDERS[letter](cfg, num_tokens=num_tokens)
