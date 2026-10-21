"""Bootstrap confidence intervals at the EXAMPLE level (mission requirement: "bootstrap
at the example level, since the five digits within each example are correlated")."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(binary_outcomes: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """binary_outcomes: 1D array of 0/1 (or bool), one value PER EXAMPLE (e.g. exact-match
    per example, already collapsed across its 5 digits -- never resample individual
    digits independently)."""
    x = np.asarray(binary_outcomes, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = x[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


_Z_BY_CI = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def wilson_ci(successes: int, n: int, ci: float = 0.95) -> dict:
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    z = _Z_BY_CI.get(ci, 1.96)  # standard-normal quantile; avoids a scipy dependency
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return {"mean": p, "lo": max(0.0, center - half), "hi": min(1.0, center + half), "n": n}
