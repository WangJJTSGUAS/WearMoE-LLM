"""Fresh-tool normalisation of multi-channel cutting signals."""
from __future__ import annotations

import numpy as np

BASELINE_FRAC = 0.1
SIGNAL_LEN = 256


def resample(sig: np.ndarray, length: int = SIGNAL_LEN) -> np.ndarray:
    """Block-smoothed resample of (C, T) to (C, length)."""
    sig = np.asarray(sig, dtype=np.float64)
    c, t = sig.shape
    if t == length:
        return sig
    xp = np.linspace(0.0, 1.0, t)
    x = np.linspace(0.0, 1.0, length)
    if t > length:
        k = max(1, t // length)
        kernel = np.ones(k) / k
        sig = np.stack([np.convolve(s, kernel, mode="same") for s in sig])
    return np.stack([np.interp(x, xp, s) for s in sig])


def baseline_normalise(
    X: np.ndarray,
    tool: np.ndarray,
    cut: np.ndarray,
    baseline_frac: float = BASELINE_FRAC,
) -> np.ndarray:
    """Scale each channel by the per-tool std of the first baseline_frac cuts."""
    Xn = np.empty_like(X)
    for t in np.unique(tool):
        m = tool == t
        order = np.argsort(cut[m])
        idx = np.flatnonzero(m)[order]
        nb = max(2, int(np.ceil(baseline_frac * len(idx))))
        base = X[idx[:nb]]
        mu = base.mean(axis=(0, 2), keepdims=True)
        sd_all = X[idx].std(axis=(0, 2), keepdims=True)
        sd = np.maximum(base.std(axis=(0, 2), keepdims=True), 0.1 * sd_all) + 1e-6
        Xn[idx] = np.clip((X[idx] - mu) / sd, -50.0, 50.0)
    return Xn.astype(np.float32)
