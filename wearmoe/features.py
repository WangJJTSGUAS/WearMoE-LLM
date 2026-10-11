"""Per-cut signal features used by the label-free mechanism estimator."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

GROUP_VOCAB = ["force", "vibration", "ae", "current"]
FEATURE_NAMES = [
    "rms", "std", "ptp", "kurt", "skew", "crest", "absdiff", "centroid",
    "band1", "band2", "band3", "band4", "spec_entropy",
]


def channel_features(X: np.ndarray) -> np.ndarray:
    """X: (N, C, L) -> (N, C, F) with F = len(FEATURE_NAMES)."""
    rms = np.sqrt((X ** 2).mean(-1))
    std = X.std(-1)
    ptp = X.max(-1) - X.min(-1)
    kurt = np.nan_to_num(kurtosis(X, axis=-1))
    skw = np.nan_to_num(skew(X, axis=-1))
    crest = np.abs(X).max(-1) / (rms + 1e-6)
    absdiff = np.abs(np.diff(X, axis=-1)).mean(-1)
    spec = np.abs(np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)) ** 2
    spec = spec[..., 1:]
    total = spec.sum(-1) + 1e-12
    freqs = np.linspace(0, 1, spec.shape[-1])
    centroid = (spec * freqs).sum(-1) / total
    edges = np.linspace(0, spec.shape[-1], 5).astype(int)
    bands = [spec[..., edges[i]:edges[i + 1]].sum(-1) / total for i in range(4)]
    p = spec / total[..., None]
    ent = -(p * np.log(p + 1e-12)).sum(-1) / np.log(spec.shape[-1])
    feats = np.stack([rms, std, ptp, kurt, skw, crest, absdiff, centroid, *bands, ent], axis=-1)
    return np.nan_to_num(feats).astype(np.float32)


def group_mask(channel_groups: list) -> np.ndarray:
    """(G, C) one-hot of channel membership over GROUP_VOCAB."""
    m = np.zeros((len(GROUP_VOCAB), len(channel_groups)), dtype=np.float32)
    for j, g in enumerate(channel_groups):
        m[GROUP_VOCAB.index(g), j] = 1.0
    return m
