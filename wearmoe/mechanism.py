"""Label-free mechanism estimator.

Three proxy indices are computed from fresh-tool-normalised signals:
  abrasive (A): rise of broadband energy in load-bearing channels;
  thermo-mechanical / adhesive (T): shift of energy towards high frequency and AE;
  chipping (K): impulsiveness (kurtosis, crest factor, sample-to-sample jumps).

Indices are z-scored with training statistics and mapped to soft mechanism weights
by a softmax. Soft stage weights come from the three-stage prior on normalised
life tau (see ``wear_priors.stage_weights_from_tau``), matching the paper.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, GROUP_VOCAB, channel_features, group_mask

MECHANISMS = ["abrasive", "thermo-mechanical", "chipping"]
STAGES = ["initial", "steady", "accelerated"]
_fi = {n: i for i, n in enumerate(FEATURE_NAMES)}


def raw_indices(F: np.ndarray, channel_groups: list) -> tuple[np.ndarray, np.ndarray]:
    """F: (N, C, F). Returns indices (N, 3) and channel-group dominance (N, 4)."""
    gm = group_mask(channel_groups)
    load = gm[GROUP_VOCAB.index("force")] + gm[GROUP_VOCAB.index("current")]
    if load.sum() == 0:
        load = gm[GROUP_VOCAB.index("vibration")]
    load = load / load.sum()
    rms = np.log1p(F[..., _fi["rms"]])
    a_idx = rms @ load
    hf = F[..., _fi["band3"]] + F[..., _fi["band4"]]
    t_idx = hf.mean(-1) + F[..., _fi["centroid"]].mean(-1)
    ae = gm[GROUP_VOCAB.index("ae")]
    if ae.sum() > 0:
        t_idx = t_idx + rms @ (ae / ae.sum())
    k_idx = (
        np.log1p(np.abs(F[..., _fi["kurt"]]))
        + np.log1p(F[..., _fi["crest"]])
        + np.log1p(F[..., _fi["absdiff"]])
    ).mean(-1)
    drift = np.abs(rms) @ gm.T / np.maximum(gm.sum(1), 1)
    dom = drift / (drift.sum(-1, keepdims=True) + 1e-6)
    return np.stack([a_idx, t_idx, k_idx], -1).astype(np.float32), dom.astype(np.float32)


class MechanismEstimator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, Xn: np.ndarray, channel_groups: list) -> "MechanismEstimator":
        F = channel_features(Xn)
        idx, _ = raw_indices(F, channel_groups)
        self.mu, self.sd = idx.mean(0), idx.std(0) + 1e-6
        return self

    def transform(self, Xn: np.ndarray, channel_groups: list) -> dict:
        """Return mechanism weights and channel-group dominance (no stage weights).

        Stage weights must be supplied by ``wear_priors.stage_weights_from_tau``
        using a fitted ``ThreeStagePrior`` on normalised life tau.
        """
        F = channel_features(Xn)
        idx, dom = raw_indices(F, channel_groups)
        z = np.clip((idx - self.mu) / self.sd, -3.0, 3.0)
        w = np.exp(z / self.temperature)
        w = w / w.sum(-1, keepdims=True)
        return {
            "features": F,
            "index_z": z.astype(np.float32),
            "mech_w": w.astype(np.float32),
            "group_dom": dom,
        }
