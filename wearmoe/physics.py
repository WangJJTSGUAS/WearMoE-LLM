"""Trajectory window sampling and physics-consistency losses."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class WindowSampler:
    """Each batch is a set of windows; a window holds L samples of one trajectory ordered by cut."""

    def __init__(self, traj, cut, n_windows: int, length: int, random_state: int | None = None):
        self.rng = np.random.default_rng(random_state)
        self.groups = []
        for t in np.unique(traj):
            idx = np.flatnonzero(traj == t)
            if len(idx) >= 2:
                self.groups.append(idx[np.argsort(cut[idx], kind="stable")])
        self.sizes = np.array([len(g) for g in self.groups], float)
        self.n_windows, self.length = n_windows, length
        self.n = int(self.sizes.sum())

    def batch(self) -> np.ndarray:
        L = int(min(self.length, self.sizes.min()))
        out = []
        gi = self.rng.choice(len(self.groups), self.n_windows, p=self.sizes / self.sizes.sum())
        for g in gi:
            idx = self.groups[g]
            max_stride = max(1, (len(idx) - 1) // (L - 1))
            s = int(self.rng.integers(1, max_stride + 1))
            start = int(self.rng.integers(0, len(idx) - s * (L - 1)))
            out.append(idx[start: start + s * (L - 1) + 1: s])
        return np.stack(out)

    def steps_per_epoch(self) -> int:
        return max(1, self.n // (self.n_windows * int(min(self.length, self.sizes.min()))))


def quantile_loss(q_pred: torch.Tensor, y: torch.Tensor, qs=(0.05, 0.95)) -> torch.Tensor:
    loss = 0.0
    for j, q in enumerate(qs):
        e = y - q_pred[:, j]
        loss = loss + torch.mean(torch.maximum(q * e, (q - 1) * e))
    return loss


def physics_losses(
    pred: torch.Tensor,
    shape: tuple,
    rate: torch.Tensor,
    load: torch.Tensor,
    dtau: torch.Tensor,
    mech_w: torch.Tensor | None,
    log_k: torch.Tensor | None,
) -> dict:
    """Monotonicity, three-stage prior shape and mechanism-dependent Archard increment."""
    B, L = shape
    p = pred.view(B, L)
    d = p[:, 1:] - p[:, :-1]
    out = {"mono": F.relu(-d).mean()}
    r = rate.view(B, L)[:, 1:] * dtau.view(B, L)[:, 1:]
    d_pos = F.relu(d)
    dn = d_pos / (d_pos.sum(1, keepdim=True) + 1e-4)
    rn = r / (r.sum(1, keepdim=True) + 1e-6)
    out["prior"] = ((dn - rn) ** 2).sum(1).mean()
    if log_k is not None and mech_w is not None:
        k = (mech_w.view(B, L, -1)[:, 1:] * F.softplus(log_k)).sum(-1)
        inc = k * F.relu(load.view(B, L)[:, 1:]) * dtau.view(B, L)[:, 1:]
        out["archard"] = F.smooth_l1_loss(d, inc)
    return out
