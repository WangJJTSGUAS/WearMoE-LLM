"""Three-stage wear-rate prior and Archard-type load proxy."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .features import FEATURE_NAMES, GROUP_VOCAB, group_mask


def trajectory_keys(tool: np.ndarray, extra: dict | None = None) -> np.ndarray:
    extra = extra or {}
    if "blade" in extra:
        return np.array([f"{t}_B{b}" for t, b in zip(tool, extra["blade"])])
    return np.asarray(tool)


def _cum(tau, a, t1, b, c, t2):
    return a * t1 * (1 - np.exp(-tau / t1)) + b * tau + c * t2 * (np.exp((tau - 1) / t2) - np.exp(-1 / t2))


def _rate(tau, a, t1, b, c, t2):
    return a * np.exp(-tau / t1) + b + c * np.exp((tau - 1) / t2)


class ThreeStagePrior:
    """Normalised cumulative wear V(tau) on normalised life tau in [0, 1]."""

    def fit(self, y: np.ndarray, traj: np.ndarray, cut: np.ndarray) -> "ThreeStagePrior":
        taus, vs = [], []
        for k in np.unique(traj):
            m = traj == k
            yy = y[m][np.argsort(cut[m])]
            if len(yy) < 4 or yy.max() - yy.min() < 1e-6:
                continue
            yy = np.maximum.accumulate(yy)
            taus.append(np.linspace(0, 1, len(yy)))
            vs.append((yy - yy[0]) / (yy[-1] - yy[0]))
        tau, v = np.concatenate(taus), np.concatenate(vs)
        p0 = [1.0, 0.1, 0.6, 1.0, 0.1]
        bounds = ([0, 0.01, 0, 0, 0.01], [20, 1, 5, 20, 1])
        try:
            self.params, _ = curve_fit(_cum, tau, v, p0=p0, bounds=bounds, maxfev=20000)
        except RuntimeError:
            self.params = np.array(p0)
        grid = np.linspace(0, 1, 501)
        r = _rate(grid, *self.params)
        i_min = int(np.argmin(r))
        thr = 1.25 * r[i_min]
        init = np.flatnonzero((grid < grid[i_min]) & (r > thr))
        acc = np.flatnonzero((grid > grid[i_min]) & (r > thr))
        self.tau_a = float(grid[init[-1]]) if len(init) else 0.0
        self.tau_b = float(grid[acc[0]]) if len(acc) else 1.0
        return self

    def cum(self, tau):
        return _cum(np.asarray(tau, dtype=float), *self.params)

    def rate(self, tau):
        return _rate(np.asarray(tau, dtype=float), *self.params)

    def stage(self, tau) -> np.ndarray:
        tau = np.asarray(tau)
        return np.where(tau < self.tau_a, 0, np.where(tau > self.tau_b, 2, 1))


def stage_weights_from_tau(
    tau: np.ndarray,
    prior: "ThreeStagePrior",
    temperature: float = 0.05,
) -> np.ndarray:
    """Soft stage weights s from the three-stage prior on normalised life tau.

    Matches the paper: initial / steady / accelerated membership from the
    prior breakpoints ``tau_a`` and ``tau_b`` (distance-softened softmax).
    If the fitted prior collapses (no break-in / acceleration on a near-linear
    tool life), mild default breakpoints are used only for the soft weights;
    hard labels still come from ``prior.stage(tau)``.
    """
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    ta, tb = float(prior.tau_a), float(prior.tau_b)
    # Degenerate fit: no distinct break-in / acceleration intervals.
    if ta <= 1e-3 or tb >= 1.0 - 1e-3 or (tb - ta) < 0.1:
        ta, tb = 0.2, 0.8
    d0 = np.where(tau <= ta, 0.0, tau - ta)
    d1 = np.where(tau < ta, ta - tau, np.where(tau > tb, tau - tb, 0.0))
    d2 = np.where(tau >= tb, 0.0, tb - tau)
    d = np.stack([d0, d1, d2], axis=-1)
    z = -d / max(float(temperature), 1e-6)
    z -= z.max(axis=-1, keepdims=True)
    w = np.exp(z)
    return (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)


def life_fraction(traj: np.ndarray, cut: np.ndarray) -> np.ndarray:
    tau = np.zeros(len(cut), dtype=np.float32)
    for k in np.unique(traj):
        m = np.flatnonzero(traj == k)
        r = np.argsort(np.argsort(cut[m]))
        tau[m] = r / max(1, len(m) - 1)
    return tau


def load_proxy(F: np.ndarray, channel_groups: list, speed: np.ndarray) -> np.ndarray:
    """Archard-type load x sliding proxy: load-channel RMS times cutting speed."""
    gm = group_mask(channel_groups)
    load = gm[GROUP_VOCAB.index("force")] + gm[GROUP_VOCAB.index("current")]
    if load.sum() == 0:
        load = gm[GROUP_VOCAB.index("vibration")]
    rms = F[..., FEATURE_NAMES.index("rms")]
    return (rms @ (load / load.sum())) * speed


def physics_features(
    F: np.ndarray,
    channel_groups: list,
    traj: np.ndarray,
    cut: np.ndarray,
    speed: np.ndarray,
    elapsed: np.ndarray,
) -> np.ndarray:
    """Causal physics descriptors: load, cumulative load-time, energy-like ratio, elapsed time."""
    load = load_proxy(F, channel_groups, speed)
    cum = np.zeros_like(load)
    for k in np.unique(traj):
        m = np.flatnonzero(traj == k)
        o = m[np.argsort(cut[m])]
        cum[o] = np.cumsum(load[o])
    energy = load / (speed + 1e-6)
    return np.stack([load, cum, energy, elapsed], -1).astype(np.float32)
