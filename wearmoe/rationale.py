"""Rule-based physics teacher: structured, checkable rationales and their parser."""
from __future__ import annotations

import re

import numpy as np

from .features import GROUP_VOCAB
from .mechanism import MECHANISMS, STAGES

LEVELS = ["low", "moderate", "high"]


def level_bins(y_train: np.ndarray) -> np.ndarray:
    return np.quantile(y_train, [1 / 3, 2 / 3])


def wear_level(y: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.digitize(y, bins)


def make_rationale(stage: int, mech_w: np.ndarray, dom: np.ndarray, level: int) -> str:
    m = int(np.argmax(mech_w))
    g = int(np.argmax(dom))
    return (
        f" Stage: {STAGES[stage]}. Mechanism: {MECHANISMS[m]} ({mech_w[m]:.2f})."
        f" Channel: {GROUP_VOCAB[g]}. Level: {LEVELS[level]}."
    )


_pat = {
    "stage": re.compile(r"Stage:\s*(initial|steady|accelerated)"),
    "mech": re.compile(r"Mechanism:\s*(abrasive|thermo-mechanical|chipping)"),
    "channel": re.compile(r"Channel:\s*(force|vibration|ae|current)"),
    "level": re.compile(r"Level:\s*(low|moderate|high)"),
}


def parse_rationale(text: str) -> dict:
    out = {}
    for k, p in _pat.items():
        hit = p.search(text)
        out[k] = hit.group(1) if hit else None
    out["stage_id"] = STAGES.index(out["stage"]) if out["stage"] else -1
    out["mech_id"] = MECHANISMS.index(out["mech"]) if out["mech"] else -1
    out["channel_id"] = GROUP_VOCAB.index(out["channel"]) if out["channel"] else -1
    out["level_id"] = LEVELS.index(out["level"]) if out["level"] else -1
    return out
