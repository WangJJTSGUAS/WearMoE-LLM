"""Mechanism knowledge graph (MKG): nodes, textual descriptions, adjacency and activations."""
from __future__ import annotations

import numpy as np

NODES = [
    ("abrasive", "abrasive wear: hard particles plough the flank face, wear grows steadily with sliding distance and load"),
    ("thermo-mechanical", "thermo-mechanical wear: high cutting temperature softens the tool, diffusion and adhesion accelerate wear"),
    ("chipping", "chipping and micro-fracture: impact and cyclic loading break the cutting edge, wear jumps abruptly"),
    ("initial", "initial wear stage: rapid break-in of the sharp edge at the beginning of tool life"),
    ("steady", "steady wear stage: slow and nearly linear flank wear growth"),
    ("accelerated", "accelerated wear stage: fast wear growth before tool failure"),
    ("force", "cutting force signals reflect mechanical load and friction on the flank"),
    ("vibration", "vibration signals reflect dynamic impact, chatter and edge damage"),
    ("ae", "acoustic emission reflects micro-fracture, friction and plastic deformation"),
    ("current", "spindle current and power reflect cutting energy and heat generation"),
    ("cutting speed", "high cutting speed raises temperature at the tool-chip interface"),
    ("feed and depth", "large feed and depth of cut raise mechanical load per tooth"),
]
EDGES = [
    ("abrasive", "steady"), ("abrasive", "force"), ("abrasive", "vibration"), ("abrasive", "feed and depth"),
    ("thermo-mechanical", "accelerated"), ("thermo-mechanical", "ae"), ("thermo-mechanical", "current"),
    ("thermo-mechanical", "cutting speed"),
    ("chipping", "accelerated"), ("chipping", "initial"), ("chipping", "vibration"), ("chipping", "ae"),
    ("chipping", "feed and depth"),
    ("initial", "force"), ("steady", "force"), ("accelerated", "vibration"), ("accelerated", "current"),
]
NAMES = [n for n, _ in NODES]
DESCS = [d for _, d in NODES]


def adjacency_hat() -> np.ndarray:
    n = len(NAMES)
    A = np.eye(n)
    for a, b in EDGES:
        i, j = NAMES.index(a), NAMES.index(b)
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1)
    return (A / np.sqrt(np.outer(d, d))).astype(np.float32)


def node_activations(
    mech_w: np.ndarray,
    stage_p: np.ndarray,
    group_dom: np.ndarray,
    speed: np.ndarray,
) -> np.ndarray:
    """(N, n_nodes) activation of MKG nodes for each sample."""
    n = len(mech_w)
    act = np.zeros((n, len(NAMES)), np.float32)
    act[:, 0:3] = mech_w
    act[:, 3:6] = stage_p
    act[:, 6:10] = group_dom
    act[:, 10] = np.clip(speed, 0, 2) / 2
    act[:, 11] = 0.5
    return act
