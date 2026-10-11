"""WearMoE-LLM: mechanism-guided multimodal LLM for physically consistent tool wear prediction."""

from .mechanism import MECHANISMS, STAGES, MechanismEstimator
from .mkg import DESCS, EDGES, NAMES, NODES, adjacency_hat, node_activations
from .model import LLMConfig, WearLLM, train_wearmoe
from .rationale import LEVELS, make_rationale, parse_rationale, wear_level
from .wear_priors import ThreeStagePrior, life_fraction, stage_weights_from_tau

__all__ = [
    "LLMConfig",
    "WearLLM",
    "train_wearmoe",
    "MechanismEstimator",
    "ThreeStagePrior",
    "life_fraction",
    "stage_weights_from_tau",
    "MECHANISMS",
    "STAGES",
    "LEVELS",
    "NODES",
    "NAMES",
    "DESCS",
    "EDGES",
    "adjacency_hat",
    "node_activations",
    "make_rationale",
    "parse_rationale",
    "wear_level",
]
