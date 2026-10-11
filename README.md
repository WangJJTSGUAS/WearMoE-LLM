# WearMoE-LLM

Core implementation of **WearMoE-LLM**, a mechanism-guided multimodal large language model for physically consistent and interpretable flank wear prediction.

This folder contains only the algorithm modules described in the paper. Dataset loaders, baselines, experiment runners and figure scripts are not included; you must prepare fold dictionaries (and optional CLIP image features) separately.

## Method overview

WearMoE-LLM combines four modules:

1. **Label-free mechanism estimation** — soft weights for abrasive, thermo-mechanical and chipping wear from fresh-tool-normalised signals.
2. **Mechanism knowledge graph (MKG)** — 12 nodes and 17 edges linking mechanisms, wear stages, sensor groups and cutting conditions; activated cut by cut.
3. **Mechanism-guided Mixture-of-LoRA-Experts (M-MoE)** — three expert groups inside a frozen decoder, gated by MKG context and mechanism / stage weights.
4. **Physics-constrained training + verifiable rationale distillation** — monotonicity, three-stage wear-rate prior and Archard-type increment losses, plus a rule-based teacher that produces fixed-format rationales (stage / mechanism / sensor / level).

Training has two stages: a signal encoder warm-up (no LLM forward), then residual MoE-LoRA refinement through the frozen decoder.

Stage weights $\mathbf{s}$ follow the paper: a **three-stage prior on normalised life** $\tau$ (not clustering on signal indices).


## Requirements

- Python 3.10+
- CUDA GPU recommended (bfloat16 training on the frozen decoder)
- A local copy or Hugging Face id of **Qwen2.5-0.5B** (or a compatible causal LM)

```bash
pip install -r requirements.txt
```

## Quick start

Prepare a fold dictionary `A` with standardised training / test arrays and teacher texts, then:

```python
from wearmoe import (
    LLMConfig,
    MechanismEstimator,
    ThreeStagePrior,
    life_fraction,
    make_rationale,
    stage_weights_from_tau,
    train_wearmoe,
    wear_level,
)
from wearmoe.preprocess import baseline_normalise
from wearmoe.rationale import level_bins

# 1) Fresh-tool normalisation and mechanism weights
Xn = baseline_normalise(X, tool, cut)
est = MechanismEstimator().fit(Xn_train, channel_groups)
mech = est.transform(Xn, channel_groups)

# 2) Stage weights from the three-stage prior on tau (paper)
tau = life_fraction(traj, cut)
prior = ThreeStagePrior().fit(y_train, traj_train, cut_train)
stage_p = stage_weights_from_tau(tau, prior)
stage_id = prior.stage(tau)  # hard labels for the teacher

# 3) Teacher rationales
bins = level_bins(y_train)
lvl = wear_level(y, bins)
rationales = [
    make_rationale(int(stage_id[i]), mech["mech_w"][i], mech["group_dom"][i], int(lvl[i]))
    for i in range(len(y))
]

# 4) Train (llm_path: local directory or Hugging Face model id)
cfg = LLMConfig()  # r=8, alpha=2, beta=1 (paper defaults)
out = train_wearmoe(
    A,
    cfg=cfg,
    llm_path="Qwen/Qwen2.5-0.5B",
    device="cuda",
)

pred = out["pred"]           # flank wear (µm)
gates = out["gates"]         # expert gates
rationales = out["rationale"]  # if cfg.rationale and cfg.gen
```

`train_wearmoe` expects:

| Key | Shape / content |
|-----|-----------------|
| `A["train"\|"test"]["X"]` | `(N, C, 256)` normalised signals |
| `...["aux"]` | auxiliary features (physics / conditions) |
| `...["ys"]` | standardised VB targets |
| `...["rate", "load", "tau", "mech_w", "stage_p", "group_dom", "speed"]` | physics and mechanism tensors (`stage_p` from `stage_weights_from_tau`) |
| `...["traj", "cut"]` | trajectory id and cut index (train) |
| `A["prep"]["train"\|"test"]["texts"]` | condition text prompts |
| `A["prep"]["train"]["rationale"]` | teacher rationale strings |
| `A["y_mu"], A["y_sd"]` | target standardisation |
| optional `...["img"]` | CLIP (or other) image features |

## Configuration

Default hyperparameters in `LLMConfig` (`wearmoe/model.py`) match the paper:

| Group | Defaults |
|-------|----------|
| M-MoE | 3 groups × 4 experts, LoRA rank $r{=}8$, $\alpha{=}2$, $\beta{=}1$ |
| Stage 1 | 60 epochs, 8 windows × 8 cuts |
| Stage 2 | ≤200 steps, 4 windows × 4 cuts |
| Losses | λ_mono = 0.5, λ_prior = 0.2, λ_Archard = 0.1, λ_rationale = 0.2 |
| Decoder | frozen causal LM in bfloat16; LoRA on q/k/v/o/up/down |

Ablations and alternative routers:

```python
cfg = LLMConfig(router="token", group_bias=False, mkg_token=False)  # token-routed MoE
cfg = LLMConfig(physics=False)                                       # no physics losses
cfg = LLMConfig(lora=False)                                          # frozen decoder only
```

## Rationale format

Teacher and model outputs use a fixed, parseable template (four fields):

```text
Stage: steady. Mechanism: abrasive (0.46). Channel: force. Level: moderate.
```

Use `parse_rationale(text)` to recover stage, mechanism, channel and level, then compare with the physics reference (prior stage, mechanism estimator, group dominance, wear tertiles).

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{wearmoe-llm,
  title   = {Mechanism-guided multimodal LLM with mixture-of-experts for physically consistent and interpretable tool wear prediction},
  author  = {},
  journal = {},
  year    = {}
}
```

## Licence

Research / academic use. Update this section before a public release if you choose a specific open-source licence (e.g. MIT, Apache-2.0).
