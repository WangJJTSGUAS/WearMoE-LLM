"""WearMoE-LLM model and two-stage training loop.

Expected batch / fold arrays (keys of ``batch`` and of ``A["train"]`` / ``A["test"]``):

- ``X``: (N, C, 256) fresh-tool-normalised signals
- ``aux``: auxiliary features (physics descriptors and/or cutting conditions)
- ``ys``: standardised flank wear targets
- ``rate``, ``load``, ``tau``, ``mech_w``, ``stage_p``, ``group_dom``, ``speed``
  (``stage_p`` from ``wear_priors.stage_weights_from_tau``, not clustering)
- ``traj``, ``cut``: trajectory identity and cut index (training only)
- optional ``img``: CLIP image features

``A`` also supplies ``y_mu``, ``y_sd`` for denormalisation and ``prep`` with
tokenised texts / teacher rationales under ``prep["train"|"test"]``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoder import SignalEncoder
from .mkg import DESCS, adjacency_hat
from .moe_lora import GateContext, inject
from .physics import WindowSampler, physics_losses, quantile_loss
from .rationale import parse_rationale

DEFAULT_DEVICE = "cuda"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"}


@dataclass
class LLMConfig:
    n_groups: int = 3
    experts_per_group: int = 4
    rank: int = 8
    alpha: float = 2.0
    router: str = "mkg"  # mkg | token | none
    group_bias: bool = True
    mkg_token: bool = True
    physics: bool = True
    rationale: bool = True
    text: bool = True
    lora: bool = True
    image: bool = True
    gen: bool = True
    n_windows: int = 4
    win_len: int = 4
    epochs: int = 12
    max_steps: int = 200
    warm_epochs: int = 60
    enc_lr_scale: float = 0.2
    lr_lora: float = 3e-4
    lr_other: float = 1e-3
    lam_lm: float = 0.2
    lam_mono: float = 0.5
    lam_prior: float = 0.2
    lam_arch: float = 0.1
    beta: float = 1.0

    @property
    def n_experts(self) -> int:
        return self.n_groups * self.experts_per_group if self.router != "none" else 1


class WearLLM(nn.Module):
    def __init__(
        self,
        cfg: LLMConfig,
        c_in: int,
        a_in: int,
        llm_path: str | Path,
        img_dim: int = 0,
        device: str = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(llm_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16
        ).to(device)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        H = self.lm.config.hidden_size
        self.ctx = GateContext()
        if cfg.lora:
            inject(self.lm, self.ctx, TARGETS, cfg.n_experts, cfg.rank, cfg.alpha)
            self.lm.to(device)
        self.sig = SignalEncoder(c_in, H)
        self.aux = nn.Linear(a_in, H)
        self.img = nn.Linear(img_dim, H) if img_dim else None
        emb = self.lm.get_input_embeddings()
        with torch.no_grad():
            ids = [self.tok(d)["input_ids"] for d in DESCS]
            E = torch.stack([emb(torch.tensor(i, device=device)).float().mean(0) for i in ids])
            A = torch.tensor(adjacency_hat(), device=device)
            self.register_buffer("node_emb", A @ A @ E)
        self.mkg_proj = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, H))
        r_in = (64 + 6) if cfg.router == "mkg" else 64
        self.r_proj = nn.Linear(H, 64)
        self.router = nn.Linear(r_in, cfg.n_experts)
        self.aux_skip = nn.Sequential(nn.Linear(a_in, 64), nn.GELU())
        self.warm_head = nn.Sequential(nn.Linear(self.sig.dim + 64 + img_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.head = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_k = nn.Parameter(torch.zeros(3))
        self.suffix = torch.tensor(self.tok(" Flank wear:")["input_ids"], device=device)
        self.to(device)
        for m in (self.sig, self.aux, self.img, self.mkg_proj, self.r_proj, self.router, self.head,
                  self.aux_skip, self.warm_head):
            if m is not None:
                m.float()

    def lora_params(self):
        return [p for n, p in self.lm.named_parameters() if p.requires_grad]

    def other_params(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("lm.")]

    def _gates(self, sig_tok, mkg_ctx, mech_w, stage_p):
        cfg = self.cfg
        if cfg.router == "none":
            return torch.ones(sig_tok.shape[0], 1, device=self.device)
        if cfg.router == "mkg":
            r = torch.cat([self.r_proj(mkg_ctx), mech_w, stage_p], -1)
        else:
            r = self.r_proj(sig_tok.mean(1))
        logits = self.router(r)
        if cfg.group_bias:
            logits = logits + cfg.beta * torch.log(mech_w + 1e-4).repeat_interleave(cfg.experts_per_group, -1)
        return torch.softmax(logits, -1)

    def warm_out(self, b, pooled=None):
        if pooled is None:
            pooled = self.sig(b["X"])[1]
        f = [pooled, self.aux_skip(b["aux"])]
        if self.img is not None and "img" in b:
            f.append(b["img"])
        return self.warm_head(torch.cat(f, -1))

    def _node_act(self, b):
        act = torch.zeros(b["mech_w"].shape[0], 12, device=self.device)
        act[:, 0:3] = b["mech_w"]
        act[:, 3:6] = b["stage_p"]
        act[:, 6:10] = b["group_dom"]
        act[:, 10] = torch.clamp(b["speed"], 0, 2) / 2
        act[:, 11] = 0.5
        return act

    def build(self, b: dict, text_ids: list, rat_ids: list | None):
        cfg = self.cfg
        emb = self.lm.get_input_embeddings()
        Bn = len(text_ids)
        Lt = max(len(t) for t in text_ids)
        tid = torch.full((Bn, Lt), self.tok.pad_token_id, device=self.device)
        tmask = torch.zeros((Bn, Lt), device=self.device, dtype=torch.long)
        for i, t in enumerate(text_ids):
            tid[i, Lt - len(t):] = torch.tensor(t, device=self.device)
            tmask[i, Lt - len(t):] = 1
        sig_tok, pooled = self.sig(b["X"])
        self._warm = self.warm_out(b, pooled)
        mkg_ctx = self._node_act(b) @ self.node_emb
        parts = [emb(tid).float(), sig_tok, self.aux(b["aux"])[:, None]]
        if cfg.mkg_token:
            parts.append(self.mkg_proj(mkg_ctx)[:, None])
        if self.img is not None and "img" in b:
            parts.append(self.img(b["img"])[:, None])
        parts.append(emb(self.suffix).float()[None].expand(Bn, -1, -1))
        prefix = torch.cat(parts, 1)
        P = prefix.shape[1]
        mask = torch.cat([tmask, torch.ones((Bn, P - Lt), device=self.device, dtype=torch.long)], 1)
        gates = self._gates(sig_tok, mkg_ctx, b["mech_w"], b["stage_p"])
        labels = None
        if rat_ids is not None:
            R = max(len(r) for r in rat_ids)
            rid = torch.full((Bn, R), self.tok.pad_token_id, device=self.device)
            labels = torch.full((Bn, R), -100, device=self.device)
            rmask = torch.zeros((Bn, R), device=self.device, dtype=torch.long)
            for i, r in enumerate(rat_ids):
                rid[i, : len(r)] = torch.tensor(r, device=self.device)
                labels[i, : len(r)] = torch.tensor(r, device=self.device)
                rmask[i, : len(r)] = 1
            prefix = torch.cat([prefix, emb(rid).float()], 1)
            mask = torch.cat([mask, rmask], 1)
        return prefix, mask, P, gates, labels

    def forward(self, b, text_ids, rat_ids=None):
        embeds, mask, P, gates, labels = self.build(b, text_ids, rat_ids)
        self.ctx.gates = gates
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.lm.model(
                inputs_embeds=embeds.to(torch.bfloat16),
                attention_mask=mask,
                position_ids=pos,
            ).last_hidden_state
            out = self._warm + self.head(h[:, P - 1].float())
            lm_loss = None
            if labels is not None:
                R = labels.shape[1]
                logits = self.lm.lm_head(h[:, P - 1: P - 1 + R]).float()
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
        self.ctx.gates = None
        return out, lm_loss, gates

    @torch.no_grad()
    def generate(self, b, text_ids, max_new=28):
        embeds, mask, P, gates, _ = self.build(b, text_ids, None)
        self.ctx.gates = gates
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        emb = self.lm.get_input_embeddings()
        Bn = embeds.shape[0]
        toks = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.lm(
                inputs_embeds=embeds.to(torch.bfloat16),
                attention_mask=mask,
                position_ids=pos,
                use_cache=True,
            )
            past = o.past_key_values
            nxt = o.logits[:, -1].argmax(-1)
            last = pos[:, -1]
            for _ in range(max_new):
                toks.append(nxt)
                mask = torch.cat([mask, torch.ones((Bn, 1), device=self.device, dtype=torch.long)], 1)
                last = last + 1
                o = self.lm(
                    inputs_embeds=emb(nxt)[:, None],
                    attention_mask=mask,
                    position_ids=last[:, None],
                    past_key_values=past,
                    use_cache=True,
                )
                past = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1)
        self.ctx.gates = None
        T = torch.stack(toks, 1).cpu().numpy()
        texts = []
        for row in T:
            stop = np.flatnonzero(row == self.tok.eos_token_id)
            row = row[: stop[0]] if len(stop) else row
            texts.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return texts


def train_wearmoe(
    A: dict,
    cfg: LLMConfig | None = None,
    llm_path: str | Path = "Qwen/Qwen2.5-0.5B",
    device: str = DEFAULT_DEVICE,
    random_state: int | None = None,
) -> dict:
    """Two-stage training: stage-1 signal regressor, then mechanism-routed MoE-LoRA refinement.

    Parameters
    ----------
    A :
        Fold dictionary with ``train``, ``test``, ``prep``, ``y_mu``, ``y_sd``.
        Optional image features may be provided as ``A["train"]["img"]`` / ``A["test"]["img"]``.
    cfg :
        Model and optimisation configuration (defaults to :class:`LLMConfig`).
    llm_path :
        Local path or Hugging Face id of the frozen decoder.
    device :
        Torch device string.
    random_state :
        Optional RNG control for sampling and initialisation.
    """
    cfg = cfg or LLMConfig()
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    tr, te = A["train"], A["test"]
    prep = A["prep"]
    img_tr = tr.get("img")
    img_te = te.get("img")
    if cfg.image and img_tr is None:
        cfg = replace(cfg, image=False)
    model = WearLLM(
        cfg,
        tr["X"].shape[1],
        tr["aux"].shape[1],
        llm_path=llm_path,
        img_dim=0 if img_tr is None else img_tr.shape[1],
        device=device,
    )
    tok = model.tok
    generic = "Signals:"
    txt_tr = [tok(t if cfg.text else generic)["input_ids"] for t in prep["train"]["texts"]]
    txt_te = [tok(t if cfg.text else generic)["input_ids"] for t in prep["test"]["texts"]]
    rat_tr = [tok(" " + r)["input_ids"] + [tok.eos_token_id] for r in prep["train"]["rationale"]]

    keys = ("X", "aux", "ys", "rate", "load", "tau", "mech_w", "stage_p", "group_dom", "speed")
    T = {k: torch.tensor(tr[k], device=device) for k in keys}
    E = {k: torch.tensor(te[k], device=device) for k in keys if k in te}
    if img_tr is not None:
        T["img"] = torch.tensor(img_tr, device=device)
        E["img"] = torch.tensor(img_te, device=device)

    def objective(out, b, w, lm_loss=None):
        y = b["ys"]
        loss = F.smooth_l1_loss(out[:, 0], y) + quantile_loss(out[:, 1:], y)
        if lm_loss is not None:
            loss = loss + cfg.lam_lm * lm_loss
        if cfg.physics:
            tau = b["tau"].view(w.shape)
            dtau = torch.cat([torch.zeros_like(tau[:, :1]), tau[:, 1:] - tau[:, :-1]], 1).reshape(-1)
            pl = physics_losses(out[:, 0], w.shape, b["rate"], b["load"], dtau, b["mech_w"], model.log_k)
            loss = loss + cfg.lam_mono * pl["mono"] + cfg.lam_prior * pl["prior"] + cfg.lam_arch * pl["archard"]
        return loss

    # Stage 1: align the signal encoder (no LLM forward).
    warm_params = [p for m in (model.sig, model.aux_skip, model.warm_head) for p in m.parameters()] + [model.log_k]
    opt = torch.optim.AdamW(warm_params, 1e-3, weight_decay=1e-4)
    smp = WindowSampler(tr["traj"], tr["cut"], 8, 8, random_state=random_state)
    steps1 = cfg.warm_epochs * smp.steps_per_epoch()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps1)
    model.train()
    for _ in range(steps1):
        w = smp.batch()
        idx = torch.tensor(w.reshape(-1), device=device)
        b = {k: v[idx] for k, v in T.items()}
        loss = objective(model.warm_out(b), b, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(warm_params, 1.0)
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        n_te = len(te["ys"])
        warm = torch.cat(
            [model.warm_out({k: v[i:i + 256] for k, v in E.items()}) for i in range(0, n_te, 256)]
        ).cpu().numpy() * A["y_sd"] + A["y_mu"]

    # Stage 2: mechanism-routed MoE-LoRA refinement (residual on the stage-1 head).
    sig_ids = {id(p) for p in model.sig.parameters()}
    others = [p for p in model.other_params() if id(p) not in sig_ids]
    groups = [
        {"params": others, "lr": cfg.lr_other},
        {"params": list(model.sig.parameters()), "lr": cfg.lr_other * cfg.enc_lr_scale},
    ]
    if cfg.lora:
        groups.append({"params": model.lora_params(), "lr": cfg.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    smp = WindowSampler(tr["traj"], tr["cut"], cfg.n_windows, cfg.win_len, random_state=random_state)
    steps = min(cfg.max_steps, cfg.epochs * smp.steps_per_epoch())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    model.train()
    model.lm.eval()
    for m in model.sig.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
    for _ in range(steps):
        w = smp.batch()
        flat = w.reshape(-1)
        idx = torch.tensor(flat, device=device)
        b = {k: v[idx] for k, v in T.items()}
        out, lm_loss, _ = model(
            b,
            [txt_tr[i] for i in flat],
            [rat_tr[i] for i in flat] if cfg.rationale else None,
        )
        loss = objective(out, b, w, lm_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()

    model.eval()
    preds, gates, texts = [], [], []
    n = len(te["ys"])
    with torch.no_grad():
        for i in range(0, n, 32):
            sl = slice(i, min(n, i + 32))
            b = {k: v[sl] for k, v in E.items()}
            out, _, g = model(b, txt_te[sl])
            preds.append(out.float().cpu().numpy())
            gates.append(g.float().cpu().numpy())
            if cfg.rationale and cfg.gen:
                texts += model.generate(b, txt_te[sl])
    out = np.concatenate(preds) * A["y_sd"] + A["y_mu"]
    q = np.sort(out[:, 1:], 1)
    qw = np.sort(warm[:, 1:], 1)
    res = {
        "pred": out[:, 0],
        "lo": q[:, 0],
        "hi": q[:, 1],
        "gates": np.concatenate(gates),
        "pred_stage1": warm[:, 0],
        "lo_stage1": qw[:, 0],
        "hi_stage1": qw[:, 1],
        "mech_w": te["mech_w"],
        "steps": steps,
        "model": model,
    }
    if texts:
        parsed = [parse_rationale(t) for t in texts]
        res["rationale"] = np.array(texts)
        res["r_stage"] = np.array([p["stage_id"] for p in parsed])
        res["r_mech"] = np.array([p["mech_id"] for p in parsed])
        res["r_channel"] = np.array([p["channel_id"] for p in parsed])
        res["r_level"] = np.array([p["level_id"] for p in parsed])
    return res
