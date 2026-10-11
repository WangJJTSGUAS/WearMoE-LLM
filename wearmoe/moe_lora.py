"""Mixture-of-LoRA-experts wrapper for frozen nn.Linear layers with shared per-sample gates."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GateContext:
    """Holds the (B, E) gate tensor for the current forward pass; shared by all wrapped layers."""

    def __init__(self):
        self.gates: torch.Tensor | None = None


class MoELoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, ctx: GateContext, n_experts: int, rank: int, alpha: float):
        super().__init__()
        self.base, self.ctx = base, ctx
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(n_experts, rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(n_experts, base.out_features, rank))
        nn.init.kaiming_uniform_(self.A.view(-1, base.in_features), a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        g = self.ctx.gates
        if g is None:
            return y
        h = torch.einsum("bti,eri->bter", x, self.A.to(x.dtype))
        h = h * g.to(x.dtype)[:, None, :, None]
        return y + self.scale * torch.einsum("bter,eor->bto", h, self.B.to(x.dtype))


def inject(model: nn.Module, ctx: GateContext, targets, n_experts: int, rank: int, alpha: float) -> list:
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                w = MoELoRALinear(child, ctx, n_experts, rank, alpha)
                setattr(mod, child_name, w)
                wrapped.append(f"{name}.{child_name}")
    return wrapped
