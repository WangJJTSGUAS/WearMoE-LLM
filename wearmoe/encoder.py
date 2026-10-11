"""Signal encoder: Conv stem + BiGRU that maps a cut window to LLM tokens."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    def __init__(self, c_in: int, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, d, 5, 2, 2), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Conv1d(d, d, 3, 2, 1), nn.BatchNorm1d(d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SignalEncoder(nn.Module):
    """(B, C, 256) -> 16 signal tokens in LLM space plus a pooled feature."""

    def __init__(self, c_in: int, hidden: int, d: int = 96):
        super().__init__()
        self.stem = ConvStem(c_in, d)
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden))
        self.dim = 2 * d

    def forward(self, x):
        z, _ = self.rnn(self.stem(x).transpose(1, 2))
        tok = F.avg_pool1d(z.transpose(1, 2), 2).transpose(1, 2)
        return self.out(tok), z.mean(1)
