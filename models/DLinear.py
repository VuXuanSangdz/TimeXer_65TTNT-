"""DLinear: Decomposition Linear baseline (Zeng et al., AAAI 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24, n_exo: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        in_dim = seq_len + n_exo * seq_len
        self.linear = nn.Linear(in_dim, pred_len)

    def forward(self, x_endo, x_exo=None, **kwargs):
        if x_exo is not None:
            flat_exo = x_exo.reshape(x_exo.size(0), -1)
            x = torch.cat([x_endo, flat_exo], dim=-1)
        else:
            x = x_endo
        return self.linear(x)
