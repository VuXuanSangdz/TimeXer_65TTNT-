"""TiDE: Time-series Dense Encoder (Das et al., 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TiDE(nn.Module):
    """Simplified TiDE — dense encoder with exogenous variables."""

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        n_exo: int = 3,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = seq_len + n_exo * seq_len
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Linear(hidden, pred_len)

    def forward(self, x_endo, x_exo=None, **kwargs):
        if x_exo is not None:
            x = torch.cat([x_endo, x_exo.reshape(x_exo.size(0), -1)], dim=-1)
        else:
            x = x_endo
        return self.decoder(self.encoder(x))
