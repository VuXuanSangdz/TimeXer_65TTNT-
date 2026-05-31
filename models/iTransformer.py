"""iTransformer: Inverted Transformer (Liu et al., ICLR 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn


class iTransformer(nn.Module):
    """Simplified iTransformer — variate-level tokens."""

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        n_exo: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = 1 + n_exo
        self.var_embed = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, e_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x_endo, x_exo=None, **kwargs):
        b = x_endo.size(0)
        endo_tok = self.var_embed(x_endo).unsqueeze(1)  # [B, 1, D]
        if x_exo is not None:
            exo_tok = self.var_embed(x_exo.reshape(b * x_exo.size(1), -1))
            exo_tok = exo_tok.reshape(b, x_exo.size(1), -1)
            tokens = torch.cat([endo_tok, exo_tok], dim=1)
        else:
            tokens = endo_tok
        out = self.encoder(tokens)
        return self.head(out[:, 0, :])
