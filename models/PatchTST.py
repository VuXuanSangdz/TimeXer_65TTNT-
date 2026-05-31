"""PatchTST: Patch-level Transformer (Nie et al., 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Simplified PatchTST — patch-level, channel independent."""

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        n_exo: int = 3,
        patch_len: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, e_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.n_exo = n_exo
        self.exo_proj = nn.Linear(n_exo * seq_len, d_model) if n_exo > 0 else None

    def forward(self, x_endo, x_exo=None, **kwargs):
        b = x_endo.size(0)
        patches = x_endo.unfold(1, self.patch_len, self.patch_len)
        tokens = self.patch_embed(patches) + self.pos
        if x_exo is not None and self.exo_proj is not None:
            exo_feat = self.exo_proj(x_exo.reshape(b, -1)).unsqueeze(1)
            tokens = tokens + exo_feat
        out = self.encoder(tokens)
        return self.head(out.reshape(b, -1))
