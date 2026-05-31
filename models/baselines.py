"""Baseline forecasters for comparison experiment (1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DLinear(nn.Module):
    """Decomposition Linear baseline (Zeng et al., AAAI 2023)."""

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


class PatchTST(nn.Module):
    """Simplified PatchTST (Nie et al., 2023) — patch-level, channel independent."""

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


class iTransformer(nn.Module):
    """Simplified iTransformer (Liu et al., ICLR 2024) — variate-level tokens."""

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


class TiDE(nn.Module):
    """Simplified TiDE (Das et al., 2023) — dense encoder with exogenous."""

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


from .TimeXer import TimeXer  # noqa: E402

BASELINE_MODELS = {
    "TimeXer": TimeXer,
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "TiDE": TiDE,
}
