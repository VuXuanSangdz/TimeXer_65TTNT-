"""TimeXer: Forecasting with Exogenous Variables (NeurIPS 2024).

Reference: https://arxiv.org/abs/2402.19072
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeXerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_p = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_g = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.concat_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(
        self,
        patches: torch.Tensor,
        global_token: torch.Tensor,
        exo_tokens: torch.Tensor,
        use_cross: bool = True,
        fusion: str = "cross",
    ):
        attn_weights = None
        # Endogenous self-attention over [patches, global]
        combined = torch.cat([patches, global_token], dim=1)
        attn_out, _ = self.self_attn(combined, combined, combined)
        combined = self.norm1(combined + self.dropout(attn_out))
        new_patches = combined[:, :-1, :]
        new_global = combined[:, -1:, :]

        if use_cross and exo_tokens is not None and exo_tokens.size(1) > 0:
            if fusion == "cross":
                cross_out, attn_weights = self.cross_attn(new_global, exo_tokens, exo_tokens)
                new_global = self.norm2(new_global + self.dropout(cross_out))
            elif fusion == "add":
                exo_summary = exo_tokens.mean(dim=1, keepdim=True)
                new_global = self.norm2(new_global + exo_summary)
            elif fusion == "concat":
                exo_summary = exo_tokens.mean(dim=1, keepdim=True)
                merged = torch.cat([new_global, exo_summary], dim=-1)
                new_global = self.norm2(new_global + self.dropout(self.concat_proj(merged)))
            else:
                attn_weights = None
        else:
            attn_weights = None

        new_patches = self.norm3(new_patches + self.dropout(self.ffn_p(new_patches)))
        new_global = self.norm4(new_global + self.dropout(self.ffn_g(new_global)))
        return new_patches, new_global, attn_weights


class TimeXer(nn.Module):
    """TimeXer forecaster with endogenous + exogenous variables."""

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        n_exo: int = 3,
        patch_len: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_exo = n_exo
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.exo_embed = nn.Linear(seq_len, d_model)

        self.blocks = nn.ModuleList(
            [TimeXerBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

        out_dim = (self.n_patches + 1) * d_model
        self.head = nn.Linear(out_dim, pred_len)

    def _embed_endogenous(self, x: torch.Tensor):
        # x: [B, seq_len]
        b = x.size(0)
        patches = x.unfold(1, self.patch_len, self.patch_len)  # [B, n_patches, patch_len]
        patch_tokens = self.patch_embed(patches) + self.pos_embed
        global_tok = self.global_token.expand(b, -1, -1)
        return patch_tokens, global_tok

    def _embed_exogenous(self, exo: torch.Tensor, mode: str = "variate"):
        # exo: [B, n_exo, seq_len]
        if mode == "variate":
            return self.exo_embed(exo)  # [B, n_exo, d_model]
        # patch mode for ablation
        b, c, t = exo.shape
        patches = exo.reshape(b, c * (t // self.patch_len), self.patch_len)
        return self.patch_embed(patches)

    def forward(
        self,
        x_endo: torch.Tensor,
        x_exo: Optional[torch.Tensor] = None,
        ablation: str = "full",
    ):
        """
        Args:
            x_endo: [B, seq_len] endogenous history
            x_exo:  [B, n_exo, seq_len] exogenous history
            ablation: full | no_global | no_exo | exo_patch | exo_add | exo_concat
        """
        patch_tokens, global_tok = self._embed_endogenous(x_endo)

        use_global = ablation not in ("no_global",)
        use_exo = ablation != "no_exo" and x_exo is not None

        if not use_global:
            global_tok = torch.zeros_like(global_tok)

        if use_exo:
            exo_mode = "patch" if ablation == "exo_patch" else "variate"
            exo_tokens = self._embed_exogenous(x_exo, mode=exo_mode)
            fusion = {
                "full": "cross",
                "no_global": "cross",
                "exo_patch": "cross",
                "exo_add": "add",
                "exo_concat": "concat",
            }.get(ablation, "cross")
        else:
            exo_tokens = None
            fusion = "cross"

        attn_weights = None
        for block in self.blocks:
            if use_global:
                patch_tokens, global_tok, attn_weights = block(
                    patch_tokens, global_tok, exo_tokens,
                    use_cross=use_exo, fusion=fusion,
                )
            else:
                # Only patch self-attention without global token bridge
                combined = patch_tokens
                attn_out, _ = block.self_attn(combined, combined, combined)
                patch_tokens = block.norm1(combined + block.dropout(attn_out))
                patch_tokens = block.norm3(patch_tokens + block.dropout(block.ffn_p(patch_tokens)))

        if use_global:
            out = torch.cat([patch_tokens, global_tok], dim=1).reshape(x_endo.size(0), -1)
        else:
            out = patch_tokens.reshape(x_endo.size(0), -1)
            # Pad to match head input if needed
            pad_size = (self.n_patches + 1) * self.d_model - out.size(1)
            if pad_size > 0:
                out = F.pad(out, (0, pad_size))

        return self.head(out)


class TimeXerAblation(TimeXer):
    """Wrapper exposing ablation modes for experiment (2)."""

    ABLATIONS = [
        ("full", "Full model (P+G, V, Cross-Attn)"),
        ("no_global", "Remove global token (P only)"),
        ("exo_patch", "Replace exo variate with patch embed"),
        ("exo_add", "Add exo summary instead of cross-attn"),
        ("exo_concat", "Concat exo instead of cross-attn"),
        ("no_exo", "Remove exogenous variables"),
    ]

    def forward(self, x_endo, x_exo=None, ablation: str = "full"):
        return super().forward(x_endo, x_exo, ablation=ablation)
