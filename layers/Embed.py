"""Embedding layers for TimeXer (patch-level and variate-level)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Endogenous patch embedding with positional encoding."""

    def __init__(self, patch_len: int, n_patches: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, self.patch_len, self.patch_len)
        return self.proj(patches) + self.pos


class VariateEmbedding(nn.Module):
    """Exogenous variate-level embedding."""

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalToken(nn.Module):
    """Learnable global bridge token for endogenous series."""

    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.token.expand(batch_size, -1, -1)
