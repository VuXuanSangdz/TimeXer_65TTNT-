"""Data loading for exogenous forecasting paradigm."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def load_csv_dataset(
    csv_path: str | Path,
    endo_col: str,
    exo_cols: list[str],
    timestamp_col: Optional[str] = "timestamp",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    endo = df[endo_col].values.astype(np.float32)
    exo = df[exo_cols].values.astype(np.float32)
    return endo, exo, df


def split_by_time(
    endo: np.ndarray,
    exo: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    n = len(endo)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = [
        (endo[:train_end], exo[:train_end]),
        (endo[train_end:val_end], exo[train_end:val_end]),
        (endo[val_end:], exo[val_end:]),
    ]
    return tuple(splits)


class ExogenousForecastDataset(Dataset):
    """Sliding window dataset for TimeXer paradigm."""

    def __init__(
        self,
        endo: np.ndarray,
        exo: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        scaler_endo: Optional[StandardScaler] = None,
        scaler_exo: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window = seq_len + pred_len

        endo = endo.reshape(-1, 1)
        if fit_scaler:
            self.scaler_endo = StandardScaler()
            self.scaler_exo = StandardScaler()
            self.endo = self.scaler_endo.fit_transform(endo).flatten()
            self.exo = self.scaler_exo.fit_transform(exo)
        else:
            self.scaler_endo = scaler_endo
            self.scaler_exo = scaler_exo
            self.endo = scaler_endo.transform(endo).flatten()
            self.exo = scaler_exo.transform(exo)

        self.n_samples = max(0, len(self.endo) - self.window + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = idx
        e = idx + self.seq_len
        p = e + self.pred_len
        x_endo = self.endo[s:e]
        y_endo = self.endo[e:p]
        x_exo = self.exo[s:e].T  # [n_exo, seq_len]
        return (
            torch.tensor(x_endo, dtype=torch.float32),
            torch.tensor(x_exo, dtype=torch.float32),
            torch.tensor(y_endo, dtype=torch.float32),
        )

    def inverse_transform_endo(self, values: np.ndarray) -> np.ndarray:
        return self.scaler_endo.inverse_transform(values.reshape(-1, 1)).flatten()
