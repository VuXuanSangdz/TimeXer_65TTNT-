"""Shared experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.loaders import ExogenousForecastDataset, load_csv_dataset, split_by_time
from models.baselines import BASELINE_MODELS
from models.TimeXer import TimeXer


@dataclass
class DatasetConfig:
    name: str
    csv_path: Path
    endo_col: str
    exo_cols: list[str]


DATASETS = {
    "private": DatasetConfig(
        name="Face Head-Pose (Private)",
        csv_path=Path("data/private/face_headpose_private.csv"),
        endo_col="head_yaw",
        exo_cols=["ambient_light", "eye_aspect_ratio", "mouth_open_ratio"],
    ),
    "public": DatasetConfig(
        name="Weather CO2 (Public)",
        csv_path=Path("data/public/weather_public.csv"),
        endo_col="co2_concentration",
        exo_cols=["temperature", "humidity", "pressure", "wind_speed"],
    ),
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(cfg: DatasetConfig, seq_len=96, pred_len=24, batch_size=64):
    endo, exo, _ = load_csv_dataset(cfg.csv_path, cfg.endo_col, cfg.exo_cols)
    (tr_e, tr_x), (va_e, va_x), (te_e, te_x) = split_by_time(endo, exo)

    train_ds = ExogenousForecastDataset(tr_e, tr_x, seq_len, pred_len, fit_scaler=True)
    val_ds = ExogenousForecastDataset(
        va_e, va_x, seq_len, pred_len, train_ds.scaler_endo, train_ds.scaler_exo
    )
    test_ds = ExogenousForecastDataset(
        te_e, te_x, seq_len, pred_len, train_ds.scaler_endo, train_ds.scaler_exo
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
    return loaders, train_ds, cfg


def build_model(name: str, n_exo: int, seq_len=96, pred_len=24):
    cls = BASELINE_MODELS[name]
    if name == "TimeXer":
        return cls(seq_len=seq_len, pred_len=pred_len, n_exo=n_exo, patch_len=8, d_model=128, e_layers=2)
    if name in ("PatchTST",):
        return cls(seq_len=seq_len, pred_len=pred_len, n_exo=n_exo, patch_len=8, d_model=128, e_layers=2)
    if name in ("iTransformer",):
        return cls(seq_len=seq_len, pred_len=pred_len, n_exo=n_exo, d_model=128, e_layers=2)
    return cls(seq_len=seq_len, pred_len=pred_len, n_exo=n_exo)
