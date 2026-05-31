from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


@torch.no_grad()
def evaluate_model(model, loader, device, ablation: str = "full") -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    for x_endo, x_exo, y in loader:
        x_endo = x_endo.to(device)
        x_exo = x_exo.to(device)
        if hasattr(model, "forward") and "ablation" in model.forward.__code__.co_varnames:
            out = model(x_endo, x_exo, ablation=ablation)
        else:
            out = model(x_endo, x_exo)
        preds.append(out.cpu().numpy())
        trues.append(y.numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    return {"MSE": mse(y_true, y_pred), "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
