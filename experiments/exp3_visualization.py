"""
Experiment (3): Real-world prediction visualization.

Research question:
  How do model predictions behave in practice?
  Evidence: forecast plots and multi-model comparison figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS, TRAIN, build_loaders, build_model, get_device
from src.utils.seed import set_seed
from src.utils.plotting import plot_comparison, plot_forecast
from src.utils.trainer import Trainer

MODELS_FOR_PLOT = ["TimeXer", "iTransformer", "PatchTST", "DLinear"]
N_SAMPLES = 3


@torch.no_grad()
def _predict_sample(model, x_endo, x_exo, device, ablation="full"):
    model.eval()
    x_endo = x_endo.unsqueeze(0).to(device)
    x_exo = x_exo.unsqueeze(0).to(device)
    if hasattr(model, "forward") and "ablation" in model.forward.__code__.co_varnames:
        return model(x_endo, x_exo, ablation=ablation).cpu().numpy().flatten()
    return model(x_endo, x_exo).cpu().numpy().flatten()


def run(seed: int = 42):
    set_seed(seed)
    device = get_device()
    out_dir = ROOT / "results" / "exp3_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for ds_key, cfg in DATASETS.items():
        loaders, train_ds, cfg = build_loaders(cfg)
        n_exo = len(cfg.exo_cols)
        test_loader = loaders["test"]

        # Train all models for comparison plot
        trained = {}
        for name in MODELS_FOR_PLOT:
            print(f"[{ds_key}] Training {name} for visualization...")
            model = build_model(name, n_exo)
            trainer = Trainer(model, device)
            trainer.fit(
                loaders["train"],
                loaders["val"],
                epochs=TRAIN.epochs,
                patience=TRAIN.patience,
            )
            trained[name] = model

        # Pick sample indices spread across test set
        test_ds = test_loader.dataset
        indices = np.linspace(0, len(test_ds) - 1, N_SAMPLES, dtype=int)

        for i, idx in enumerate(indices):
            x_endo, x_exo, y = test_ds[idx]
            hist = train_ds.inverse_transform_endo(x_endo.numpy())
            gt = train_ds.inverse_transform_endo(y.numpy())

            preds = {}
            for name, model in trained.items():
                pred_scaled = _predict_sample(model, x_endo, x_exo, device)
                # inverse scale single-step vector
                pred = train_ds.scaler_endo.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                preds[name] = pred

            base = f"{ds_key}_sample{i+1}"
            plot_forecast(
                hist, gt, preds["TimeXer"],
                title=f"{cfg.name} — TimeXer Forecast (sample {i+1})",
                save_path=out_dir / f"{base}_timexer.png",
            )
            plot_comparison(
                hist, gt, preds,
                title=f"{cfg.name} — Multi-Model Comparison (sample {i+1})",
                save_path=out_dir / f"{base}_comparison.png",
            )
            manifest.append({"dataset": ds_key, "sample": i + 1, "files": [f"{base}_timexer.png", f"{base}_comparison.png"]})

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "research_question": "RQ3: Forecast evaluation",
                "seed": seed,
                "samples_per_dataset": N_SAMPLES,
                "models": MODELS_FOR_PLOT,
                "note": "Figures saved under results/exp3_visualization/ (not embedded in README/Pages).",
                "cases": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved visualizations to {out_dir}")


if __name__ == "__main__":
    run()
