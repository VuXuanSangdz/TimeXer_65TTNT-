"""
Experiment (1): Compare TimeXer with similar forecasting models.

Research question:
  How does TimeXer compare to models with similar ideas?
  Evidence: qualitative + quantitative result tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS, build_loaders, build_model, get_device
from src.utils.metrics import evaluate_model
from src.utils.trainer import Trainer

MODELS = ["TimeXer", "iTransformer", "PatchTST", "TiDE", "DLinear"]
EPOCHS = 12


QUALITATIVE = [
    {
        "Criterion": "Exogenous variate-level embedding",
        "TimeXer": "Yes (V token)",
        "iTransformer": "No (all variates equal)",
        "PatchTST": "No (concat/add only)",
        "TiDE": "Concat at input",
        "DLinear": "Concat at input",
    },
    {
        "Criterion": "Patch-level temporal modeling",
        "TimeXer": "Yes",
        "iTransformer": "No (linear projection)",
        "PatchTST": "Yes",
        "TiDE": "No",
        "DLinear": "No",
    },
    {
        "Criterion": "Global bridge token (G)",
        "TimeXer": "Yes",
        "iTransformer": "No",
        "PatchTST": "No",
        "TiDE": "No",
        "DLinear": "No",
    },
    {
        "Criterion": "Cross-attention exo→endo",
        "TimeXer": "Yes",
        "iTransformer": "Self-attn only",
        "PatchTST": "No",
        "TiDE": "No",
        "DLinear": "No",
    },
    {
        "Criterion": "Handles irregular exo length",
        "TimeXer": "Designed for it",
        "iTransformer": "Limited",
        "PatchTST": "Limited",
        "TiDE": "Limited",
        "DLinear": "Limited",
    },
]


def run():
    device = get_device()
    results_dir = ROOT / "results" / "exp1_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    quant_rows = []
    for ds_key, cfg in DATASETS.items():
        loaders, _, cfg = build_loaders(cfg)
        n_exo = len(cfg.exo_cols)
        for model_name in MODELS:
            print(f"\n[{ds_key}] Training {model_name}...")
            model = build_model(model_name, n_exo)
            trainer = Trainer(model, device, lr=1e-3)
            trainer.fit(loaders["train"], loaders["val"], epochs=EPOCHS, patience=4)
            metrics = evaluate_model(model, loaders["test"], device)
            quant_rows.append(
                {
                    "Dataset": cfg.name,
                    "DatasetKey": ds_key,
                    "Model": model_name,
                    "MSE": round(metrics["MSE"], 4),
                    "MAE": round(metrics["MAE"], 4),
                    "RMSE": round(metrics["RMSE"], 4),
                }
            )
            print(f"  -> MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

    quant_df = pd.DataFrame(quant_rows)
    qual_df = pd.DataFrame(QUALITATIVE)

    quant_df.to_csv(results_dir / "quantitative_comparison.csv", index=False)
    qual_df.to_csv(results_dir / "qualitative_comparison.csv", index=False)

    summary = {
        "research_question": "RQ1: Model comparison with similar-idea baselines",
        "best_private": quant_df[quant_df["DatasetKey"] == "private"].sort_values("MSE").iloc[0].to_dict(),
        "best_public": quant_df[quant_df["DatasetKey"] == "public"].sort_values("MSE").iloc[0].to_dict(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved results to {results_dir}")
    return quant_df, qual_df


if __name__ == "__main__":
    run()
