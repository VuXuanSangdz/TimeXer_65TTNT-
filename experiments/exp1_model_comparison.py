"""
Experiment (1): Compare TimeXer with similar forecasting models.

Research question:
  How does TimeXer compare to models with similar ideas?
  Evidence: qualitative + quantitative result tables (multi-seed when enabled).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS, TRAIN, build_loaders, build_model, experiment_manifest, get_device
from src.utils.metrics import evaluate_model
from src.utils.seed import set_seed
from src.utils.trainer import Trainer

MODELS = ["TimeXer", "iTransformer", "PatchTST", "TiDE", "DLinear"]

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


def _aggregate_metrics(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = per_seed_df.groupby(["Dataset", "DatasetKey", "Model"], as_index=False)
    agg = grouped.agg(
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        n_seeds=("Seed", "count"),
    )
    for col in ("MSE_std", "MAE_std", "RMSE_std"):
        agg[col] = agg[col].fillna(0.0)
    agg["MSE"] = agg["MSE_mean"].round(4)
    agg["MAE"] = agg["MAE_mean"].round(4)
    agg["RMSE"] = agg["RMSE_mean"].round(4)
    return agg


def run(seeds: list[int] | None = None):
    seeds = list(seeds or TRAIN.default_seeds)
    device = get_device()
    results_dir = ROOT / "results" / "exp1_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for seed in seeds:
        set_seed(seed)
        for ds_key, cfg in DATASETS.items():
            loaders, _, cfg = build_loaders(cfg)
            n_exo = len(cfg.exo_cols)
            for model_name in MODELS:
                print(f"\n[seed={seed}][{ds_key}] Training {model_name}...")
                model = build_model(model_name, n_exo)
                trainer = Trainer(model, device, lr=TRAIN.lr)
                trainer.fit(
                    loaders["train"],
                    loaders["val"],
                    epochs=TRAIN.epochs,
                    patience=TRAIN.patience,
                )
                metrics = evaluate_model(model, loaders["test"], device)
                per_seed_rows.append(
                    {
                        "Seed": seed,
                        "Dataset": cfg.name,
                        "DatasetKey": ds_key,
                        "Model": model_name,
                        "MSE": round(metrics["MSE"], 4),
                        "MAE": round(metrics["MAE"], 4),
                        "RMSE": round(metrics["RMSE"], 4),
                    }
                )
                print(f"  -> MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(results_dir / "quantitative_comparison_per_seed.csv", index=False)

    agg_df = _aggregate_metrics(per_seed_df)
    agg_df.to_csv(results_dir / "quantitative_comparison_aggregate.csv", index=False)

    # Backward-compatible primary table (mean metrics)
    quant_df = agg_df[["Dataset", "DatasetKey", "Model", "MSE", "MAE", "RMSE"]].copy()
    qual_df = pd.DataFrame(QUALITATIVE)
    quant_df.to_csv(results_dir / "quantitative_comparison.csv", index=False)
    qual_df.to_csv(results_dir / "qualitative_comparison.csv", index=False)

    summary = {
        "research_question": "RQ1: Model comparison with similar-idea baselines",
        "seeds": seeds,
        "train_config": experiment_manifest()["train_config"],
        "data_provenance": {
            k: {"data_source": v.data_source, "note": v.collection_note}
            for k, v in DATASETS.items()
        },
        "best_private": quant_df[quant_df["DatasetKey"] == "private"].sort_values("MSE").iloc[0].to_dict(),
        "best_public": quant_df[quant_df["DatasetKey"] == "public"].sort_values("MSE").iloc[0].to_dict(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest(), indent=2), encoding="utf-8"
    )

    print(f"\nSaved results to {results_dir}")
    return quant_df, qual_df, agg_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in TRAIN.default_seeds),
        help="Comma-separated random seeds (default: 42,123,456)",
    )
    args = parser.parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run(seed_list)
