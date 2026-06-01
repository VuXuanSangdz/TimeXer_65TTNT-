"""
Experiment (2): Ablation study — remove/replace each TimeXer module.

Research question:
  Does removing each module affect overall performance?
  Evidence: qualitative + quantitative ablation tables (multi-seed when enabled).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS, TRAIN, build_loaders, experiment_manifest, get_device
from models.TimeXer import TimeXerAblation
from src.utils.metrics import evaluate_model
from src.utils.seed import set_seed
from src.utils.trainer import Trainer

QUALITATIVE = [
    {"Variant": "full", "Description": "P+G endo, V exo, Cross-Attn", "Expected": "Best overall"},
    {"Variant": "no_global", "Description": "Remove global token G", "Expected": "Weaker exo→patch bridge"},
    {"Variant": "exo_patch", "Description": "Exo patch embed instead of variate", "Expected": "Noisy exo representation"},
    {"Variant": "exo_add", "Description": "Add exo summary instead of cross-attn", "Expected": "Simpler fusion, worse"},
    {"Variant": "exo_concat", "Description": "Concat fusion", "Expected": "Moderate degradation"},
    {"Variant": "no_exo", "Description": "Remove all exogenous vars", "Expected": "Largest drop if exo informative"},
]


def _aggregate_ablation(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = per_seed_df.groupby(["Dataset", "DatasetKey", "Ablation", "Description"], as_index=False)
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
    results_dir = ROOT / "results" / "exp2_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for seed in seeds:
        set_seed(seed)
        for ds_key, cfg in DATASETS.items():
            loaders, _, cfg = build_loaders(cfg)
            n_exo = len(cfg.exo_cols)

            for ablation, desc in TimeXerAblation.ABLATIONS:
                print(f"\n[seed={seed}][{ds_key}] Ablation: {ablation} — {desc}")
                model = TimeXerAblation(
                    seq_len=TRAIN.seq_len,
                    pred_len=TRAIN.pred_len,
                    n_exo=n_exo,
                    patch_len=TRAIN.patch_len,
                    d_model=TRAIN.d_model,
                    e_layers=TRAIN.e_layers,
                )
                trainer = Trainer(model, device, lr=TRAIN.lr, ablation=ablation)
                trainer.fit(
                    loaders["train"],
                    loaders["val"],
                    epochs=TRAIN.epochs,
                    patience=TRAIN.patience,
                )
                metrics = evaluate_model(model, loaders["test"], device, ablation=ablation)
                per_seed_rows.append(
                    {
                        "Seed": seed,
                        "Dataset": cfg.name,
                        "DatasetKey": ds_key,
                        "Ablation": ablation,
                        "Description": desc,
                        "MSE": round(metrics["MSE"], 4),
                        "MAE": round(metrics["MAE"], 4),
                        "RMSE": round(metrics["RMSE"], 4),
                    }
                )
                print(f"  -> MSE={metrics['MSE']:.4f}")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(results_dir / "ablation_quantitative_per_seed.csv", index=False)

    agg_df = _aggregate_ablation(per_seed_df)
    agg_df.to_csv(results_dir / "ablation_quantitative_aggregate.csv", index=False)

    df = agg_df[["Dataset", "DatasetKey", "Ablation", "Description", "MSE", "MAE", "RMSE"]].copy()
    df.to_csv(results_dir / "ablation_quantitative.csv", index=False)
    pd.DataFrame(QUALITATIVE).to_csv(results_dir / "ablation_qualitative.csv", index=False)

    delta_rows = []
    for ds_key in DATASETS:
        sub = df[df["DatasetKey"] == ds_key]
        full_mse = sub[sub["Ablation"] == "full"]["MSE"].values[0]
        for _, r in sub.iterrows():
            delta_rows.append(
                {
                    "DatasetKey": ds_key,
                    "Ablation": r["Ablation"],
                    "MSE_delta_pct": round((r["MSE"] - full_mse) / full_mse * 100, 2),
                }
            )
    pd.DataFrame(delta_rows).to_csv(results_dir / "ablation_delta.csv", index=False)

    (results_dir / "summary.json").write_text(
        json.dumps(
            {
                "research_question": "RQ2: Module ablation study",
                "seeds": seeds,
                "variants": len(TimeXerAblation.ABLATIONS),
                "train_config": experiment_manifest()["train_config"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved ablation results to {results_dir}")
    return df


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
