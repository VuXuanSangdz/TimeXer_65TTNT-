"""
Experiment (2): Ablation study — remove/replace each TimeXer module.

Research question:
  Does removing each module affect overall performance?
  Evidence: qualitative + quantitative ablation tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import DATASETS, build_loaders, get_device
from src.models.timexer import TimeXerAblation
from src.utils.metrics import evaluate_model
from src.utils.trainer import Trainer

EPOCHS = 12

QUALITATIVE = [
    {"Variant": "full", "Description": "P+G endo, V exo, Cross-Attn", "Expected": "Best overall"},
    {"Variant": "no_global", "Description": "Remove global token G", "Expected": "Weaker exo→patch bridge"},
    {"Variant": "exo_patch", "Description": "Exo patch embed instead of variate", "Expected": "Noisy exo representation"},
    {"Variant": "exo_add", "Description": "Add exo summary instead of cross-attn", "Expected": "Simpler fusion, worse"},
    {"Variant": "exo_concat", "Description": "Concat fusion", "Expected": "Moderate degradation"},
    {"Variant": "no_exo", "Description": "Remove all exogenous vars", "Expected": "Largest drop if exo informative"},
]


def run():
    device = get_device()
    results_dir = ROOT / "results" / "exp2_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_key, cfg in DATASETS.items():
        loaders, _, cfg = build_loaders(cfg)
        n_exo = len(cfg.exo_cols)

        for ablation, desc in TimeXerAblation.ABLATIONS:
            print(f"\n[{ds_key}] Ablation: {ablation} — {desc}")
            model = TimeXerAblation(seq_len=96, pred_len=24, n_exo=n_exo, patch_len=8, d_model=128, e_layers=2)
            trainer = Trainer(model, device, lr=1e-3, ablation=ablation)
            trainer.fit(loaders["train"], loaders["val"], epochs=EPOCHS, patience=4)
            metrics = evaluate_model(model, loaders["test"], device, ablation=ablation)
            rows.append(
                {
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

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ablation_quantitative.csv", index=False)
    pd.DataFrame(QUALITATIVE).to_csv(results_dir / "ablation_qualitative.csv", index=False)

    # Compute delta vs full model
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
        json.dumps({"research_question": "RQ2: Module ablation study", "variants": len(TimeXerAblation.ABLATIONS)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved ablation results to {results_dir}")
    return df


if __name__ == "__main__":
    run()
