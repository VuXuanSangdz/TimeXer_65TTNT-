# Experimental Setup

This document specifies the full protocol used in RQ1–RQ3 so results can be audited and reproduced.

## 1. Scope of this repository

| Aspect | This repo | Official [thuml/TimeXer](https://github.com/thuml/TimeXer) |
|--------|-----------|-----------------------------------------------------------|
| Goal | Course reproduction + custom private/public datasets | Full paper benchmarks (Weather, EPF, etc.) |
| Code | Standalone simplified reimplementation | `run.py` + Time-Series-Library integration |
| Data in repo | **Simulated CSV** (schema-valid, pipeline-ready) | Official benchmark downloads |
| Claim | Architectural study under exogenous forecasting paradigm | State-of-the-art benchmark numbers |

We do **not** claim numerical equivalence with NeurIPS paper tables. We report results on our datasets under a fixed, documented protocol.

## 2. Data provenance

### Private — Face Head-Pose

| Field | Value |
|-------|-------|
| File | `data/private/face_headpose_private.csv` |
| Source type | **`simulated`** (see `src/data/generate_datasets.py`) |
| Rationale | Demonstrates end-to-end pipeline before real webcam collection |
| Endogenous | `head_yaw` (degrees) |
| Exogenous | `ambient_light`, `eye_aspect_ratio`, `mouth_open_ratio` |
| Sampling | 5 Hz, 5 sessions × 600 s ≈ 15,000 rows |
| Real collection | `scripts/extract_from_webcam.py` (MediaPipe) |
| Raw videos | Not in repo; upload to Google Drive and update link in README |

### Public — Weather CO2

| Field | Value |
|-------|-------|
| File | `data/public/weather_public.csv` |
| Source type | **`simulated`** (Weather-schema subset) |
| Reference benchmark | [Time-Series-Library Weather](https://github.com/thuml/Time-Series-Library) |
| Endogenous | `co2_concentration` |
| Exogenous | `temperature`, `humidity`, `pressure`, `wind_speed` |
| Sampling | 10-minute intervals, 8,000 rows |

> **Transparency:** Both CSV files are synthetically generated with controlled dynamics and noise. Metadata in `data/*/metadata.json` records `data_source: simulated`.

## 3. Train / validation / test split

Chronological split (no shuffle across time):

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Fit model + scaler |
| Validation | 10% | Early stopping |
| Test | 20% | Reported metrics only |

Implementation: `src/data/loaders.py` → `split_by_time()`.

Scaling: `StandardScaler` fit **only on training** endogenous/exogenous features; applied to val/test.

## 4. Forecasting task

| Parameter | Value |
|-----------|-------|
| Input length `seq_len` | 96 |
| Horizon `pred_len` | 24 |
| Paradigm | Exogenous forecasting (exo known over input window) |
| Loss | MSE |
| Optimizer | Adam, lr = 1e-3 |
| Batch size | 64 |
| Max epochs | 12 |
| Early stopping | Patience = 4 on validation MSE |
| Random seeds | **42, 123, 456** (report mean ± std) |

## 5. Model configuration

Shared settings unless noted:

| Hyperparameter | TimeXer | PatchTST | iTransformer | TiDE | DLinear |
|----------------|---------|----------|--------------|------|---------|
| `d_model` | 128 | 128 | 128 | hidden 256 | — |
| `e_layers` | 2 | 2 | 2 | 2 MLP blocks | — |
| `patch_len` | 8 | 8 | — | — | — |
| `n_heads` | 4 | 4 | 4 | — | — |
| Exo handling | Variate token + cross-attn | Additive exo proj | Variate tokens | Concat input | Concat input |

All models share `seq_len`, `pred_len`, `n_exo` per dataset.

## 6. Evaluation metrics

Reported on **test set** (inverse-transformed where applicable):

- **MSE** — Mean Squared Error (primary)
- **MAE** — Mean Absolute Error
- **RMSE** — Root MSE

Multi-seed runs: mean ± standard deviation across seeds per (dataset, model).

## 7. Research questions & outputs

| RQ | Script | Primary outputs |
|----|--------|-----------------|
| RQ1 | `experiments/exp1_model_comparison.py` | `quantitative_comparison.csv`, `*_per_seed.csv`, `*_aggregate.csv` |
| RQ2 | `experiments/exp2_ablation.py` | `ablation_quantitative.csv`, `ablation_delta.csv` |
| RQ3 | `experiments/exp3_visualization.py` | `results/exp3_visualization/*.png`, `manifest.json` |

## 8. Reproducibility commands

```bash
pip install -r requirements.txt
python src/data/generate_datasets.py
python experiments/exp1_model_comparison.py --seeds 42,123,456
python experiments/exp2_ablation.py --seeds 42,123,456
python experiments/exp3_visualization.py
python scripts/build_github_pages.py
```

Or end-to-end: `python scripts/run_all.py`

Machine manifest is written to `results/exp1_comparison/experiment_manifest.json`.

## 9. Limitations

1. **Simulated data** — Results validate pipeline and module design, not real-world benchmark SOTA.
2. **Simplified baselines** — Not full Time-Series-Library configs (layer depth, channel independence details may differ).
3. **Single hardware run** — Document CPU/GPU in run log; no distributed training.
4. **No statistical tests** — We report mean ± std over seeds; formal significance tests (e.g. paired t-test) are future work.
5. **Private raw media** — Placeholder Drive link until real videos are uploaded.

## 10. Relation to TimeXer paper

The original paper evaluates on standard benchmarks with official preprocessing. Our contribution is:

- Applying the **exogenous forecasting paradigm** to a **face biometric** private task
- Systematic **RQ1–RQ3** comparison on a **fixed, documented protocol**
- Open code + results for course reproducibility

For paper-level numbers, use the official repository and datasets.
