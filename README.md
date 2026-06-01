# TimeXer_65TTNT

Course reproduction of [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072) (NeurIPS 2024), inspired by [thuml/TimeXer](https://github.com/thuml/TimeXer).

**Demo:** [https://vuxuansangdz.github.io/TimeXer_65TTNT-/](https://vuxuansangdz.github.io/TimeXer_65TTNT-/)  
**Full protocol:** [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md) · **Research report (VI):** [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md)

---

## Scope & Scientific Positioning

| | This repository | Official TimeXer |
|---|-----------------|------------------|
| **Goal** | RQ1–RQ3 on custom private + public datasets | Paper benchmarks (Weather, EPF, …) |
| **Code** | Standalone simplified reimplementation | `run.py` + Time-Series-Library |
| **Data in repo** | Simulated CSV (documented, reproducible) | Official benchmark downloads |
| **Claim** | Documented architectural comparison under fixed protocol | SOTA benchmark numbers |

We report results transparently on **simulated datasets** (see [Data Provenance](#data-provenance)). This supports reproducible methodology; it does **not** claim numerical equivalence with NeurIPS paper tables.

---

## Introduction

This project studies **forecasting with exogenous variables**: predict an endogenous target using auxiliary covariates that need not be forecast.

- **Private task:** forecast **head yaw** from face biometric time series (webcam / MediaPipe paradigm)
- **Public task:** forecast **CO₂ concentration** with climate covariates (Weather-style benchmark)

TimeXer uses **patch-level** endogenous tokens, **variate-level** exogenous tokens, and a **learnable global token** bridged via self- and cross-attention.

---

## Data Provenance

> **Important:** CSV files currently in the repository are **synthetically generated** by `src/data/generate_datasets.py` to validate the pipeline. Metadata records `data_source: simulated`.

### Private — Face Head-Pose

| Variable | Role | Description |
|----------|------|-------------|
| `head_yaw` | Endogenous | Horizontal head rotation (degrees) |
| `ambient_light` | Exogenous | Frame brightness |
| `eye_aspect_ratio` | Exogenous | Eye openness (EAR) |
| `mouth_open_ratio` | Exogenous | Mouth openness |

- **File:** `./data/private/face_headpose_private.csv` (15,000 rows, 5 Hz, 5 sessions)
- **Source type:** `simulated` — replace with `scripts/extract_from_webcam.py` for real data
- **Raw videos:** [Google Drive placeholder](https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID) — update when uploaded

### Public — Weather CO₂

| Variable | Role |
|----------|------|
| `co2_concentration` | Endogenous |
| `temperature`, `humidity`, `pressure`, `wind_speed` | Exogenous |

- **File:** `./data/public/weather_public.csv` (8,000 rows, 10-min intervals)
- **Source type:** `simulated` — schema follows [Time-Series-Library Weather](https://github.com/thuml/Time-Series-Library)

See `data/*/metadata.json` and `data/*/DATA_LINK.md` for full provenance.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| `seq_len` / `pred_len` | 96 / 24 |
| Train / Val / Test split | 70% / 10% / 20% (chronological) |
| Scaler | StandardScaler (fit on train only) |
| Optimizer / LR | Adam / 1e-3 |
| Loss | MSE |
| Epochs / Patience | 12 / 4 |
| Random seeds | 42, 123, 456 |
| Metrics | MSE, MAE, RMSE (test set) |

Models compared: **TimeXer**, **iTransformer**, **PatchTST**, **TiDE**, **DLinear**.

Full hyperparameters and limitations: [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md).

---

## Usage

```bash
pip install -r requirements.txt
python src/data/generate_datasets.py
python scripts/run_all.py
```

Or run each research question:

```bash
python experiments/exp1_model_comparison.py --seeds 42,123,456   # RQ1
python experiments/exp2_ablation.py --seeds 42,123,456             # RQ2
python experiments/exp3_visualization.py                           # RQ3
python scripts/build_github_pages.py
```

Collect real private data (optional):

```bash
python scripts/extract_from_webcam.py --output data/private/face_headpose_real.csv --seconds 120
```

---

## Main Results

Metrics are **mean ± std over 3 seeds** (42, 123, 456). Per-seed CSVs: `results/exp1_comparison/quantitative_comparison_per_seed.csv`.

### RQ1 — Model Comparison

**Private dataset (Face Head-Pose, simulated)**

| Model | MSE | MAE | RMSE |
|-------|-----|-----|------|
| TiDE | **1.0205 ± 0.0004** | **0.8603 ± 0.0005** | **1.0102 ± 0.0002** |
| iTransformer | 1.0225 ± 0.0006 | 0.8577 ± 0.0005 | 1.0112 ± 0.0003 |
| TimeXer | 1.0243 ± 0.0029 | 0.8603 ± 0.0013 | 1.0121 ± 0.0014 |
| PatchTST | 1.0264 ± 0.0027 | 0.8602 ± 0.0004 | 1.0131 ± 0.0013 |
| DLinear | 1.0507 ± 0.0004 | 0.8652 ± 0.0003 | 1.0251 ± 0.0002 |

**Public dataset (Weather CO₂, simulated)**

| Model | MSE | MAE | RMSE |
|-------|-----|-----|------|
| TiDE | **0.3720 ± 0.0013** | **0.4882 ± 0.0006** | **0.6099 ± 0.0010** |
| PatchTST | 0.3741 ± 0.0032 | 0.4890 ± 0.0019 | 0.6116 ± 0.0026 |
| iTransformer | 0.3745 ± 0.0024 | 0.4886 ± 0.0012 | 0.6119 ± 0.0020 |
| TimeXer | 0.3798 ± 0.0011 | 0.4919 ± 0.0008 | 0.6162 ± 0.0009 |
| DLinear | 0.3877 ± 0.0020 | 0.4973 ± 0.0019 | 0.6226 ± 0.0017 |

**Qualitative comparison**

| Criterion | TimeXer | iTransformer | PatchTST | TiDE | DLinear |
|-----------|---------|--------------|----------|------|---------|
| Exogenous variate-level embedding | Yes | No | No | Partial | Partial |
| Patch-level temporal modeling | Yes | No | Yes | No | No |
| Global bridge token | Yes | No | No | No | No |
| Cross-attention exo→endo | Yes | No | No | No | No |

Full tables: `./results/exp1_comparison/`

### RQ2 — Ablation Study

Ablation on **both** datasets (mean ± std over seeds 42, 123, 456).

**Public dataset (Weather CO₂)**

| Design | MSE | MAE | RMSE | ΔMSE vs full |
|--------|-----|-----|------|--------------|
| exo_add | **0.3753 ± 0.0036** | 0.4888 ± 0.0025 | 0.6125 ± 0.0029 | −0.74% |
| full (P+G, V, Cross-Attn) | 0.3781 ± 0.0004 | 0.4905 ± 0.0005 | 0.6149 ± 0.0003 | 0% |
| exo_concat | 0.3777 ± 0.0019 | 0.4906 ± 0.0010 | 0.6146 ± 0.0016 | −0.11% |
| exo_patch | 0.3855 ± 0.0015 | 0.4968 ± 0.0009 | 0.6209 ± 0.0013 | +1.96% |
| no_exo | 0.3856 ± 0.0019 | 0.4963 ± 0.0016 | 0.6210 ± 0.0016 | +1.99% |
| no_global | 0.3856 ± 0.0030 | 0.4959 ± 0.0012 | 0.6210 ± 0.0024 | +1.99% |

Removing exogenous variables or the global token increases MSE by ~2% on public data, supporting their role in the full architecture. Private ablation deltas are smaller (see `ablation_delta.csv`).

Full tables: `./results/exp2_ablation/`

### RQ3 — Forecast Evaluation

12 forecast figures (3 test windows × 2 datasets × 2 plot types) in `./results/exp3_visualization/`:

- `*_timexer.png` — TimeXer: history + ground truth + prediction
- `*_comparison.png` — TimeXer vs iTransformer, PatchTST, DLinear

Manifest: `results/exp3_visualization/manifest.json`

---

## Limitations

1. Simulated data — results validate pipeline, not real-world SOTA
2. Simplified reimplementation — not official `run.py` fork
3. No formal statistical significance tests (mean ± std over seeds only)
4. Private Drive link is a placeholder until real videos are uploaded
5. Paper benchmark numbers require official Time-Series-Library datasets

---

## Project Structure

```
├── models/
│   ├── TimeXer.py         # Proposed model (NeurIPS 2024)
│   ├── iTransformer.py    # Baseline
│   ├── PatchTST.py        # Baseline
│   ├── TiDE.py            # Baseline
│   ├── DLinear.py         # Baseline
│   └── baselines.py       # Model registry
├── layers/                # Embedding layers
├── data/private/          # Private CSV + metadata
├── data/public/           # Public CSV + metadata
├── experiments/           # RQ1, RQ2, RQ3
├── results/               # Quantitative + qualitative + figures
├── scripts/               # run_all, extract_from_webcam, build_github_pages
├── docs/                  # GitHub Pages
├── EXPERIMENTAL_SETUP.md  # Full scientific protocol
└── RESEARCH_REPORT.md     # Báo cáo nghiên cứu (Tiếng Việt)
```

---

## Authors

**VuXuanSangdz** — Course project 65TTNT, NorthWest University.

---

## Citation

If you use this repo, please cite the original TimeXer paper:

```
@article{wang2024timexer,
  title={Timexer: Empowering transformers for time series forecasting with exogenous variables},
  author={Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Qiu, Yunzhong and Zhang, Haoran and Wang, Jianmin and Long, Mingsheng},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## Acknowledgement

- [TimeXer](https://github.com/thuml/TimeXer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)
- [PatchTST](https://github.com/yuqinie98/PatchTST)

---

## Contact

Repository: [VuXuanSangdz/TimeXer_65TTNT-](https://github.com/VuXuanSangdz/TimeXer_65TTNT-)
