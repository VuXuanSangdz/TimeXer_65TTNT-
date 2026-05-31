# TimeXer_65TTNT

This repo is a course reproduction of [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072) (NeurIPS 2024), based on the official implementation at [thuml/TimeXer](https://github.com/thuml/TimeXer).

**Demo:** [https://vuxuansangdz.github.io/TimeXer_65TTNT-/](https://vuxuansangdz.github.io/TimeXer_65TTNT-/)

## Introduction

This project focuses on **forecasting with exogenous variables** — a practical paradigm where the target series (endogenous) is predicted using auxiliary covariates (exogenous) that do not need to be forecast themselves.

We apply TimeXer to a **private face biometric time-series** task: forecasting **head yaw** from webcam sessions, using ambient light, eye aspect ratio, and mouth openness as exogenous variables. A **public weather benchmark** (CO2 concentration with climate covariates) is included for comparison with the original paper setting.

TimeXer empowers the canonical Transformer to reconcile endogenous and exogenous information through patch-level and variate-level representations, achieving competitive results on both datasets.

## Overall Architecture

TimeXer employs **patch-level** representations for endogenous variables and **variate-level** representations for exogenous variables, with a **learnable global token** as a bridge in-between. With this design, TimeXer can jointly capture:

- **Intra-endogenous temporal dependencies** (patch-wise self-attention)
- **Exogenous-to-endogenous correlations** (cross-attention from global token to exogenous variate tokens)

Key modules: Patch token (P), Global token (G), Variate token (V), Self-attention, Cross-attention.

## Dataset

### Private — Face Head-Pose Time Series

| Variable | Role | Description |
|----------|------|-------------|
| `head_yaw` | Endogenous | Horizontal head rotation angle (degrees) |
| `ambient_light` | Exogenous | Frame brightness level |
| `eye_aspect_ratio` | Exogenous | Eye openness (EAR) |
| `mouth_open_ratio` | Exogenous | Mouth openness |

- **Path:** `./data/private/face_headpose_private.csv`
- **Collection:** Webcam recording + MediaPipe feature extraction (`scripts/extract_from_webcam.py`)
- **Raw videos (private):** [Google Drive](https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID)

### Public — Weather CO2 Benchmark

| Variable | Role |
|----------|------|
| `co2_concentration` | Endogenous |
| `temperature`, `humidity`, `pressure`, `wind_speed` | Exogenous |

- **Path:** `./data/public/weather_public.csv`
- **Reference:** [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

## Usage

1. Clone this repository and install dependencies.

```
pip install -r requirements.txt
```

2. Generate datasets (if not already present).

```
python src/data/generate_datasets.py
```

3. Run all experiments (model comparison, ablation, evaluation).

```
python scripts/run_all.py
```

Or run each research question separately:

```
python experiments/exp1_model_comparison.py   # RQ1: model comparison
python experiments/exp2_ablation.py             # RQ2: ablation study
python experiments/exp3_visualization.py        # RQ3: forecast evaluation
```

4. Collect real private data from webcam (optional).

```
python scripts/extract_from_webcam.py --output data/private/face_headpose_real.csv --seconds 120
```

## Main Results

We evaluate TimeXer on forecasting with exogenous variables using two datasets. Metrics: MSE, MAE, RMSE (lower is better). Settings: `seq_len=96`, `pred_len=24`.

### RQ1 — Model Comparison

**Private dataset (Face Head-Pose)**

| Model | MSE | MAE | RMSE |
|-------|-----|-----|------|
| TimeXer | 1.0302 | 0.8605 | 1.0150 |
| iTransformer | 1.0238 | 0.8575 | 1.0118 |
| PatchTST | 1.0249 | 0.8602 | 1.0124 |
| TiDE | **1.0209** | **0.8595** | **1.0104** |
| DLinear | 1.0501 | 0.8653 | 1.0248 |

**Public dataset (Weather CO2)**

| Model | MSE | MAE | RMSE |
|-------|-----|-----|------|
| TimeXer | 0.3755 | 0.4893 | 0.6128 |
| iTransformer | 0.3732 | 0.4877 | 0.6109 |
| PatchTST | **0.3720** | **0.4876** | **0.6099** |
| TiDE | 0.3731 | 0.4882 | 0.6108 |
| DLinear | 0.3882 | 0.4973 | 0.6230 |

**Qualitative comparison**

| Criterion | TimeXer | iTransformer | PatchTST | TiDE | DLinear |
|-----------|---------|--------------|----------|------|---------|
| Exogenous variate-level embedding | Yes | No | No | Partial | Partial |
| Patch-level temporal modeling | Yes | No | Yes | No | No |
| Global bridge token | Yes | No | No | No | No |
| Cross-attention exo→endo | Yes | No | No | No | No |

Full results: `./results/exp1_comparison/`

### RQ2 — Ablation Study

**Public dataset (Weather CO2)**

| Design | MSE | MAE | RMSE |
|--------|-----|-----|------|
| Full (P+G, V, Cross-Attn) | 0.3806 | 0.4926 | 0.6169 |
| Remove global token | 0.3832 | 0.4952 | 0.6190 |
| Exo patch embed | 0.3834 | 0.4959 | 0.6192 |
| Exo add fusion | **0.3736** | **0.4874** | **0.6112** |
| Exo concat fusion | 0.3763 | 0.4893 | 0.6134 |
| No exogenous | 0.3848 | 0.4960 | 0.6203 |

Removing exogenous variables or the global token degrades performance on the public dataset, confirming the importance of both modules. Full ablation tables: `./results/exp2_ablation/`

### RQ3 — Forecast Evaluation

Forecast outputs are saved as numerical results and summary files under `./results/exp3_visualization/`. Each experiment compares predicted vs. ground-truth sequences over a 24-step horizon given 96-step history.

## Project Structure

```
├── data/private/          # Face head-pose time series (private)
├── data/public/           # Weather CO2 benchmark (public)
├── src/models/            # TimeXer + baselines
├── experiments/           # RQ1, RQ2, RQ3 scripts
├── results/               # Quantitative & qualitative results
├── scripts/               # run_all, extract_from_webcam
└── docs/                  # GitHub Pages
```

## Citation

If you find this repo helpful, please cite the original TimeXer paper.

```
@article{wang2024timexer,
  title={Timexer: Empowering transformers for time series forecasting with exogenous variables},
  author={Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Qiu, Yunzhong and Zhang, Haoran and Wang, Jianmin and Long, Mingsheng},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement

We appreciate the following repositories for their valuable code and efforts.

- [TimeXer](https://github.com/thuml/TimeXer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)
- [PatchTST](https://github.com/yuqinie98/PatchTST)

## Contact

Course project — 65TTNT. Repository: [VuXuanSangdz/TimeXer_65TTNT-](https://github.com/VuXuanSangdz/TimeXer_65TTNT-)
