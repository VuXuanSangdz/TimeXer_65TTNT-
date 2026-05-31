# TimeXer Head-Pose Forecast Project

Triển khai mô hình **TimeXer** (NeurIPS 2024) cho bài toán **dự báo chuỗi thời gian với biến ngoại sinh** (*forecasting with exogenous variables*).

## Chủ đề nghiên cứu

**Dự báo góc quay đầu (Head Yaw) từ chuỗi sinh trắc khuôn mặt thu thập qua webcam**

| Loại biến | Tên | Mô tả |
|-----------|-----|-------|
| **Endogenous** (dự báo) | `head_yaw` | Góc quay ngang của đầu (độ) |
| **Exogenous** (hỗ trợ) | `ambient_light` | Độ sáng khung hình |
| | `eye_aspect_ratio` | Mức mở mắt (EAR) |
| | `mouth_open_ratio` | Mức mở miệng |

**Ứng dụng thực tế:** Giám sát mức tập trung, driver monitoring, HCI.

Dataset **public** bổ sung: **Weather CO2** (theo benchmark TimeXer paper).

## Links

| Mục | URL |
|-----|-----|
| **GitHub Code** | https://github.com/VuXuanSangdz/TimeXer_65TTNT- |
| **GitHub Pages Demo** | https://vuxuansangdz.github.io/TimeXer_65TTNT-/ |
| **Public Data** | `data/public/weather_public.csv` |
| **Private Data (CSV)** | `data/private/face_headpose_private.csv` |
| **Private Raw Videos** | [Google Drive – Private Link](https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID) |
| **TimeXer Paper** | https://arxiv.org/abs/2402.19072 |
| **Official Code** | https://github.com/thuml/TimeXer |

## Cấu trúc project

```
timexer-headpose-forecast/
├── data/
│   ├── public/          # Weather CO2 benchmark
│   └── private/         # Face head-pose time series
├── src/
│   ├── models/          # TimeXer + baselines
│   ├── data/            # Loaders + dataset generation
│   └── utils/           # Trainer, metrics, plotting
├── experiments/
│   ├── exp1_model_comparison.py   # RQ1
│   ├── exp2_ablation.py           # RQ2
│   └── exp3_visualization.py      # RQ3
├── results/             # Bảng kết quả + hình minh họa
├── docs/                # GitHub Pages (index.html)
└── scripts/
    ├── run_all.py
    ├── extract_from_webcam.py
    └── build_github_pages.py
```

## Cài đặt

```bash
cd timexer-headpose-forecast
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Chạy toàn bộ thí nghiệm

```bash
python scripts/run_all.py
```

Hoặc từng bước:

```bash
# 1. Tạo dataset
python src/data/generate_datasets.py

# 2. RQ1 — So sánh mô hình
python experiments/exp1_model_comparison.py

# 3. RQ2 — Ablation
python experiments/exp2_ablation.py

# 4. RQ3 — Visualization
python experiments/exp3_visualization.py

# 5. Build GitHub Pages
python scripts/build_github_pages.py
```

## Thu thập dữ liệu private thật (webcam)

```bash
python scripts/extract_from_webcam.py --output data/private/face_headpose_real.csv --seconds 120
```

Yêu cầu: webcam + `mediapipe` + `opencv-python`.

## 3 câu hỏi nghiên cứu

### RQ1: TimeXer so với mô hình cùng tư tưởng?

- **Models:** TimeXer, iTransformer, PatchTST, TiDE, DLinear
- **Bằng chứng:** `results/exp1_comparison/quantitative_comparison.csv` + `qualitative_comparison.csv`

### RQ2: Loại bỏ từng module có ảnh hưởng không?

- **Variants:** full, no_global, exo_patch, exo_add, exo_concat, no_exo
- **Bằng chứng:** `results/exp2_ablation/ablation_quantitative.csv`

### RQ3: Dự đoán thực tế như thế nào?

- **Bằng chứng:** `results/exp3_visualization/*.png`

## Deploy GitHub Pages

1. Push repo lên GitHub
2. **Settings → Pages → Source:** Deploy from branch `main`, folder `/docs`
3. Truy cập: https://vuxuansangdz.github.io/TimeXer_65TTNT-/

## Tham khảo

```bibtex
@inproceedings{wang2024timexer,
  title={TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and others},
  booktitle={NeurIPS},
  year={2024}
}
```
