# Báo cáo nghiên cứu — TimeXer Head-Pose Forecast

**Môn:** 65TTNT  
**Chủ đề:** Dự báo chuỗi thời gian với biến ngoại sinh (Exogenous Forecasting)  
**Paper tham chiếu:** [TimeXer (NeurIPS 2024)](https://arxiv.org/abs/2402.19072)

---

## 1. Tóm tắt (Abstract)

Dự án tái hiện **paradigm TimeXer** — dự báo biến nội sinh (endogenous) bằng biến ngoại sinh (exogenous) — trên hai bộ dữ liệu:

1. **Private:** dự báo góc quay đầu `head_yaw` từ webcam (sinh trắc khuôn mặt)
2. **Public:** dự báo nồng độ CO₂ từ benchmark thời tiết

Ba câu hỏi nghiên cứu (RQ1–RQ3) được trả lời bằng so sánh định lượng/định tính, ablation và đánh giá dự báo trực quan. Toàn bộ protocol được ghi trong [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md).

---

## 2. Minh bạch dữ liệu (Data Provenance)

| Dataset | Loại nguồn | Ghi chú |
|---------|------------|---------|
| Private CSV | **Mô phỏng** | Sinh bởi `generate_datasets.py`, mô phỏng động lực webcam + MediaPipe |
| Public CSV | **Mô phỏng** | Schema giống Weather benchmark (CO₂ + khí hậu) |
| Webcam thật | Tùy chọn | `scripts/extract_from_webcam.py` |
| Benchmark gốc | Chưa tải | [Time-Series-Library](https://github.com/thuml/Time-Series-Library) |

**Lý do dùng dữ liệu mô phỏng:** Đảm bảo pipeline end-to-end, tái lập được, và bảo vệ quyền riêng tư trước khi thu thập video thật. Metadata (`data/*/metadata.json`) ghi rõ `data_source: simulated`.

---

## 3. Phương pháp (Methodology)

### 3.1 Task formulation

- Input: 96 bước endogenous + exogenous đồng bộ
- Output: dự báo 24 bước endogenous
- Chia dữ liệu theo thời gian: 70% train / 10% val / 20% test
- Chuẩn hóa StandardScaler (fit trên train)

### 3.2 Mô hình so sánh (RQ1)

| Model | Vai trò |
|-------|---------|
| **TimeXer** | Mô hình đề xuất (patch + global token + cross-attn) |
| iTransformer | Transformer biến (variate token) |
| PatchTST | Transformer patch-level |
| TiDE | MLP encoder–decoder |
| DLinear | Baseline tuyến tính |

### 3.3 Thiết lập huấn luyện

- Optimizer: Adam, lr = 1e-3
- Loss: MSE
- Epochs: 12, early stopping patience = 4
- Seeds: **42, 123, 456** → báo cáo mean ± std

Chi tiết: [`experiments/config.py`](experiments/config.py), [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md).

---

## 4. RQ1 — So sánh mô hình

**Câu hỏi:** TimeXer so với các mô hình cùng hướng tư tưởng?

**Bằng chứng:**
- Định lượng: `results/exp1_comparison/quantitative_comparison_aggregate.csv`
- Định tính: `results/exp1_comparison/qualitative_comparison.csv`

**Kết luận (3 seeds: 42, 123, 456 — mean ± std):**
- Private: TiDE tốt nhất (MSE 1.0205 ± 0.0004); TimeXer 1.0243 ± 0.0029; DLinear kém nhất.
- Public: TiDE tốt nhất (MSE 0.3720 ± 0.0013); TimeXer 0.3798 ± 0.0011; chênh lệch giữa các seed nhỏ (std ~0.001–0.003).
- **Định tính:** TimeXer là mô hình duy nhất kết hợp patch-level + variate-level exo + global token + cross-attention — phù hợp khi exogenous có cấu trúc khác endogenous.

> Chênh lệch số liệu giữa các seed nhỏ; xem cột `*_std` trong file aggregate.

---

## 5. RQ2 — Ablation

**Câu hỏi:** Loại bỏ từng module có ảnh hưởng performance không?

**Biến thể:** `full`, `no_global`, `exo_patch`, `exo_add`, `exo_concat`, `no_exo`

**Bằng chứng:** `results/exp2_ablation/ablation_quantitative_aggregate.csv`, `ablation_delta.csv`

**Kết luận (public, mean ± std):**
- Bỏ exogenous (`no_exo`): MSE tăng **+1.99%** so với full
- Bỏ global token: MSE tăng **+1.98%**
- Exo patch thay variate: MSE tăng **+1.96%**
- Biến thể `exo_add` cho MSE thấp hơn full (−0.74%) trên dữ liệu mô phỏng — cần kiểm chứng trên benchmark thật

Trên private dataset, chênh lệch ablation nhỏ hơn — có thể do quy mô/dynamics mô phỏng.

---

## 6. RQ3 — Đánh giá dự báo thực tế

**Câu hỏi:** Dự báo trông như thế nào trên từng cửa sổ test?

**Bằng chứng:** `results/exp3_visualization/` — 12 PNG (3 mẫu × 2 dataset × 2 loại hình)

- `*_timexer.png`: input 96 bước + ground truth + prediction 24 bước
- `*_comparison.png`: TimeXer vs iTransformer, PatchTST, DLinear

Manifest: `results/exp3_visualization/manifest.json`

---

## 7. Hạn chế (Limitations)

1. Dữ liệu mô phỏng — chưa phản ánh benchmark thật hay webcam thật
2. Reimplementation đơn giản — không phải fork `run.py` gốc
3. Chưa so sánh số liệu paper trên Weather/EPF official
4. Chưa có kiểm định thống kê formal (p-value)
5. Link Google Drive video private cần cập nhật khi có dữ liệu thật

---

## 8. Kết luận

Repo cung cấp **framework nghiên cứu có thể kiểm chứng** cho paradigm TimeXer trên task sinh trắc private và benchmark weather-style public. Kết quả cho thấy kiến trúc TimeXer có lợi thế **cấu trúc** (exo handling) dù không luôn thắng mọi baseline trên dữ liệu mô phỏng quy mô nhỏ. Ablation xác nhận vai trò của exogenous variables và global token trên public dataset.

**Hướng phát triển:** Thu thập webcam thật, tải Weather dataset gốc, tích hợp code official TimeXer, chạy thêm seeds và statistical tests.

---

## 9. Tài liệu liên quan

- [`README.md`](README.md) — Tổng quan repo
- [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md) — Protocol đầy đủ
- [GitHub Pages Demo](https://vuxuansangdz.github.io/TimeXer_65TTNT-/)
