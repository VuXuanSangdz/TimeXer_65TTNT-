# Báo cáo nghiên cứu — TimeXer Head-Pose Forecast

## Chủ đề

**Dự báo góc quay đầu (Head Yaw) từ chuỗi sinh trắc khuôn mặt webcam** theo paradigm *Forecasting with Exogenous Variables* của TimeXer (NeurIPS 2024).

| | Private | Public |
|---|---|---|
| **Endogenous** | `head_yaw` | `co2_concentration` |
| **Exogenous** | ambient_light, eye_aspect_ratio, mouth_open_ratio | temperature, humidity, pressure, wind_speed |
| **Nguồn** | Webcam tự thu thập (MediaPipe) | Weather benchmark |

---

## RQ1: So sánh với mô hình cùng tư tưởng

Xem `results/exp1_comparison/quantitative_comparison.csv`.

**Kết luận định lượng:** Trên private data, TiDE và iTransformer có MSE thấp nhất (~1.02); TimeXer đạt MSE ~1.03, tốt hơn DLinear. Trên public Weather, PatchTST/TiDE hơi tốt hơn TimeXer (~0.372 vs 0.376).

**Kết luận định tính:** TimeXer là mô hình duy nhất kết hợp patch-level temporal + variate-level exogenous + global bridge token + cross-attention — phù hợp khi exogenous có cấu trúc khác endogenous (xem `qualitative_comparison.csv`).

---

## RQ2: Ablation — loại bỏ từng module

Xem `results/exp2_ablation/ablation_quantitative.csv`.

| Ablation | Ý nghĩa | Ảnh hưởng (public) |
|---|---|---|
| `no_exo` | Bỏ biến ngoại sinh | MSE tăng ~1.1% |
| `no_global` | Bỏ global token | MSE tăng ~0.7% |
| `exo_add` | Thay cross-attn bằng cộng | MSE giảm nhẹ (dataset nhỏ) |
| `exo_patch` | Exo dùng patch thay variate | MSE tăng ~0.7% |

**Kết luận:** Exogenous variables và thiết kế embedding có ảnh hưởng đo được; global token đóng vai trò cầu nối quan trọng theo paper.

---

## RQ3: Dự đoán thực tế

Xem `results/exp3_visualization/*.png` và demo GitHub Pages.

Hình `*_timexer.png`: input 96 bước + ground truth + prediction 24 bước.  
Hình `*_comparison.png`: so sánh TimeXer vs iTransformer, PatchTST, DLinear trên cùng cửa sổ.

---

## Triển khai lại code gốc

Implementation dựa trên kiến trúc paper và repo [thuml/TimeXer](https://github.com/thuml/TimeXer), tích hợp trong project độc lập để dễ chạy và báo cáo.
