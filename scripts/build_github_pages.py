"""Build static GitHub Pages site from experiment results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _df_to_html(df: pd.DataFrame, table_id: str) -> str:
    return df.to_html(index=False, classes="data-table", table_id=table_id, border=0)


def build():
    DOCS.mkdir(parents=True, exist_ok=True)
    assets = DOCS / "assets"
    assets.mkdir(exist_ok=True)

    # Copy visualization images
    viz_src = RESULTS / "exp3_visualization"
    if viz_src.exists():
        for png in viz_src.glob("*.png"):
            shutil.copy2(png, assets / png.name)

    exp1_q = pd.read_csv(RESULTS / "exp1_comparison" / "quantitative_comparison.csv") if (RESULTS / "exp1_comparison" / "quantitative_comparison.csv").exists() else pd.DataFrame()
    exp1_qual = pd.read_csv(RESULTS / "exp1_comparison" / "qualitative_comparison.csv") if (RESULTS / "exp1_comparison" / "qualitative_comparison.csv").exists() else pd.DataFrame()
    exp2 = pd.read_csv(RESULTS / "exp2_ablation" / "ablation_quantitative.csv") if (RESULTS / "exp2_ablation" / "ablation_quantitative.csv").exists() else pd.DataFrame()

    images = sorted(assets.glob("*.png"))
    gallery = "\n".join(
        f'<figure><img src="assets/{p.name}" alt="{p.stem}"><figcaption>{p.stem.replace("_", " ")}</figcaption></figure>'
        for p in images
    )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TimeXer Head-Pose Forecast — Demo</title>
  <style>
    :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --accent:#38bdf8; --muted:#94a3b8; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
    header {{ padding:2.5rem 1.5rem; text-align:center; background:linear-gradient(135deg,#0f172a,#1e3a5f); border-bottom:1px solid #334155; }}
    header h1 {{ font-size:1.8rem; color:var(--accent); }}
    header p {{ color:var(--muted); max-width:720px; margin:.5rem auto 0; }}
    main {{ max-width:1100px; margin:0 auto; padding:2rem 1.5rem 4rem; }}
    section {{ background:var(--card); border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; border:1px solid #334155; }}
    h2 {{ color:var(--accent); margin-bottom:1rem; font-size:1.25rem; }}
    h3 {{ color:#7dd3fc; margin:1rem 0 .5rem; font-size:1rem; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
    .data-table th {{ background:#334155; padding:.5rem; text-align:left; }}
    .data-table td {{ padding:.45rem .5rem; border-bottom:1px solid #334155; }}
    .data-table tr:hover td {{ background:#263248; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1rem; }}
    figure {{ background:#0f172a; border-radius:8px; overflow:hidden; border:1px solid #334155; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ padding:.5rem .75rem; font-size:.8rem; color:var(--muted); }}
    .links a {{ color:var(--accent); margin-right:1rem; }}
    .badge {{ display:inline-block; background:#334155; padding:.2rem .6rem; border-radius:999px; font-size:.75rem; margin:.2rem; }}
    footer {{ text-align:center; padding:2rem; color:var(--muted); font-size:.85rem; }}
  </style>
</head>
<body>
  <header>
    <h1>TimeXer: Dự báo Head Pose từ Webcam</h1>
    <p>Triển khai mô hình TimeXer (NeurIPS 2024) cho bài toán forecasting with exogenous variables.
       Private data: sinh trắc khuôn mặt tự thu thập. Public data: Weather CO2 benchmark.</p>
    <div style="margin-top:1rem">
      <span class="badge">Endogenous: head_yaw / CO2</span>
      <span class="badge">Exogenous: light, EAR, mouth / climate</span>
      <span class="badge">seq_len=96, pred_len=24</span>
    </div>
  </header>
  <main>
    <section class="links">
      <h2>Links</h2>
      <p>
        <a href="https://github.com/VuXuanSangdz/TimeXer_65TTNT-">GitHub Repository</a>
        <a href="https://vuxuansangdz.github.io/TimeXer_65TTNT-/">GitHub Pages Demo</a>
        <a href="https://github.com/thuml/TimeXer">TimeXer Official</a>
      </p>
      <h3>Data</h3>
      <p><strong>Public:</strong> <code>data/public/weather_public.csv</code> — Weather-style benchmark</p>
      <p><strong>Private:</strong> <code>data/private/face_headpose_private.csv</code> + raw videos on Google Drive</p>
    </section>

    <section>
      <h2>RQ1: So sánh mô hình (Định lượng)</h2>
      {_df_to_html(exp1_q, "rq1-quant") if len(exp1_q) else "<p>Chưa chạy experiment.</p>"}
    </section>

    <section>
      <h2>RQ1: So sánh mô hình (Định tính)</h2>
      {_df_to_html(exp1_qual, "rq1-qual") if len(exp1_qual) else ""}
    </section>

    <section>
      <h2>RQ2: Ablation Study (Định lượng)</h2>
      {_df_to_html(exp2, "rq2-ablation") if len(exp2) else "<p>Chưa chạy experiment.</p>"}
    </section>

    <section>
      <h2>RQ3: Minh họa dự đoán thực tế</h2>
      <div class="gallery">{gallery if gallery else "<p>Chưa có hình.</p>"}</div>
    </section>
  </main>
  <footer>TimeXer Head-Pose Forecast Project — NeurIPS 2024 reproduction</footer>
</body>
</html>"""

    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"Built GitHub Pages -> {DOCS / 'index.html'}")


if __name__ == "__main__":
    build()
