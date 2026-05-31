"""Build static GitHub Pages site from experiment results (text/tables only, no figures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _df_to_html(df: pd.DataFrame, table_id: str) -> str:
    return df.to_html(index=False, classes="data-table", table_id=table_id, border=0)


def build():
    DOCS.mkdir(parents=True, exist_ok=True)

    exp1_q = pd.read_csv(RESULTS / "exp1_comparison" / "quantitative_comparison.csv")
    exp1_qual = pd.read_csv(RESULTS / "exp1_comparison" / "qualitative_comparison.csv")
    exp2 = pd.read_csv(RESULTS / "exp2_ablation" / "ablation_quantitative.csv")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TimeXer_65TTNT</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; line-height: 1.6; color: #24292f; }}
    h1 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }}
    h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.25em; margin-top: 2rem; }}
    h3 {{ margin-top: 1.5rem; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f6f8fa; padding: 0.15em 0.35em; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; margin: 1rem 0; }}
    .data-table th {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 0.5rem; text-align: left; }}
    .data-table td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.5rem; }}
    .meta {{ color: #57606a; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <h1>TimeXer_65TTNT</h1>
  <p class="meta">
    Reproduction of
    <a href="https://arxiv.org/abs/2402.19072">TimeXer (NeurIPS 2024)</a>
    for forecasting with exogenous variables.
    <a href="https://github.com/VuXuanSangdz/TimeXer_65TTNT-">GitHub Repository</a>
  </p>

  <h2>Introduction</h2>
  <p>
    This project applies TimeXer to forecast <strong>head yaw</strong> from face biometric time series (private dataset)
    and <strong>CO2 concentration</strong> from weather data (public benchmark), using exogenous variables to improve prediction.
  </p>

  <h2>Overall Architecture</h2>
  <p>
    TimeXer uses patch-level representations for endogenous variables and variate-level representations for exogenous variables,
    connected by a learnable global token via self-attention and cross-attention.
  </p>

  <h2>Main Results</h2>

  <h3>RQ1 — Model Comparison (Quantitative)</h3>
  {_df_to_html(exp1_q, "rq1-quant")}

  <h3>RQ1 — Model Comparison (Qualitative)</h3>
  {_df_to_html(exp1_qual, "rq1-qual")}

  <h3>RQ2 — Ablation Study</h3>
  {_df_to_html(exp2, "rq2-ablation")}

  <h2>Usage</h2>
  <pre>pip install -r requirements.txt
python scripts/run_all.py</pre>

  <h2>Citation</h2>
  <pre>@article{{wang2024timexer,
  title={{Timexer: Empowering transformers for time series forecasting with exogenous variables}},
  author={{Wang, Yuxuan and others}},
  journal={{Advances in Neural Information Processing Systems}},
  year={{2024}}
}}</pre>
</body>
</html>"""

    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"Built GitHub Pages -> {DOCS / 'index.html'}")


if __name__ == "__main__":
    build()
