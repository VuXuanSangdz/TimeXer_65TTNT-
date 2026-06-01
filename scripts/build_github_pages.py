"""Build static GitHub Pages site from experiment results (text/tables only, no figures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _df_to_html(df: pd.DataFrame, table_id: str) -> str:
    return df.to_html(index=False, classes="data-table", table_id=table_id, border=0)


def _load_aggregate(exp_dir: Path, name: str) -> pd.DataFrame:
    agg_path = exp_dir / name
    if agg_path.exists():
        df = pd.read_csv(agg_path)
        if "MSE_mean" in df.columns:
            display = df.copy()
            display["MSE"] = display.apply(
                lambda r: f"{r['MSE_mean']:.4f} ± {r['MSE_std']:.4f}", axis=1
            )
            display["MAE"] = display.apply(
                lambda r: f"{r['MAE_mean']:.4f} ± {r['MAE_std']:.4f}", axis=1
            )
            display["RMSE"] = display.apply(
                lambda r: f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f}", axis=1
            )
            keep = [c for c in display.columns if c in df.columns or c in ("MSE", "MAE", "RMSE")]
            return display[keep]
        return df
    return pd.read_csv(exp_dir / name.replace("_aggregate", ""))


def build():
    DOCS.mkdir(parents=True, exist_ok=True)

    exp1_dir = RESULTS / "exp1_comparison"
    exp1_q = _load_aggregate(exp1_dir, "quantitative_comparison_aggregate.csv")
    exp1_qual = pd.read_csv(exp1_dir / "qualitative_comparison.csv")
    exp2 = _load_aggregate(RESULTS / "exp2_ablation", "ablation_quantitative_aggregate.csv")

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
    .notice {{ background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px; padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>TimeXer_65TTNT</h1>
  <p class="meta">
    Course reproduction of
    <a href="https://arxiv.org/abs/2402.19072">TimeXer (NeurIPS 2024)</a>.
    <a href="https://github.com/VuXuanSangdz/TimeXer_65TTNT-">GitHub Repository</a>
  </p>

  <div class="notice">
    <strong>Data provenance:</strong> CSV datasets in this project are <em>simulated</em> for reproducible pipeline validation.
    See <code>EXPERIMENTAL_SETUP.md</code> for full protocol. Metrics below are mean ± std over seeds 42, 123, 456.
  </div>

  <h2>Introduction</h2>
  <p>
    Forecast <strong>head yaw</strong> (private face biometric series) and <strong>CO₂ concentration</strong>
    (public weather-style benchmark) using exogenous variables. Compare TimeXer with iTransformer, PatchTST, TiDE, DLinear.
  </p>

  <h2>Experimental Setup</h2>
  <ul>
    <li><code>seq_len=96</code>, <code>pred_len=24</code></li>
    <li>Split: 70% train / 10% val / 20% test (chronological)</li>
    <li>Adam lr=1e-3, MSE loss, 12 epochs, early stopping patience=4</li>
    <li>Seeds: 42, 123, 456</li>
  </ul>

  <h2>Main Results</h2>

  <h3>RQ1 — Model Comparison (mean ± std)</h3>
  {_df_to_html(exp1_q, "rq1-quant")}

  <h3>RQ1 — Qualitative</h3>
  {_df_to_html(exp1_qual, "rq1-qual")}

  <h3>RQ2 — Ablation (mean ± std)</h3>
  {_df_to_html(exp2, "rq2-ablation")}

  <h2>RQ3 — Forecast Figures</h2>
  <p>PNG outputs: <code>results/exp3_visualization/</code> (12 files, not embedded here).</p>

  <h2>Limitations</h2>
  <ul>
    <li>Simulated data — not official Time-Series-Library benchmarks</li>
    <li>Simplified reimplementation — not official <code>run.py</code></li>
    <li>No formal statistical significance tests</li>
  </ul>

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
