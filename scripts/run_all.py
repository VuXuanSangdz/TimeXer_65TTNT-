"""Run all experiments sequentially."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generate_datasets import generate_private_dataset, generate_public_weather_subset


def main():
    print("=" * 60)
    print("Step 0: Generate datasets")
    print("=" * 60)
    generate_private_dataset(ROOT / "data" / "private")
    generate_public_weather_subset(ROOT / "data" / "public")

    print("\n" + "=" * 60)
    print("Step 1: Model comparison (RQ1)")
    print("=" * 60)
    from experiments.exp1_model_comparison import run as run_exp1
    run_exp1()

    print("\n" + "=" * 60)
    print("Step 2: Ablation study (RQ2)")
    print("=" * 60)
    from experiments.exp2_ablation import run as run_exp2
    run_exp2()

    print("\n" + "=" * 60)
    print("Step 3: Visualization (RQ3)")
    print("=" * 60)
    from experiments.exp3_visualization import run as run_exp3
    run_exp3()

    print("\n" + "=" * 60)
    print("Step 4: Build GitHub Pages")
    print("=" * 60)
    from scripts.build_github_pages import build
    build()

    print("\nAll experiments completed. See results/ and docs/")


if __name__ == "__main__":
    main()
