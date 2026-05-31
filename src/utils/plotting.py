from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_forecast(
    history: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str,
    save_path: str | Path,
    seq_len: int = 96,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    t_hist = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + len(ground_truth))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_hist, history, label="Input history", color="#2563eb", linewidth=1.5)
    ax.plot(t_pred, ground_truth, label="Ground truth", color="#16a34a", linewidth=2)
    ax.plot(t_pred, prediction, "--", label="Prediction", color="#dc2626", linewidth=2)
    ax.axvline(seq_len, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    history: np.ndarray,
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    save_path: str | Path,
    seq_len: int = 96,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    t_hist = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + len(ground_truth))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t_hist, history, color="#94a3b8", label="History", linewidth=1.2)
    ax.plot(t_pred, ground_truth, color="black", label="Ground truth", linewidth=2.2)
    colors = ["#dc2626", "#2563eb", "#9333ea", "#ea580c", "#0891b2"]
    for (name, pred), c in zip(predictions.items(), colors):
        ax.plot(t_pred, pred, "--", label=name, color=c, linewidth=1.8)
    ax.axvline(seq_len, color="gray", linestyle=":", alpha=0.7)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(attn_matrix: np.ndarray, exo_names: list[str], save_path: str | Path, title: str = ""):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sns.heatmap(attn_matrix.reshape(1, -1), xticklabels=exo_names, yticklabels=["Global token"], cmap="YlOrRd", ax=ax)
    ax.set_title(title or "Cross-attention: Global token → Exogenous variables")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
