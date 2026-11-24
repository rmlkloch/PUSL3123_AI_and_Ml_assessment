"""Reusable visualization and performance comparison utilities for CIFAR-10 experiments.

Provides unified 300-DPI Matplotlib plotting functions and tabular summary helpers
for Exercise 1 (MLP), Exercise 2 (CNN), Exercise 3 (Regularisation), and
Exercise 4 (Network Width & Depth).
"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    model_title: str = "CIFAR-10 CNN",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot side-by-side Training vs. Validation Loss and Accuracy curves.

    Follows the unified team visual standard (14x5 inches, 300 DPI, circle/square
    markers, dashed grid lines) established for the final experimental report.

    Args:
        train_losses: Per-epoch training cross-entropy loss values.
        val_losses: Per-epoch validation cross-entropy loss values.
        train_accs: Per-epoch training accuracy percentages (0-100).
        val_accs: Per-epoch validation accuracy percentages (0-100).
        model_title: Prefix used in subplot titles (e.g., ``"CIFAR-10 CNN"``).
        save_path: Optional file path to save the 300-DPI PNG figure.
        show: Whether to call ``plt.show()``.

    Returns:
        The generated Matplotlib ``Figure`` object.
    """
    num_epochs = len(train_losses)
    epochs_range = range(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left subplot: Training vs. Validation Loss
    ax1.plot(epochs_range, train_losses, marker="o", linewidth=2, label="Train Loss")
    ax1.plot(epochs_range, val_losses, marker="s", linewidth=2, label="Validation Loss")
    ax1.set_title(
        f"{model_title}: Training vs. Validation Loss",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_xticks(list(epochs_range))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Right subplot: Training vs. Validation Accuracy
    ax2.plot(epochs_range, train_accs, marker="o", linewidth=2, label="Train Accuracy")
    ax2.plot(epochs_range, val_accs, marker="s", linewidth=2, label="Validation Accuracy")
    ax2.set_title(
        f"{model_title}: Training vs. Validation Accuracy",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xticks(list(epochs_range))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")

    if show:
        plt.show()

    return fig


def plot_mlp_vs_cnn_comparison(
    mlp_history: Dict[str, Sequence[float]],
    cnn_history: Dict[str, Sequence[float]],
    mlp_summary: Dict[str, float],
    cnn_summary: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a 2x2 comparative analysis figure of the Exercise 1 MLP vs. Exercise 2 CNN.

    Subplots:
        1. Validation Loss curves across epochs (MLP vs. CNN).
        2. Validation Accuracy curves across epochs (MLP vs. CNN).
        3. Grouped bar chart of Train, Validation (Best), and Test Accuracy (%).
        4. Bar chart comparing Trainable Parameter Counts (log/linear scale annotation).

    Args:
        mlp_history: Dict with keys ``train_losses``, ``val_losses``, ``train_accs``, ``val_accs``.
        cnn_history: Dict with keys ``train_losses``, ``val_losses``, ``train_accs``, ``val_accs``.
        mlp_summary: Dict with ``train_acc``, ``best_val_acc``, ``test_acc``, ``params``.
        cnn_summary: Dict with ``train_acc``, ``best_val_acc``, ``test_acc``, ``params``.
        save_path: Optional destination path for the 300-DPI PNG figure.
        show: Whether to display the figure with ``plt.show()``.

    Returns:
        The generated Matplotlib ``Figure`` object.
    """
    num_epochs = len(cnn_history["val_losses"])
    epochs_range = list(range(1, num_epochs + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    (ax_loss, ax_acc), (ax_bars, ax_params) = axes

    # 1. Validation Loss Comparison
    ax_loss.plot(
        epochs_range,
        mlp_history["val_losses"],
        marker="o",
        linewidth=2,
        linestyle="--",
        color="#d95f02",
        label="MLP Validation Loss",
    )
    ax_loss.plot(
        epochs_range,
        mlp_history["train_losses"],
        linewidth=1.5,
        linestyle=":",
        color="#d95f02",
        alpha=0.7,
        label="MLP Train Loss",
    )
    ax_loss.plot(
        epochs_range,
        cnn_history["val_losses"],
        marker="s",
        linewidth=2,
        color="#1b9e77",
        label="CNN Validation Loss",
    )
    ax_loss.plot(
        epochs_range,
        cnn_history["train_losses"],
        linewidth=1.5,
        linestyle=":",
        color="#1b9e77",
        alpha=0.7,
        label="CNN Train Loss",
    )
    ax_loss.set_title("MLP vs. CNN: Training & Validation Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax_loss.set_xticks(epochs_range)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=9.5)

    # 2. Validation Accuracy Comparison
    ax_acc.plot(
        epochs_range,
        mlp_history["val_accs"],
        marker="o",
        linewidth=2,
        linestyle="--",
        color="#d95f02",
        label="MLP Validation Accuracy",
    )
    ax_acc.plot(
        epochs_range,
        mlp_history["train_accs"],
        linewidth=1.5,
        linestyle=":",
        color="#d95f02",
        alpha=0.7,
        label="MLP Train Accuracy",
    )
    ax_acc.plot(
        epochs_range,
        cnn_history["val_accs"],
        marker="s",
        linewidth=2,
        color="#1b9e77",
        label="CNN Validation Accuracy",
    )
    ax_acc.plot(
        epochs_range,
        cnn_history["train_accs"],
        linewidth=1.5,
        linestyle=":",
        color="#1b9e77",
        alpha=0.7,
        label="CNN Train Accuracy",
    )
    ax_acc.set_title("MLP vs. CNN: Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xticks(epochs_range)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(fontsize=9.5)

    # 3. Partition Accuracy Comparison (Grouped Bar Chart)
    metrics_labels = ["Final Train Acc", "Best Val Acc", "Final Test Acc"]
    mlp_vals = [
        mlp_summary["train_acc"],
        mlp_summary["best_val_acc"],
        mlp_summary["test_acc"],
    ]
    cnn_vals = [
        cnn_summary["train_acc"],
        cnn_summary["best_val_acc"],
        cnn_summary["test_acc"],
    ]

    x = np.arange(len(metrics_labels))
    width = 0.35
    bars_mlp = ax_bars.bar(x - width / 2, mlp_vals, width, label="3-Layer MLP (Ex 1)", color="#d95f02", alpha=0.88)
    bars_cnn = ax_bars.bar(x + width / 2, cnn_vals, width, label="3-Layer CNN (Ex 2)", color="#1b9e77", alpha=0.88)

    ax_bars.set_title("MLP vs. CNN: Classification Accuracy by Split", fontsize=13, fontweight="bold")
    ax_bars.set_ylabel("Accuracy (%)", fontsize=11)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(metrics_labels, fontsize=11)
    ax_bars.set_ylim(0, 105)
    ax_bars.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bars.legend(fontsize=10)

    for bar in list(bars_mlp) + list(bars_cnn):
        height = bar.get_height()
        ax_bars.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # 4. Model Complexity (Trainable Parameters)
    models = ["3-Layer MLP\n(Exercise 1)", "3-Layer CNN\n(Exercise 2)"]
    param_counts = [int(mlp_summary["params"]), int(cnn_summary["params"])]
    colors = ["#d95f02", "#1b9e77"]

    bars_p = ax_params.bar(models, param_counts, color=colors, width=0.45, alpha=0.88)
    ax_params.set_title("Model Complexity: Trainable Parameter Count", fontsize=13, fontweight="bold")
    ax_params.set_ylabel("Number of Trainable Parameters", fontsize=11)
    ax_params.set_ylim(0, max(param_counts) * 1.18)
    ax_params.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars_p, param_counts):
        ax_params.annotate(
            f"{count:,} params",
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved MLP vs. CNN comparison figure to {save_path}")

    if show:
        plt.show()

    return fig


def build_comparison_table(
    mlp_summary: Dict[str, float],
    cnn_summary: Dict[str, float],
) -> pd.DataFrame:
    """Create a structured Pandas DataFrame comparing Exercise 1 MLP and Exercise 2 CNN.

    Args:
        mlp_summary: Dictionary of summary metrics for the 3-layer MLP.
        cnn_summary: Dictionary of summary metrics for the 3-layer CNN.

    Returns:
        A formatted ``pd.DataFrame`` ready for display or export.
    """
    mlp_gap = mlp_summary["train_acc"] - mlp_summary["test_acc"]
    cnn_gap = cnn_summary["train_acc"] - cnn_summary["test_acc"]
    param_ratio = mlp_summary["params"] / cnn_summary["params"]

    rows: List[Dict[str, str]] = [
        {
            "Metric": "Trainable Layers",
            "Exercise 1: 3-Layer MLP": "3 (3 Fully Connected)",
            "Exercise 2: 3-Layer CNN": "3 (2 Conv2d + 1 Linear)",
            "Difference (CNN vs. MLP)": "Spatial Conv Feature Extractor",
        },
        {
            "Metric": "Trainable Parameters",
            "Exercise 1: 3-Layer MLP": f"{int(mlp_summary['params']):,}",
            "Exercise 2: 3-Layer CNN": f"{int(cnn_summary['params']):,}",
            "Difference (CNN vs. MLP)": f"-{int(mlp_summary['params'] - cnn_summary['params']):,} ({param_ratio:.1f}x fewer)",
        },
        {
            "Metric": "Final Training Loss (Epoch 20)",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['final_train_loss']:.4f}",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['final_train_loss']:.4f}",
            "Difference (CNN vs. MLP)": f"{cnn_summary['final_train_loss'] - mlp_summary['final_train_loss']:+.4f}",
        },
        {
            "Metric": "Lowest Validation Loss",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['min_val_loss']:.4f}",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['min_val_loss']:.4f}",
            "Difference (CNN vs. MLP)": f"{cnn_summary['min_val_loss'] - mlp_summary['min_val_loss']:+.4f}",
        },
        {
            "Metric": "Final Validation Loss (Epoch 20)",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['final_val_loss']:.4f}",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['final_val_loss']:.4f}",
            "Difference (CNN vs. MLP)": f"{cnn_summary['final_val_loss'] - mlp_summary['final_val_loss']:+.4f}",
        },
        {
            "Metric": "Final Training Accuracy (Epoch 20)",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['train_acc']:.2f}%",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['train_acc']:.2f}%",
            "Difference (CNN vs. MLP)": f"{cnn_summary['train_acc'] - mlp_summary['train_acc']:+.2f}%",
        },
        {
            "Metric": "Best Validation Accuracy",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['best_val_acc']:.2f}%",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['best_val_acc']:.2f}%",
            "Difference (CNN vs. MLP)": f"{cnn_summary['best_val_acc'] - mlp_summary['best_val_acc']:+.2f}%",
        },
        {
            "Metric": "Unseen Test Accuracy (10,000 imgs)",
            "Exercise 1: 3-Layer MLP": f"{mlp_summary['test_acc']:.2f}%",
            "Exercise 2: 3-Layer CNN": f"{cnn_summary['test_acc']:.2f}%",
            "Difference (CNN vs. MLP)": f"{cnn_summary['test_acc'] - mlp_summary['test_acc']:+.2f}%",
        },
        {
            "Metric": "Generalisation Gap (Train - Test Acc)",
            "Exercise 1: 3-Layer MLP": f"{mlp_gap:.2f}%",
            "Exercise 2: 3-Layer CNN": f"{cnn_gap:.2f}%",
            "Difference (CNN vs. MLP)": f"{cnn_gap - mlp_gap:+.2f}%",
        },
    ]

    return pd.DataFrame(rows)
