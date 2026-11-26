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


def build_regularisation_comparison_table(
    summaries: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Create a structured Pandas DataFrame comparing all Exercise 3 regularization setups.

    Args:
        summaries: Dict mapping model names to metric dicts containing:
            ``params``, ``final_train_loss``, ``min_val_loss``, ``final_val_loss``,
            ``train_acc``, ``best_val_acc``, ``test_acc``.

    Returns:
        Formatted DataFrame with comprehensive generalization comparison.
    """
    rows: List[Dict[str, str]] = []
    baseline_test_acc = summaries.get("Baseline CNN", {}).get("test_acc", 71.20)

    for name, s in summaries.items():
        train_acc = s["train_acc"]
        test_acc = s["test_acc"]
        gen_gap = train_acc - test_acc
        delta = test_acc - baseline_test_acc

        rows.append(
            {
                "Model Variant": name,
                "Parameters": f"{int(s['params']):,}",
                "Final Train Loss": f"{s['final_train_loss']:.4f}",
                "Lowest Val Loss": f"{s['min_val_loss']:.4f}",
                "Final Val Loss": f"{s['final_val_loss']:.4f}",
                "Final Train Acc": f"{train_acc:.2f}%",
                "Best Val Acc": f"{s['best_val_acc']:.2f}%",
                "Test Acc (10k)": f"{test_acc:.2f}%",
                "Generalisation Gap": f"{gen_gap:.2f}%",
                "vs. Baseline": f"{delta:+.2f}%" if name != "Baseline CNN" else "Baseline",
            }
        )

    return pd.DataFrame(rows)


def build_scaling_comparison_table(
    summaries: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Create a structured Pandas DataFrame comparing all Exercise 4 Width and Depth variants.

    Args:
        summaries: Dict mapping model names to metric dicts containing:
            ``dimension``, ``type``, ``channels``, ``params``, ``final_train_loss``,
            ``min_val_loss``, ``final_val_loss``, ``train_acc``, ``best_val_acc``, ``test_acc``.

    Returns:
        Formatted DataFrame with comprehensive network scaling comparisons.
    """
    rows: List[Dict[str, str]] = []
    baseline_test_acc = summaries.get("Width: Baseline [32, 64]", {}).get("test_acc", 71.20)

    for name, s in summaries.items():
        train_acc = s["train_acc"]
        test_acc = s["test_acc"]
        gen_gap = train_acc - test_acc
        delta = test_acc - baseline_test_acc

        rows.append(
            {
                "Scaling Variant": name,
                "Scaling Dimension": s.get("dimension", "Width"),
                "Channel Structure": s.get("channels", ""),
                "Trainable Parameters": f"{int(s['params']):,}",
                "Final Train Loss": f"{s['final_train_loss']:.4f}",
                "Lowest Val Loss": f"{s['min_val_loss']:.4f}",
                "Final Val Loss": f"{s['final_val_loss']:.4f}",
                "Final Train Acc": f"{train_acc:.2f}%",
                "Best Val Acc": f"{s['best_val_acc']:.2f}%",
                "Test Acc (10k)": f"{test_acc:.2f}%",
                "Generalisation Gap": f"{gen_gap:.2f}%",
                "vs. Baseline": f"{delta:+.2f}%" if "Baseline" not in name else "Baseline",
            }
        )

    return pd.DataFrame(rows)


def plot_scaling_comparison(
    width_summaries: Dict[str, Dict[str, float]],
    depth_summaries: Dict[str, Dict[str, float]],
    width_histories: Dict[str, Dict[str, Sequence[float]]],
    depth_histories: Dict[str, Dict[str, Sequence[float]]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a comprehensive 2x2 comparison of network width and depth scaling.

    Subplots:
        1. Validation Accuracy Trajectories across Width Variations.
        2. Validation Accuracy Trajectories across Depth Variations.
        3. Parameter Count vs. Test Accuracy (Scaling Efficiency Trade-off).
        4. Generalisation Gap across all 6 Architecture Variations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    (ax_w, ax_d), (ax_tradeoff, ax_gap) = axes

    # Subplot 1: Width Validation Acc Trajectories
    w_palette = {"Width: Narrow [16, 32]": "#377eb8", "Width: Baseline [32, 64]": "#4daf4a", "Width: Wide [64, 128]": "#984ea3"}
    for name, hist in width_histories.items():
        epochs = list(range(1, len(hist["val_accs"]) + 1))
        color = w_palette.get(name, "#333333")
        ax_w.plot(epochs, hist["val_accs"], marker="o", markersize=4, linewidth=2, label=name, color=color)
    ax_w.set_title("Network Width: Validation Accuracy Across Epochs", fontsize=12, fontweight="bold")
    ax_w.set_xlabel("Epoch", fontsize=10)
    ax_w.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax_w.grid(True, linestyle="--", alpha=0.5)
    ax_w.legend(fontsize=9, loc="lower right")

    # Subplot 2: Depth Validation Acc Trajectories
    d_palette = {"Depth: 2 Layers (1 Conv)": "#ff7f00", "Depth: 3 Layers (2 Conv - Base)": "#4daf4a", "Depth: 4 Layers (3 Conv)": "#e41a1c"}
    for name, hist in depth_histories.items():
        epochs = list(range(1, len(hist["val_accs"]) + 1))
        color = d_palette.get(name, "#333333")
        ax_d.plot(epochs, hist["val_accs"], marker="s", markersize=4, linewidth=2, label=name, color=color)
    ax_d.set_title("Network Depth: Validation Accuracy Across Epochs", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Epoch", fontsize=10)
    ax_d.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9, loc="lower right")

    # Subplot 3: Parameter Count vs. Test Accuracy (Trade-off Scatter / Curve)
    all_models = {**width_summaries, **depth_summaries}
    params = [s["params"] for s in all_models.values()]
    accs = [s["test_acc"] for s in all_models.values()]
    labels = list(all_models.keys())

    # Width points (blue/purple) and Depth points (orange/red)
    for name, s in all_models.items():
        p, acc = s["params"], s["test_acc"]
        if "Width" in name:
            marker, color = "o", w_palette.get(name, "#377eb8")
        else:
            marker, color = "s", d_palette.get(name, "#e41a1c")
        ax_tradeoff.scatter(p, acc, s=120, color=color, marker=marker, zorder=5, label=name)
        ax_tradeoff.annotate(
            f"{name}\n({p:,} p, {acc:.1f}%)",
            xy=(p, acc),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="semibold",
        )

    ax_tradeoff.set_title("Scaling Efficiency: Model Parameters vs. Test Accuracy", fontsize=12, fontweight="bold")
    ax_tradeoff.set_xlabel("Trainable Parameters", fontsize=10)
    ax_tradeoff.set_ylabel("Unseen Test Accuracy (%)", fontsize=10)
    ax_tradeoff.set_xlim(10000, 180000)
    ax_tradeoff.set_ylim(63, 76)
    ax_tradeoff.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Generalisation Gap across all 6 models
    gaps = [s["train_acc"] - s["test_acc"] for s in all_models.values()]
    colors = [w_palette.get(n, d_palette.get(n, "#555555")) for n in labels]
    short_labels = [n.replace("Width: ", "W: ").replace("Depth: ", "D: ") for n in labels]

    bars = ax_gap.bar(range(len(labels)), gaps, color=colors, alpha=0.85, width=0.5)
    ax_gap.set_title("Generalisation Gap (Final Train Acc − Test Acc)", fontsize=12, fontweight="bold")
    ax_gap.set_ylabel("Gap (percentage points)", fontsize=10)
    ax_gap.set_xticks(range(len(labels)))
    ax_gap.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=9)
    ax_gap.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, gap in zip(bars, gaps):
        ax_gap.annotate(
            f"{gap:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, gap),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved scaling comparison figure to {save_path}")

    if show:
        plt.show()

    return fig


def plot_regularisation_comparison(
    histories: Dict[str, Dict[str, Sequence[float]]],
    summaries: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a comprehensive 2x2 comparison of regularization techniques.

    Subplots:
        1. Training vs. Validation Loss trajectories across all configurations.
        2. Validation Accuracy across 20 epochs.
        3. Train vs. Test Generalisation Gap bar chart.
        4. Final Test Accuracy improvement over baseline.
    """
    palette = {
        "Baseline CNN": "#e41a1c",
        "Dropout (p=0.5)": "#377eb8",
        "Data Augmentation": "#4daf4a",
        "Dropout + Data Aug": "#984ea3",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    (ax_loss, ax_acc), (ax_gap, ax_summary) = axes

    # Subplot 1: Validation Loss
    for name, hist in histories.items():
        epochs = list(range(1, len(hist["val_losses"]) + 1))
        color = palette.get(name, "#333333")
        ax_loss.plot(epochs, hist["val_losses"], marker="o", markersize=4, linewidth=2, label=f"{name} (Val)", color=color)
        ax_loss.plot(epochs, hist["train_losses"], linestyle=":", linewidth=1.5, alpha=0.6, label=f"{name} (Train)", color=color)

    ax_loss.set_title("Training & Validation Loss Over 20 Epochs", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=8, loc="upper right", ncol=2)

    # Subplot 2: Validation Accuracy
    for name, hist in histories.items():
        epochs = list(range(1, len(hist["val_accs"]) + 1))
        color = palette.get(name, "#333333")
        ax_acc.plot(epochs, hist["val_accs"], marker="s", markersize=4, linewidth=2, label=name, color=color)

    ax_acc.set_title("Validation Accuracy Trajectory (%)", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(fontsize=9, loc="lower right")

    # Subplot 3: Generalisation Gap (Train Acc - Test Acc)
    names = list(summaries.keys())
    train_accs = [summaries[n]["train_acc"] for n in names]
    test_accs = [summaries[n]["test_acc"] for n in names]
    gaps = [tr - te for tr, te in zip(train_accs, test_accs)]

    bar_colors = [palette.get(n, "#555555") for n in names]
    bars = ax_gap.bar(names, gaps, color=bar_colors, alpha=0.85, width=0.5)
    ax_gap.set_title("Generalisation Gap (Final Train Acc − Test Acc)", fontsize=12, fontweight="bold")
    ax_gap.set_ylabel("Generalisation Gap (percentage points)", fontsize=10)
    ax_gap.set_xticks(range(len(names)))
    ax_gap.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)
    ax_gap.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar, gap in zip(bars, gaps):
        ax_gap.annotate(
            f"{gap:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, gap),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    # Subplot 4: Final Test Accuracy Breakdown
    x = np.arange(len(names))
    width = 0.35
    b_val = ax_summary.bar(x - width/2, [summaries[n]["best_val_acc"] for n in names], width, label="Best Val Acc", color="#3182bd", alpha=0.85)
    b_test = ax_summary.bar(x + width/2, test_accs, width, label="Unseen Test Acc", color="#2ca02c", alpha=0.85)

    ax_summary.set_title("Best Validation vs. Final Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax_summary.set_ylabel("Accuracy (%)", fontsize=10)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)
    ax_summary.set_ylim(50, 85)
    ax_summary.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_summary.legend(fontsize=9.5, loc="lower right")

    for bar in list(b_val) + list(b_test):
        h = bar.get_height()
        ax_summary.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved regularisation comparison figure to {save_path}")

    if show:
        plt.show()

    return fig

