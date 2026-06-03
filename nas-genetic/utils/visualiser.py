"""
utils/visualiser.py
===================
Plotting utilities for the NAS project report.

Functions:
  - plot_convergence()     : Fitness vs. generation curve with shaded band
  - plot_architecture()    : Block diagram of a decoded architecture
  - plot_gene_heatmap()    : Heatmap of gene distributions in top architectures
  - plot_comparison_bar()  : Bar chart comparing GA vs. baselines
"""

import os
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("muted")


# ---------------------------------------------------------------------------
# 1. Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    log_path: str,
    out_path: str = None,
    show: bool = False,
) -> str:
    """
    Plot best and mean fitness per generation with a shaded min/max band.

    Args:
        log_path: Path to a run_*.csv file or a directory containing csv files
                  (in which case the most-recent file is used).
        out_path: Where to save the figure. Defaults to same dir as csv.
        show:     If True, call plt.show() after saving.

    Returns:
        Path to the saved figure.
    """
    # Resolve CSV path
    if os.path.isdir(log_path):
        csvs = sorted(
            [f for f in os.listdir(log_path) if f.endswith(".csv")],
            reverse=True,
        )
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {log_path}")
        log_path = os.path.join(log_path, csvs[0])

    df = pd.read_csv(log_path)
    gens = df["gen"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    if "std_fitness" in df.columns:
        ax.fill_between(gens, df["mean_fitness"] - df["std_fitness"], df["mean_fitness"] + df["std_fitness"],
                        alpha=0.2, color=PALETTE[0], label="Mean ± 1 Std")
    else:
        ax.fill_between(gens, df["min_fitness"], df["best_fitness"],
                        alpha=0.2, color=PALETTE[0], label="Min–Best range")
                        
    ax.plot(gens, df["best_fitness"], "o-", color=PALETTE[0],
            linewidth=2, label="Best fitness")
    ax.plot(gens, df["mean_fitness"], "s--", color=PALETTE[1],
            linewidth=1.5, label="Mean fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("GA-NAS Convergence")
    ax.legend()
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if out_path is None:
        out_path = log_path.replace(".csv", "_convergence.png")
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Convergence plot saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. Architecture block diagram
# ---------------------------------------------------------------------------

def plot_architecture(
    arch: dict,
    out_path: str = "experiments/best_architectures/architecture.png",
    show: bool = False,
) -> str:
    """
    Draw a simple block diagram of the given architecture dict.

    Args:
        arch:     Decoded architecture dict from chromosome.decode()
        out_path: Output PNG path
        show:     Whether to display interactively

    Returns:
        Path to the saved figure.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_blocks = arch["num_blocks"]
    filter_keys = ["filters_1", "filters_2", "filters_3"]
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, num_blocks + 3))

    fig, ax = plt.subplots(figsize=(max(8, num_blocks * 2 + 4), 4))
    ax.set_xlim(0, num_blocks + 4)
    ax.set_ylim(0, 3)
    ax.axis("off")

    x = 0.5

    def _draw_block(ax, x, label, color, width=1.0):
        rect = mpatches.FancyBboxPatch(
            (x - 0.4, 0.8), 0.8 * width, 1.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, 1.5, label, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", wrap=True)

    # Input
    _draw_block(ax, x, "Input\n3×32×32", colors[0])
    x += 1.2

    # Conv blocks
    for i in range(num_blocks):
        fkey = filter_keys[min(i, len(filter_keys) - 1)]
        filters = arch.get(fkey, arch["filters_3"])
        label = (
            f"Conv{arch['kernel_size']}×{arch['kernel_size']}\n"
            f"{filters}ch\n"
            f"+{arch['pooling'].capitalize()}Pool"
        )
        ax.annotate("", xy=(x - 0.4, 1.5), xytext=(x - 0.85, 1.5),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        _draw_block(ax, x, label, colors[i + 1])
        x += 1.2

    # FC head
    ax.annotate("", xy=(x - 0.4, 1.5), xytext=(x - 0.85, 1.5),
                arrowprops=dict(arrowstyle="->", color="gray"))
    _draw_block(ax, x, f"FC\n{arch['fc_hidden']}", colors[-2])
    x += 1.2
    ax.annotate("", xy=(x - 0.4, 1.5), xytext=(x - 0.85, 1.5),
                arrowprops=dict(arrowstyle="->", color="gray"))
    _draw_block(ax, x, "Output\n10 classes", colors[-1])

    title = (
        f"Blocks={num_blocks}  "
        f"Act={arch['activation']}  "
        f"BN={arch['batch_norm']}  "
        f"Drop={arch['dropout']}"
    )
    ax.set_title(title, pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Architecture diagram saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 3. Gene heatmap
# ---------------------------------------------------------------------------

def plot_gene_heatmap(
    population: list,
    fitnesses: list,
    top_k: int = None,
    out_path: str = "experiments/gene_heatmap.png",
    show: bool = False,
) -> str:
    """
    Heatmap of gene index distributions across the population (or top-k).

    Args:
        population: List of chromosomes
        fitnesses:  Corresponding fitness values
        top_k:      If set, only plot top-k individuals
        out_path:   Output PNG path
        show:       Whether to display interactively

    Returns:
        Path to saved figure.
    """
    from ga.chromosome import GENE_NAMES, NUM_GENES

    if top_k is not None:
        paired = sorted(zip(fitnesses, population), reverse=True)[:top_k]
        population = [p for _, p in paired]

    matrix = np.array(population, dtype=float)  # shape: (N, NUM_GENES)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        matrix.T,
        ax=ax,
        yticklabels=GENE_NAMES,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Gene index"},
    )
    title = f"Gene Distribution (n={len(population)})"
    if top_k is not None:
        title += f" — Top {top_k}"
    ax.set_title(title)
    ax.set_xlabel("Individual")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Gene heatmap saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 4. Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    results: dict,
    out_path: str = "experiments/comparison.png",
    show: bool = False,
) -> str:
    """
    Bar chart comparing test accuracies of GA-NAS vs. baselines.

    Args:
        results: Dict of {method_name: test_accuracy}
                 e.g. {"GA-NAS": 0.85, "Random Search": 0.81, "ResNet": 0.83}
        out_path: Output PNG path
        show:     Whether to display interactively

    Returns:
        Path to saved figure.
    """
    methods = list(results.keys())
    accs = [results[m] * 100 for m in methods]
    colors = [PALETTE[0] if "GA" in m else PALETTE[2] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("GA-NAS vs. Baselines on CIFAR-10")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Comparison bar chart saved: {out_path}")
    return out_path
