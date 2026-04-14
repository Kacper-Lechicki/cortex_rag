import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.manifold import TSNE

# Dark background palette
_BG_COLOR = "#1a1a2e"
_TEXT_COLOR = "#e0e0e0"

_BAR_COLORS = [
    "#4cc9f0", "f72585", "#7209b7", "#3a0ca3", "#4361ee"
]


def visualize_documents(
    embeddings: list[list[float]],
    metadatas: list[dict],
    output_path: Path = Path("cortex_docs.png")
) -> Path:
    """
    Create a t-SNE scatter plot of all document embeddings.

    Each point represents one chunk. Points are colored by domain (source website).
    Clusters of points with the same color indicate thematically similar chunks.

    Args:
        embeddings: List of embedding vectors (one per chunk). Shape: [N, 384].
        metadatas: Parallel list of metadata dicts. Must contain 'domain' key.
        ooutput_path: Where to save the PNG image.

    Returns:
        Path to saved image.

    Raises:
        ValueError: If fewer than 10 chunks are provided (t-SNE needs enough points).
    """

    n = len(embeddings)

    if n < 10:
        raise ValueError(
            f"Need at least 10 chunks for t-SNE, got {n}. "
            "Add more articles with 'cortex add'."
        )

    # Convert to numpy array: shape (N, 384)
    # numpy arrays are more efficient than python lists for numerical operations
    X = np.array(embeddings, dtype=np.float32)

    # t-SNE hyperparameters:
    # - perplexity: rougly "number of neighbours" - lower for small datasets
    # - n_iter: optimization steps - 1000 is usually enough for convergence
    # - random_state: seed for reproducibility
    perplexity = min(30, n - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42,
        init="pca" # PCA initialization is more stable than random
    )

    # fit_transform: fits the model AND returns the 2D coordinates
    # Shape: (N, 2) — each row is [x, y] for one chunk
    coords_2d = tsne.fit_transform(X)

    # Group by domain for coloring
    domains = [m.get("domain", "unknown") for m in metadatas]
    unique_domains = sorted(set(domains))
    color_map = plt.cm.get_cmap("tab10", len(unique_domains))
    domain_to_color = {d: color_map(i) for i, d in enumerate(unique_domains)}
    point_colors = [domain_to_color[d] for d in domains]

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Scatter plot: one point per chunk
    ax.scatter(
        coords_2d[:, 0],   # x coordinates (all rows, column 0)
        coords_2d[:, 1],   # y coordinates (all rows, column 1)
        c=point_colors,
        s=10,              # point size
        alpha=0.75,        # slight transparency for overlapping points
        linewidths=0,
    )

    # Legend: one entry per domain
    legend_patches = [
        mpatches.Patch(color=domain_to_color[d], label=d)
        for d in unique_domains
    ]

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        framealpha=0.2,
        labelcolor=_TEXT_COLOR,
        fontsize=9,
    )

    # Labels and styling
    ax.set_title(
        f"Knowledge Base — Document Map (t-SNE)  [{n} chunks]",
        color=_TEXT_COLOR,
        fontsize=13,
        pad=14,
    )

    ax.tick_params(colors=_TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)  # Release memory — important for long-running processes

    return output_path


def visualize_metrics(
    reports: list[dict],
    output_path: Path = Path("cortex_metrics.png"),
) -> Path:
    """
    Create a grouped bar chart comparing MRR and Hit Rate across experiments.

    Each experiment is one group of bars. Running this after each configuration
    change shows visually how your tweaks affect retrieval quality.

    Args:
        reports: List of report dicts from EvalReport.load_all().
                 Each dict has keys: 'experiment', 'mrr', 'hit_rate'.
        output_path: Where to save the PNG image.

    Returns:
        Path to saved image.
    """

    if not reports:
        raise ValueError("No reports to visualize.")

    experiment_names = [r["experiment"] for r in reports]
    mrr_scores = [r["mrr"] for r in reports]
    hit_rate_scores = [r["hit_rate"] for r in reports]
    n_queries = [r.get("n_queries", "?") for r in reports]

    # x: array of integer positions [0, 1, 2, ...]
    x = np.arange(len(experiment_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Two groups of bars: MRR (left offset) and Hit Rate (right offset)
    bars_mrr = ax.bar(
        x - bar_width / 2, mrr_scores, bar_width,
        label="MRR", color=_BAR_COLORS[0], alpha=0.9,
    )

    bars_hr = ax.bar(
        x + bar_width / 2, hit_rate_scores, bar_width,
        label="Hit Rate", color=_BAR_COLORS[1], alpha=0.9,
    )

    # Add score labels on top of each bar
    for bar_group in [bars_mrr, bars_hr]:
        for bar in bar_group:
            height = bar.get_height()

            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=_TEXT_COLOR,
                fontsize=9,
            )

    # Reference lines for quality thresholds
    ax.axhline(y=0.6, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.8, color="#777", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(x) - 0.1, 0.61, "good (0.6)", color="#777", fontsize=8, ha="right")
    ax.text(len(x) - 0.1, 0.81, "excellent (0.8)", color="#777", fontsize=8, ha="right")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=20, ha="right", color=_TEXT_COLOR)
    ax.set_ylabel("Score", color=_TEXT_COLOR)
    ax.set_title("RAG Evaluation — Experiment Comparison", color=_TEXT_COLOR, fontsize=13, pad=14)
    ax.legend(framealpha=0.2, labelcolor=_TEXT_COLOR)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)

    return output_path
