import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def visualization_channels(targets, preds, save_path=None, channels=None, ylim=None, channel_names=None):
    """
    targets: (1, len, channels)
    preds: (1, len, channels)
    channel_names: optional list of names for each channel; if None, use sequential numbering.
    """
    assert targets.shape == preds.shape, "targets and preds must have the same shape"
    assert targets.shape[0] == 1, "targets and preds must have batch size of 1"
    assert preds.shape[0] == 1, "targets and preds must have batch size of 1"

    if channels is None:
        num_panels = targets.shape[-1]
        indices = list(range(num_panels))
    else:
        num_panels = len(channels)
        indices = list(channels)

    if channel_names is None:
        channel_names = [f"channel {i + 1}" for i in range(num_panels)]
    else:
        channel_names = list(channel_names)
        if len(channel_names) != num_panels:
            raise ValueError(
                f"channel_names length ({len(channel_names)}) must match number of panels ({num_panels})"
            )

    plt.figure(figsize=(15, 4 * num_panels))
    for i, idx in enumerate(indices):
        plot_target = targets.squeeze(0)[:, idx]
        plot_pred = preds.squeeze(0)[:, idx]

        plt.subplot(num_panels * 2, 1, i * 2 + 1)
        plt.plot(plot_target, color="#779a92")
        plt.gca().text(
            0.05, 0.9, f"Target {channel_names[i]}",
            transform=plt.gca().transAxes,
            fontsize=10, color="#9aadbe",
        )

        plt.subplot(num_panels * 2, 1, i * 2 + 2)
        plt.plot(plot_pred, color="#9aadbe")
        plt.gca().text(
            0.05, 0.9, f"Pred {channel_names[i]}",
            transform=plt.gca().transAxes,
            fontsize=10, color="#9aadbe",
        )

    if ylim is not None:
        for ax in plt.gcf().axes:
            ax.set_ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

def visualization_channels_means(
    targets: np.ndarray,
    preds: np.ndarray,
    mean_order: dict,
    save_path=None,
    channels=None,
    ylim=None,
):
    """
    means the replication within both target and pred channels
    targets: (1, len, channels)
    preds: (1, len, channels)
    """
    assert targets.shape == preds.shape, "targets and preds must have the same shape"
    assert targets.shape[0] == 1, "targets and preds must have batch size of 1"
    assert preds.shape[0] == 1, "targets and preds must have batch size of 1"
    _, seq_len, num_all_channels = targets.shape

    mean_channels = list(mean_order.keys())
    mean_target = np.zeros((1, seq_len, len(mean_channels)))
    mean_pred = np.zeros((1, seq_len, len(mean_channels)))

    for i in range(len(mean_channels)):

        mc = mean_channels[i]
        select_channel = mean_order[mc]
        mean_target[:, :, i] = targets[:, :, select_channel].mean(axis=-1)
        mean_pred[:, :, i] = preds[:, :, select_channel].mean(axis=-1)
    visualization_channels(mean_target, mean_pred, save_path, channels, ylim=ylim, channel_names=mean_channels)


EXPERT_CMAP_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
]


def _downsample_axis(arr: np.ndarray, axis: int, step: int) -> np.ndarray:
    if step <= 1:
        return arr
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(None, None, step)
    return arr[tuple(sl)]


def plot_moe_expert_routing(
    topk_idx: np.ndarray,
    expert_names: list[str],
    *,
    topk_probs: np.ndarray | None = None,
    router_probs: np.ndarray | None = None,
    region_label: str = "",
    downsample: int = 100,
    save_path: str | None = None,
    dpi: int = 200,
) -> None:
    """Visualize per-position MoE expert routing along the sequence.

    Args:
        topk_idx: (seq_len, top_k) expert indices.
        expert_names: Human-readable expert names indexed by expert id.
        topk_probs: Optional (seq_len, top_k) routing probabilities.
        router_probs: Optional (seq_len, num_experts) full router distribution.
        region_label: Title suffix for the figure.
        downsample: Plot every Nth position for long sequences.
        save_path: If set, save figure to this path.
        dpi: Figure DPI.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    topk_idx = np.asarray(topk_idx)
    if topk_idx.ndim != 2:
        raise ValueError(f"topk_idx must be 2D (seq_len, top_k), got shape {topk_idx.shape}")

    seq_len, top_k = topk_idx.shape
    num_experts = len(expert_names)
    plot_idx = _downsample_axis(topk_idx, axis=0, step=downsample)

    n_panels = 1 + top_k + (1 if router_probs is not None else 0)
    height_ratios = [1.0] + [0.8] * top_k
    if router_probs is not None:
        height_ratios.append(2.0)
    fig_height = 1.0 + top_k * 0.8 + (2.0 if router_probs is not None else 0.0)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(14, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    cmap = ListedColormap(EXPERT_CMAP_COLORS[:num_experts])
    norm = BoundaryNorm(np.arange(num_experts + 1) - 0.5, num_experts)

    top1 = plot_idx[:, 0][None, :]
    axes[0].imshow(
        top1,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axes[0].set_ylabel("Top-1")
    axes[0].set_yticks([])

    for rank in range(top_k):
        ax = axes[1 + rank]
        rank_data = plot_idx[:, rank][None, :]
        ax.imshow(
            rank_data,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        prob_label = ""
        if topk_probs is not None and topk_probs.shape[1] > rank:
            mean_prob = float(np.mean(topk_probs[:, rank]))
            prob_label = f" (mean prob={mean_prob:.3f})"
        ax.set_ylabel(f"Rank {rank + 1}{prob_label}")
        ax.set_yticks([])

    panel_offset = 1 + top_k
    if router_probs is not None:
        router_probs = np.asarray(router_probs)
        if router_probs.shape[0] == seq_len:
            router_plot = router_probs.T
        else:
            router_plot = router_probs
        router_plot = _downsample_axis(router_plot, axis=1, step=downsample)
        ax_router = axes[panel_offset]
        im_router = ax_router.imshow(
            router_plot,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=max(router_plot.max(), 1e-6),
            interpolation="nearest",
        )
        ax_router.set_ylabel("Router prob")
        ax_router.set_yticks(np.arange(num_experts))
        ax_router.set_yticklabels(expert_names, fontsize=7)
        fig.colorbar(im_router, ax=ax_router, fraction=0.02, pad=0.01)

    axes[-1].set_xlabel(f"Position (downsample x{downsample}, full length={seq_len})")
    title = "MoE expert routing"
    if region_label:
        title = f"{title} — {region_label}"
    axes[0].set_title(title, fontsize=11)

    legend_handles = [
        Patch(facecolor=EXPERT_CMAP_COLORS[i % len(EXPERT_CMAP_COLORS)], label=name)
        for i, name in enumerate(expert_names)
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.35),
        ncol=min(4, num_experts),
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

