from model.dataset import GenomeBigWigDataset
from model.utils import load_Data, transform_fn
from model.decorator import NUC_CONFIG
from model.head import HFModelWithHead_Infer
from model.utils import load_config, init_config, init_model, load_ckpt_with_compile

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from typing import Dict, Tuple
import pyfaidx
import os
import pandas as pd
import numpy as np

# ============================================================
# Pyramid-style Attention Map Plotting (45-degree rotated triangular)
# ============================================================
from matplotlib.colors import TwoSlopeNorm
from matplotlib import pyplot as plt

# Global settings
LINE_WIDTH = 0.5

def _get_45deg_mesh(mat):
    """Create 45-degree rotated mesh for triangular attention visualization.

    Args:
        mat: Input matrix to create mesh for.

    Returns:
        tuple: (X, Y) mesh coordinates rotated 45 degrees.
    """
    theta = -np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    K = len(mat) + 1
    g = np.arange(0, K) - 0.5
    X = np.tile(g[:, None], (1, K))
    Y = np.tile(g[None, :], (K, 1))
    xy = np.array([X.ravel(), Y.ravel()])
    xy_rot = R @ xy
    return xy_rot[0, :].reshape(K, K), xy_rot[1, :].reshape(K, K).T

def plot_attention_panel(mat, filename=None, *, cmap="Blues", vmin=0.0001,
                         vmax=0.005, colorbar=True, dpi=600, figsize=(10, 10),
                         variant_id="", show_titles=True, show_xlabel=True,
                         show_xticks=True, show_yticks=False,
                         positions='TOKEN', token_resolution=128, show=True):
    """Plot triangular attention heatmap with 45-degree rotation (pyramid style).

    Args:
        mat: Attention matrix to plot.
        filename: Output filename. If None and show=True, displays plot.
        cmap: Colormap name. Defaults to "Blues".
        vmin: Minimum value for color scale. Defaults to 0.0001.
        vmax: Maximum value for color scale. Defaults to 0.005.
        colorbar: Whether to show colorbar. Defaults to True.
        dpi: Figure DPI. Defaults to 600.
        figsize: Figure size tuple. Defaults to (10, 10).
        variant_id: Title for the plot. Defaults to "".
        show_titles: Whether to show title. Defaults to True.
        show_xlabel: Whether to show x-axis label. Defaults to True.
        show_xticks: Whether to show x-axis ticks. Defaults to True.
        show_yticks: Whether to show y-axis ticks. Defaults to False.
        positions: 'TOKEN' or 'BP' for position units. Defaults to 'TOKEN'.
        token_resolution: Base pairs per token. Defaults to 128.
        show: Whether to display plot. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    # Symmetrize matrix
    mat = 0.5 * (mat + mat.T)

    # Mask lower triangle
    mat[np.tril_indices_from(mat, k=-1)] = np.nan
    n = mat.shape[0]
    X, Y = _get_45deg_mesh(mat)

    # Coordinate normalization
    C = 1
    half_pixel_diag = 1 / (2*C)
    pixel_side = 1 / (C * np.sqrt(2))
    X = X * pixel_side + half_pixel_diag
    Y = Y * pixel_side
    Y = -Y  # Flip Y

    # Clean up coordinates
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1.0, neginf=-1.0)

    # Set up normalization
    vcenter = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.pcolormesh(X, Y, mat, cmap=cmap, norm=norm, shading="auto")

    # Set axis properties
    ax.set_aspect("equal")

    # Calculate coordinate extents
    finite_mask = np.isfinite(mat)
    if finite_mask.any():
        vertex_mask = np.zeros_like(X, dtype=bool)
        vertex_mask[:-1, :-1] |= finite_mask
        vertex_mask[:-1, 1:] |= finite_mask
        vertex_mask[1:, :-1] |= finite_mask
        vertex_mask[1:, 1:] |= finite_mask

        valid_x = X[vertex_mask]
        valid_y = Y[vertex_mask]
        x_min, x_max = np.nanmin(valid_x), np.nanmax(valid_x)
        y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
    else:
        x_min, x_max = np.nanmin(X), np.nanmax(X)
        y_min, y_max = np.nanmin(Y), np.nanmax(Y)

    triangle_bottom_y = y_min
    triangle_top_y = y_max

    xlim_pad = 0.1
    ylim_pad = 0.1
    y_range = (y_max - y_min) if y_max > y_min else 1.0
    bottom_pad = 0.001 * y_range

    ax.set_xlim(x_min - xlim_pad, x_max + xlim_pad)
    ax.set_ylim(y_min - bottom_pad, y_max + ylim_pad)

    # Add colorbar
    if colorbar:
        pyramid_y_min = y_min
        pyramid_y_max = y_max

        cb = plt.colorbar(im, ax=ax, shrink=1.0, aspect=20, pad=0.1)
        cb.ax.tick_params(width=LINE_WIDTH*0.8, length=3, labelsize=7, direction="in")
        cb.outline.set_linewidth(LINE_WIDTH*0.8)

        ax_pos = ax.get_position()
        axes_y_min_data = y_min - bottom_pad
        axes_y_max_data = y_max + ylim_pad
        axes_height_data = axes_y_max_data - axes_y_min_data

        if axes_height_data > 0:
            pyramid_bottom_frac = (
                (pyramid_y_min - axes_y_min_data) / axes_height_data
            )
            pyramid_top_frac = (
                (pyramid_y_max - axes_y_min_data) / axes_height_data
            )
        else:
            pyramid_bottom_frac = 0
            pyramid_top_frac = 1

        cb_pos = cb.ax.get_position()
        new_y0 = ax_pos.y0 + ax_pos.height * pyramid_bottom_frac
        new_height = ax_pos.height * (pyramid_top_frac - pyramid_bottom_frac)
        cb.ax.set_position(
            [cb_pos.x0, new_y0, cb_pos.width, new_height]
        )

    # Format axes
    use_bp_display = positions.upper() == 'BP'

    ax.tick_params(width=LINE_WIDTH, length=3, labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH)
    ax.spines["bottom"].set_position(('data', triangle_bottom_y))
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    if show_xticks and mat.size > 0:
        n_tokens = mat.shape[0]
        tick_positions = [0, n_tokens - 1]
        if use_bp_display:
            max_bp = (n_tokens - 1) * token_resolution
            tick_labels = ['0', str(max_bp)]
        else:
            tick_labels = ['0', str(n_tokens - 1)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks([])

    if not show_yticks:
        ax.set_yticks([])

    if use_bp_display:
        xlabel_text = "Position (base pairs)"
    else:
        xlabel_text = "Token position"
    if show_xlabel:
        ax.set_xlabel(xlabel_text, fontsize=8)

    if show_titles and variant_id:
        ax.set_title(variant_id, fontsize=9, pad=10)

    ax.set_facecolor("white")

    if filename:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(
            filename, dpi=600, bbox_inches="tight", facecolor='white'
        )
        print(f"✅ Saved attention plot: {filename}")
        if not show:
            plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# setting
# Extract attention from last layer
layer_index = -1  # Last layer
config = load_config("config/fineturn_my.toml")
config = init_config(config)
model, tokenizer = init_model(config, HFModelWithHead_Infer)

# load ckpt
device = "cuda:1"
ckpt_path = "/vepfs-C/vepfs_public/daijc/lncRNA/checkpoints/NTv3-pre-100M_fineturn_6k_epcho71-val_pcc0.5.ckpt"
model = load_ckpt_with_compile(model, ckpt_path, device, compile=True, strict=False)
model = model.to(device)
model.eval()

# data
faidx = pyfaidx.Fasta(config["fasta_path"])
gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
gene_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"]) # TSS up 1500, TSS down 500
faidx.close()

# 随机打乱gene bed
gene_bed = gene_bed.sample(frac=1, random_state=123).reset_index(drop=True)
infer_bed = gene_bed.iloc[:16] # 取前16个样本

infer_dataset = GenomeBigWigDataset(
    fasta_path=config["fasta_path"],
    bigwig_path_list=[os.path.join(config["training_data_dir"], f) for f in config["bigwig_files"]],
    chrom_regions = infer_bed,
    sequence_length=config["sequence_length"],
    tokenizer=tokenizer,
    transform_fn = transform_fn,
    keep_target_center_fraction=config["keep_target_center_fraction"],
    num_samples=len(infer_bed)
)
infer_dataloader = DataLoader(
    infer_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=config["num_workers"]
)

att_dict = {}
for idx, batch in enumerate(infer_dataloader):
    tokens, bigwig_targets, chrom, start, end = batch["tokens"].to(device), batch["bigwig_targets"].to(device), batch["chrom"], batch["start"], batch["end"]
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(tokens, return_dict=True)
        
        attention_maps = outputs.attentions[-1] # List[(batch, heads, seq_len, seq_len)] length:layers
    
    for i in range(attention_maps.shape[0]):
        att_dict[f'{chrom[i]}_{start[i]}_{end[i]}'] = attention_maps[i, :, :, :].cpu().numpy() # (heads, seq_len, seq_len))

def attention_map(attention_last_layer):
    # Average over heads and symmetrize
    # [seq_len, seq_len]
    attention_mean = attention_last_layer.mean(axis=0)
    attention_mean = 0.5 * (attention_mean + attention_mean.T)  # Symmetrize

    print(f"Attention map shape: {attention_mean.shape}")
    attn_min = attention_mean.min()
    attn_max = attention_mean.max()
    print(f"Attention value range: [{attn_min:.6f}, {attn_max:.6f}]")

    return attention_mean

attention_mean = attention_map(att_dict['A157_chr06_57751086_57753958'])

fig = plot_attention_panel(
    attention_mean,
    cmap="Blues",
    vmin=0.0001,
    vmax=0.005,
    colorbar=True,
    figsize=(10, 10),
    show_xlabel=True,
    show_xticks=True,
    positions='TOKEN',
    show=True,
)