import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Callable, Optional, Union
import numpy as np
import pandas as pd
import tqdm

def gene_filter(gene_df, faidx, TSS_region_len_up, TSS_region_len_down):
    """
    过滤基因数据框，保留在基因组中的基因
    :param gene_df: 基因数据框
    :param faidx: pyfaidx
    :return: 过滤后的基因数据框
    """
    # 过滤基因数据框，保留在基因组中的基因
    input_gene_sets = []
    for _, row in gene_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        gene_id = row["id"]
        if start - TSS_region_len_up > 0 and start + TSS_region_len_down < len(faidx[chrom]):
            input_gene_sets.append(gene_id)
    filted_gene_df = gene_df[gene_df["id"].isin(input_gene_sets)]

    return filted_gene_df

def load_Data(gene_df, faidx, TSS_region_len_up, TSS_region_len_down):
    """
    加载基因数据和h5ad文件
    :param gene_df: 基因数据框
    :param TSS_region_len_up: TSS区域长度 up
    :param TSS_region_len_down: TSS区域长度 down
    """

    filted_gene_df = gene_filter(gene_df, faidx, TSS_region_len_up, TSS_region_len_down)
    filted_gene_df['region_start'] = filted_gene_df['start'] - TSS_region_len_up
    filted_gene_df['region_end'] = filted_gene_df['start'] + TSS_region_len_down

    return filted_gene_df

def crop_center(x: Optional[Union[torch.Tensor,np.ndarray]], keep_target_center_fraction: float = 0.375):
    """Crop the central sequence-length fraction for arrays of size (..., seq_len, num_tracks)"""
    if x is None:
        raise ValueError("Input array cannot be None.")
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]

def create_targets_scaling_fn(
    metadata_df: pd.DataFrame
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a scaling function that uses the track means to normalise and softclip the targets.
    """
    # Open bigwig files and compute track statistics
    track_means = metadata_df["mean"].to_numpy()
    print(f"Track means: {track_means}")
    print(f"Number of tracks: {track_means.shape}")

    # Create tensor from computed means
    track_means_tensor = torch.tensor(track_means, dtype=torch.float32)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        # Move constants to correct device then normalize
        means = track_means_tensor.to(x.device)
        scaled = x / means

        # Smooth clipping: if > 10, apply formula
        clipped = torch.where(
            scaled > 10.0,
            2.0 * torch.sqrt(scaled * 10.0) - 10.0,
            scaled,
        )
        return clipped

    return transform_fn

def transform_fn(x: torch.Tensor) -> torch.Tensor:
    """
    对tracks进行3/4幂次变换, 若幂次大于384则使用384+(y-384)^(1/2)
    :param y: 输入的track值
    :return: 变换后的值
    """
    transformed = torch.pow(x, 0.75)
    mask = transformed > 384
    transformed[mask] = 384 + torch.sqrt(transformed[mask] - 384)

    return transformed

# def transform_fn(x: torch.Tensor) -> torch.Tensor:
#     """
#     Transform the input tensor by normalizing and clipping.
#     """
#     # Move constants to correct device then normalize
# 
#     means = x.mean(dim=-1, keepdim=True)
# 
#     eps = 1e-8
#     means = torch.clamp(means, min=eps)
#     
#     scaled = x / means
# 
#     # Smooth clipping: if > 10, apply formula
#     clipped = torch.where(
#         scaled > 10.0,
#         2.0 * torch.sqrt(scaled * 10.0) - 10.0,
#         scaled,
#     )
#     return clipped

