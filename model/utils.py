import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, List, Callable, Optional, Union, Tuple
import numpy as np
import pandas as pd
import tqdm
import toml
from pytorch_lightning import seed_everything

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = toml.load(f)
    return config

def init_config(config: Dict) -> Dict:
    # Set random seed
    seed_everything(config["seed"], workers=True)
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    return config

def init_model(config: Dict, model_cls) -> Tuple[nn.Module, AutoTokenizer]:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    # Create model
    init_model = model_cls(
        model_name=config["model_name"],
        num_tracks=config["num_tracks"],
        keep_target_center_fraction=config["keep_target_center_fraction"],
    )

    return init_model, tokenizer

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

    filted_gene_df = gene_filter(gene_df, faidx, TSS_region_len_up, TSS_region_len_down).copy()
    filted_gene_df['region_start'] = filted_gene_df['start'] - TSS_region_len_up
    filted_gene_df['region_end'] = filted_gene_df['start'] + TSS_region_len_down
    print("before drop duplicates: ", filted_gene_df.shape)
    filted_gene_df = filted_gene_df.drop_duplicates(subset=['region_start', 'region_end'])
    print("after drop duplicates: ", filted_gene_df.shape)

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

def load_ckpt_with_compile(
    my_models: nn.Module, 
    ckpt_path: str, 
    device: str = "cuda", 
    prefix_in_lightning: str = "mymodel.",
    compile: bool = False,
    strict: bool = True,
):
    """Load checkpoint.

    Args:
        my_models (pl.LightningModule): pl.LightningModule to load checkpoint.
        ckpt_path (str): Checkpoint path.
        device (str, optional): Device to load checkpoint. Defaults to "cuda".
        prefix_in_lightning (str, optional): Prefix in Lightning checkpoint. Defaults to "mymodel.".
        compile (bool, optional): Whether to compile model. Defaults to False.
        strict (bool, optional): Whether to strictly load state_dict (for RoPE cache, False). Defaults to True.

    Return:
        my_models: Loaded model for eval
    """
    checkpoint = torch.load(ckpt_path, map_location=device)  # 或 "cuda" if GPU

    # 2. 提取 state_dict（注意：Lightning 保存的 key 带有 "model." 前缀）
    state_dict = checkpoint["state_dict"]

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix_in_lightning):
            new_k = k[len(prefix_in_lightning):]  # 去掉 "model." 前缀
            if compile: 
                if "_orig_mod." in new_k: # 模型训练过程中使用torch.compile会在ckpt的权重文件中增加_orig_mod 需要去掉 否则无法加载
                    new_k = new_k.replace("_orig_mod.", "")
                
            new_state_dict[new_k] = v
        else:
            # 可能还有其他非 model 的参数（如 loss_fn 等），跳过
            continue

    # 4. 加载到 my_models（纯 nn.Module）
    my_models.load_state_dict(new_state_dict, strict=strict)

    return my_models