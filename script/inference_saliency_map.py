from model.head import HFModelWithHead_Saliency, SaliencyComputer
from model.utils import (
    load_config, init_config, 
    init_model, load_ckpt_with_compile, 
    load_Data, transform_fn
)
from model.dataset import GenomeBigWigDataset
from model.analysis import visualization_channels_means
from model.decorator import NUC_CONFIG
from model.metrics import InferMetrics

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
import toml
from tqdm import tqdm

config = load_config("config/fineturn_my.toml")
config = init_config(config)
model, tokenizer = init_model(config, HFModelWithHead_Saliency)

# load ckpt
device = "cuda:1"
ckpt_path = "/vepfs-C/vepfs_public/daijc/lncRNA/checkpoints/NTv3-pre-100M_fineturn_12.8k_epcho41-val_pcc0.55.ckpt"
model = load_ckpt_with_compile(model, ckpt_path, device, compile=True, strict=False)
model = model.to(device)
model.eval()

# init saliency computer
saliency_computer = SaliencyComputer(
    model=model, # HFModelWithHead_Saliency
    tokenizer=tokenizer,
    sequence_length=config["sequence_length"],
    track_indices=None,
    region=(10800, 14800),
    device=device
)

# data
faidx = pyfaidx.Fasta(config["fasta_path"])
gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
gene_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"]) # TSS up 1500, TSS down 500
faidx.close()

# 随机打乱gene bed
gene_bed = gene_bed.sample(frac=1, random_state=123).reset_index(drop=True)
infer_bed = gene_bed.iloc[:16]

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

for idx, batch in enumerate(infer_dataloader):

    tokens, bigwig_targets, chrom, start, end = (
        batch["tokens"].to(device), 
        batch["bigwig_targets"].to(device), 
        batch["chrom"], batch["start"], batch["end"]
    )
    with autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs["bigwig_tracks_logits"]

    metrics.update(logits, bigwig_targets)
    corr = metrics.compute()
    for b in tqdm(range(logits.shape[0]), desc=f"Processing batch {idx}, total batch {len(infer_dataloader)}"):
        corr_dict[f"{chrom[b]}_{start[b]}_{end[b]}"] = corr[b]
        visualization_channels_means(
            bigwig_targets[b, :, :].unsqueeze(0).detach().cpu().numpy(), logits[b, :, :].unsqueeze(0).detach().cpu().numpy(), 
            order_dict, 
            f"visualization/{chrom[b]}_{start[b]}_{end[b]}.png",
        )
