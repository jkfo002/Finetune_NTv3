from model.head import HFModelWithHead_Infer
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
model, tokenizer = init_model(config, HFModelWithHead_Infer)

# load ckpt
device = "cuda:1"
ckpt_path = "/vepfs-C/vepfs_public/daijc/lncRNA/checkpoints/NTv3-pre-100M_fineturn_12.8k_epcho41-val_pcc0.55.ckpt"
model = load_ckpt_with_compile(model, ckpt_path, device, compile=True, strict=False)
model = model.to(device)
model.eval()

# data
faidx = pyfaidx.Fasta(config["fasta_path"])
gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
gene_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"]) # TSS up 1500, TSS down 500
faidx.close()
# order dict
with open("./config/sample_order.toml", "r") as f:
    mean_order = toml.load(f)

# 随机打乱gene bed
gene_bed = gene_bed.sample(frac=1, random_state=123).reset_index(drop=True)
infer_bed = gene_bed

# init dataset
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

from multiprocessing import Pool
from functools import partial
def process_single_item(i, chrom, start, end, bigwig_targets, logits, mean_order):

    """处理单个项目的函数"""
    target = bigwig_targets[i, :, :]
    pred = logits[i, :, :]
    
    # 可视化
    visualization_channels_means(
        np.expand_dims(target, axis=0), 
        np.expand_dims(pred, axis=0),
        mean_order=mean_order,
        save_path=f'/vepfs-C/vepfs_public/daijc/lncRNA/results/visual/{chrom[i]}/{chrom[i]}_{start[i]}_{end[i]}.png'
    )
    
    return f"{chrom[i]}_{start[i]}_{end[i]}"

# init metrics
metrics = InferMetrics(num_tracks=config["num_tracks"])
corr_dict = {}
for idx, batch in tqdm(enumerate(infer_dataloader), desc="Inference", total=len(infer_dataloader)):

    tokens, bigwig_targets, chrom, start, end = batch["tokens"].to(device), batch["bigwig_targets"].to(device), batch["chrom"], batch["start"], batch["end"]

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(tokens, return_dict=True)
            logits = outputs['bigwig_tracks_logits'].detach().cpu().numpy()

    # calculate PCC
    metrics.update(outputs['bigwig_tracks_logits'], bigwig_targets)
    metrics_list = metrics.compute()
    for i in range(len(chrom)):
        corr_dict[f"{chrom[i]}_{start[i]}_{end[i]}"] = metrics_list[i]
    
    # 使用多进程处理
    bigwig_targets = bigwig_targets.detach().cpu().numpy()

    params = [(i, chrom, start, end, bigwig_targets, logits, mean_order) for i in range(len(chrom))]
    with Pool(processes=8) as pool:
        results = pool.starmap(process_single_item, params)

import pickle
with open("/vepfs-C/vepfs_public/daijc/lncRNA/results/visual/corr_dict.pkl", "wb") as f:
    pickle.dump(corr_dict, f)
