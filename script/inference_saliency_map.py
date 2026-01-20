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
    region=None,
    device=device
)

# data
faidx = pyfaidx.Fasta(config["fasta_path"])
gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
gene_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"]) # TSS up 1500, TSS down 500

# 随机打乱gene bed
gene_bed = gene_bed.sample(frac=1, random_state=123).reset_index(drop=True)
infer_bed = gene_bed.iloc[:]

# grads
save_embeddings = []
save_gradients = []
save_genes = []
batch_interval = 8000

file_counter = 0
for idx, row in tqdm(infer_bed.iterrows(), total=len(infer_bed)):

    chrom, start, end, id, _, region_start, region_end = row
    seq = faidx[chrom][region_start:region_end].seq
    gradient, one_hots = saliency_computer.compute_saliency(sequence=seq)
    # change order into acgt (order in ntv3 atcg)
    gradient, one_hots = gradient[:,6:10], one_hots[:, 6:10][:, [0, 2, 3, 1]]
    
    save_embeddings.append(one_hots)
    save_gradients.append(gradient)

    # 每8000个batch存储一次
    if (idx + 1) % batch_interval == 0 or (idx + 1) == len(infer_bed):
        save_embeddings_concat = np.concatenate(save_embeddings, axis=0)
        save_gradients_concat = np.concatenate(save_gradients, axis=0)
        
        # 保存文件
        np.save(f"/vepfs-C/vepfs_public/daijc/lncRNA/results/grads/embeddings_ordered_{file_counter}.npy", save_embeddings_concat)
        np.save(f"/vepfs-C/vepfs_public/daijc/lncRNA/results/grads/gradients_ordered_{file_counter}.npy", save_gradients_concat)
        
        # 清空缓存
        save_embeddings = []
        save_gradients = []
        save_genes = []
        file_counter += 1

