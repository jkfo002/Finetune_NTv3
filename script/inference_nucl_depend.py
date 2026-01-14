from model.head import HFModelWithHead_Infer
from model.dataset import GenomeBigWigDataset_Nucl_Depend
from model.utils import load_Data, transform_fn, load_config, init_config, init_model, load_ckpt_with_compile
from model.decorator import NUC_CONFIG

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


def Infer_for_Nucleotide_dependencies(model, infer_dataloader, device):
    output_arrays = []
    for idx, batch in enumerate(
        tqdm(
            infer_dataloader, 
            desc="Processing batches", 
            total=len(infer_dataloader)
        )
    ):

        tokens = batch["tokens"].to(device)

        with autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():
                outputs = model(tokens, return_dict=True)

        output_probs = F.softmax(outputs['logits'], dim=-1)[:, :(TSS_UP+TSS_DOWN), ACGT_IDX]
        output_arrays.append(output_probs) 

    snp_reconstruct = torch.concat(output_arrays, axis=0)
    snp_reconstruct.detach().cpu().to(torch.float32).numpy()

    return snp_reconstruct

def compute_dependency_map(snp_reconstruct, dataset, epsilon=1e-10):

     # for the logit add a small value epsilon and renormalize such that every prob in one position sums to 1
    snp_reconstruct = snp_reconstruct + epsilon
    snp_reconstruct = snp_reconstruct/snp_reconstruct.sum(axis=-1)[:,:, np.newaxis]

    seq_len = snp_reconstruct.shape[1]
    snp_effect = np.zeros((seq_len, seq_len,4, 4))
    reference_probs = snp_reconstruct[dataset[dataset['nuc'] == 'real_sequence'].index[0]]
    dataset = dataset[dataset['mutation_pos'] < seq_len]

    snp_effect[dataset.iloc[1:]['mutation_pos'].values, : ,  dataset.iloc[1:]['var_nt_idx'].values - 6,:] = np.log2(snp_reconstruct[1:]) - np.log2(1 - snp_reconstruct[1:]) \
        - np.log2(reference_probs) + np.log2(1-reference_probs)

    dep_map = np.max(np.abs(snp_effect), axis=(2,3))
    #zero main diagonal values
    dep_map[np.arange(dep_map.shape[0]), np.arange(dep_map.shape[0])] = 0

    return dep_map

import seaborn as sns
import matplotlib.pyplot as plt

def plot_map_with_seq(matrix, dna_sequence,  plot_size=10, vmax=5, tick_label_fontsize=8):

    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    
   
    sns.heatmap(matrix, cmap='coolwarm', vmax=vmax, ax=ax, 
                xticklabels=False, yticklabels=False)  
    ax.set_aspect('equal')

    tick_positions = np.arange(len(dna_sequence)) + 0.5 # Center the ticks

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(list(dna_sequence), fontsize=tick_label_fontsize, rotation=0)
    ax.set_yticklabels(list(dna_sequence), fontsize=tick_label_fontsize)

    plt.show()
    
def plot_map(matrix, vmax=None, display_values=False, annot_size=8, fig_size=10, save_path=None):
   
    plt.figure(figsize=(fig_size, fig_size))

    ax = sns.heatmap(matrix, cmap="coolwarm", vmax=vmax, annot=display_values, 
                     fmt=".2f", annot_kws={"size": annot_size})

    ax.set_aspect('equal')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# settings
TSS_UP, TSS_DOWN = (1500, 500)
NUC_TAB = NUC_CONFIG.NUC_TAB
ACGT_IDX = NUC_CONFIG.ACGT_IDX
config = load_config("config/fineturn_my.toml")
config = init_config(config)
model, tokenizer = init_model(config, HFModelWithHead_Infer)

# load ckpt
device = "cuda:1"
ckpt_path = "/vepfs-C/vepfs_public/daijc/lncRNA/checkpoints/NTv3-pre-100M_fineturn_6k_epcho71-val_pcc0.5.ckpt"
model = load_ckpt_with_compile(model, ckpt_path, device, compile=True, strict=False)
model = model.to(device)
model.eval()

# load data
faidx = pyfaidx.Fasta(config["fasta_path"])
gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
gene_bed = load_Data(gene_bed, faidx, TSS_UP, TSS_DOWN) # TSS up 1500, TSS down 500
faidx.close()

# get one region
gene_bed = gene_bed.sample(frac=1, random_state=123).reset_index(drop=True)
infer_bed = gene_bed.iloc[1]
chrom_dict = {
    'chrom': infer_bed['chrom'],
    'start': infer_bed['region_start'],
    'end': infer_bed['region_end'],
}

# dataset
infer_dataset = GenomeBigWigDataset_Nucl_Depend(
    fasta_path=config["fasta_path"],
    chrom_regions=chrom_dict,
    sequence_length=config["sequence_length"],
    tokenizer=tokenizer,
    transform_fn=transform_fn
)

infer_dataloader = DataLoader(infer_dataset, batch_size=16, shuffle=False)
output = Infer_for_Nucleotide_dependencies(model, infer_dataloader, device)
dep_map = compute_dependency_map(output.to('cpu'), infer_dataset.mutations_df)

plot_map(dep_map, vmax=2)