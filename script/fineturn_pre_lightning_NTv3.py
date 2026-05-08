import functools
import os

# Set distributed debug env vars before importing torch/lightning so DDP child
# processes inherit them early enough for NCCL flight recorder setup.
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "200000"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
import toml
import pyfaidx
import argparse
import swanlab

from model.head import HFModelWithHead
from model.backbone import MyDataModule_NTv3, MyModel
from model.utils import load_config, init_config, init_model, load_ckpt_with_compile, load_Data

def set_callbacks(config: Dict) -> Tuple[LearningRateMonitor, ModelCheckpoint]:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_pearson",          # 监控指标
        mode="max",                 # 寻找最大值
        save_top_k=2,              # 只保存最好的一个模型
        filename=f"{config['logger_prefix']}_"+"-{epoch}-{val_epoch_pearson:.2f}",
        dirpath=config['checkpoints'],          # 保存路径
    )

    return lr_monitor, checkpoint_callback

def main():
    # config
    parser = argparse.ArgumentParser(description="Finetune NTv3")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file, default: config/fineturn_my.toml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = init_config(config)

    model, tokenizer = init_model(config, HFModelWithHead) # init model and tokenizer

    # data path
    faidx = pyfaidx.Fasta(config["fasta_path"])
    gene_bed = os.path.join(config["training_data_dir"], config["gene_bed"])
    gene_bed = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
    gene_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"])
    faidx.close()

    # 随机打乱gene bed
    gene_bed = gene_bed.sample(frac=1, random_state=42).reset_index(drop=True)
    total_size = len(gene_bed)
    train_size, val_size = int(total_size * 0.8), int(total_size * 0.1)
    train_chrom_regions = gene_bed.iloc[:train_size]
    valid_chrom_regions = gene_bed.iloc[train_size:train_size + val_size]
    test_chrom_regions = gene_bed.iloc[train_size + val_size:]
    print("="*30)
    print(f"train_size: {train_size}, val_size: {val_size}, test_size: {total_size - train_size - val_size}")
    print("="*30)

    # lightning dataset
    pl_datamodule = MyDataModule_NTv3(
        fasta_path=config["fasta_path"],
        bigwig_path_list=[os.path.join(config["training_data_dir"], f) for f in config["bigwig_files"]],
        train_chrom_regions=train_chrom_regions,
        val_chrom_regions=valid_chrom_regions,
        test_chrom_regions=test_chrom_regions,
        sequence_length=config["sequence_length"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        keep_target_center_fraction=config["keep_target_center_fraction"],
    )

    # lightning model
    pl_model = MyModel(
        model=model,
        config=config,
        device=config["device"],
    )

    # trainer
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    if not os.path.exists(config["checkpoints"]):
        os.makedirs(config["checkpoints"])
    lr_monitor, checkpoint_callback = set_callbacks(config)
    if config['num_devices'] > 1:
        trainer = Trainer(
            max_steps=config["max_steps"],
            precision="bf16-mixed",
            accelerator="gpu",
            devices=config['num_devices'],
            strategy="ddp",  # 添加分布式策略
            log_every_n_steps=config["log_every_n_steps"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            logger=TensorBoardLogger(
                save_dir=config["log_dir"],
                name=config["logger_prefix"],
            ),
            callbacks=[lr_monitor, checkpoint_callback]
        )
    else:
        trainer = Trainer(
            max_steps=config["max_steps"],
            precision="bf16-mixed",
            accelerator="gpu",
            devices=config['num_devices'],
            log_every_n_steps=config["log_every_n_steps"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            logger=TensorBoardLogger(
                save_dir=config["log_dir"],
                name=config["logger_prefix"],
            ),
            callbacks=[lr_monitor, checkpoint_callback]
        )
    

    trainer.fit(pl_model, datamodule=pl_datamodule) # 训练模型
    trainer.test(pl_model, datamodule=pl_datamodule) # 测试模型

    # 保存最终模型
    torch.save(
        pl_model.state_dict(), 
        f"{config['checkpoints']}/{config['logger_prefix']}_final.pth"
    )

if __name__ == "__main__":
    main()
