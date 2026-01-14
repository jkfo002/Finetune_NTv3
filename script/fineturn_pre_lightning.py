import functools
import os
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

from model.head import HFModelWithHead
from model.backbone import MyDataModule, MyModel

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

def init_model(config: Dict) -> Tuple[nn.Module, AutoTokenizer]:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    # Create model
    model = HFModelWithHead(
        model_name=config["model_name"],
        num_tracks=config["num_tracks"],
        keep_target_center_fraction=config["keep_target_center_fraction"],
    )

    return model, tokenizer

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
    config = load_config("config/pretrain.toml")
    config = init_config(config)

    model, tokenizer = init_model(config) # init model and tokenizer

    # data path
    data_dir = Path(config["fasta_path"]).parent
    train_chrom_regions = [os.path.join(data_dir, r) for r in config["train_chrom_regions"]]
    valid_chrom_regions = [os.path.join(data_dir, r) for r in config["valid_chrom_regions"]]
    test_chrom_regions = [os.path.join(data_dir, r) for r in config["test_chrom_regions"]]

    # lightning dataset
    pl_datamodule = MyDataModule(
        fasta_path=config["fasta_path"],
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
