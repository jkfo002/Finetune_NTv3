from pytorch_lightning import LightningDataModule, LightningModule
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import AutoTokenizer
from typing import Callable
import functools

from .dataset import GenomeBigWigDataset_myIterable, GenomeBigWigDataset
from .loss import poisson_multinomial_loss
from .metrics import TracksMetrics
from .utils import transform_fn


class MyDataModule(LightningDataModule):

    def __init__(
        self, 
        fasta_path: str,
        train_chrom_regions: list[str],
        val_chrom_regions: list[str],
        test_chrom_regions: list[str],
        sequence_length: int,
        tokenizer: AutoTokenizer,
        keep_target_center_fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.train_chrom_regions = train_chrom_regions
        self.val_chrom_regions = val_chrom_regions
        self.test_chrom_regions = test_chrom_regions

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.keep_target_center_fraction = keep_target_center_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.create_dataset_fn = functools.partial(
            GenomeBigWigDataset_myIterable,
            fasta_path=self.fasta_path,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            keep_target_center_fraction=self.keep_target_center_fraction,
            batch_size=self.batch_size
        )

    
    def train_dataloader(self):
        self.train_dataset = self.create_dataset_fn(
            parquet_path=self.train_chrom_regions
        )
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def val_dataloader(self):
        self.val_dataset = self.create_dataset_fn(
            parquet_path=self.val_chrom_regions,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def test_dataloader(self):
        self.test_dataset = self.create_dataset_fn(
            parquet_path=self.test_chrom_regions,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=2
        )


class MyDataModule_NTv3(LightningDataModule):

    def __init__(
        self, 
        fasta_path: str,
        bigwig_path_list: list[str],
        train_chrom_regions: pd.DataFrame,
        val_chrom_regions: pd.DataFrame,
        test_chrom_regions: pd.DataFrame,
        sequence_length: int,
        tokenizer: AutoTokenizer,
        keep_target_center_fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.bigwig_path_list = bigwig_path_list
        self.train_chrom_regions = train_chrom_regions
        self.val_chrom_regions = val_chrom_regions
        self.test_chrom_regions = test_chrom_regions

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.keep_target_center_fraction = keep_target_center_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.create_dataset_fn = functools.partial(
            GenomeBigWigDataset,
            fasta_path=self.fasta_path,
            bigwig_path_list=self.bigwig_path_list,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            transform_fn=transform_fn,
            keep_target_center_fraction=self.keep_target_center_fraction,
        )
    
    def train_dataloader(self):
        self.train_dataset = self.create_dataset_fn(
            chrom_regions=self.train_chrom_regions,
            num_samples=len(self.train_chrom_regions)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def val_dataloader(self):
        self.val_dataset = self.create_dataset_fn(
            chrom_regions=self.val_chrom_regions,
            num_samples=len(self.val_chrom_regions)
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def test_dataloader(self):
        self.test_dataset = self.create_dataset_fn(
            chrom_regions=self.test_chrom_regions,
            num_samples=len(self.test_chrom_regions)
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2
        )


class MyModel(LightningModule):
    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.mymodel = model
        if config["freeze_backbone"]:
            for param in self.mymodel.parameters():
                param.requires_grad = False

        self.loss_fn = poisson_multinomial_loss
        self.to_device = device

        # optimizer and scheduler
        self.optimizer = AdamW(
            self.mymodel.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["max_steps"],
            eta_min=0
        )
        self.init_metric_target()
        
    def forward(self, x):
        return self.mymodel(x)

    def init_metric_target(self):

        # for step metrics
        self.train_metrics = TracksMetrics(self.config["num_tracks"], "train")
        self.val_metrics = TracksMetrics(self.config["num_tracks"], "val")
        self.test_metrics = TracksMetrics(self.config["num_tracks"], "test")

        # for epoch metrics
        self.train_metrics_epoch = TracksMetrics(self.config["num_tracks"], "train")
        self.val_metrics_epoch = TracksMetrics(self.config["num_tracks"], "val")
        self.test_metrics_epoch = TracksMetrics(self.config["num_tracks"], "test")


    def configure_optimizers(self):
        
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": {'scheduler':self.scheduler, "interval": "step", "frequency": 1,}
        }

    def _sync_scalar_mean(self, value: torch.Tensor) -> torch.Tensor:
        synced = value.detach().clone()
        if self.trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(synced, op=dist.ReduceOp.SUM)
            synced /= self.trainer.world_size
        return synced

    def training_step(self, batch, batch_idx):

        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]

        outputs = self(tokens)
        bigwig_logits = outputs["bigwig_tracks_logits"]
        
        loss = self.loss_fn(logits=bigwig_logits, targets=bigwig_targets)
        synced_loss = self._sync_scalar_mean(loss)
        self.log("train_loss", synced_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        self.train_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.train_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]

        outputs = self(tokens)
        bigwig_logits = outputs["bigwig_tracks_logits"]
        
        loss = self.loss_fn(logits=bigwig_logits, targets=bigwig_targets)
        synced_loss = self._sync_scalar_mean(loss)
        self.log("val_loss", synced_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.val_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.val_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())
        
        
        return loss

    def test_step(self, batch, batch_idx):
        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]

        outputs = self(tokens)
        bigwig_logits = outputs["bigwig_tracks_logits"]
        
        loss = self.loss_fn(logits=bigwig_logits, targets=bigwig_targets)
        synced_loss = self._sync_scalar_mean(loss)
        self.log("test_loss", synced_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        # Log metrics
        self.test_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.test_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())
        
        return loss

    def predict_step(self, batch, batch_idx):
        tokens = batch["tokens"]

        outputs = self(tokens)
        bigwig_logits = outputs["bigwig_tracks_logits"]

        return bigwig_logits

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 每100个batch计算并记录一次指标
        if (batch_idx + 1) % 100 == 0:
            # 计算当前累积的指标
            train_metrics_result = self.train_metrics.compute(sync_dist=self.trainer.world_size > 1)
            
            # 记录平均损失和平均皮尔逊相关系数
            self.log("train_avg_loss", train_metrics_result["loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("train_avg_pearson", train_metrics_result["mean/pearson"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
            # 重置step metrics以便下一个100步的累积
            self.train_metrics.reset()
        
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # 每100个batch计算并记录一次指标
        if (batch_idx + 1) % 100 == 0:
            # 计算当前累积的指标
            val_metrics_result = self.val_metrics.compute(sync_dist=self.trainer.world_size > 1)
            
            # 记录平均损失和平均皮尔逊相关系数
            self.log("val_avg_loss", val_metrics_result["loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("val_avg_pearson", val_metrics_result["mean/pearson"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
            # 重置step metrics以便下一个100步的累积
            self.val_metrics.reset()
    
    def on_test_batch_end(self, outputs, batch, batch_idx):
        # 每100个batch计算并记录一次指标
        if (batch_idx + 1) % 100 == 0:
            # 计算当前累积的指标
            test_metrics_result = self.test_metrics.compute(sync_dist=self.trainer.world_size > 1)
            
            # 记录平均损失和平均皮尔逊相关系数
            self.log("test_avg_loss", test_metrics_result["loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("test_avg_pearson", test_metrics_result["mean/pearson"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
            # 重置step metrics以便下一个100步的累积
            self.test_metrics.reset()

    def on_train_epoch_end(self):
        # 计算并记录训练epoch级别的指标
        train_epoch_metrics = self.train_metrics_epoch.compute(sync_dist=self.trainer.world_size > 1)
        self.log("train_epoch_loss", train_epoch_metrics["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("train_epoch_pearson", train_epoch_metrics["mean/pearson"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        # 重置epoch metrics以便下一个epoch的累积
        self.train_metrics_epoch.reset()

    def on_validation_epoch_end(self):
        # 计算并记录验证epoch级别的指标
        val_epoch_metrics = self.val_metrics_epoch.compute(sync_dist=self.trainer.world_size > 1)
        self.log("val_epoch_loss", val_epoch_metrics["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("val_epoch_pearson", val_epoch_metrics["mean/pearson"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        # 重置epoch metrics以便下一个epoch的累积
        self.val_metrics_epoch.reset()
    
    def on_test_epoch_end(self):
        # 计算并记录测试epoch级别的指标
        test_epoch_metrics = self.test_metrics_epoch.compute(sync_dist=self.trainer.world_size > 1)
        self.log("test_epoch_loss", test_epoch_metrics["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("test_epoch_pearson", test_epoch_metrics["mean/pearson"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        
        # 重置epoch metrics以便下一个epoch的累积
        self.test_metrics_epoch.reset()
