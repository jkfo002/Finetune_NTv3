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
from .moe import compute_moe_aux_loss
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
        track_label_list: list[int] | None = None,
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
        self.track_label_list = track_label_list
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
        track_label_list: list[int] | None = None,
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
        self.track_label_list = track_label_list

        self.create_dataset_fn = functools.partial(
            GenomeBigWigDataset,
            fasta_path=self.fasta_path,
            bigwig_path_list=self.bigwig_path_list,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            transform_fn=transform_fn,
            keep_target_center_fraction=self.keep_target_center_fraction,
            track_label_list=self.track_label_list,
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


class MyModelMOE(LightningModule):
    """MoE finetuning with staged backbone unfreezing."""

    PHASE_HEAD_WARMUP = 0
    PHASE_FULL_FINETUNE = 1

    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.mymodel = model
        self.to_device = device

        self.moe_warmup_epochs = int(config.get("moe_warmup_epochs", 0))
        self.moe_warmup_router_only = bool(config.get("moe_warmup_router_only", False))
        self.head_learning_rate = float(
            config.get("head_learning_rate", config["learning_rate"])
        )
        self._training_phase: int | None = None

        self._init_metrics()
        self._apply_initial_trainable_state()

    def forward(self, tokens: torch.Tensor):
        return self.mymodel(tokens)

    def _init_metrics(self) -> None:
        num_tracks = self.config["num_tracks"]
        self.train_metrics = TracksMetrics(num_tracks, "train")
        self.val_metrics = TracksMetrics(num_tracks, "val")
        self.test_metrics = TracksMetrics(num_tracks, "test")
        self.train_metrics_epoch = TracksMetrics(num_tracks, "train")
        self.val_metrics_epoch = TracksMetrics(num_tracks, "val")
        self.test_metrics_epoch = TracksMetrics(num_tracks, "test")

    def _resolve_training_phase(self, epoch: int) -> int:
        if self.moe_warmup_epochs <= 0:
            return self.PHASE_FULL_FINETUNE
        if epoch < self.moe_warmup_epochs:
            return self.PHASE_HEAD_WARMUP
        return self.PHASE_FULL_FINETUNE

    def _set_training_phase(self, phase: int) -> None:
        if phase == self.PHASE_HEAD_WARMUP:
            self.mymodel.set_backbone_trainable(False)
            self.mymodel.set_moe_head_trainable(router_only=self.moe_warmup_router_only)
        else:
            self.mymodel.set_backbone_trainable(True)
            self.mymodel.set_moe_head_trainable(router_only=False)
        self._training_phase = phase

    def _apply_initial_trainable_state(self) -> None:
        if self.moe_warmup_epochs > 0:
            self._set_training_phase(self._resolve_training_phase(0))
            return

        if self.config.get("freeze_backbone", False):
            self.mymodel.set_backbone_trainable(False)
            self.mymodel.set_moe_head_trainable(router_only=False)
            self._training_phase = self.PHASE_HEAD_WARMUP
            return

        self._set_training_phase(self.PHASE_FULL_FINETUNE)

    def _create_optimizer(self) -> AdamW:
        if self._training_phase == self.PHASE_HEAD_WARMUP:
            head_params = [
                p for p in self.mymodel.bigwig_head.parameters() if p.requires_grad
            ]
            param_groups = [{"params": head_params, "lr": self.head_learning_rate}]
        else:
            trainable_params = [p for p in self.mymodel.parameters() if p.requires_grad]
            param_groups = [{"params": trainable_params, "lr": self.config["learning_rate"]}]

        if not param_groups or not any(group["params"] for group in param_groups):
            raise RuntimeError("MoE model has no trainable parameters.")

        return AdamW(param_groups, weight_decay=self.config["weight_decay"])

    def _cosine_t_max(self) -> int:
        max_steps = int(self.config["max_steps"])
        if self.moe_warmup_epochs <= 0:
            return max_steps

        if self.trainer is not None:
            remaining = max_steps - int(self.trainer.global_step)
            return max(1, remaining)

        return max_steps

    def configure_optimizers(self):
        optimizer = self._create_optimizer()

        # Head warmup: constant LR, no scheduler.
        if (
            self.moe_warmup_epochs > 0
            and self._training_phase == self.PHASE_HEAD_WARMUP
        ):
            return optimizer

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self._cosine_t_max(),
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _reconfigure_optimizers(self) -> None:
        if self.trainer is None:
            return

        from lightning_fabric.utilities.optimizer import _optimizers_to_device
        from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers

        optimizers, lr_scheduler_configs = _init_optimizers_and_lr_schedulers(self)
        strategy = self.trainer.strategy
        strategy.optimizers = optimizers
        strategy.lr_scheduler_configs = lr_scheduler_configs
        _optimizers_to_device(strategy.optimizers, strategy.root_device)

    def on_train_epoch_start(self) -> None:
        if self.moe_warmup_epochs <= 0:
            return

        phase = self._resolve_training_phase(self.current_epoch)
        if phase == self._training_phase:
            return

        self._set_training_phase(phase)
        self._reconfigure_optimizers()

        phase_name = (
            "head_warmup"
            if phase == self.PHASE_HEAD_WARMUP
            else "full_finetune"
        )
        if self.trainer.is_global_zero:
            remaining_steps = self._cosine_t_max()
            print(
                f"[MoE] Switched to training phase {phase} ({phase_name}) "
                f"at epoch {self.current_epoch}; cosine T_max={remaining_steps}."
            )
        self.log(
            "moe_training_phase",
            float(phase),
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def _sync_scalar_mean(self, value: torch.Tensor) -> torch.Tensor:
        synced = value.detach().clone()
        if self.trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(synced, op=dist.ReduceOp.SUM)
            synced /= self.trainer.world_size
        return synced

    def _compute_loss(
        self,
        outputs: dict,
        bigwig_targets: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bigwig_logits = outputs["bigwig_tracks_logits"]
        loss = poisson_multinomial_loss(logits=bigwig_logits, targets=bigwig_targets)

        aux_loss, aux_logs = compute_moe_aux_loss(
            outputs,
            self.config.get("moe_config", {}),
        )
        if aux_loss is not None:
            loss = loss + aux_loss
            for name, value in aux_logs.items():
                synced_value = self._sync_scalar_mean(value)
                self.log(
                    f"{stage}_{name}",
                    synced_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

        return loss, bigwig_logits

    def training_step(self, batch, batch_idx):
        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]
        outputs = self(tokens)
        loss, bigwig_logits = self._compute_loss(outputs, bigwig_targets, "train")

        synced_loss = self._sync_scalar_mean(loss)
        self.log(
            "train_loss",
            synced_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.train_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.train_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]
        outputs = self(tokens)
        loss, bigwig_logits = self._compute_loss(outputs, bigwig_targets, "val")

        synced_loss = self._sync_scalar_mean(loss)
        self.log(
            "val_loss",
            synced_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.val_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.val_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        tokens, bigwig_targets = batch["tokens"], batch["bigwig_targets"]
        outputs = self(tokens)
        loss, bigwig_logits = self._compute_loss(outputs, bigwig_targets, "test")

        synced_loss = self._sync_scalar_mean(loss)
        self.log(
            "test_loss",
            synced_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.test_metrics.update(bigwig_logits, bigwig_targets, loss.item())
        self.test_metrics_epoch.update(bigwig_logits, bigwig_targets, loss.item())
        return loss

    def predict_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        outputs = self(tokens)
        return outputs["bigwig_tracks_logits"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % 100 != 0:
            return

        train_metrics_result = self.train_metrics.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "train_avg_loss",
            train_metrics_result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "train_avg_pearson",
            train_metrics_result["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.train_metrics.reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % 100 != 0:
            return

        val_metrics_result = self.val_metrics.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "val_avg_loss",
            val_metrics_result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "val_avg_pearson",
            val_metrics_result["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.val_metrics.reset()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % 100 != 0:
            return

        test_metrics_result = self.test_metrics.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "test_avg_loss",
            test_metrics_result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "test_avg_pearson",
            test_metrics_result["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.test_metrics.reset()

    def on_train_epoch_end(self):
        train_epoch_metrics = self.train_metrics_epoch.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "train_epoch_loss",
            train_epoch_metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_epoch_pearson",
            train_epoch_metrics["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.train_metrics_epoch.reset()

    def on_validation_epoch_end(self):
        val_epoch_metrics = self.val_metrics_epoch.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "val_epoch_loss",
            val_epoch_metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "val_epoch_pearson",
            val_epoch_metrics["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.val_metrics_epoch.reset()

    def on_test_epoch_end(self):
        test_epoch_metrics = self.test_metrics_epoch.compute(
            sync_dist=self.trainer.world_size > 1
        )
        self.log(
            "test_epoch_loss",
            test_epoch_metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "test_epoch_pearson",
            test_epoch_metrics["mean/pearson"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.test_metrics_epoch.reset()
