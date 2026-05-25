"""Finetune NTv3 model package."""

from .analysis import visualization_channels, visualization_channels_means, plot_moe_expert_routing
from .backbone import MyDataModule, MyDataModule_NTv3, MyModel, MyModelMOE
from .dataset import (
    GenomeBigWigDataset,
    GenomeBigWigDataset_my,
    GenomeBigWigDataset_myIterable,
    GenomeBigWigDataset_Nucl_Depend,
)
from .decorator import NUCConfig, NUC_CONFIG
from .head import (
    GatedHead,
    HFModelWithHead,
    HFModelWithHead_Infer,
    HFModelWithHead_Saliency,
    LinearHead,
    SaliencyComputer,
    build_bigwig_head,
)
from .loss import poisson_loss, poisson_multinomial_loss, safe_for_grad_log_torch
from .metrics import InferMetrics, TracksMetrics
from .moe import (
    HardMoEHead,
    HFModelWithMoE,
    HFModelWithMoE_Infer,
    MoEHeadBase,
    SoftMoEHead,
    build_moe_head,
    compute_moe_aux_loss,
    load_moe_config,
    switch_load_balance_loss,
)
from .utils import (
    crop_center,
    create_targets_scaling_fn,
    gene_filter,
    init_config,
    init_model,
    init_moe_model,
    load_Data,
    load_ckpt_with_compile,
    load_config,
    transform_fn,
)

__all__ = [
    # analysis
    "visualization_channels",
    "visualization_channels_means",
    "plot_moe_expert_routing",
    # backbone
    "MyDataModule",
    "MyDataModule_NTv3",
    "MyModel",
    "MyModelMOE",
    # dataset
    "GenomeBigWigDataset",
    "GenomeBigWigDataset_my",
    "GenomeBigWigDataset_myIterable",
    "GenomeBigWigDataset_Nucl_Depend",
    # decorator
    "NUCConfig",
    "NUC_CONFIG",
    # head
    "LinearHead",
    "GatedHead",
    "build_bigwig_head",
    "HFModelWithHead",
    "HFModelWithHead_Infer",
    "HFModelWithHead_Saliency",
    "SaliencyComputer",
    # loss
    "poisson_loss",
    "poisson_multinomial_loss",
    "safe_for_grad_log_torch",
    # metrics
    "TracksMetrics",
    "InferMetrics",
    # moe
    "load_moe_config",
    "switch_load_balance_loss",
    "compute_moe_aux_loss",
    "MoEHeadBase",
    "SoftMoEHead",
    "HardMoEHead",
    "build_moe_head",
    "HFModelWithMoE",
    "HFModelWithMoE_Infer",
    # utils
    "load_config",
    "init_config",
    "init_model",
    "init_moe_model",
    "gene_filter",
    "load_Data",
    "crop_center",
    "create_targets_scaling_fn",
    "transform_fn",
    "load_ckpt_with_compile",
]
