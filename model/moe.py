import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM

from .utils import crop_center

MOE_ROUTINGS = ("soft", "hard")


def load_moe_config(config_path: str) -> Dict[str, Any]:
    """Load MoE design from MOE.json."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"MoE config not found: {config_path}")

    with path.open("r") as handle:
        cfg = json.load(handle)

    if "num_experts" not in cfg:
        raise ValueError("MOE.json must contain 'num_experts'.")

    cfg.setdefault("routing", "soft")
    cfg.setdefault("top_k", 1)
    cfg.setdefault("loss", {})

    routing = cfg["routing"]
    if routing not in MOE_ROUTINGS:
        raise ValueError(f"'routing' must be one of {MOE_ROUTINGS}, got {routing!r}.")

    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]
    if not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError(f"'num_experts' must be a positive integer, got {num_experts!r}.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"'top_k' must be a positive integer, got {top_k!r}.")
    if routing == "hard" and top_k > num_experts:
        raise ValueError(f"'top_k' ({top_k}) cannot exceed 'num_experts' ({num_experts}).")

    experts = cfg.get("experts", [])
    if experts and len(experts) != num_experts:
        raise ValueError(
            f"Length of 'experts' metadata ({len(experts)}) must match 'num_experts' ({num_experts})."
        )

    expert_design = cfg.get("expert_design")
    if expert_design:
        design_path = Path(expert_design)
        if not design_path.is_file():
            design_path = path.parent / expert_design
        if design_path.is_file():
            with design_path.open("r") as handle:
                cfg["expert_design_data"] = json.load(handle)

    return cfg


def switch_load_balance_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-style load balancing loss to avoid expert collapse."""
    probs = F.softmax(router_logits, dim=-1)
    top1 = probs.argmax(dim=-1)
    counts = torch.bincount(top1.reshape(-1), minlength=num_experts).float()
    f = counts / counts.sum().clamp(min=1.0)
    p = probs.mean(dim=(0, 1))
    return num_experts * (f * p).sum()


def compute_moe_aux_loss(
    outputs: Dict[str, torch.Tensor],
    moe_config: Dict[str, Any],
) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute optional MoE auxiliary losses configured in MOE.json."""
    router_logits = outputs.get("router_logits")
    if router_logits is None:
        return None, {}

    loss_cfg = moe_config.get("loss", {})
    aux_loss: Optional[torch.Tensor] = None
    logs: Dict[str, torch.Tensor] = {}

    load_balance_cfg = loss_cfg.get("load_balance", {})
    if load_balance_cfg.get("enabled", False):
        num_experts = moe_config["num_experts"]
        lb_loss = switch_load_balance_loss(router_logits, num_experts)
        lb_weight = load_balance_cfg.get("weight", 0.01)
        weighted_lb_loss = lb_weight * lb_loss
        aux_loss = weighted_lb_loss if aux_loss is None else aux_loss + weighted_lb_loss
        logs["moe_load_balance_loss"] = lb_loss.detach()
        logs["moe_weighted_load_balance_loss"] = weighted_lb_loss.detach()

    return aux_loss, logs


class MoEHeadBase(nn.Module):
    """Per-position MoE head with shared router and track experts."""

    routing: str = "soft"

    def __init__(self, embed_dim: int, num_labels: int, moe_config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.moe_config = moe_config
        self.num_experts = moe_config["num_experts"]
        self.top_k = moe_config["top_k"]
        self.expert_metadata = moe_config.get("experts", [])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.router = nn.Linear(embed_dim, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, num_labels) for _ in range(self.num_experts)
        ])

    def _combine_experts(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=-2,
        )

        if self.routing == "soft":
            combined = (probs.unsqueeze(-1) * expert_outputs).sum(dim=-2)
            return combined, probs

        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.num_labels)
        selected = torch.gather(expert_outputs, dim=-2, index=gather_idx)
        combined = (topk_probs.unsqueeze(-1) * selected).sum(dim=-2)
        return combined, probs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.layer_norm(x)
        router_logits = self.router(x)
        combined, router_probs = self._combine_experts(x, router_logits)
        return {
            "logits": F.softplus(combined),
            "router_logits": router_logits,
            "router_probs": router_probs,
        }


class SoftMoEHead(MoEHeadBase):
    routing = "soft"


class HardMoEHead(MoEHeadBase):
    routing = "hard"


def build_moe_head(
    embed_dim: int,
    num_tracks: int,
    moe_config: Dict[str, Any],
) -> nn.Module:
    routing = moe_config["routing"]
    if routing == "soft":
        return SoftMoEHead(embed_dim, num_tracks, moe_config)
    if routing == "hard":
        return HardMoEHead(embed_dim, num_tracks, moe_config)
    raise ValueError(f"Unknown MoE routing: {routing!r}")


def _forward_moe_head(
    model: nn.Module,
    embedding: torch.Tensor,
) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
    head_out = model.bigwig_head(embedding)
    return {
        "embedding": embedding,
        "bigwig_tracks_logits": head_out["logits"],
        "router_logits": head_out["router_logits"],
        "router_probs": head_out["router_probs"],
    }


class HFModelWithMoE(nn.Module):
    """HF backbone + MoE bigwig head configured by MOE.json."""

    def __init__(
        self,
        model_name: str,
        num_tracks: int,
        moe_config: Dict[str, Any],
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=self.config,
        )
        self.backbone = torch.compile(backbone)

        self.keep_target_center_fraction = keep_target_center_fraction
        self.moe_config = moe_config
        embed_dim = self.config.embed_dim

        self.bigwig_head = build_moe_head(embed_dim, num_tracks, moe_config)
        self.model_name = model_name

    def forward(
        self,
        tokens: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]

        if self.keep_target_center_fraction < 1.0:
            embedding = crop_center(embedding, self.keep_target_center_fraction)

        return _forward_moe_head(self, embedding)


class HFModelWithMoE_Infer(nn.Module):
    """HF backbone + MoE head for inference."""

    def __init__(
        self,
        model_name: str,
        num_tracks: int,
        moe_config: Dict[str, Any],
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=self.config,
        )
        self.backbone = backbone

        self.keep_target_center_fraction = keep_target_center_fraction
        self.moe_config = moe_config
        embed_dim = self.config.embed_dim

        self.bigwig_head = build_moe_head(embed_dim, num_tracks, moe_config)
        self.model_name = model_name

    def forward(
        self,
        tokens: torch.Tensor,
        return_logits_direct: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        outputs = self.backbone(
            input_ids=tokens,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if return_logits_direct:
            return outputs

        embedding = outputs.hidden_states[-1]
        if self.keep_target_center_fraction < 1.0:
            embedding = crop_center(embedding, self.keep_target_center_fraction)

        return _forward_moe_head(self, embedding)
