import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union
import numpy as np
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from .utils import crop_center

class LinearHead(nn.Module):
    """A linear head that predicts one scalar value per track."""
    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.head(x)
        x = F.softplus(x)  # Ensure positive values
        return x


class HFModelWithHead(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head."""
    
    def __init__(
        self,
        model_name: str,
        num_tracks: int,
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.config,
        )
        self.backbone = torch.compile(backbone)
        
        self.keep_target_center_fraction = keep_target_center_fraction
        embed_dim = self.config.embed_dim
        
        # Bigwig head (NTv3 outputs at single-nucleotide resolution)
        self.bigwig_head = LinearHead(embed_dim, num_tracks)
        self.model_name = model_name
    
    def forward(self, tokens: torch.Tensor, **kwargs) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        # Forward through backbone
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]  # Last hidden state
        
        # Crop to center fraction
        if self.keep_target_center_fraction < 1.0:
            embedding = crop_center(embedding, self.keep_target_center_fraction)
        
        # Predict bigwig tracks
        bigwig_logits = self.bigwig_head(embedding)
        
        return {
            "embedding": embedding,
            "bigwig_tracks_logits": bigwig_logits
        }

class HFModelWithHead_Infer(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head.
    This model is used for inference only.
    """
    
    def __init__(
        self,
        model_name: str,
        num_tracks: int,
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()
        
        # Load config and model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.config,
        )
        # self.backbone = torch.compile(backbone)
        self.backbone = backbone
        
        self.keep_target_center_fraction = keep_target_center_fraction
        embed_dim = self.config.embed_dim
        
        # Bigwig head (NTv3 outputs at single-nucleotide resolution)
        self.bigwig_head = LinearHead(embed_dim, num_tracks)
        self.model_name = model_name
    
    def forward(self, tokens: torch.Tensor, return_dict: bool = False, **kwargs) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        
        # Forward through backbone
        outputs = self.backbone(
            input_ids=tokens,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if return_dict:
            return outputs

        embedding = outputs.hidden_states[-1]  # Last hidden state
        
        # Crop to center fraction
        if self.keep_target_center_fraction < 1.0:
            embedding = crop_center(embedding, self.keep_target_center_fraction)
        
        # Predict bigwig tracks
        bigwig_logits = self.bigwig_head(embedding)
        
        return {
            "embedding": embedding,
            "bigwig_tracks_logits": bigwig_logits
        }