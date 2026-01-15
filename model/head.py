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
    
    def forward(self, tokens: torch.Tensor, return_logits_direct: bool = False, **kwargs) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        
        # Forward through backbone
        outputs = self.backbone(
            input_ids=tokens,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if return_logits_direct:
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

class HFModelWithHead_Saliency(nn.Module):
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
    
    def forward(self, input_embeds: torch.Tensor, return_logits_direct: bool = False, **kwargs) -> Dict[str, Optional[Union[torch.Tensor, np.ndarray]]]:
        
        # Forward through backbone
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=input_embeds,
            return_dict=True,
        )
        if return_logits_direct:
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

class SaliencyComputer:
    """Compute gradient-based saliency maps (new model API)."""

    def __init__(
        self, 
        model: nn.Module,
        tokenizer: AutoTokenizer, 
        sequence_length: int,
        track_indices: Optional[List[int]] = None, 
        device: Optional[str] = None, 
        region: Optional[tuple[int, int]] = None
    ):
        """Initialize saliency computer.

        Args:
            model: The model to compute saliency for. only BACKBONE is used.
            tokenizer: Tokenizer for input sequences.
            track_indices: List of track indices to compute saliency for.
            device: Device to use. If None, auto-detects. Defaults to None.
            promoter_window_bp: Window size in base pairs. Defaults to 512.
            token_resolution: Base pairs per token. Defaults to 128.
        """
        # Auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.track_indices = track_indices
        self.device = device
        self.region = region

        self.vocab_size = tokenizer.vocab_size
        
    def compute_saliency(self, sequence: str):
        """Compute saliency map for given sequence.

        Args:
            sequence: DNA sequence string to compute saliency for.

        Returns:
            tuple: (gradients, one_hot_sequence) where gradients is the
                saliency scores and one_hot_sequence is the one-hot encoded
                input sequence.
        """
        # Tokenize
        batch = self.tokenizer(
            [sequence],
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        token_ids = batch["input_ids"][0].to(self.device)

        # Create one-hot encoding
        one_hot = F.one_hot(
            token_ids, num_classes=self.vocab_size
        ).float()
        one_hot.requires_grad_(True)

        # Forward pass with gradient tracking
        # Get embedding layer (try common paths)
        embedding_layer = None
        if hasattr(self.model.backbone, 'core') and hasattr(self.model.backbone.core, 'embed_layer'):
            embedding_layer = self.model.backbone.core.embed_layer
        elif hasattr(self.model.backbone, 'embeddings'):
            embedding_layer = self.model.backbone.embeddings
        elif hasattr(self.model.backbone, 'embed_tokens'):
            embedding_layer = self.model.backbone.embed_tokens

        if embedding_layer is None:
            raise ValueError(
                "Could not find embedding layer in model. "
                "Saliency computation requires access to embedding weights."
            )

        inputs_embeds = torch.matmul(
            one_hot, embedding_layer.weight
        ).unsqueeze(0)

        # Forward pass with inputs_embeds (NOT input_ids - model requires exactly one)
        outputs = self.model(inputs_embeds=inputs_embeds)

        # Access bigwig logits using attribute-style access
        logits = outputs['bigwig_tracks_logits']

        # Focus on center window
        if self.region is None:
            prom_logits = logits
        else:
            prom_start, prom_end = self.region
            prom_logits = logits[:, prom_start:prom_end, :] 

        # Select tracks and compute score
        if self.track_indices is not None:
            selected_logits = prom_logits[:, :, self.track_indices]
        else:
            selected_logits = prom_logits
        score = torch.log(selected_logits.sum()/selected_logits.shape[2] + 1e-8)

        # Backward pass
        score.backward()

        gradients = one_hot.grad.cpu().detach().numpy()
        one_hot_np = one_hot.detach().cpu().numpy()

        return gradients, one_hot_np