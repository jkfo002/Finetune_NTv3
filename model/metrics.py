import torch
from torchmetrics import PearsonCorrCoef
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class TracksMetrics:
    """Metrics to handle multi-track pearson correlations and losses"""
    
    def __init__(self, num_tracks: int, split: str, device: str):
        self.num_tracks = num_tracks
        self.split = split

        # Initialise metrics 
        self.pearson = PearsonCorrCoef(num_outputs=self.num_tracks).to(device)
        self.pearson.set_dtype(torch.float64) # Use float64 for improved numerical stability
        self.losses = []

        # Record mean metrics per logging interval
        self.step_idxs = []
        self.mean_pearsons = []
        self.mean_losses = []
    
    def reset(self):
        self.pearson.reset()
        self.losses = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss : Optional[float] = None
    ):
        """
        Update the metrics with predictions and targets of shape (..., num_tracks) and a scalar loss.
        """
        # Flatten batch and sequence dimensions
        pred_flat = predictions.detach().reshape(-1, self.num_tracks).to(torch.float64)  # (N, num_tracks)
        target_flat = targets.detach().reshape(-1, self.num_tracks).to(torch.float64)  # (N, num_tracks)
        
        # Update metrics
        self.pearson.update(pred_flat, target_flat)
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute the pearson correlations and loss and return a dictionary of metrics."""
        # Per-track Pearson correlations
        correlations = self.pearson.compute().cpu().numpy()
        metrics_dict = {
            f"track{i}/pearson": correlations[i] for i in range(self.num_tracks)
        }
        metrics_dict["mean/pearson"] = correlations.mean()
        
        # Mean loss
    
        metrics_dict["loss"] = np.mean(self.losses)
        
        return metrics_dict


class InferMetrics:
    def __init__(self, num_tracks: int):
        self.num_tracks = num_tracks

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Just store or compute immediately
        self.predictions = predictions.detach()
        self.targets = targets.detach()

    def pearson_corrcoef_batch(
        self, 
        pred: torch.Tensor, target: torch.Tensor, 
        eps=1e-8
    ):
        """
        Compute Pearson correlation coefficient for each track in a batch.

        Args:
            pred:   (B, L, N)
            target: (B, L, N)

        Returns:
            corr:   (B, N)  # Pearson r per sample and track
        """
        pred = pred.to(torch.float64)
        target = target.to(torch.float64)

        # Mean over sequence length (L)
        pred_mean = pred.mean(dim=1, keepdim=True)      # (B, 1, N)
        target_mean = target.mean(dim=1, keepdim=True)  # (B, 1, N)

        pred_centered = pred - pred_mean                # (B, L, N)
        target_centered = target - target_mean          # (B, L, N)

        # Numerator: covariance (mean of product)
        cov = (pred_centered * target_centered).mean(dim=1)  # (B, N)

        # Denominator: stds
        pred_std = pred_centered.std(dim=1, unbiased=False)   # (B, N)
        target_std = target_centered.std(dim=1, unbiased=False)  # (B, N)

        corr = cov / (pred_std * target_std + eps)      # (B, N)

        # Clamp to [-1, 1] for numerical stability
        corr = torch.clamp(corr, -1.0, 1.0)

        return corr

    def compute(self) -> list[Dict[str, float]]:
        # Shape: (B, L, N)
        corr = self.pearson_corrcoef_batch(self.predictions, self.targets)  # (B, N)
        corr = corr.cpu().numpy()

        metrics_list = []
        for b in range(corr.shape[0]):
            batch_metrics_dict = {
                f"track{i}/pearson": float(corr[b, i]) for i in range(self.num_tracks)
            }
            batch_metrics_dict["mean/pearson"] = float(corr[b].mean())
            metrics_list.append(batch_metrics_dict)

        return metrics_list