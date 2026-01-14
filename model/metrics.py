import torch
from torchmetrics import PearsonCorrCoef
import numpy as np
import pandas as pd
from typing import List, Dict

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
        loss: float
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