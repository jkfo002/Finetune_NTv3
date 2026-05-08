import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional

class TracksMetrics:
    """Metrics to handle multi-track pearson correlations and losses"""
    
    def __init__(self, num_tracks: int, split: str):
        self.num_tracks = num_tracks
        self.split = split
        self.reset()
    
    def reset(self):
        self.sum_x = None
        self.sum_y = None
        self.sum_x2 = None
        self.sum_y2 = None
        self.sum_xy = None
        self.n = None
        self.loss_sum = None
        self.loss_count = None

    def _ensure_state(self, device: torch.device):
        if self.sum_x is None:
            zeros = torch.zeros(self.num_tracks, dtype=torch.float64, device=device)
            self.sum_x = zeros.clone()
            self.sum_y = zeros.clone()
            self.sum_x2 = zeros.clone()
            self.sum_y2 = zeros.clone()
            self.sum_xy = zeros.clone()
            self.n = torch.zeros(1, dtype=torch.float64, device=device)
            self.loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
            self.loss_count = torch.zeros(1, dtype=torch.float64, device=device)
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss : Optional[float] = None
    ):
        """
        Update the metrics with predictions and targets of shape (..., num_tracks) and a scalar loss.
        """
        self._ensure_state(predictions.device)
        pred_flat = predictions.detach().reshape(-1, self.num_tracks).to(dtype=torch.float64)
        target_flat = targets.detach().reshape(-1, self.num_tracks).to(dtype=torch.float64)

        self.sum_x += pred_flat.sum(dim=0)
        self.sum_y += target_flat.sum(dim=0)
        self.sum_x2 += (pred_flat ** 2).sum(dim=0)
        self.sum_y2 += (target_flat ** 2).sum(dim=0)
        self.sum_xy += (pred_flat * target_flat).sum(dim=0)
        self.n += pred_flat.new_tensor([pred_flat.shape[0]], dtype=torch.float64)

        if loss is not None:
            self.loss_sum += pred_flat.new_tensor([float(loss)], dtype=torch.float64)
            self.loss_count += pred_flat.new_tensor([1.0], dtype=torch.float64)
    
    def compute(self, sync_dist: bool = False) -> Dict[str, float]:
        """Compute the pearson correlations and loss and return a dictionary of metrics."""
        if self.sum_x is None:
            if sync_dist and dist.is_available() and dist.is_initialized():
                device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
                self._ensure_state(device)
            else:
                metrics_dict = {f"track{i}/pearson": 0.0 for i in range(self.num_tracks)}
                metrics_dict["mean/pearson"] = 0.0
                metrics_dict["loss"] = 0.0
                return metrics_dict

        stats = torch.cat([
            self.sum_x,
            self.sum_y,
            self.sum_x2,
            self.sum_y2,
            self.sum_xy,
            self.n,
            self.loss_sum,
            self.loss_count,
        ]).clone()

        if sync_dist and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        offset = 0
        sum_x = stats[offset : offset + self.num_tracks]
        offset += self.num_tracks
        sum_y = stats[offset : offset + self.num_tracks]
        offset += self.num_tracks
        sum_x2 = stats[offset : offset + self.num_tracks]
        offset += self.num_tracks
        sum_y2 = stats[offset : offset + self.num_tracks]
        offset += self.num_tracks
        sum_xy = stats[offset : offset + self.num_tracks]
        offset += self.num_tracks
        n = stats[offset : offset + 1]
        offset += 1
        loss_sum = stats[offset : offset + 1]
        offset += 1
        loss_count = stats[offset : offset + 1]

        eps = 1e-12
        numerator = n * sum_xy - sum_x * sum_y
        denominator = torch.sqrt((n * sum_x2 - sum_x ** 2).clamp_min(0.0) * (n * sum_y2 - sum_y ** 2).clamp_min(0.0) + eps)
        correlations = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
        correlations = correlations.clamp(-1.0, 1.0).cpu().numpy()
        metrics_dict = {
            f"track{i}/pearson": float(correlations[i]) for i in range(self.num_tracks)
        }
        metrics_dict["mean/pearson"] = float(correlations.mean())
        
        if loss_count.item() > 0:
            metrics_dict["loss"] = float((loss_sum / loss_count).item())
        else:
            metrics_dict["loss"] = 0.0
        
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