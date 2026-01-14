import torch

def poisson_loss(ytrue: torch.Tensor, ypred: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Poisson loss per element: ypred - ytrue * log(ypred)."""
    return ypred - ytrue * torch.log(ypred + epsilon)


def safe_for_grad_log_torch(x: torch.Tensor) -> torch.Tensor:
    """Guarantees that the log is defined for all x > 0 in a differentiable way."""
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))


def poisson_multinomial_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    shape_loss_coefficient: float = 5.0,
    epsilon: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Regression loss for bigwig tracks (Poisson-Multinomial). The logits and targets are
    expected to be of shape (batch, seq_length, num_tracks).
    """
    batch_size, seq_length, num_tracks = logits.shape
    
    # Scale loss: Poisson loss on total counts per sequence per track
    # Sum over sequence dimension (axis=1)
    sum_pred = logits.sum(dim=1)  # (batch, num_tracks)
    sum_true = targets.sum(dim=1)  # (batch, num_tracks)
    
    # Compute poisson loss per (batch, track)
    scale_loss = poisson_loss(sum_true, sum_pred, epsilon=epsilon)  # (batch, num_tracks)
    
    # Normalize by sequence length
    scale_loss = scale_loss / (seq_length + epsilon)
    
    # Average over batch and tracks
    scale_loss = scale_loss.mean()
    
    # Shape loss: Multinomial loss
    # Add epsilon to all positions
    predicted_counts = logits + epsilon
    targets_with_epsilon = targets + epsilon
    
    # Normalize predictions to get probabilities
    denom = predicted_counts.sum(dim=1, keepdim=True) + epsilon  # (batch, 1, num_tracks)
    p_pred = predicted_counts / denom
    
    # Compute shape loss: -sum(targets * log(p_pred))
    pl_pred = safe_for_grad_log_torch(p_pred)
    shape_loss = -(targets_with_epsilon * pl_pred)
    
    # Sum over all dimensions and normalize by total number of positions
    shape_denom = batch_size * seq_length * num_tracks + epsilon
    shape_loss = shape_loss.sum() / shape_denom
    
    # Combine losses
    loss = shape_loss + scale_loss / shape_loss_coefficient

    return loss