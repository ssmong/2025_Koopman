import torch
import torch.nn as nn
from typing import Optional, Tuple

class BaseLoss(nn.Module):
    def __init__(self, 
                 weight: float = 1.0, 
                 weight_decay: float = 0.99, 
                 key_pred: str = None, 
                 key_target: str = None,
                 n_step_max: int = 100,
                 **kwargs):
        super().__init__()
        self.weight = weight
        self.weight_decay = weight_decay
        self.key_pred = key_pred
        self.key_target = key_target
        self.n_step_max = n_step_max
        
        self.loss_fn = nn.MSELoss(reduction='none')

        # Pre-compute decay weights
        if self.weight_decay != 1.0:
            steps = torch.arange(n_step_max)
            weights = self.weight_decay ** steps
            # register_buffer handles device movement automatically
            # persistent=False prevents saving this to state_dict
            self.register_buffer('decay_weights', weights, persistent=False)
        else:
            self.register_buffer('decay_weights', None, persistent=False)

    def get_inputs(self, results: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract prediction and target tensors from results dictionary."""
        if self.key_pred is None or self.key_target is None:
            raise ValueError(f"key_pred and key_target must be specified. Current: {self.key_pred}, {self.key_target}")
        
        pred = results.get(self.key_pred)
        target = results.get(self.key_target)

        if pred is None:
            raise KeyError(f"Prediction key '{self.key_pred}' not found in results.")
        if target is None:
            raise KeyError(f"Target key '{self.key_target}' not found in results.")

        return pred, target

    def apply_weight_decay(self, loss_per_step: torch.Tensor) -> torch.Tensor:
        if loss_per_step.dim() > 1:
            loss_per_step = loss_per_step.mean(dim=0)
        T = loss_per_step.size(0)
        if self.weight_decay == 1.0:
            weighted_loss = loss_per_step.mean()
        else:
            weights = self.decay_weights[:T].to(loss_per_step.device)
            weighted_loss = (loss_per_step * weights).sum() / weights.sum()
        return weighted_loss

    def forward(self, results: dict) -> torch.Tensor:
        """
        Default forward pass: MSE Loss with weight decay.
        """
        pred, target = self.get_inputs(results)
        
        # pred, target: [B, T, D]
        # MSE per element: [B, T, D]
        # In this codebase, T = N + 1, where N is the number of prediction steps.
        loss = self.loss_fn(pred, target)
        loss = loss.mean(dim=-1)    # Average over dimensions (D): [B, T]
        final_loss = self.apply_weight_decay(loss)
        
        return self.weight * final_loss
