import torch
from src.losses.components.base import BaseLoss

class VeroneseTraceLoss(BaseLoss):
    def __init__(self, veronese_dim: int = 10, start_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.veronese_dim = veronese_dim
        self.start_dim = start_dim
        
        triu_idx = torch.triu_indices(4, 4)
        diag_mask = (triu_idx[0] == triu_idx[1])
        self.diag_indices = torch.where(diag_mask)[0].tolist()

    def forward(self, results: dict) -> torch.Tensor:
        z_pred, _ = self.get_inputs(results)
        
        # Slice the veronese part from z
        z_v = z_pred[..., self.start_dim : self.start_dim + self.veronese_dim]
        
        squared_terms = z_v[..., self.diag_indices]
        trace = squared_terms.sum(dim=-1)
        
        loss = (trace - 1.0) ** 2
        
        return self.weight * self.apply_weight_decay(loss)
