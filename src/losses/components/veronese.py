import torch
import torch.nn.functional as F
import math
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

class VeroneseConsistencyLoss(BaseLoss):
    """
    Consistency Loss: || v_pred - Veronese(q_pred) ||^2
    """
    def __init__(self, 
                 veronese_dim: int = 10, 
                 veronese_start_dim: int = 7, # 4 (quat) + 3 (omega)
                 quat_start_dim: int = 0,
                 quat_dim: int = 4,
                 scale_off_diagonals: bool = True, # True for V3, False for V4
                 **kwargs):
        super().__init__(**kwargs)
        self.veronese_dim = veronese_dim
        self.veronese_start_dim = veronese_start_dim
        self.quat_start_dim = quat_start_dim
        self.quat_dim = quat_dim
        self.scale_off_diagonals = scale_off_diagonals

        triu_idx = torch.triu_indices(self.quat_dim, self.quat_dim)
        self.register_buffer('idx_i', triu_idx[0], persistent=False)
        self.register_buffer('idx_j', triu_idx[1], persistent=False)
        
        scale = torch.ones(self.veronese_dim)
        if self.scale_off_diagonals:
            mask_cross = self.idx_i != self.idx_j
            scale[mask_cross] = math.sqrt(2)
        self.register_buffer('scale', scale, persistent=False)

    def forward(self, results: dict) -> torch.Tensor:
        # z_pred: [B, T, latent_dim]
        z_pred, _ = self.get_inputs(results)
        
        # Extract q and v from z
        q_pred = z_pred[..., self.quat_start_dim : self.quat_start_dim + self.quat_dim]
        v_pred = z_pred[..., self.veronese_start_dim : self.veronese_start_dim + self.veronese_dim]
        
        # Normalize q to ensure unit quaternion property
        q_norm = F.normalize(q_pred, p=2, dim=-1)
        
        # Compute explicit Veronese map
        v_target = q_norm[..., self.idx_i] * q_norm[..., self.idx_j] * self.scale
        
        loss = ((v_pred - v_target) ** 2).sum(dim=-1).mean()
        
        return self.weight * self.apply_weight_decay(loss)
