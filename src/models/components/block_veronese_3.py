import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import hydra
import torch.nn.functional as F

class VeroneseLifting3(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int], 
                 omega_indices: List[int],
                 backbone: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        
        self.quat_indices = quat_indices
        self.omega_indices = omega_indices
        self.n_quat = len(quat_indices)
        self.n_omega = len(omega_indices)
        self.veronese_dim = (self.n_quat * (self.n_quat + 1)) // 2 
        
        # z structure: [q, w, veronese(q), z_nn]
        self.fixed_dim = self.n_quat + self.n_omega + self.veronese_dim
        self.nn_out_dim = out_features - self.fixed_dim
        
        if self.nn_out_dim < 0:
             raise ValueError(f"Output features ({out_features}) must be >= Fixed dim ({self.fixed_dim}) = |q| + |w| + |Veronese(q)|.")

        if self.nn_out_dim > 0 and backbone is not None:
            self.backbone = hydra.utils.instantiate(
                backbone,
                in_features=in_features,
                out_features=self.nn_out_dim
            )
        else:
            self.backbone = None

        triu_idx = torch.triu_indices(self.n_quat, self.n_quat)
        self.register_buffer('idx_i', triu_idx[0], persistent=False)
        self.register_buffer('idx_j', triu_idx[1], persistent=False)
        
        scale = torch.ones(self.veronese_dim)
        mask_cross = self.idx_i != self.idx_j
        scale[mask_cross] = math.sqrt(2)
        self.register_buffer('scale', scale, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x[..., self.quat_indices]
        w = x[..., self.omega_indices]
        
        # Safe normalization to prevent NaN gradients if q is 0
        q = F.normalize(q, p=2, dim=-1)
        
        z_veronese = q[..., self.idx_i] * q[..., self.idx_j] * self.scale
        
        parts = [q, w, z_veronese]
        
        if self.backbone is not None:
            z_nn = self.backbone(x)
            if z_nn.dtype != z_veronese.dtype:
                z_nn = z_nn.to(z_veronese.dtype)
            parts.append(z_nn)
            
        return torch.cat(parts, dim=-1)


class VeroneseDecoding3(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int], 
                 omega_indices: List[int],
                 backbone: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        self.quat_indices = quat_indices
        self.omega_indices = omega_indices
        
        self.n_quat = len(quat_indices)
        self.n_omega = len(omega_indices)
        
        self.nn_out_dim = out_features - (self.n_quat + self.n_omega)
        
        if self.nn_out_dim > 0 and backbone is not None:
            self.backbone = hydra.utils.instantiate(
                backbone,
                in_features=in_features, 
                out_features=self.nn_out_dim
            )
        else:
            self.backbone = None
            
        all_indices = torch.arange(out_features)
        is_known = torch.zeros(out_features, dtype=torch.bool)
        is_known[quat_indices] = True
        is_known[omega_indices] = True
        self.register_buffer('unknown_indices', all_indices[~is_known], persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape
        z_flat = z.view(-1, original_shape[-1])
        
        # z structure: [q, w, veronese(q), z_nn]
        q_recon = z_flat[:, :self.n_quat]
        w_recon = z_flat[:, self.n_quat : self.n_quat + self.n_omega]
        
        # Normalize q
        q_recon = F.normalize(q_recon, p=2, dim=-1)
        
        if self.backbone is not None:
            others_pred = self.backbone(z_flat) 
        else:
            others_pred = torch.empty(z_flat.size(0), 0, device=z.device, dtype=z.dtype)

        x_recon = torch.zeros(z_flat.size(0), self.n_quat + self.n_omega + self.nn_out_dim, 
                            device=z.device, dtype=z.dtype)
        
        if q_recon.dtype != x_recon.dtype: q_recon = q_recon.to(x_recon.dtype)
        if w_recon.dtype != x_recon.dtype: w_recon = w_recon.to(x_recon.dtype)
        if others_pred.dtype != x_recon.dtype: others_pred = others_pred.to(x_recon.dtype)
        
        x_recon[:, self.quat_indices] = q_recon
        x_recon[:, self.omega_indices] = w_recon
        
        if self.nn_out_dim > 0:
            x_recon[:, self.unknown_indices] = others_pred
        
        return x_recon.view(*original_shape[:-1], -1)
