import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import hydra
import torch.nn.functional as F

class VeroneseLifting2(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int], 
                 backbone: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        
        self.quat_indices = quat_indices
        self.n_quat = len(quat_indices)
        self.veronese_dim = (self.n_quat * (self.n_quat + 1)) // 2 

        self.nn_out_dim = out_features - self.veronese_dim
        
        if self.nn_out_dim < 0:
             raise ValueError(f"Output features ({out_features}) must be >= Veronese dim ({self.veronese_dim}).")

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
        z_veronese = q[..., self.idx_i] * q[..., self.idx_j] * self.scale
        
        if self.backbone is not None:
            z_nn = self.backbone(x)
            if z_nn.dtype != z_veronese.dtype:
                z_nn = z_nn.to(z_veronese.dtype)
            return torch.cat([z_veronese, z_nn], dim=-1)
            
        return z_veronese


class VeroneseDecoding2(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int], 
                 backbone: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        self.quat_indices = quat_indices
        self.n_quat = len(quat_indices)
        self.veronese_dim = (self.n_quat * (self.n_quat + 1)) // 2

        self.nn_out_dim = out_features - self.n_quat
        
        if self.nn_out_dim > 0 and backbone is not None:
            self.backbone = hydra.utils.instantiate(
                backbone,
                in_features=in_features, 
                out_features=self.nn_out_dim
            )
        else:
            self.backbone = None
        
        triu_idx = torch.triu_indices(self.n_quat, self.n_quat)
        row, col = triu_idx[0], triu_idx[1]
        
        diag_mask = row == col
        self.register_buffer('diag_indices', torch.nonzero(diag_mask).squeeze(), persistent=False)
        
        all_indices = torch.arange(out_features)
        is_quat = torch.zeros(out_features, dtype=torch.bool)
        is_quat[quat_indices] = True
        self.register_buffer('non_quat_indices', all_indices[~is_quat], persistent=False)

    def _recover_quat(self, z_veronese: torch.Tensor) -> torch.Tensor:
        B = z_veronese.size(0)
        
        # 1. Anchor Selection: Largest diagonal element
        z_diag = z_veronese[:, self.diag_indices] # [B, 4]
        val_sq, k_idx = torch.max(z_diag, dim=1)  # [B], [B]
        
        q_k_abs = torch.sqrt(torch.clamp(val_sq, min=1e-8))
        
        # 2. Reconstruct M to extract row k
        M = torch.zeros(B, self.n_quat, self.n_quat, device=z_veronese.device, dtype=z_veronese.dtype)
        
        triu_indices = torch.triu_indices(self.n_quat, self.n_quat, device=z_veronese.device)
        rows, cols = triu_indices
        
        z_vals = z_veronese.clone()
        mask_cross = (rows != cols)
        z_vals[:, mask_cross] /= math.sqrt(2)
        
        M[:, rows, cols] = z_vals
        M[:, cols, rows] = z_vals 
        
        # 3. Extract row k: q_k * [q_0, ..., q_3]
        k_expanded = k_idx.view(B, 1, 1).expand(B, 1, self.n_quat)
        row_k = torch.gather(M, 1, k_expanded).squeeze(1) # [B, 4]
        
        # 4. Normalize by q_k to get q
        q_recon = row_k / (q_k_abs.unsqueeze(1) + 1e-8)
        
        # 5. Enforce q0 > 0
        sign_q0 = torch.sign(q_recon[:, 0:1])
        sign_q0[sign_q0 == 0] = 1.0 
        
        q_recon = q_recon * sign_q0
        q_recon = F.normalize(q_recon, p=2, dim=-1)
        
        return q_recon

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape
        z_flat = z.view(-1, original_shape[-1])

        z_v = z_flat[:, :self.veronese_dim]
        q_pred = self._recover_quat(z_v)
        
        if self.backbone is not None:
            others_pred = self.backbone(z_flat) 
        else:
            others_pred = torch.empty(z_flat.size(0), 0, device=z.device, dtype=z.dtype)

        x_recon = torch.zeros(z_flat.size(0), self.n_quat + self.nn_out_dim, 
                            device=z.device, dtype=z.dtype)
        
        if q_pred.dtype != x_recon.dtype: q_pred = q_pred.to(x_recon.dtype)
        if others_pred.dtype != x_recon.dtype: others_pred = others_pred.to(x_recon.dtype)
        
        x_recon[:, self.quat_indices] = q_pred
        if self.nn_out_dim > 0:
            x_recon[:, self.non_quat_indices] = others_pred
        
        return x_recon.view(*original_shape[:-1], -1)
