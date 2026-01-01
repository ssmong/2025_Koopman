import torch
import torch.nn as nn
import hydra
from typing import List, Dict

class VeroneseLifting(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int], 
                 backbone: Dict):
        # out_features: Veronese Dim + NN Dim
        # backbone: Hydra config for backbone model (learnable Feature Map) (ResNetBlock)
        
        super().__init__()
        
        self.quat_indices = quat_indices
        self.n_quat = len(quat_indices)
        self.veronese_dim = 10

        self.nn_out_dim = out_features - self.veronese_dim
        
        if self.nn_out_dim <= 0:
            raise ValueError(
                f"Latent dim({out_features}) must be > Veronese dim({self.veronese_dim})."
            )

        self.backbone = hydra.utils.instantiate(
            backbone,
            in_features=in_features,
            out_features=self.nn_out_dim
        )
        
        # Pre-compute indices for Veronese calculation
        # Use python list for direct indexing (works on any device)
        triu_idx = torch.triu_indices(self.n_quat, self.n_quat)
        self.idx_i = triu_idx[0].tolist()
        self.idx_j = triu_idx[1].tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x[..., self.quat_indices] # [..., 4]
        
        z_veronese = q[..., self.idx_i] * q[..., self.idx_j]    # [..., 10] : [q1^2, q1q2, ..., q4^2]
        # Learnable Feature Map: Neural Network Lifting
        z_nn = self.backbone(x) # [..., nn_out_dim]
        
        z = torch.cat([z_veronese, z_nn], dim=-1)    
        return z

class VeroneseDecoding(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 quat_indices: List[int],
                 backbone: Dict):
        # out_features: State Dim
        # backbone: Hydra config for backbone model (learnable Feature Map) (ResNetBlock)
        
        super().__init__()
        self.quat_indices = quat_indices
        self.n_quat = len(quat_indices)
        self.veronese_dim = 10

        # NN part reconstructs the non-quaternion state dimensions
        self.nn_out_dim = out_features - self.n_quat
        
        self.backbone = hydra.utils.instantiate(
            backbone,
            in_features=in_features, 
            out_features=self.nn_out_dim
        )
        
        triu_idx = torch.triu_indices(self.n_quat, self.n_quat)
        row, col = triu_idx[0], triu_idx[1]
        
        # Indices for filling the upper triangle
        self.triu_rows = row.tolist()
        self.triu_cols = col.tolist()
        
        # Indices for filling the symmetric lower triangle (where row != col)
        mask = row != col
        self.sym_rows = col[mask].tolist() # swapped
        self.sym_cols = row[mask].tolist() # swapped
        self.sym_z_idx = torch.nonzero(mask).squeeze().tolist() # indices in z_veronese
        
        # Pre-compute non-quat indices for direct assignment
        all_indices = set(range(out_features))
        quat_set = set(quat_indices)
        self.non_quat_indices = list(all_indices - quat_set)
        self.non_quat_indices.sort()

    def _recover_quat(self, z_veronese: torch.Tensor) -> torch.Tensor:
        """
        z_veronese (B, 10) -> Symmetric Matrix M (B, 4, 4) -> Eigen Decomposition -> q (B, 4)
        """
        B = z_veronese.size(0)
        
        M = torch.zeros(B, 4, 4, device=z_veronese.device, dtype=z_veronese.dtype)
        
        # Fill upper triangle
        M[:, self.triu_rows, self.triu_cols] = z_veronese
        # Fill symmetric lower triangle
        M[:, self.sym_rows, self.sym_cols] = z_veronese[:, self.sym_z_idx]
        
        _, eigenvectors = torch.linalg.eigh(M) # [B, 4], [B, 4, 4]
        q_recon = eigenvectors[:, :, -1]       # [B, 4]
        
        sign = torch.sign(q_recon[:, 0:1] + 1e-9)
        q_recon = q_recon * sign
        
        return q_recon

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape
        z = z.view(-1, original_shape[-1])
        
        z_v = z[:, :self.veronese_dim]
        q_pred = self._recover_quat(z_v)
        
        others_pred = self.backbone(z)
        
        # Reconstruct full state x
        # n_quat + nn_out_dim = out_features
        x_recon = torch.zeros(z.size(0), self.n_quat + self.nn_out_dim, device=z.device, dtype=z.dtype)
        
        x_recon[:, self.quat_indices] = q_pred
        x_recon[:, self.non_quat_indices] = others_pred
        
        return x_recon.view(*original_shape[:-1], -1)
