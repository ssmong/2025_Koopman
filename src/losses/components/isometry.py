import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.components.quat import BaseLoss, QuatLoss

class IsometryLoss(QuatLoss):
    # Loss calculation: || D_x^2 - s * D_z^2 ||^2
    # Force latent space distance to be proportional to actual space distance (Isometry)
    # s: scale parameter = min_scale + Softplus(param) >= 1
        
    def __init__(self, 
                 min_scale: float = 1.0,
                 sample_size: int = 512,
                 **kwargs):
        # By enforcing quat_norm_weight=0 (in config), 
        # we prevent the loss from being contaminated by non-quaternion parts.
        super().__init__(**kwargs)
        
        self.min_scale = min_scale
        self.sample_size = sample_size
        
        # Learnable scale parameter
        self.scale_param = nn.Parameter(torch.zeros(1))

    def _compute_dist_sq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise squared Euclidean distance using torch.cdist for memory efficiency.
        Input: [N, D] -> Output: [N, N]
        """
        # torch.cdist calculates L2 norm (Euclidean distance)
        dist = torch.cdist(x, x, p=2)
        return dist ** 2

    def forward(self, results: dict) -> torch.Tensor:
        z, x = self.get_inputs(results)     # pred: z_traj_gt, target: x_traj_gt
        
        # Random subsampling
        B, T, D_x = x.shape
        D_z = z.shape[-1]
        
        flat_x = x.reshape(-1, D_x)
        flat_z = z.reshape(-1, D_z)
        
        total_samples = flat_x.size(0)
        n_samples = min(total_samples, self.sample_size)
        
        # Random permutation to get indices
        indices = torch.randperm(total_samples, device=x.device)[:n_samples]
        
        sub_x = flat_x[indices] # [N, D_x]
        sub_z = flat_z[indices] # [N, D_z]
        
        # X Space distance calculation (Composite Distance)
        if self.quat_indices:
            is_quat = torch.zeros(D_x, dtype=torch.bool, device=x.device)
            is_quat[self.quat_indices] = True
            
            # Non-quaternion part (Squared Euclidean)
            if (~is_quat).any():
                dist_nq_sq = self._compute_dist_sq(sub_x[:, ~is_quat])
            else:
                dist_nq_sq = 0.0
                
            # Quaternion Part (Reusing QuatLoss logic with Broadcasting)
            q_part = sub_x[:, self.quat_indices]
            # Parent method (QuatLoss._compute_quat_loss)
            dist_q_sq = self._compute_quat_loss(
                q_part.unsqueeze(1), 
                q_part.unsqueeze(0)
            )
            dist_x_sq = dist_nq_sq + dist_q_sq   
        else:
            # If no quaternions, use cdist for entire space
            dist_x_sq = self._compute_dist_sq(sub_x)
            
        dist_z_sq = self._compute_dist_sq(sub_z)
        current_scale = self.min_scale + F.softplus(self.scale_param)
        loss = F.mse_loss(dist_x_sq, current_scale * dist_z_sq)
        
        # Isometry Loss is not applicable to time axis due to sampling
        # Simply perform weight mulpvplication and return
        return self.weight * loss


class PartialIsometryLoss(BaseLoss):
    def __init__(self, 
                 exclude_x_idx: list = None, 
                 exclude_z_idx: list = None, 
                 min_scale: float = 1.0,
                 sample_size: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
    
        self.exclude_x_idx = set(exclude_x_idx) if exclude_x_idx else None
        self.exclude_z_idx = set(exclude_z_idx) if exclude_z_idx else None
        
        self.min_scale = min_scale
        self.sample_size = sample_size
        
        # Learnable scale parameter: s = min_scale + Softplus(param)
        self.scale_param = nn.Parameter(torch.zeros(1))

    def _compute_dist_sq(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, x, p=2)
        return dist ** 2

    def forward(self, results: dict) -> torch.Tensor:
        z, x = self.get_inputs(results)
        
        # 1. Filter Indices (Exclude Quaternion & Veronese parts)
        if self.exclude_x_idx:
            full_dim = x.shape[-1]
            keep_idx = [i for i in range(full_dim) if i not in self.exclude_x_idx]
            x = x[..., keep_idx]

        if self.exclude_z_idx:
            full_dim = z.shape[-1]
            keep_idx = [i for i in range(full_dim) if i not in self.exclude_z_idx]
            z = z[..., keep_idx]
    
        # 2. Flatten for Pairwise Distance
        x_flat = x.reshape(-1, x.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        
        # 3. Random Subsampling (Efficiency)
        N = x_flat.size(0)
        n_samples = min(N, self.sample_size)
        
        if n_samples > 0:
            idx = torch.randperm(N, device=x.device)[:n_samples]
            x_sub = x_flat[idx]
            z_sub = z_flat[idx]
        else:
            x_sub = x_flat
            z_sub = z_flat
        
        # 4. Compute Squared Distances
        # Since x is filtered (no quaternions), we use Euclidean distance for both
        dist_x_sq = self._compute_dist_sq(x_sub)
        dist_z_sq = self._compute_dist_sq(z_sub)
        
        # 5. Apply Learnable Scale
        current_scale = self.min_scale + F.softplus(self.scale_param)
        
        # 6. Loss Calculation (MSE between scaled distances)
        loss = F.mse_loss(dist_x_sq, current_scale * dist_z_sq)

        return self.weight * loss