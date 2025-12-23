import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.components.quat import QuatLoss

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