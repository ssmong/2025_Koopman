import torch
import torch.nn as nn
from src.losses.components.base import BaseLoss

class BackwardLossBase(BaseLoss):
    def __init__(self,
                 key_z: str = "z_traj_gt",
                 key_x: str = "x_traj_gt",
                 key_A_params: str = "A_params",
                 key_B: str = "B",
                 key_u: str = "u_future",
                 eigval_max: float = 1.0,
                 weight_z: float = 1.0,
                 weight_x: float = 0.0,
                 **kwargs):
        # Pass dummy keys to BaseLoss
        super().__init__(key_pred=key_z, key_target=key_z, **kwargs)
        
        self.key_z = key_z
        self.key_x = key_x
        self.key_A_params = key_A_params
        self.key_B = key_B
        self.key_u = key_u
        self.eigval_max = eigval_max
        self.weight_z = weight_z
        self.weight_x = weight_x
        self.decoder = None

    def bind_model(self, model: nn.Module):
        self.decoder = model.decoder

    def _analytic_backward_step(self, z_next: torch.Tensor, u: torch.Tensor, 
                              r: torch.Tensor, c: torch.Tensor, s: torch.Tensor, 
                              B: torch.Tensor) -> torch.Tensor:
        """
        Computes z_prev = A^{-1} (z_next - B u) analytically.
        """
        if u.dim() == 3: # [B, T, Du]
            Bu = torch.einsum('b d u, b t u -> b t d', B, u)
        else: # [B, Du]
            Bu = torch.einsum('b d u, b u -> b d', B, u)
            
        rhs = z_next - Bu # [B, ..., D]
        
        # Analytic Inversion of Block Diagonal A
        shape = rhs.shape
        batch_shape = shape[:-1]
        dim = shape[-1]
        n_blocks = dim // 2
        
        rhs_reshaped = rhs.view(*batch_shape, n_blocks, 2)
        rhs_even = rhs_reshaped[..., 0] # [..., D/2]
        rhs_odd = rhs_reshaped[..., 1]  # [..., D/2]
        
        if r.dim() < rhs_even.dim():
            while r.dim() < rhs_even.dim():
                 r = r.unsqueeze(1)
                 c = c.unsqueeze(1)
                 s = s.unsqueeze(1)
        
        inv_r = 1.0 / (r + 1e-6)
        
        z_prev_even = inv_r * (c * rhs_even + s * rhs_odd)
        z_prev_odd = inv_r * (-s * rhs_even + c * rhs_odd)
        
        z_prev = torch.stack([z_prev_even, z_prev_odd], dim=-1).view(*shape)
        
        return z_prev

    def _get_params(self, results):
        z = results.get(self.key_z)             # [B, N+1, D]
        # x is required only if weight_x > 0
        x = results.get(self.key_x) if self.weight_x > 0 else None
        
        A_params = results.get(self.key_A_params) # [B, D]
        B = results.get(self.key_B)             # [B, D, Du]
        u = results.get(self.key_u)             # [B, N, Du]

        if z is None or A_params is None or B is None or u is None:
             missing = []
             if z is None: missing.append(self.key_z)
             if A_params is None: missing.append(self.key_A_params)
             if B is None: missing.append(self.key_B)
             if u is None: missing.append(self.key_u)
             raise KeyError(f"BackwardLoss missing keys: {missing}")
        
        if self.weight_x > 0 and x is None:
             raise KeyError(f"BackwardLoss x-space loss enabled but key_x '{self.key_x}' missing.")
             
        dim = A_params.size(-1)
        n_blocks = dim // 2
        log_r = A_params[:, :n_blocks]
        theta = A_params[:, n_blocks:]
        
        r = self.eigval_max * torch.sigmoid(log_r) # [B, D/2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        return z, x, B, u, r, c, s


class BackwardRollingLoss(BackwardLossBase):
    def forward(self, results: dict) -> torch.Tensor:
        z, x, B, u, r, c, s = self._get_params(results)
        
        # Rolling Backward
        # Start from last step z_N, roll back to z_0
        
        z_roll = z[:, -1, :] # [B, D]
        N = u.size(1)
        
        # Loop backwards: N-1 down to 0
        for k in range(N - 1, -1, -1):
            u_k = u[:, k, :] # [B, Du]
            z_roll = self._analytic_backward_step(z_roll, u_k, r, c, s, B)
        
        total_loss = 0.0
        
        # 1. Z-space Loss
        if self.weight_z > 0:
            z_0_gt = z[:, 0, :]
            loss_z = ((z_roll - z_0_gt) ** 2).mean(dim=-1).mean()
            total_loss = total_loss + self.weight_z * loss_z

        # 2. X-space Loss
        if self.weight_x > 0:
            if self.decoder is None:
                raise ValueError("Decoder is not bound. Call bind_model() first.")
            
            x_pred_bwd_0 = self.decoder(z_roll) # [B, D_x]
            x_0_gt = x[:, 0, :]
            loss_x = self.loss_fn(x_pred_bwd_0, x_0_gt).mean(dim=-1).mean()
            total_loss = total_loss + self.weight_x * loss_x
        
        # Do not apply weight decay for rolling loss
        return self.weight * total_loss
