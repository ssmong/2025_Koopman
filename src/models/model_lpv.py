from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

class LPVModel(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int,
                 latent_dim: int,
                 quat_indices: List[int],
                 eigval_max: float,
                 context: DictConfig, 
                 lifting: DictConfig,
                 matrix: DictConfig,
                 decoding: DictConfig = None,):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim  
        self.latent_dim = latent_dim
        self.eigval_max = eigval_max
        self.quat_indices = list(quat_indices) if quat_indices is not None else None

        assert latent_dim % 2 == 0, f"Latent dim must be even for block-diagonal LPV, got {latent_dim}"

        self.encoder = hydra.utils.instantiate(
            lifting, 
            in_features=state_dim, 
            out_features=latent_dim
        )   
        decoding = decoding if decoding is not None else lifting
        self.decoder = hydra.utils.instantiate(
            decoding, 
            in_features=latent_dim, 
            out_features=state_dim
        )

        if hasattr(self.decoder, 'set_mixing'):
            self.decoder.set_mixing(self.encoder.mixing)

        self.ctxt_encoder = hydra.utils.instantiate(context)
        self._to_A = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim 
        )
        
        self._to_B = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * control_dim
        )
        
        self._init_weights()

    def _init_weights(self):
        def _init_layer(module, bias=None, is_A=False):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    with torch.no_grad():
                        if is_A:
                            # A init: r ~ 1 (large logit), theta ~ 0
                            n_blocks = self.latent_dim // 2
                            last_layer.bias[:n_blocks].fill_(5.0) # Sigmoid(5.0) ~= 0.993
                            last_layer.bias[n_blocks:].fill_(0.0)
                        elif bias is not None:
                            last_layer.bias.copy_(bias)

        _init_layer(self._to_A, is_A=True)
        
        noise = torch.randn(self.latent_dim * self.control_dim) * 1e-4
        _init_layer(self._to_B, bias=noise)
    
    def _forward_dynamics(self, z, A_params, B_flat, u):
        batch_size = z.size(0)
        n_blocks = self.latent_dim // 2

        log_r = A_params[:, :n_blocks]
        theta = A_params[:, n_blocks:]

        r = self.eigval_max * torch.sigmoid(log_r)
        c = torch.cos(theta)
        s = torch.sin(theta)

        z_reshaped = z.view(batch_size, n_blocks, 2)
        z_even = z_reshaped[:, :, 0]
        z_odd = z_reshaped[:, :, 1]

        z_next_even = r * (c * z_even - s * z_odd)
        z_next_odd = r * (s * z_even + c * z_odd)

        Az = torch.stack([z_next_even, z_next_odd], dim=-1).view(batch_size, self.latent_dim)
        B_mat = B_flat.view(batch_size, self.latent_dim, self.control_dim)
        Bu = torch.bmm(B_mat, u.unsqueeze(-1)).squeeze(-1)

        return Az + Bu


    def forward(self, 
                x_history: torch.Tensor,    # [B, W, D]     t=k-W, ..., k-1
                u_history: torch.Tensor,    # [B, W, D]     t=k-W, ..., k-1
                x_init: torch.Tensor,       # [B, D]        t=k
                x_future: torch.Tensor,     # [B, N, D]     t=k+1, ..., k+N
                u_future: torch.Tensor,     # [B, N, D]     t=k, ..., k+N-1
                n_steps: int):
        """
        Returns:
            z_traj: [B, N+1, D]
            x_traj: [B, N+1, D]
            z_re_encoded: [B, N+1, D]
        """
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        A_params = self._to_A(h)
        B_flat = self._to_B(h)

        z_curr = self.encoder(x_init)

        z_traj = [z_curr]

        for k in range(n_steps):
            u_k = u_future[:, k, :]
            z_curr = self._forward_dynamics(z_curr, A_params, B_flat, u_k)
            z_traj.append(z_curr)
        
        z_traj = torch.stack(z_traj, dim=1)     # [B, N+1, D]
        x_traj = self.decoder(z_traj)           # [B, N+1, D]

        if not self.training and self.quat_indices is not None:
            x_traj = x_traj.clone()
            q_part = x_traj[..., self.quat_indices]
            q_norm = F.normalize(q_part, p=2, dim=-1)
            x_traj[..., self.quat_indices] = q_norm.to(dtype=x_traj.dtype)
        
        results = {
            "z_traj": z_traj,
            "x_traj": x_traj,
        }

        # --- Expose params for Backward/Consistency Losses ---
        # Pass parameters instead of constructing full A matrix for analytic inversion
        results["A_params"] = A_params
        
        batch_size = A_params.size(0)
        results["B"] = B_flat.view(batch_size, self.latent_dim, self.control_dim)
        results["u_future"] = u_future
        
        # 1. Latent Consistency: z_traj_re = E(x_traj) should match z_traj
        results["z_traj_re"] = self.encoder(x_traj) 

        # 2. GT Trajectories
        x_traj_gt = torch.cat([x_init.unsqueeze(1), x_future], dim=1)  # [B, N+1, D]
        results["x_traj_gt"] = x_traj_gt
        
        z_traj_gt = self.encoder(x_traj_gt)
        results["z_traj_gt"] = z_traj_gt           
            
        return results
    
    #
    # --- Get Methods ---
    #

    def get_A(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        A_params = self._to_A(h)

        batch_size = A_params.size(0)
        n_blocks = self.latent_dim // 2

        log_r = A_params[:, :n_blocks]
        theta = A_params[:, n_blocks:]

        r = self.eigval_max * torch.sigmoid(log_r)
        c = torch.cos(theta)
        s = torch.sin(theta)

        A = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=A_params.device)
        
        indices = torch.arange(n_blocks, device=A_params.device)
        even_indices = 2 * indices
        odd_indices = 2 * indices + 1

        # Diagonal elements: r * cos(theta)
        rc = r * c
        A[:, even_indices, even_indices] = rc
        A[:, odd_indices, odd_indices] = rc
        
        # Off-diagonal elements
        rs = r * s
        A[:, even_indices, odd_indices] = -rs
        A[:, odd_indices, even_indices] = rs
        
        return A
    
    def get_B(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        B_flat = self._to_B(h)

        batch_size = B_flat.size(0)
        B = B_flat.view(batch_size, self.latent_dim, self.control_dim)

        return B
