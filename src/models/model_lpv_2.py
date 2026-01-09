from typing import List, Dict, Optional, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)

"""
Dense LPV Model with modified veronese lifting
"""
class LPVModel2(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int,
                 latent_dim: int,
                 quat_indices: List[int],
                 context: DictConfig, 
                 lifting: DictConfig,
                 matrix: DictConfig,
                 decoding: DictConfig = None,
                 dt: float = 0.01,
                 **kwargs):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim  
        self.latent_dim = latent_dim
        self.dt = dt
        self.quat_indices = list(quat_indices) if quat_indices is not None else None
        
        if self.quat_indices is not None and len(self.quat_indices) == 4:
            self.veronese_dim = 10
        else:
            logger.error("Quat indices are not provided or the length is not 4. Veronese dimension is set to 0.")
            raise ValueError("Quat indices are not provided or the length is not 4. Veronese dimension is set to 0.")


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

        if hasattr(self.encoder, 'mixing') and hasattr(self.decoder, 'set_mixing'):
            self.decoder.set_mixing(self.encoder.mixing)

        self.ctxt_encoder = hydra.utils.instantiate(context)
        
        # Dense Matrix Generation Parameters
        # Output size: (N*N for Coupling W) + (N for Damping Sigma)
        self._to_A = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * self.latent_dim + self.latent_dim 
        )
        
        self._to_B = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * control_dim
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize A
        # Coupling(W) is around 0, 
        # Damping(Sigma) is slightly negative (before softplus)
        def _init_A(module):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    n_sq = self.latent_dim * self.latent_dim
                    # W (Coupling): Small random noise
                    last_layer.bias.data[:n_sq].normal_(0, 0.01)
                    # Sigma (Damping): Init to give small positive damping after softplus
                    # Softplus(-3.0) approx 0.05
                    last_layer.bias.data[n_sq:].fill_(-3.0)
                    
        def _init_B(module, bias=None):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if bias is not None:
                    last_layer.bias.data.copy_(bias)
                elif last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

        _init_A(self._to_A)
        
        noise = torch.randn(self.latent_dim * self.control_dim) * 1e-4
        _init_B(self._to_B, bias=noise)
    
    def get_A_ct(self, h: torch.Tensor):
        """
        Constructs Continuous-time Dense A matrix: A = (W - W.T) - Mask * D
        Args:
            h: Context embedding [B, hidden_size]
        """
        params = self._to_A(h)
        
        batch_size = h.size(0)
        n_sq = self.latent_dim * self.latent_dim
        
        W_flat = params[:, :n_sq]
        W = W_flat.view(batch_size, self.latent_dim, self.latent_dim)
        J = W - W.transpose(1, 2)
        
        sigma = F.softplus(params[:, n_sq:])
        
        # Selective Masking
        mask = torch.ones_like(sigma)
        if self.veronese_dim > 0:
            mask[:, :self.veronese_dim] = 0.0        
        sigma_masked = sigma * mask
        D = torch.diag_embed(sigma_masked)
        
        A_ct = J - D
        return A_ct

    def _cayley_map(self, A_ct: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.latent_dim, device=A_ct.device).unsqueeze(0)
        
        A_half = A_ct * (self.dt * 0.5)
        
        U = I - A_half
        V = I + A_half
        return torch.linalg.solve(U, V)

    def _forward_dynamics(self, z, A_ct, B_flat, u):
        batch_size = z.size(0)
        A_dt = self._cayley_map(A_ct)
        
        Az = torch.einsum('bij,bj->bi', A_dt, z)
        
        B_mat = B_flat.view(batch_size, self.latent_dim, self.control_dim)
        Bu = torch.einsum('bij,bj->bi', B_mat, u)

        return Az + Bu

    def forward(self, 
                x_history: torch.Tensor, 
                u_history: torch.Tensor, 
                x_init: torch.Tensor, 
                x_future: torch.Tensor, 
                u_future: torch.Tensor, 
                n_steps: int):
        
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        
        # Compute A_continuous once (Frozen-time LPV)
        A_ct = self.get_A_ct(h)
        B_flat = self._to_B(h)

        z_curr = self.encoder(x_init)
        z_traj = [z_curr]

        for k in range(n_steps):
            u_k = u_future[:, k, :]
            z_curr = self._forward_dynamics(z_curr, A_ct, B_flat, u_k)
            z_traj.append(z_curr)
        
        z_traj = torch.stack(z_traj, dim=1)
        x_traj = self.decoder(z_traj)

        # Normalize quaternion output during inference/validation
        if not self.training and self.quat_indices is not None:
            x_traj = x_traj.clone()
            q_part = x_traj[..., self.quat_indices]
            q_norm = F.normalize(q_part, p=2, dim=-1)
            x_traj[..., self.quat_indices] = q_norm.to(dtype=x_traj.dtype)
        
        results = {
            "z_traj": z_traj,
            "x_traj": x_traj,
            "A_params": A_ct, # Returning A_ct for loss/analysis
            "B": B_flat.view(A_ct.size(0), self.latent_dim, self.control_dim),
            "u_future": u_future,
            "z_traj_re": self.encoder(x_traj) 
        }

        # GT Trajectories for loss
        x_traj_gt = torch.cat([x_init.unsqueeze(1), x_future], dim=1)
        results["x_traj_gt"] = x_traj_gt
        results["z_traj_gt"] = self.encoder(x_traj_gt)            
            
        return results
    
    # --- Public Methods ---

    def get_A(self, x_history: torch.Tensor, u_history: torch.Tensor):
        """
        Returns Discrete-time A matrix: A_dt = exp(A_ct * dt)
        """
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        
        A_ct = self.get_A_ct(h)
        A_dt = torch.matrix_exp(A_ct * self.dt)
        
        return A_dt
    
    def get_B(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        B_flat = self._to_B(h)

        batch_size = B_flat.size(0)
        B = B_flat.view(batch_size, self.latent_dim, self.control_dim)

        return B
