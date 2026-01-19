from typing import List, Dict, Optional, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)

"""
Dense LPV Model (V5) - "Direct Discrete & Residual"
Reflecting conversation:
1. z = [q, w, veronese(q), z_nn]
2. Dynamics: z_{k+1} = A_dt * z_k + B * u_k
3. A_dt is directly learned as Residual: A_dt = I + delta_A
4. Supports Low-Rank + Diagonal structure
"""
class LPVModel5(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int,
                 latent_dim: int,
                 quat_indices: List[int],
                 omega_indices: List[int],
                 context: DictConfig, 
                 lifting: DictConfig,
                 matrix: DictConfig,
                 decoding: DictConfig = None,
                 dt: float = 0.01,
                 noise_std: float = 0.0,
                 rank: int = 0,
                 **kwargs):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim  
        self.latent_dim = latent_dim
        self.dt = dt
        self.noise_std = noise_std
        self.rank = rank
        self.quat_indices = list(quat_indices) if quat_indices is not None else []
        self.omega_indices = list(omega_indices) if omega_indices is not None else []
        
        if len(self.quat_indices) == 4:
            self.veronese_dim = 10
        else:
            self.veronese_dim = 0
            logger.warning("Quat indices length is not 4. Veronese dimension is set to 0.")
            
        self.n_quat = len(self.quat_indices)
        self.n_omega = len(self.omega_indices)
        
        # z = [q, w, veronese, z_nn]
        self.veronese_start_idx = self.n_quat + self.n_omega

        # 1. Encoder (Lifting)
        self.encoder = hydra.utils.instantiate(
            lifting, 
            in_features=state_dim, 
            out_features=latent_dim,
            quat_indices=self.quat_indices,
            omega_indices=self.omega_indices
        )   
        
        # 2. Decoder (Retraction)
        decoding = decoding if decoding is not None else lifting
        self.decoder = hydra.utils.instantiate(
            decoding, 
            in_features=latent_dim, 
            out_features=state_dim,
            quat_indices=self.quat_indices,
            omega_indices=self.omega_indices
        )

        if hasattr(self.encoder, 'mixing') and hasattr(self.decoder, 'set_mixing'):
            self.decoder.set_mixing(self.encoder.mixing)

        # 3. Context Encoder
        self.ctxt_encoder = hydra.utils.instantiate(context)
        
        # 4. Dense Matrix Generation Parameters
        if self.rank > 0:
            # Low-Rank + Diagonal: A = I + (U*V^T + D)
            # Output size: (N*r for U) + (N*r for V) + (N for D)
            self.u_size = self.latent_dim * self.rank
            self.v_size = self.latent_dim * self.rank
            self.d_size = self.latent_dim
            out_features = self.u_size + self.v_size + self.d_size
            logger.info(f"LPVModel5: Using Low-Rank A (rank={self.rank}) + Diagonal. Output dim: {out_features}")
        else:
            # Full Rank: NxN
            out_features = self.latent_dim * self.latent_dim
            
        self._to_A = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=out_features 
        )
        
        self._to_B = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * control_dim
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize delta_A to be near zero -> A_dt near Identity
        def _init_A(module):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    # Initialize to small random noise to break symmetry, 
                    # but close to 0 to ensure A_dt starts as Identity.
                    nn.init.normal_(last_layer.bias, mean=0.0, std=1e-5)
                    
        def _init_B(module, bias=None):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if bias is not None:
                    last_layer.bias.data.copy_(bias)
                elif last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

        _init_A(self._to_A)
        
        noise = torch.randn(self.latent_dim * self.control_dim) * 1e-5 
        _init_B(self._to_B, bias=noise)

    def get_A_dt(self, h):
        """
        Reshape output of _to_A into (Batch, N, N) and add Identity.
        A_dt = I + delta_A
        """
        batch_size = h.size(0)
        device = h.device
        
        params = self._to_A(h)
        
        if self.rank > 0:
            # Split params into U, V, D
            u_flat = params[:, :self.u_size]
            v_flat = params[:, self.u_size : self.u_size + self.v_size]
            d_flat = params[:, self.u_size + self.v_size :]
            
            U = u_flat.view(batch_size, self.latent_dim, self.rank)
            V = v_flat.view(batch_size, self.latent_dim, self.rank)
            
            # Low-rank part: U @ V.T
            delta_A = torch.matmul(U, V.transpose(1, 2))
            
            # Diagonal part
            D = torch.diag_embed(d_flat)
            
            delta_A = delta_A + D
        else:
            # Full Rank
            delta_A = params.view(batch_size, self.latent_dim, self.latent_dim)
            
        # Residual Connection: A_dt = I + delta_A
        I = torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        A_dt = I + delta_A
        
        return A_dt

    def _forward_dynamics(self, z, A_dt, B_flat, u):
        batch_size = z.size(0)
        # Linear Dynamics: z_{k+1} = A_dt @ z_k + B @ u_k
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
        
        # Compute Discrete A directly (Frozen-time LPV)
        A_dt = self.get_A_dt(h)
        B_flat = self._to_B(h)

        z_curr = self.encoder(x_init)
        z_traj = [z_curr]

        for k in range(n_steps):
            # Add Gaussian noise to latent state during training (Robustness/Denoising)
            z_in = z_curr
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(z_curr) * self.noise_std
                z_in = z_curr + noise

            u_k = u_future[:, k, :]
            z_curr = self._forward_dynamics(z_in, A_dt, B_flat, u_k)
            z_traj.append(z_curr)
        
        z_traj = torch.stack(z_traj, dim=1)
        x_traj = self.decoder(z_traj)

        # Normalize quaternion output during inference/validation
        if not self.training and self.quat_indices:
            x_traj = x_traj.clone()
            q_part = x_traj[..., self.quat_indices]
            q_norm = F.normalize(q_part, p=2, dim=-1)
            x_traj[..., self.quat_indices] = q_norm.to(dtype=x_traj.dtype)
        
        results = {
            "z_traj": z_traj,
            "x_traj": x_traj,
            "A_dt": A_dt, 
            "B": B_flat.view(A_dt.size(0), self.latent_dim, self.control_dim),
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
        Returns Discrete-time A matrix directly
        """
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        
        A_dt = self.get_A_dt(h)
        
        return A_dt
    
    def get_B(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        B_flat = self._to_B(h)

        batch_size = B_flat.size(0)
        B = B_flat.view(batch_size, self.latent_dim, self.control_dim)

        return B
