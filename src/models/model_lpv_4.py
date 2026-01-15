from typing import List, Dict, Optional, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)

"""
Dense LPV Model (V4) - "Unconstrained & Invertible"
Reflecting conversation:
1. z = [q, w, veronese(q), z_nn]
2. Dynamics: z_{k+1} = Cayley(A_ct) * z_k + B * u_k
3. A_ct is fully learnable (no skew-symmetric constraints)
4. Cayley map ensures invertibility and handles large dt better than simple Euler.
"""
class LPVModel4(nn.Module):
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
                 **kwargs):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim  
        self.latent_dim = latent_dim
        self.dt = dt
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
        # Changed: Just output a full NxN matrix. No structural constraints.
        self._to_A = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * self.latent_dim 
        )
        
        self._to_B = hydra.utils.instantiate(
            matrix,
            in_features=context.hidden_size,
            out_features=self.latent_dim * control_dim
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize A to be near zero -> A_dt near Identity
        def _init_A(module):
            last_layer = list(module.modules())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    # Initialize to small random noise to break symmetry, 
                    # but close to 0 to ensure A_dt starts as Identity.
                    nn.init.normal_(last_layer.bias, mean=0.0, std=1e-4)
                    
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

    def get_A_ct(self, h):
        """Reshape output of _to_A into (Batch, N, N)"""
        batch_size = h.size(0)
        A_flat = self._to_A(h)
        return A_flat.view(batch_size, self.latent_dim, self.latent_dim)

    def _cayley_map(self, A_ct):
        """
        Cayley transform for fast and invertible discretization.
        A_dt = (I - dt/2 * A_ct)^-1 @ (I + dt/2 * A_ct)
        Approximates exp(A_ct * dt).
        """
        B, N, _ = A_ct.shape
        dt = self.dt
        device = A_ct.device
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        
        # Terms for Crank-Nicolson / Cayley
        # T1 = I - 0.5*dt*A
        # T2 = I + 0.5*dt*A
        factor = 0.5 * dt * A_ct
        lhs = I - factor
        rhs = I + factor
        
        # Solve linear system instead of explicit inverse for speed & stability
        # Solves X such that lhs @ X = rhs
        A_dt = torch.linalg.solve(lhs, rhs)
        
        return A_dt

    def _forward_dynamics(self, z, A_ct, B_flat, u):
        batch_size = z.size(0)
        
        # Apply Cayley Map to get Discrete A
        A_dt = self._cayley_map(A_ct)
        
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
        
        # Compute A_continuous and B once (Frozen-time LPV)
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
        if not self.training and self.quat_indices:
            x_traj = x_traj.clone()
            q_part = x_traj[..., self.quat_indices]
            q_norm = F.normalize(q_part, p=2, dim=-1)
            x_traj[..., self.quat_indices] = q_norm.to(dtype=x_traj.dtype)
        
        results = {
            "z_traj": z_traj,
            "x_traj": x_traj,
            "A_params": A_ct,  # Return continuous A for regularization (e.g., Eigenvalue penalty)
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
        Returns Discrete-time A matrix using Cayley Map
        """
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        
        A_ct = self.get_A_ct(h)
        A_dt = self._cayley_map(A_ct)
        
        return A_dt
    
    def get_B(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        h, _, _ = self.ctxt_encoder(ctxt)
        B_flat = self._to_B(h)

        batch_size = B_flat.size(0)
        B = B_flat.view(batch_size, self.latent_dim, self.control_dim)

        return B