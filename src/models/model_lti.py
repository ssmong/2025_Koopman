from typing import List, Dict, Optional, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)

"""
LTI Model (Linear Time-Invariant)
"""
class LTIModel(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 control_dim: int,
                 latent_dim: int,
                 quat_indices: List[int],
                 omega_indices: List[int],
                 lifting: DictConfig,
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

        # 3. Static Matrix Parameters
        self.A_ct = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.B = nn.Parameter(torch.zeros(latent_dim, control_dim))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.A_ct, mean=0.0, std=1e-4)
        nn.init.normal_(self.B, mean=0.0, std=1e-5)

    def _cayley_map(self, A_ct):
        """
        Cayley transform for fast and invertible discretization.
        A_dt = (I - dt/2 * A_ct)^-1 @ (I + dt/2 * A_ct)
        Approximates exp(A_ct * dt).
        """
        # Handle both batched and unbatched inputs
        if A_ct.dim() == 3:
            B, N, _ = A_ct.shape
            I = torch.eye(N, device=A_ct.device).unsqueeze(0).expand(B, -1, -1)
        else:
            N, _ = A_ct.shape
            I = torch.eye(N, device=A_ct.device)
        
        dt = self.dt
        
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

    def _forward_dynamics(self, z, A_dt, B, u):
        # Linear Dynamics: z_{k+1} = A_dt @ z_k + B @ u_k
        
        # Check if A_dt is batched
        if A_dt.dim() == 3:
            Az = torch.einsum('bij,bj->bi', A_dt, z)
        else:
            Az = torch.mm(z, A_dt.T) # z: [B, D], A: [D, D] -> zA^T = (Az)^T
            
        # Check if B is batched (it won't be for LTI unless expanded, but good for robustness)
        if B.dim() == 3:
            Bu = torch.einsum('bij,bj->bi', B, u)
        else:
            Bu = torch.mm(u, B.T)

        return Az + Bu

    def forward(self, 
                x_history: torch.Tensor, 
                u_history: torch.Tensor, 
                x_init: torch.Tensor, 
                x_future: torch.Tensor, 
                u_future: torch.Tensor, 
                n_steps: int):
        
        # Compute Discrete A once (N, N) - No context dependency
        A_dt = self._cayley_map(self.A_ct)

        z_curr = self.encoder(x_init)
        z_traj = [z_curr]

        for k in range(n_steps):
            u_k = u_future[:, k, :]
            # Pass (N, N) matrix directly to allow efficient mm()
            z_curr = self._forward_dynamics(z_curr, A_dt, self.B, u_k)
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
            # Return (1, N, N) to avoid redundant computations in Loss (e.g. Eigen decomposition)
            "A_ct": self.A_ct.unsqueeze(0), 
            "A_dt": A_dt.unsqueeze(0),
            "B": self.B.unsqueeze(0),
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
        Returns Discrete-time A matrix using Cayley Map.
        Ignores inputs, returns batch of static A matrices.
        """
        batch_size = x_history.size(0)
        A_dt = self._cayley_map(self.A_ct)
        return A_dt.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_B(self, x_history: torch.Tensor, u_history: torch.Tensor):
        """
        Returns B matrix.
        Ignores inputs, returns batch of static B matrices.
        """
        batch_size = x_history.size(0)
        return self.B.unsqueeze(0).expand(batch_size, -1, -1)

