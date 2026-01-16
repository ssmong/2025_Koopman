import torch
import torch.nn as nn
from src.losses.components.base import BaseLoss

class BackwardRollingLossDense(BaseLoss):
    def __init__(self,
                 key_z: str = "z_traj_gt",
                 key_x: str = "x_traj_gt",
                 key_A_ct: str = "A_ct",
                 key_B: str = "B",
                 key_u: str = "u_future",
                 weight_z: float = 1.0,
                 weight_x: float = 0.0,
                 dt: float = None, # Optional: if None, try to get from model
                 **kwargs):
        super().__init__(key_pred=key_z, key_target=key_z, **kwargs)
        
        self.key_z = key_z
        self.key_x = key_x
        self.key_A_ct = key_A_ct
        self.key_B = key_B
        self.key_u = key_u
        self.weight_z = weight_z
        self.weight_x = weight_x
        
        self.dt = dt
        self.decoder = None

    def bind_model(self, model: nn.Module):
        self.decoder = getattr(model, 'decoder', None)
        # 모델에서 dt를 가져오기 시도
        if self.dt is None and hasattr(model, 'dt'):
            self.dt = model.dt

    def _get_backward_A(self, A_ct: torch.Tensor) -> torch.Tensor:
        """
        Computes Discrete-time Backward A matrix: A_dt_inv = Cayley(A_ct, -dt)
        """
        if self.dt is None:
            raise ValueError("dt is not set. Provide it in config or bind a model with .dt attribute.")

        batch_size, dim, _ = A_ct.shape
        I = torch.eye(dim, device=A_ct.device).unsqueeze(0)
        
        # Use -dt for inverse dynamics
        # Cayley(-dt) = (I + 0.5*A*dt)^{-1} (I - 0.5*A*dt)
        A_half = A_ct * (-self.dt * 0.5)
        
        U = I - A_half
        V = I + A_half
        
        # Solve U X = V
        return torch.linalg.solve(U, V)

    def forward(self, results: dict) -> torch.Tensor:
        z = results.get(self.key_z)             # [B, N+1, D]
        x = results.get(self.key_x) if self.weight_x > 0 else None
        
        A_ct = results.get(self.key_A_ct)       # [B, D, D] (Dense A_ct)
        B = results.get(self.key_B)             # [B, D, Du]
        u = results.get(self.key_u)             # [B, N, Du]

        if z is None or A_ct is None or B is None or u is None:
             raise KeyError(f"BackwardLossDense missing keys. Check if model returns '{self.key_A_ct}' as dense matrix.")
        
        # 1. Compute Backward Dynamics Matrix for each batch
        # A_inv: [B, D, D]
        A_inv = self._get_backward_A(A_ct)
        
        # 2. Rolling Backward
        # Start from last step z_N, roll back to z_0
        z_roll = z[:, -1, :] # [B, D]
        N = u.size(1)
        
        # Loop backwards: N-1 down to 0
        for k in range(N - 1, -1, -1):
            u_k = u[:, k, :] # [B, Du]
            
            # z_{k} = A^{-1} (z_{k+1} - B u_k)
            # 1. Bu term
            Bu = torch.einsum('b d u, b u -> b d', B, u_k)
            
            # 2. Subtract input
            rhs = z_roll - Bu
            
            # 3. Apply Inverse Dynamics
            z_roll = torch.einsum('bij,bj->bi', A_inv, rhs)
        
        total_loss = 0.0
        
        # 3. Z-space Loss (Consistency at t=0)
        if self.weight_z > 0:
            z_0_gt = z[:, 0, :]
            loss_z = ((z_roll - z_0_gt) ** 2).mean(dim=-1).mean()
            total_loss = total_loss + self.weight_z * loss_z

        # 4. X-space Loss (Consistency at t=0)
        if self.weight_x > 0:
            if self.decoder is None:
                 raise ValueError("Decoder is not bound. Call bind_model() first.")
            
            x_pred_bwd_0 = self.decoder(z_roll) # [B, D_x]
            x_0_gt = x[:, 0, :]
            
            # Use the loss function defined in BaseLoss (usually MSE)
            loss_x = self.loss_fn(x_pred_bwd_0, x_0_gt).mean(dim=-1).mean()
            total_loss = total_loss + self.weight_x * loss_x
        
        return self.weight * total_loss
