import torch
from src.losses.components.base import BaseLoss

class SpectralRadiusLoss(BaseLoss):
    def __init__(self, 
                 margin: float = 0.0,
                 key_A: str = "A_dt",
                 **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.key_A = key_A
            
    def forward(self, results: dict) -> torch.Tensor:
        A = results.get(self.key_A)
        
        if A is None:
            return torch.tensor(0.0, device=self.device)

        # Ensure A is [B, N, N]
        if A.ndim == 2:
             batch_size = A.shape[0]
             dim = int(A.shape[1]**0.5)
             A = A.view(batch_size, dim, dim)
        
        # Use PyTorch built-in Matrix Norm (Spectral Norm when ord=2)
        # Spectral Norm = Max Singular Value >= Max Absolute Eigenvalue (Spectral Radius)
        # Enforcing Spectral Norm <= 1 guarantees Stability (Spectral Radius <= 1)
        # Requires float32 for SVD-based computation
        rho = torch.linalg.matrix_norm(A.float(), ord=2) # [B]
        
        # Loss: max(0, rho - (1 - margin))^2
        loss = torch.relu(rho - (1.0 - self.margin)).pow(2).mean()
        
        # Ensure loss is at least 1D [1] for apply_weight_decay
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)

        return self.weight * self.apply_weight_decay(loss)
