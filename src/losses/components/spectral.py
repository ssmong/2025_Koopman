import torch
from src.losses.components.base import BaseLoss

class SpectralRadiusLoss(BaseLoss):
    def __init__(self, 
                 key_A: str = "A_dt",
                 mode: str = "discrete",
                 subsample: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.key_A = key_A
        self.mode = mode
        self.subsample = subsample
            
    def forward(self, results: dict) -> torch.Tensor:
        A = results.get(self.key_A)
        
        if A is None:
            return torch.tensor(0.0, device=self.device)

        if A.ndim == 2:
             batch_size = A.shape[0]
             dim = int(A.shape[1]**0.5)
             A = A.view(batch_size, dim, dim)
        
        # [Speed Optimization] Random Subsampling
        # O(N^3) Eigenvalue computation is expensive. 
        # We compute loss on a random subset of the batch.
        # This provides an unbiased estimator of the true loss gradients.
        if self.subsample is not None and A.size(0) > self.subsample:
            idx = torch.randperm(A.size(0), device=A.device)[:self.subsample]
            A = A[idx]
        
        eigvals = torch.linalg.eigvals(A.float()) # [B_sub, N]
        
        if self.mode == "discrete":
            rho = torch.max(eigvals.abs(), dim=1).values # [B]
            loss = torch.relu(rho - 1.0).pow(2).mean()

        elif self.mode == "continuous":
            max_real = torch.max(eigvals.real, dim=1).values # [B]
            loss = torch.relu(max_real).pow(2).mean()
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)

        return self.weight * self.apply_weight_decay(loss)
