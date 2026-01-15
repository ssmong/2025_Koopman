import torch
from src.losses.components.base import BaseLoss

class SpectralRadiusLoss(BaseLoss):
    def __init__(self, 
                 mode: str = 'continuous', # 'discrete' or 'continuous'
                 margin: float = 0.0,
                 key_A_params: str = "A_params",
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.margin = margin
        self.key_A_params = key_A_params
        
        if mode not in ['discrete', 'continuous']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'discrete' or 'continuous'.")

    def forward(self, results: dict) -> torch.Tensor:
        A_params = results.get(self.key_A_params)
        
        if A_params is None:
            return torch.tensor(0.0, device=self.device)

        # Ensure A_params is [B, N, N]
        if A_params.ndim == 2:
             batch_size = A_params.shape[0]
             dim = int(A_params.shape[1]**0.5)
             A_params = A_params.view(batch_size, dim, dim)
        
        vals = torch.linalg.eigvals(A_params)

        if self.mode == 'continuous':
            # Continuous stability: Re(eigenvalues) <= margin (usually 0)
            real_parts = vals.real
            loss = torch.relu(real_parts - self.margin).pow(2).mean()
            
        else: # discrete
            # Discrete stability: |eigenvalues| <= 1 - margin
            abs_vals = vals.abs()
            loss = torch.relu(abs_vals - (1.0 - self.margin)).pow(2).mean()
        
        return self.weight * self.apply_weight_decay(loss)
