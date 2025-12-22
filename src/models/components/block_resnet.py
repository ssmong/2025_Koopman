import torch
import torch.nn as nn
from fla.modules import RMSNorm

class ResNetBlock(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation: nn.Module = nn.SiLU(),
        use_norm: bool = True,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features, hidden_size)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                RMSNorm(hidden_size) if use_norm else nn.Identity(),
                nn.Linear(hidden_size, hidden_size),
                activation,
                nn.Linear(hidden_size, hidden_size)
            ))
            
        self.output_proj = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + block(x)
        x = self.output_proj(x)
        
        return x
