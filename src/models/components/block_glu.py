import torch.nn as nn
from fla.modules import GatedMLP

class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_features: int, hidden_dim: int = 128, hidden_ratio: int = 2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            GatedMLP(
                hidden_size=hidden_dim,
                hidden_ratio=hidden_ratio,
                fuse_swiglu=False
            ),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)

