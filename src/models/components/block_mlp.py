import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 128, num_layers: int = 2, activation=nn.SiLU()):
        super().__init__()
        
        layers = []
        # Input Layer
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(activation)
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, out_features))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

