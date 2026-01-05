import torch
import torch.nn as nn
import torch.nn.functional as F

class LULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features == out_features, "Mixing Layer must be square for inversion."
        self.features = in_features

        self.l_lower = nn.Parameter(torch.eye(in_features) + torch.randn(in_features, in_features) * 0.01)
        
        self.u_diag = nn.Parameter(torch.zeros(in_features))
        self.u_upper = nn.Parameter(torch.randn(in_features, in_features) * 0.01)

        self.register_buffer('mask_lower', torch.tril(torch.ones(in_features, in_features), diagonal=-1))
        self.register_buffer('mask_upper', torch.triu(torch.ones(in_features, in_features), diagonal=1))
        self.register_buffer('eye', torch.eye(in_features))

    def get_matrix(self):
        L = (self.l_lower * self.mask_lower) + self.eye
        
        d = F.softplus(self.u_diag) + 1e-6 
        U = (self.u_upper * self.mask_upper) + torch.diag(d)
        
        W = torch.matmul(L, U)
        return W

    def forward(self, x):
        W = self.get_matrix()
        return F.linear(x, W)

    def inverse(self, z):
        W = self.get_matrix()
        
        # Convert to float32 for inverse calculation
        # torch.linalg.inv does not support float16 (Half)
        orig_dtype = W.dtype
        if orig_dtype == torch.float16:
            W = W.float()
            
        W_inv = torch.linalg.inv(W)
        
        if orig_dtype == torch.float16:
            W_inv = W_inv.to(orig_dtype)
            
        return F.linear(z, W_inv)