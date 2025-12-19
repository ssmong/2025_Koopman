import torch.nn as nn
from typing import Dict


class Objective(nn.Module):
    def __init__(self, targets: Dict[str, nn.Module]):
        super().__init__()

        self.targets = nn.ModuleDict(targets)
        self.encoder = None

    def bind_model(self, model: nn.Module):
        self.encoder = model.encoder

    def forward(self, results: dict):
        if self.encoder is None:
            raise ValueError("Encoder is not bound to the objective. Please call bind_model() first.")
        
        total_loss = 0.0
        log_metrics = {}

        for name, criterion in self.targets.items():
            loss_val = criterion(results)
            total_loss += loss_val
            log_metrics[f"loss/{name}"] = loss_val.item()

        return total_loss, log_metrics