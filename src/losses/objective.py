import torch.nn as nn
from typing import Dict


class Objective(nn.Module):
    def __init__(self, targets: Dict[str, nn.Module], **kwargs):
        super().__init__()

        self.targets = nn.ModuleDict(targets)
        self.encoder = None

    def bind_model(self, model: nn.Module):
        self.encoder = model.encoder
        
        # Propagate bind_model to child loss components if they have it
        for _, criterion in self.targets.items():
            if hasattr(criterion, 'bind_model'):
                criterion.bind_model(model)

    def forward(self, results: dict, epoch: int = 0):
        if self.encoder is None:
            raise ValueError("Encoder is not bound to the objective. Please call bind_model() first.")
        
        total_loss = 0.0
        loss_metrics = {}

        for name, criterion in self.targets.items():
            # Check for start_epoch (Curriculum Learning)
            start_epoch = getattr(criterion, 'start_epoch', 0)
            if epoch < start_epoch:
                # Log 0.0 for monitoring but don't add to total_loss
                loss_metrics[f"loss/{name}"] = 0.0
                continue

            loss_val = criterion(results)
            total_loss += loss_val
            loss_metrics[f"loss/{name}"] = loss_val.item()
        
        loss_metrics["loss/total"] = total_loss.item()

        return total_loss, loss_metrics