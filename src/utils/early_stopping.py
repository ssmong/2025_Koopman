# src/utils/early_stopping.py
import numpy as np
import torch
import logging
from pathlib import Path

log = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, 
                 patience: int = 20, 
                 delta: float = 0.0, 
                 path: str = 'best_model.pt', 
                 verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss + self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                log.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def set_path(self, path: str):
        self.path = path

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            log.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss