import os
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, HydraConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       1. Dataset & DataLoader
    # ------------------------------------------------------------------
    log.info(f"Initializing dataset...")

    train_dataset = hydra.utils.instantiate(cfg.data.train, split="train")
    val_dataset = hydra.utils.instantiate(cfg.data.val, split="val")

    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=True,
        persistent_workers=True
    )

    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    #       2. Model & Optimizer & Scheduler & Loss
    # ------------------------------------------------------------------
    log.info(f"Initializing model...")
    model = hydra.utils.instantiate(cfg.model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")
    
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    criterion = hydra.utils.instantiate(cfg.loss)
    criterion.bind_model(model)

    # ------------------------------------------------------------------
    #       3. Training Loop
    # ------------------------------------------------------------------
    log.info(f"Starting training...")

    epochs = cfg.train.epochs
    early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    model_path = os.path.join(output_dir, "best_model.pt")
    early_stopping.set_path(model_path)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
        