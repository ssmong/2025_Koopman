import os
import torch
import hydra
from omegaconf import OmegaConf
import logging
import json

log = logging.getLogger(__name__)

def load_model(checkpoint_dir: str, device: str = "cuda"):
    cfg_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    weights_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    cfg = OmegaConf.load(cfg_path)
    
    log.info(f"Loading model from {weights_path}...")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(weights_path, map_location=device), weights_only=True)

    model.to(device)
    model.eval()

    data_name = cfg.data.name
    stats_path = os.path.join(checkpoint_dir, "stats", f"{data_name}_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    stats_dict = {}
    keys = ["mean", "std", "ctrl_mean", "ctrl_std"]
    for key in keys:
        if key in stats:
            stats_dict[key] = torch.tensor(stats[key], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Key {key} not found in stats")

    log.info(f"Loaded stats from {stats_path}")

    return model, cfg, stats_dict
