import os
import torch
import hydra
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def load_finetune_model(cfg: DictConfig, pretrained_dir: str, device: str = "cuda", strict: bool = False, criterion = None):
    """
    Load model for fine-tuning.
    
    Args:
        cfg (DictConfig): Current training configuration.
        pretrained_dir (str): Path string like 'YYYY-MM-DD/HH-MM-SS'.
                              Weights are loaded from 'outputs/learning/{pretrained_dir}/best_model.pt'.
        device (str): Device to load the model on.
        strict (bool): Strict mode for load_state_dict.
        criterion (torch.nn.Module): Optional criterion to load state_dict into.
        
    Returns:
        model: Model with loaded weights.
    """
    log.info("Initializing model from current configuration for fine-tuning...")
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)

    weights_path = os.path.join("outputs", "learning", pretrained_dir, "best_model.pt")
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    log.info(f"Loading pretrained weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle both old format (model only) and new format (dict with 'model' and 'criterion')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
        if criterion is not None and 'criterion' in checkpoint:
            log.info("Loading criterion state_dict...")
            criterion.load_state_dict(checkpoint['criterion'])
    else:
        model_state = checkpoint
        if criterion is not None:
            log.warning("Checkpoint does not contain criterion state. Criterion will use initialized weights.")
    
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=strict)
    
    if missing_keys:
        log.warning(f"Missing keys (initialized randomly): {missing_keys}")
    if unexpected_keys:
        log.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")
        
    log.info("Model weights loaded successfully.")
    return model
