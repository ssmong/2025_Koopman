import sys
import os

# Set Project Root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)

import torch
import hydra
import logging
import scipy.io
import numpy as np
import h5py
from omegaconf import OmegaConf
from src.data.dataset import KoopmanDataProcessor

log = logging.getLogger(__name__)

def export_lpv():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load parameters from Environment
    data_path = os.path.join("data/raw", os.environ.get("DATA_PATH")) # Example: "attitude_1000_1000_0.1.h5"
    seq_id = os.environ.get("SEQ_ID")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") # Example: "2025-01-14/10-00-00"

    if not data_path or not seq_id or not checkpoint_dir:
        log.error("Environment variables DATA_PATH, SEQ_ID, or CHECKPOINT_DIR are missing.")
        return

    # Load Config & Model (From Checkpoint)
    base_dir = os.path.join("outputs", "learning", checkpoint_dir)
    cfg_path = os.path.join(base_dir, ".hydra", "config.yaml")
    weights_path = os.path.join(base_dir, "best_model.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    log.info(f"Loading config from {cfg_path}...")
    cfg = OmegaConf.load(cfg_path)
    
    log.info(f"Loading model from {weights_path}...")
    try:
        model = hydra.utils.instantiate(cfg.model)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return

    # Load Processor & Statistics (Pre-processing)
    log.info("Initializing KoopmanDataProcessor...")
    
    stats_path = os.path.join(base_dir, "stats", f"{cfg.data.name}_stats.json")
    
    processor = KoopmanDataProcessor(
        raw_state_dim=cfg.data.raw_state_dim,
        state_dim=cfg.data.state_dim,
        control_dim=cfg.data.control_dim,
        angle_indices=list(cfg.data.angle_indices),
        quat_indices=list(cfg.data.quat_indices),
        normalization=cfg.data.normalization,
        stats_dir=os.path.dirname(stats_path),
        name=cfg.data.name,
        device=device
    )
    
    try:
        if os.path.exists(stats_path):
            processor.load_stats(stats_path)
            log.info(f"Statistics loaded from {stats_path}")
        else:
            log.error(f"Statistics file not found at {stats_path}")
            return
            
    except Exception as e:
        log.error(f"Failed to load statistics: {e}")
        return


    # Load Specific Sequence & Prepare Input
    log.info(f"Loading sequence {seq_id} from {data_path}...")
    try:
        with h5py.File(data_path, 'r') as f:
            seq_key = f"timeseries/sequence_{seq_id}"
            state_key = cfg.data.state_key
            control_key = cfg.data.control_key
            
            if seq_key not in f:
                raise KeyError(f"Sequence {seq_id} not found in H5 file.")
            
            # Load Raw Data
            s_raw_np = f[f"{seq_key}/{state_key}"][:]  
            c_raw_np = f[f"{seq_key}/{control_key}"][:] 
            
            s_raw = torch.from_numpy(s_raw_np).float().to(device)
            c_raw = torch.from_numpy(c_raw_np).float().to(device)

            # Feature Expansion (Angle -> Cos/Sin)
            s_expanded = processor.expand_features(s_raw)

            # History Length check
            hist_len = cfg.data.hist_len
            if s_expanded.shape[0] < hist_len + 1:
                raise ValueError(f"Sequence length ({s_expanded.shape[0]}) is shorter than required history ({hist_len}).")

            # Extract History from the BEGINNING (t=0 to t=hist_len)
            x_hist_raw = s_expanded[:hist_len, :]      # [hist_len, Dim]
            u_hist_raw = c_raw[:hist_len, :]           # [hist_len, Dim]
            x_init_raw = s_expanded[hist_len, :]       # [Dim]
            
            # Normalize
            x_hist = processor.normalize_state(x_hist_raw, is_expanded=True)
            u_hist = processor.normalize_control(u_hist_raw)
            x_init = processor.normalize_state(x_init_raw, is_expanded=True)

            # Add Batch Dimension [1, Window, Dim]
            x_history = x_hist.unsqueeze(0)
            u_history = u_hist.unsqueeze(0)
            x_init_batch = x_init.unsqueeze(0) # [1, Dim]

    except Exception as e:
        log.error(f"Data loading failed: {e}")
        return

    # Extract Linear Matrices
    log.info("Extracting linearized matrices...")
    with torch.no_grad():
        A_tensor = model.get_A(x_history, u_history) 
        B_tensor = model.get_B(x_history, u_history)
        z0_tensor = model.encoder(x_init_batch)
        
        A_np = A_tensor.squeeze(0).cpu().numpy()
        B_np = B_tensor.squeeze(0).cpu().numpy()
        z0_np = z0_tensor.squeeze(0).cpu().numpy()

        # C Matrix
        state_dim = cfg.model.state_dim
        latent_dim = cfg.model.latent_dim
        C_np = np.hstack([np.eye(state_dim), np.zeros((state_dim, latent_dim - state_dim))])

        # Get Normalized Constraints
        u_min_raw = torch.full((1, cfg.data.control_dim), float(cfg.constraint.u_min), device=device)
        u_max_raw = torch.full((1, cfg.data.control_dim), float(cfg.constraint.u_max), device=device)
        
        u_lb_norm = processor.normalize_control(u_min_raw)
        u_ub_norm = processor.normalize_control(u_max_raw)

        u_lb_norm = u_lb_norm.squeeze(0).cpu().numpy()
        u_ub_norm = u_ub_norm.squeeze(0).cpu().numpy()

        w_lb_raw = -0.5
        w_ub_raw = 0.5
        
        w_indices = cfg.data.omega_indices
        
        w_mean = processor.mean[w_indices].cpu().numpy()
        w_std = processor.std[w_indices].cpu().numpy()
        
        w_lb_norm = (w_lb_raw - w_mean) / w_std
        w_ub_norm = (w_ub_raw - w_mean) / w_std

        # Get Origin in Lifted Space
        # Raw origin: [qw, qx, qy, qz, wx, wy, wz] = [1, 0, 0, 0, 0, 0, 0]
        x_origin_raw = torch.zeros((1, cfg.data.raw_state_dim), device=device)
        x_origin_raw[0, 0] = 1.0 # qw = 1

        # Expand & Normalize
        x_origin_expanded = processor.expand_features(x_origin_raw)
        x_origin_norm = processor.normalize_state(x_origin_expanded, is_expanded=True)

        # Encode to Lifted Space
        z_origin_tensor = model.encoder(x_origin_norm)
        z_origin_np = z_origin_tensor.detach().cpu().numpy().flatten() # [LatentDim]

    # Prepare Stats for MATLAB
    def to_np(t):
        return t.cpu().numpy() if t is not None else np.array([])

    stats_export = {
        "x_mean": to_np(processor.mean),
        "x_std": to_np(processor.std),
        "x_min": to_np(processor.min),
        "x_max": to_np(processor.max),
        "u_mean": to_np(processor.ctrl_mean),
        "u_std": to_np(processor.ctrl_std),
        "u_min": to_np(processor.ctrl_min),
        "u_max": to_np(processor.ctrl_max),
    }

    # Save
    output_path = "data/brs/brs_data.mat"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mat_data = {
        "A": A_np, 
        "B": B_np, 
        "z0": z0_np, 
        "C": C_np,
        "u_lb": u_lb_norm[0], # Assuming symmetric/scalar bound derived, take first elem or keep vector
        "u_ub": u_ub_norm[0],
        "w_lb": w_lb_norm,
        "w_ub": w_ub_norm,
        "z_origin": z_origin_np,
        "seq_id": seq_id,
        **stats_export
    }
    
    scipy.io.savemat(output_path, mat_data)
    log.info(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    export_lpv()