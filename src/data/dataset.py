import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple

log = logging.getLogger(__name__)

class KoopmanDataset(Dataset):
    def __init__(
        self, 
        name: str,
        data_root: str,
        raw_state_dim: int,
        state_dim: int,
        control_dim: int,
        state_key: str,
        control_key: str,
        angle_indices: List[int],
        quat_indices: List[int],
        num_sequences: int,
        sequence_len: int,
        raw_dt: float,
        hist_dt: float,
        hist_len: int,
        pred_dt: float,
        pred_len: int,
        normalization: bool = True,
        stats_dir: str = "data/stats"
    ):
        super().__init__()
        
        self.name = name
        self.data_root = Path(data_root)
        self.raw_state_dim = raw_state_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.state_key = state_key
        self.control_key = control_key
        self.angle_indices = angle_indices
        self.quat_indices = quat_indices

        self.num_sequences = num_sequences
        self.sequence_len = sequence_len
        self.raw_dt = raw_dt
        
        self.hist_dt = hist_dt
        self.hist_len = hist_len
        self.pred_dt = pred_dt
        self.pred_len = pred_len

        self.normalization = normalization
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Discovery & Loading
        states, self.controls = self._load_data()
        
        # 2. Feature Expansion
        self.expanded_states, self.skip_norm_mask = self._expand_features(states)
        self.current_state_dim = self.expanded_states[0].shape[-1]
        log.info(f"State dim expanded from {self.raw_state_dim} to {self.current_state_dim}")
        del states
        
        if self.state_dim != self.current_state_dim:
             raise ValueError(f"Config state_dim ({self.state_dim}) != Expanded dim ({self.current_state_dim})")

        # 3. Stats & Caching
        if self.normalization:
            self.stats = self._get_stats()
            self._apply_normalization()
            self._print_stats()
            
        # 4. Indexing
        self.hist_stride = int(round(self.hist_dt / self.raw_dt))
        self.pred_stride = int(round(self.pred_dt / self.raw_dt))
        
        if not np.isclose(self.hist_stride * self.raw_dt, self.hist_dt, atol=1e-5):
            raise ValueError(f"hist_dt ({self.hist_dt}) must be a multiple of raw_dt ({self.raw_dt})")
        if not np.isclose(self.pred_stride * self.raw_dt, self.pred_dt, atol=1e-5):
            raise ValueError(f"pred_dt ({self.pred_dt}) must be a multiple of raw_dt ({self.raw_dt})")
            
        self.indices = self._build_index()

    def _load_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        filename = f"{self.name}_{self.num_sequences}_{self.sequence_len}_{self.raw_dt}.h5"
        file_path = self.data_root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        log.info(f"Loading data from {file_path}")        
        states_list, controls_list = [], []
        
        with h5py.File(file_path, 'r') as hf:
            if 'timeseries' not in hf:
                raise ValueError(f"File {file_path} missing 'timeseries' group")
            
            ts_grp = hf['timeseries']
            sorted_keys = sorted(ts_grp.keys(), key=lambda k: int(k) if k.isdigit() else k)

            for seq_id in sorted_keys:
                grp = ts_grp[seq_id]
                try:
                    s = torch.from_numpy(grp[self.state_key][:]).float()
                    c = torch.from_numpy(grp[self.control_key][:]).float()
                except KeyError:
                    raise ValueError(f"Missing state/control keys in {file_path} (seq: {seq_id})")

                if s.shape[-1] != self.raw_state_dim:
                    raise ValueError(f"Dim mismatch in {file_path}: expected {self.raw_state_dim}, got {s.shape[-1]}")
                
                states_list.append(s)
                controls_list.append(c)

        log.info(f"Loaded {len(states_list)} sequences.")
        return states_list, controls_list

    def _expand_features(self, states_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        expanded_list = []
        mask_list = []
        
        # Determine mask structure
        for i in range(self.raw_state_dim):
            if i in self.angle_indices:
                mask_list.append(True) # cos
                mask_list.append(True) # sin
            else:
                if i in self.quat_indices:
                    mask_list.append(True)
                else:
                    mask_list.append(False)
                    
        skip_norm_mask = torch.tensor(mask_list, dtype=torch.bool)
        
        for s in states_list:
            new_feats = []
            for i in range(self.raw_state_dim):
                col = s[..., i]
                if i in self.angle_indices:
                    new_feats.append(torch.cos(col))
                    new_feats.append(torch.sin(col))
                else:
                    new_feats.append(col)
            
            expanded_s = torch.stack(new_feats, dim=-1)
            expanded_list.append(expanded_s)
            
        return expanded_list, skip_norm_mask

    def _get_stats(self) -> Dict[str, torch.Tensor]:
        stats_file = self.stats_dir / f"{self.name}_stats.json"
        
        if stats_file.exists():
            log.info(f"Loading stats from {stats_file}")
            with open(stats_file, 'r') as f:
                data = json.load(f)
            stats = {k: torch.tensor(v) for k, v in data.items()}
            return stats
            
        log.info("Computing stats...")
        
        s_sum = 0
        s_count = 0
        s_min, s_max = None, None

        c_sum = 0
        c_count = 0
        c_min, c_max = None, None
        
        for s, c in zip(self.expanded_states, self.controls):
            s_sum += s.sum(dim=0)
            s_count += s.shape[0]
            
            cur_s_min = s.min(dim=0)[0]
            cur_s_max = s.max(dim=0)[0]
            
            if s_min is None:
                s_min, s_max = cur_s_min, cur_s_max
            else:
                s_min = torch.min(s_min, cur_s_min)
                s_max = torch.max(s_max, cur_s_max)

            # --- Control ---
            c_sum += c.sum(dim=0)
            c_count += c.shape[0]
            
            cur_c_min = c.min(dim=0)[0]
            cur_c_max = c.max(dim=0)[0]
            
            if c_min is None:
                c_min, c_max = cur_c_min, cur_c_max
            else:
                c_min = torch.min(c_min, cur_c_min)
                c_max = torch.max(c_max, cur_c_max)
        
        s_mean = s_sum / s_count
        c_mean = c_sum / c_count
        
        s_var_sum = 0
        c_var_sum = 0
        
        for s, c in zip(self.expanded_states, self.controls):
            s_var_sum += ((s - s_mean) ** 2).sum(dim=0)
            c_var_sum += ((c - c_mean) ** 2).sum(dim=0)
        
        s_std = torch.sqrt(s_var_sum / s_count)
        s_std[s_std < 1e-6] = 1.0
        c_std = torch.sqrt(c_var_sum / c_count)
        c_std[c_std < 1e-6] = 1.0
        
        stats = {
            "mean": s_mean,
            "std": s_std,
            "min": s_min,
            "max": s_max,
            "ctrl_mean": c_mean,
            "ctrl_std": c_std,
            "ctrl_min": c_min,
            "ctrl_max": c_max
        }
        
        # 5. 저장
        with open(stats_file, 'w') as f:
            json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=4)
            
        return stats
    
    def _apply_normalization(self):
        # 1. State Normalization
        s_mean = self.stats['mean']
        s_std = self.stats['std']
        # Mask broadcasting: [D] -> [1, D]
        mask = self.skip_norm_mask.unsqueeze(0) 
        
        for i in range(len(self.expanded_states)):
            # If mask is True, keep original value, otherwise normalize
            self.expanded_states[i] = torch.where(
                mask, 
                self.expanded_states[i], 
                (self.expanded_states[i] - s_mean) / s_std
            )

        # 2. Control Normalization
        c_mean = self.stats['ctrl_mean']
        c_std = self.stats['ctrl_std']
        
        for i in range(len(self.controls)):
            self.controls[i] = (self.controls[i] - c_mean) / c_std

    def _print_stats(self):
        print(f"\n{'='*20} Dataset Statistics {'='*20}")
        print(f"{'Idx':<5} {'Skip':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        for i in range(self.current_state_dim):
            skip = "Yes" if self.skip_norm_mask[i] else "No"
            m = self.stats['mean'][i].item()
            s = self.stats['std'][i].item()
            mn = self.stats['min'][i].item()
            mx = self.stats['max'][i].item()
            print(f"{i:<5} {skip:<6} {m:<10.4f} {s:<10.4f} {mn:<10.4f} {mx:<10.4f}")
        print("="*60 + "\n")

    def _build_index(self) -> List[Tuple[int, int]]:
        indices = []
        min_idx = self.hist_len * self.hist_stride
        
        for seq_i, s in enumerate(self.expanded_states):
            L = s.shape[0]
            max_idx = L - 1 - (self.pred_len * self.pred_stride)
            
            if max_idx >= min_idx:
                for k in range(min_idx, max_idx + 1):
                    indices.append((seq_i, k))
                    
        log.info(f"Generated {len(indices)} samples.")
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_i, k = self.indices[idx]
        
        s_full = self.expanded_states[seq_i]
        c_full = self.controls[seq_i]
        
        hist_indices = torch.arange(k - self.hist_len * self.hist_stride, k, self.hist_stride)
        future_indices = torch.arange(k + self.pred_stride, k + self.pred_len * self.pred_stride + 1, self.pred_stride)
        u_future_indices = torch.arange(k, k + self.pred_len * self.pred_stride, self.pred_stride)
        
        x_hist = s_full[hist_indices]
        u_hist = c_full[hist_indices]
        x_init = s_full[k]
        x_future = s_full[future_indices]
        u_future = c_full[u_future_indices]
            
        return {
            "x_history": x_hist,
            "u_history": u_hist,
            "x_init": x_init,
            "x_future": x_future,
            "u_future": u_future
        }