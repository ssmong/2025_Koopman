import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)

class KoopmanDataProcessor:
    def __init__(
        self,
        raw_state_dim: int,
        state_dim: int,
        control_dim: int,
        angle_indices: List[int],
        quat_indices: List[int],
        normalization: bool = True,
        stats_dir: str = "data/stats"
        name: str = "dataset",
        device: str = "cpu"
    ):
        self.raw_state_dim = raw_state_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.angle_indices = set(angle_indices)
        self.quat_indices = set(quat_indices)
        self.normalization = normalization
        self.stats_dir = Path(stats_dir)
        self.name = name
        self.device = device

        self.mean = None
        self.std = None
        self.ctrl_mean = None
        self.ctrl_std = None
        
        self.min = None
        self.max = None
        self.ctrl_min = None
        self.ctrl_max = None

        self.skip_norm_mask = self._build_mask()
    
    def to(self, device: str):
        self.device = device
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            self.ctrl_mean = self.ctrl_mean.to(device)
            self.ctrl_std = self.ctrl_std.to(device)
            self.skip_norm_mask = self.skip_norm_mask.to(device)
        return self
    
    def _build_mask(self) -> torch.Tensor:
        mask_list = []
        for i in range(self.raw_state_dim):
            if i in self.angle_indices:
                mask_list.extend([True, True])
            elif i in self.quat_indices:
                mask_list.append(True)
            else:
                mask_list.append(False)
        
        # Dimension verification
        current_dim = len(mask_list)
        if current_dim != self.state_dim:
            log.warning(f"State dim mismatch: {current_dim} != {self.state_dim}")
        return torch.tensor(mask_list, dtype=torch.bool, device=self.device)

    def set_stats(self, stats_data: Dict[str, torch.Tensor]):
        """
        Insert the stats computed from the outside into the processor.
        """
        self.mean = stats_data['mean'].to(self.device).float()
        self.std = stats_data['std'].to(self.device).float()
        self.ctrl_mean = stats_data['ctrl_mean'].to(self.device).float()
        self.ctrl_std = stats_data['ctrl_std'].to(self.device).float()

        self.min = stats_data['min'].to(self.device).float()
        self.max = stats_data['max'].to(self.device).float()
        self.ctrl_min = stats_data['ctrl_min'].to(self.device).float()
        self.ctrl_max = stats_data['ctrl_max'].to(self.device).float()
        
        self.std[self.std < 1e-6] = 1.0
        self.ctrl_std[self.ctrl_std < 1e-6] = 1.0

    def load_stats(self, stats_path: Optional[Union[str, Path]] = None):
        if stats_path is None:
            stats_path = self.stats_dir / f"{self.name}_stats.json"
        else:
            stats_path = Path(stats_path)
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file missing: {stats_path}")

        log.info(f"Loading stats from {stats_path}")
        with open(stats_path, 'r') as f:
            data = json.load(f)
        
        tensor_data = {k: torch.tensor(v) for k, v in data.items()}
        self.set_stats(tensor_data)
    
    def save_stats(self, stats_path: Optional[Union[str, Path]] = None):
        if stats_path is None:
            stats_path = self.stats_dir / f"{self.name}_stats.json"
        else:
            stats_path = Path(stats_path)
        
        states_path.parent.mkdir(parents=True, exist_ok=True)

        stats_data = {
            "mean": self.mean, "std": self.std,
            "min": self.min, "max": self.max,
            "ctrl_mean": self.ctrl_mean, "ctrl_std": self.ctrl_std,
            "ctrl_min": self.ctrl_min, "ctrl_max": self.ctrl_max
        }
        
        serializable_stats = {k: v.cpu().tolist() if v is not None else None for k, v in stats_data.items()}
        
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        log.info(f"Stats saved to {stats_path}")

    def expand_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        if not self.angle_indices:
            return x_raw
        
        expanded_feats = []
        for i in range(self.raw_state_dim):
            col = x_raw[..., i]
            if i in self.angle_indices:
                expanded_feats.append(torch.cos(col))
                expanded_feats.append(torch.sin(col))
            else:
                expanded_feats.append(col)
        return torch.stack(expanded_feats, dim=-1)

    def normalize_state(self, x: torch.Tensor, is_expanded: bool = False) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device).float()

        if not is_expanded:
            x = self.expand_features(x)        
        if self.normalization and self.mean is not None:
            return torch.where(self.skip_norm_mask, x, (x - self.mean) / self.std)
        return x

    def normalize_control(self, u: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(u):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
        else:
            u = u.to(self.device).float()
        if self.normalization and self.ctrl_mean is not None:
            return (u - self.ctrl_mean) / self.ctrl_std
        return u

    def denormalize_state(self, x: torch.Tensor, is_expanded: bool = False) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device).float()
        if not is_expanded:
            x = self.expand_features(x)
        if self.normalization and self.mean is not None:
            return torch.where(self.skip_norm_mask, x, x * self.std + self.mean)
        return x

    def denormalize_control(self, u: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(u):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
        else:
            u = u.to(self.device).float()
        if self.normalization and self.ctrl_mean is not None:
            return u * self.ctrl_std + self.ctrl_mean
        return u
    

class KoopmanDataset(Dataset):
    def __init__(
        self, 
        # Dataset general
        name: str,
        data_root: str,
        raw_state_dim: int,
        state_dim: int,
        control_dim: int,
        state_key: str,
        control_key: str,
        angle_indices: List[int],
        quat_indices: List[int],
        # File specific
        num_seqs: int,
        seq_len: int,
        raw_dt: float,
        hist_dt: float,
        hist_len: int,
        pred_dt: float,
        train_steps: int,
        sample_stride: int,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
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
        
        self.processor = KoopmanDataProcessor(
            raw_state_dim=raw_state_dim,
            state_dim=state_dim,
            control_dim=control_dim,
            angle_indices=angle_indices,
            quat_indices=quat_indices,
            normalization=normalization,
            stats_dir=stats_dir,
            name=name,
            device=device,
        )
        
        self.num_seqs = num_seqs
        self.seq_len = seq_len
        self.raw_dt = raw_dt
        
        self.hist_dt = hist_dt
        self.hist_len = hist_len
        self.pred_dt = pred_dt
        self.train_steps = train_steps

        self.sample_stride = sample_stride

        self.split = split.lower()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.normalization = normalization
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        self.states, self.controls = self._load_data()

        self.current_state_dim = self.states[0].shape[-1]
        log.info(f"State dim: {self.raw_state_dim} -> {self.current_state_dim} (Configured: {self.state_dim})")
        
        if self.state_dim != self.current_state_dim:
             raise ValueError(f"Config state_dim ({self.state_dim}) != Expanded dim ({self.current_state_dim})")

        if self.processor.normalization:
            if self.split == "train":
                log.info("Computing stats from training data...")
                stats_data = self._compute_stats()
                
                self.processor.set_stats(stats_data)
                self.processor.save_stats()
                
                self._print_stats()
            else:
                self.processor.load_stats()
            
        self.hist_stride = int(round(self.hist_dt / self.raw_dt))
        self.pred_stride = int(round(self.pred_dt / self.raw_dt))
        
        self._validate_strides()
        
        # Pre-calc offsets (relative to index k)
        # History: [k - H*s, ..., k - s]
        self.hist_offsets = torch.arange(-self.hist_len * self.hist_stride, 0, self.hist_stride, dtype=torch.long)
        # Future State: [k + s, ..., k + P*s]
        self.future_offsets = torch.arange(self.pred_stride, self.train_steps * self.pred_stride + 1, self.pred_stride, dtype=torch.long)
        # Future Control: [k, ..., k + (P-1)*s] -> often u_k is applied to get x_{k+1}
        self.u_future_offsets = torch.arange(0, self.train_steps * self.pred_stride, self.pred_stride, dtype=torch.long)

        # 4. Build Index Map
        self.indices = self._build_index()

    def _validate_strides(self):
        if not np.isclose(self.hist_stride * self.raw_dt, self.hist_dt, atol=1e-5):
            raise ValueError(f"hist_dt ({self.hist_dt}) must be mulpvple of raw_dt ({self.raw_dt})")
        if not np.isclose(self.pred_stride * self.raw_dt, self.pred_dt, atol=1e-5):
            raise ValueError(f"pred_dt ({self.pred_dt}) must be mulpvple of raw_dt ({self.raw_dt})")

    def _load_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        filename = f"{self.name}_{self.num_seqs}_{self.seq_len}_{self.raw_dt}.h5"
        file_path = self.data_root / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        log.info(f"Loading and processing data from {file_path}")
        
        states_list, controls_list = [], []

        with h5py.File(file_path, 'r') as hf:
            if 'timeseries' not in hf:
                raise ValueError(f"Missing 'timeseries' group in {file_path}")
            
            ts_grp = hf['timeseries']
            all_keys = sorted(ts_grp.keys(), key=lambda k: int(k) if k.isdigit() else k)
            total_seqs = len(all_keys)
            
            # Stable shuffle
            rng = np.random.RandomState(seed=42)
            rng.shuffle(all_keys)
            
            n_train = int(total_seqs * self.train_ratio)
            n_val = int(total_seqs * self.val_ratio)
            
            if self.split == 'train':
                target_keys = all_keys[:n_train]
            elif self.split == 'val':
                target_keys = all_keys[n_train : n_train + n_val]
            elif self.split == 'test':
                target_keys = all_keys[n_train + n_val :]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            # Load and Expand immediately to save memory spikes
            for seq_id in target_keys:
                grp = ts_grp[seq_id]
                
                # Load Raw
                s_raw = torch.from_numpy(grp[self.state_key][:]).float()
                c_raw = torch.from_numpy(grp[self.control_key][:]).float()
                
                if torch.isinf(s_raw).any():
                    log.error(f"Seq {seq_id}: Float32 overflow detected in RAW state data!")
                if torch.isnan(s_raw).any():
                    log.error(f"Seq {seq_id}: NaN detected in RAW state data!")
                if torch.isinf(c_raw).any():
                    log.error(f"Seq {seq_id}: Float32 overflow detected in RAW control data!")
                if torch.isnan(c_raw).any():
                    log.error(f"Seq {seq_id}: NaN detected in RAW control data!")

                if s_raw.shape[-1] != self.raw_state_dim:
                    raise ValueError(f"Seq {seq_id}: Dim mismatch {s_raw.shape[-1]} != {self.raw_state_dim}")

                # Feature Expansion (Angle -> Cos/Sin)
                # If angle_indices is empty, this loop is skipped or minimal overhead
                s_final = self.processor.expand_features(s_raw)

                if torch.isnan(s_final).any():
                    log.error(f"Seq {seq_id}: NaN generated during feature expansion!")

                states_list.append(s_final)
                controls_list.append(c_raw)

        log.info(f"Loaded {len(states_list)} seqs for split '{self.split}'.")
        return states_list, controls_list, skip_norm_mask

    def _compute_stats(self) -> Dict[str, torch.Tensor]:
        # Concatenate all seqs to calculate global stats efficiently
        # Since we have plenty of RAM, we are not using lazy slicing.
        all_s = torch.cat(self.states, dim=0)   # [Total_Frames, State_Dim]
        all_c = torch.cat(self.controls, dim=0) # [Total_Frames, Control_Dim]
        
        return {
            "mean": all_s.mean(dim=0),
            "std": all_s.std(dim=0),
            "min": all_s.min(dim=0)[0],
            "max": all_s.max(dim=0)[0],
            "ctrl_mean": all_c.mean(dim=0),
            "ctrl_std": all_c.std(dim=0),
            "ctrl_min": all_c.min(dim=0)[0],
            "ctrl_max": all_c.max(dim=0)[0]
        }

    def _print_stats(self):
        print(f"\n{'='*20} Dataset Statistics ({self.split}) {'='*20}")
        print(f"{'Idx':<5} {'Skip':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        for i in range(self.current_state_dim):
            skip = "Yes" if self.skip_norm_mask[i] else "No"
            m = self.mean[i].item()
            s = self.std[i].item()
            mn = self.min[i].item()
            mx = self.max[i].item()
            print(f"{i:<5} {skip:<6} {m:<10.4f} {s:<10.4f} {mn:<10.4f} {mx:<10.4f}")
        print("="*65 + "\n")

        print(f"\n{'='*25} CONTROL Statistics ({self.split}) {'='*24}")
        print(f"{'Idx':<5} {'Skip':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 75)
        for i in range(self.control_dim):
            skip = "No" 
            m = self.ctrl_mean[i].item()
            s = self.ctrl_std[i].item()
            mn = self.ctrl_min[i].item()
            mx = self.ctrl_max[i].item()
            print(f"{i:<5} {skip:<6} {m:<10.4f} {s:<10.4f} {mn:<10.4f} {mx:<10.4f}")
        print("="*75 + "\n")

    def _build_index(self) -> List[Tuple[int, int]]:
        indices = []
        # Minimum required history length
        min_idx = self.hist_len * self.hist_stride
        
        for seq_i, s in enumerate(self.states):
            L = s.shape[0]
            # Max index allowing for prediction horizon
            max_idx = L - 1 - (self.train_steps * self.pred_stride)
            
            if max_idx >= min_idx:
                valid_range = range(min_idx, max_idx + 1, self.sample_stride)
                indices.extend([(seq_i, k) for k in valid_range])
                    
        log.info(f"Generated {len(indices)} samples.")
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_i, k = self.indices[idx]
        
        # 1. Read-only access to source tensors
        s_full = self.states[seq_i]
        c_full = self.controls[seq_i]
        
        # 2. Fast Indexing using pre-computed offsets
        # Use broadcasting if needed, but here simple indexing works
        x_hist = s_full[k + self.hist_offsets]
        u_hist = c_full[k + self.hist_offsets]
        
        x_init = s_full[k]
        
        x_future = s_full[k + self.future_offsets]
        u_future = c_full[k + self.u_future_offsets]
            
        # 3. Normalization (Optimized)
        if self.processor.normalization:
            x_hist = self.processor.normalize_state(x_hist, is_expanded=True)
            x_init = self.processor.normalize_state(x_init, is_expanded=True)
            x_future = self.processor.normalize_state(x_future, is_expanded=True)
            
            u_hist = self.processor.normalize_control(u_hist)
            u_future = self.processor.normalize_control(u_future)

        return {
            "x_history": x_hist,
            "u_history": u_hist,
            "x_init": x_init,
            "x_future": x_future,
            "u_future": u_future
        }