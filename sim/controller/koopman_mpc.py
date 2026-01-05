from typing import Dict
import logging
import time
from collections import deque
import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import traceback

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

from src.data.dataset import KoopmanDataProcessor
from sim.utils.load import load_model
from sim.utils.profiler import ControllerProfiler

log = logging.getLogger(__name__)

class BskKoopmanMPC(sysModel.SysModel):
    def __init__(
        self,
        checkpoint_dir: str,
        mpc_params: Dict,
        ctrl_dt: float = 1.0,
        device: str = "cuda"
        ):
        super().__init__()
        self.ModelTag = "KoopmanMPC"

        self.model, self.prev_cfg, self.stats_dict = load_model(checkpoint_dir, device)
        
        self.device = device
        self.ctrl_dt = ctrl_dt
        self.last_ctrl_update_time = -self.ctrl_dt

        self.processor = KoopmanDataProcessor(
            raw_state_dim=self.prev_cfg.data.raw_state_dim,
            state_dim=self.prev_cfg.data.state_dim,
            control_dim=self.prev_cfg.data.control_dim,
            angle_indices=self.prev_cfg.data.angle_indices,
            quat_indices=self.prev_cfg.data.quat_indices,
            normalization=self.prev_cfg.data.normalization,
            stats_dir="data/stats",          
            name=self.prev_cfg.data.name,
            device=device,
        )

        self.processor.set_stats(self.stats_dict)

        self.hist_len = self.prev_cfg.data.hist_len
        self.control_dim = self.prev_cfg.data.control_dim
        
        self.hist_dt = self.prev_cfg.data.get('dt', 0.1)

        self.controller = KoopmanMPC(
            model=self.model,
            processor=self.processor,
            device=device,
            **mpc_params,
        )

        self.guidInMsg = messaging.AttGuidMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        self.vehConfigInMsg = messaging.VehicleConfigMsgReader()

        x_ref = [1, 0, 0, 0, 0, 0, 0]
        self.x_ref_norm = self.processor.normalize_state(x_ref, is_expanded=False)
        
        if isinstance(self.x_ref_norm, torch.Tensor):
            self.x_ref_norm = self.x_ref_norm.detach().cpu().numpy()

        self.x_history = deque(maxlen=self.hist_len)
        self.u_history = deque(maxlen=self.hist_len)
        self.prev_u = np.zeros(self.control_dim)
        
        self.warmup_time = 0.0
        self.last_hist_update_time = -1.0
        
        self.profiler = ControllerProfiler(skip_first=True)

        log.info(f"Basilisk Koopman MPC initialized successfully from {checkpoint_dir}")

    def set_warmup_time(self, time: float):
        self.warmup_time = time

    def Reset(self, CurrentSimNanos):
        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"Reset called at {CurrentSimNanos}")
        if not self.guidInMsg.isLinked():
            self.bskLogger.bskLog(bskLogging.BSK_ERROR, "guidInMsg not linked")
            return
        
        payload = messaging.CmdTorqueBodyMsgPayload()
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos)

        self.x_history.clear()
        self.u_history.clear()
        self.prev_u = np.zeros(self.control_dim)
        self.last_hist_update_time = -1.0
        self.last_ctrl_update_time = -self.ctrl_dt

        if self.guidInMsg.isWritten():
            guid_msg = self.guidInMsg()
            sigma_BR = np.array(guid_msg.sigma_BR)
            omega_BR_B = np.array(guid_msg.omega_BR_B)
            ep_BR = rbk.MRP2EP(sigma_BR)
            x_init_np = np.concatenate([ep_BR, omega_BR_B])
        else:
            x_init_np = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        x_init_norm = self.processor.normalize_state(x_init_np, is_expanded=False)
        u_init_norm = self.processor.normalize_control(self.prev_u)

        for _ in range(self.hist_len):
            self.x_history.append(x_init_norm)
            self.u_history.append(u_init_norm)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, "Reset Koopman MPC successfully")

    def UpdateState(self, CurrentSimNanos):
        current_time_sec = CurrentSimNanos * 1e-9
        
        if not self.guidInMsg.isLinked():
            return

        guid_msg = self.guidInMsg()
        sigma_BR = np.array(guid_msg.sigma_BR)
        omega_BR_B = np.array(guid_msg.omega_BR_B)
        ep_BR = rbk.MRP2EP(sigma_BR)

        x_curr = np.concatenate([ep_BR, omega_BR_B])
        x_curr_norm = self.processor.normalize_state(x_curr, is_expanded=False)
        
        if isinstance(x_curr_norm, torch.Tensor):
            x_curr_norm = x_curr_norm.detach().cpu().numpy()
        
        # --- Update History (Always based on history time step) ---
        if self.last_hist_update_time < 0 or (current_time_sec - self.last_hist_update_time) >= self.hist_dt - 1e-6:
            u_prev_norm = self.processor.normalize_control(self.prev_u)
            
            if isinstance(u_prev_norm, torch.Tensor):
                u_prev_norm = u_prev_norm.detach().cpu().numpy()
                
            self.x_history.append(x_curr_norm)
            self.u_history.append(u_prev_norm)
            self.last_hist_update_time = current_time_sec

        # --- Control Update (Decimated) ---
        if current_time_sec < self.warmup_time:
             # Warmup: Output zero torque, but continue updating history
            u_opt = np.zeros(self.control_dim)
            self.prev_u = u_opt
            
            out_payload = messaging.CmdTorqueBodyMsgPayload()
            out_payload.torqueRequestBody = u_opt.tolist()
            self.cmdTorqueOutMsg.write(out_payload, CurrentSimNanos, self.moduleID)
            return

        # Check if it's time for new control (based on ctrl_dt)
        if self.last_ctrl_update_time < 0 or (current_time_sec - self.last_ctrl_update_time) >= self.ctrl_dt - 1e-6:
             self.last_ctrl_update_time = current_time_sec
        else:
             # Not time yet: Hold previous control
             out_payload = messaging.CmdTorqueBodyMsgPayload()
             out_payload.torqueRequestBody = self.prev_u.tolist()
             self.cmdTorqueOutMsg.write(out_payload, CurrentSimNanos, self.moduleID)
             return

        obs = {
            "x_curr": x_curr_norm,
            "x_ref": self.x_ref_norm,
            "x_history": list(self.x_history),
            "u_history": list(self.u_history),
        }

        t_start = time.perf_counter()
        
        try:
            u_opt_norm = self.controller.getControl(obs)
        except Exception as e:
            self.bskLogger.bskLog(bskLogging.BSK_ERROR, f"MPC Error: {e}")
            u_opt_norm = np.zeros(self.control_dim)
            
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        solver_time = self.controller.last_solver_time
        
        self.profiler.update(wall_time, solver_time)
        
        u_opt = self.processor.denormalize_control(u_opt_norm)
        if isinstance(u_opt, torch.Tensor):
            u_opt = u_opt.detach().cpu().numpy()

        self.prev_u = u_opt

        out_payload = messaging.CmdTorqueBodyMsgPayload()
        out_payload.torqueRequestBody = u_opt.tolist()
        self.cmdTorqueOutMsg.write(out_payload, CurrentSimNanos, self.moduleID)


class KoopmanMPC:
    def __init__(self, 
                 model,
                 processor,
                 horizon: int, 
                 u_min: float, 
                 u_max: float, 
                 Q_diag: list, 
                 R_diag: list,
                 F_diag: list,
                 rest_ratio: float = 1,
                 constraint_cfg: dict = None,
                 device: str = "cuda"):
        
        self.model = model
        self.processor = processor
        self.horizon = horizon
        self.device = device
        self.latent_dim = model.latent_dim
        self.control_dim = model.control_dim
        self.constraint_cfg = constraint_cfg
        
        u_min_phys = np.full(self.control_dim, u_min)
        u_max_phys = np.full(self.control_dim, u_max)
        
        self.u_min_norm = self.processor.normalize_control(torch.from_numpy(u_min_phys).float().to(self.processor.device)).cpu().numpy()
        self.u_max_norm = self.processor.normalize_control(torch.from_numpy(u_max_phys).float().to(self.processor.device)).cpu().numpy()
           
        self.z = cp.Variable((horizon + 1, self.latent_dim))
        self.u = cp.Variable((horizon, self.control_dim))
        
        self.z_init = cp.Parameter(self.latent_dim)
        self.z_ref = cp.Parameter((horizon + 1, self.latent_dim))
        
        # A is block-diagonal (theoretically sparse ~6% non-zeros), but CVXPY Parameter
        # doesn't properly support time-varying sparse matrices. Use dense instead.
        # OSQP will still detect sparsity in the KKT system during solve.
        self.A_dyn = cp.Parameter((self.latent_dim, self.latent_dim))
        self.B_dyn = cp.Parameter((self.latent_dim, self.control_dim))

        self.prev_z_seq = None
        self.prev_u_seq = None
        self.last_solver_time = None
        
        self.Q_np = np.array(Q_diag)
        self.R_np = np.array(R_diag)
        self.F_np = np.array(F_diag)
        
        cost = 0

        # Expand Q and F matrices if they don't match latent dimension
        if self.Q_np.shape[0] != self.latent_dim:
            log.warning(f"Q_diag dimension {self.Q_np.shape[0]} != latent_dim {self.latent_dim}. Padding with defaults.")
            Q_expanded = np.ones(self.latent_dim)
            Q_expanded[:self.Q_np.shape[0]] = self.Q_np
            Q_expanded[self.Q_np.shape[0]:] = np.mean(self.Q_np) * rest_ratio 
            self.Q_np = Q_expanded

        if self.F_np.shape[0] != self.latent_dim:
             log.warning(f"F_diag dimension {self.F_np.shape[0]} != latent_dim {self.latent_dim}. Padding with defaults.")
             F_expanded = np.ones(self.latent_dim)
             F_expanded[:self.F_np.shape[0]] = self.F_np
             F_expanded[self.F_np.shape[0]:] = np.mean(self.F_np) * rest_ratio
             self.F_np = F_expanded
             
        # --- Prepare AFZ Constraints ---
        A_constr = None
        if self.constraint_cfg is not None:
            try:
                if hasattr(self.model, 'mixing'):
                    with torch.no_grad():
                        W = self.model.mixing.get_matrix()
                        # Ensure W is float32 for inverse if it's float16
                        if W.dtype == torch.float16:
                            W = W.float()
                        W_inv = torch.linalg.inv(W)
                        W_inv_np = W_inv.cpu().numpy()
                else:
                    log.warning("No mixing layer found in model for AFZ constraints. Assuming Identity.")
                    W_inv_np = np.eye(self.latent_dim)
                
                boresight_vec = np.array(self.constraint_cfg['boresight_vec'])
                afz_list = self.constraint_cfg.get('afz_list', [])
                
                M_list = []
                for item in afz_list:
                    theta = np.deg2rad(item['theta'])
                    afz_vec = np.array(item['afz_vec'])
                    M_prime = self._get_afz_matrix(theta, afz_vec, boresight_vec)
                    M_list.append(M_prime)
                
                if M_list:
                    M_total = np.vstack(M_list) # Shape: (N_zones, 10)
                    # We only care about the first 10 dimensions of the unmixed state (Veronese part)
                    W_inv_trunc = W_inv_np[:10, :] # Shape: (10, latent_dim)
                    A_constr = M_total @ W_inv_trunc # Shape: (N_zones, latent_dim)
                    log.info(f"Initialized AFZ constraints with {len(M_list)} zones.")
            except Exception as e:
                 log.error(f"Failed to initialize AFZ constraints: {e}")
                 traceback.print_exc()

        state_err = self.z[:-1] - self.z_ref[:-1]
        cost += cp.sum(cp.multiply(self.Q_np, cp.square(state_err)))
        cost += cp.sum(cp.multiply(self.R_np, cp.square(self.u)))
        term_err = self.z[horizon] - self.z_ref[horizon]
        cost += cp.sum(cp.multiply(self.F_np, cp.square(term_err)))

        constraints = [self.z[0] == self.z_init]
        for k in range(horizon):
            constraints.append(self.z[k+1] == self.A_dyn @ self.z[k] + self.B_dyn @ self.u[k])
            # Apply element-wise normalized constraints
            constraints.append(self.u[k] >= self.u_min_norm)
            constraints.append(self.u[k] <= self.u_max_norm)
            
            # Apply AFZ constraints
            if A_constr is not None:
                constraints.append(A_constr @ self.z[k+1] <= 2)
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def _get_A_safe(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            h, _, _ = self.model.ctxt_encoder(ctxt)
            
        h = h.float()
        A_params = self.model._to_A(h)

        batch_size = A_params.size(0)
        n_blocks = self.model.latent_dim // 2

        log_r = A_params[:, :n_blocks]
        theta = A_params[:, n_blocks:]

        r = self.model.eigval_max * torch.sigmoid(log_r)
        c = torch.cos(theta)
        s = torch.sin(theta)

        A = torch.zeros(batch_size, self.model.latent_dim, self.model.latent_dim, device=A_params.device, dtype=A_params.dtype)
        
        indices = torch.arange(n_blocks, device=A_params.device)
        even_indices = 2 * indices
        odd_indices = 2 * indices + 1

        rc = r * c
        A[:, even_indices, even_indices] = rc
        A[:, odd_indices, odd_indices] = rc
        
        rs = r * s
        A[:, even_indices, odd_indices] = -rs
        A[:, odd_indices, even_indices] = rs
        
        return A

    def _get_B_safe(self, x_history: torch.Tensor, u_history: torch.Tensor):
        ctxt = torch.cat([x_history, u_history], dim=-1)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            h, _, _ = self.model.ctxt_encoder(ctxt)
            
        h = h.float()
        B_flat = self.model._to_B(h)

        batch_size = B_flat.size(0)
        B = B_flat.view(batch_size, self.model.latent_dim, self.model.control_dim)

        return B

    def getControl(self, obs: dict):
        x_ref  = obs["x_ref"]
        x_curr = obs["x_curr"]
        x_hist = obs["x_history"]
        u_hist = obs["u_history"]
        
        with torch.no_grad():
            x_hist_t = torch.FloatTensor(np.array(x_hist)).unsqueeze(0).to(self.device)
            u_hist_t = torch.FloatTensor(np.array(u_hist)).unsqueeze(0).to(self.device)
            x_curr_t = torch.FloatTensor(np.array(x_curr)).unsqueeze(0).to(self.device)
            x_ref_t = torch.FloatTensor(np.array(x_ref)).unsqueeze(0).to(self.device)

            A_val = self._get_A_safe(x_hist_t, u_hist_t).squeeze(0)
            B_val = self._get_B_safe(x_hist_t, u_hist_t).squeeze(0)
            z_curr = self.model.encoder(x_curr_t).squeeze(0).cpu().numpy()
            z_ref_point = self.model.encoder(x_ref_t).squeeze(0).cpu().numpy()
            
            z_ref = np.tile(z_ref_point, (self.horizon + 1, 1))
            z_ref = np.tile(z_ref_point, (self.horizon + 1, 1))
            
        self.z_init.value = z_curr
        self.z_ref.value = z_ref
        
        self.A_dyn.value = A_val.float().cpu().numpy()
        self.B_dyn.value = B_val.float().cpu().numpy()

        if self.prev_z_seq is not None:
            self.z.value = np.vstack([self.prev_z_seq[1:], self.prev_z_seq[-1:]])
            self.u.value = np.vstack([self.prev_u_seq[1:], self.prev_u_seq[-1:]])
        else:
            self.z.value = np.tile(z_curr, (self.horizon + 1, 1))
            self.u.value = np.zeros((self.horizon, self.control_dim))
        
        try:
            self.prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-3,
                eps_rel=1e-3,
                adaptive_rho=True,
                polish=False,
                verbose=False
            )
            self.last_solver_time = self.prob.solver_stats.solve_time
            
                
        except cp.SolverError:
            log.error("Solver error in Koopman MPC: cp.SolverError")
            return np.zeros(self.control_dim)
            
        if self.u.value is None:
            log.error("Solver error in Koopman MPC: u.value is None")
            return np.zeros(self.control_dim)
        
        self.prev_z_seq = self.z.value
        self.prev_u_seq = self.u.value

        return self.u.value[0]

    def _get_afz_matrix(self, theta: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        :param theta: Forbidden zone angle [radians]
        :param x: Forbidden zone vector in the inertial frame (3x1)
        :param y: Boresight vector in the body frame (3x1)
        :return: M_prime (1x10 matrix): Attitude forbidden constraint: M' @ veronese(q) <= 2
        """
        x = np.array(x, dtype=float).flatten()  
        y = np.array(y, dtype=float).flatten()  

        # Basic terms
        dot_xy = np.dot(x, y)
        cross_xy = np.cross(x, y)
        alpha = 2 - np.cos(theta)

        m_ss = dot_xy + alpha
        m_vs = -cross_xy 
        m_vv = np.outer(x, y) + np.outer(y, x) - (dot_xy - alpha) * np.eye(3)

        M = np.zeros((4, 4))
        M[0, 0] = m_ss
        M[0, 1:] = m_vs
        M[1:, 0] = m_vs
        M[1:, 1:] = m_vv

        # Map to Veronese M' (1x10)
        # Off-diagonals are multiplied by 2
        M_prime = np.array([
            M[0,0],                         # q0^2
            2*M[0,1], 2*M[0,2], 2*M[0,3],   # q0q1, q0q2, q0q3
            M[1,1],                         # q1^2
            2*M[1,2], 2*M[1,3],             # q1q2, q1q3
            M[2,2],                         # q2^2
            2*M[2,3],                       # q2q3
            M[3,3]                          # q3^2
        ])

        return M_prime.reshape(1, -1)