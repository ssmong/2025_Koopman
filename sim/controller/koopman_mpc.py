from typing import Dict
import logging
import time
from collections import deque
import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as spa

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
        device: str = "cuda"
        ):
        super().__init__()
        self.ModelTag = "KoopmanMPC"

        self.model, self.prev_cfg, self.stats_dict = load_model(checkpoint_dir, device)
        self.device = device

        self.processor = KoopmanDataProcessor(
            raw_state_dim=self.prev_cfg.data.raw_state_dim,
            state_dim=self.prev_cfg.data.state_dim,
            control_dim=self.prev_cfg.data.control_dim,
            angle_indices=self.prev_cfg.data.angle_indices,
            quat_indices=self.prev_cfg.data.quat_indices,
            normalization=self.prev_cfg.data.normalization,
            stats_dir=None,          
            name=self.prev_cfg.data.name,
            device=device,
        )

        self.processor.set_stats(self.stats_dict)

        self.hist_len = self.prev_cfg.data.hist_len
        self.control_dim = self.prev_cfg.data.control_dim
        
        self.hist_dt = self.prev_cfg.data.get('dt', 0.1)

        self.controller = KoopmanMPC(
            model=self.model,
            device=device,
            **mpc_params,
        )

        self.guidInMsg = messaging.AttGuidMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

        x_ref = [1, 0, 0, 0, 0, 0, 0]
        self.x_ref_norm = self.processor.normalize_state(x_ref, is_expanded=False)

        self.x_history = deque(maxlen=self.hist_len)
        self.u_history = deque(maxlen=self.hist_len)
        self.prev_u = np.zeros(self.control_dim)
        
        self.warmup_time = 0.0
        self.last_hist_update_time = -1.0
        
        self.profiler = ControllerProfiler(skip_first=True)

        log.info(f"Basilisk Koopman MPC initialized successfully from {checkpoint_dir}")

    def set_warmup_time(self, time: float):
        self.warmup_time = time

    def Reset(self, CurrentSimNanos: int):        
        if not self.guidInMsg.isLinked():
            self.bskLogger.bskLog(bskLogging.BSK_ERROR, "guidInMsg not linked")
            return
        
        payload = messaging.CmdTorqueBodyMsgPayload()
        self.cmdTorqueOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        self.x_history.clear()
        self.u_history.clear()
        self.prev_u = np.zeros(self.control_dim)
        self.last_hist_update_time = -1.0

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

    def UpdateState(self, CurrentSimNanos: int):
        current_time_sec = CurrentSimNanos * 1e-9
        
        if not self.guidInMsg.isLinked():
            return

        guid_msg = self.guidInMsg()
        sigma_BR = np.array(guid_msg.sigma_BR)
        omega_BR_B = np.array(guid_msg.omega_BR_B)
        ep_BR = rbk.MRP2EP(sigma_BR)

        x_curr = np.concatenate([ep_BR, omega_BR_B])
        x_curr_norm = self.processor.normalize_state(x_curr, is_expanded=False)
        
        if self.last_hist_update_time < 0 or (current_time_sec - self.last_hist_update_time) >= self.hist_dt - 1e-6:
            u_prev_norm = self.processor.normalize_control(self.prev_u)
            self.x_history.append(x_curr_norm)
            self.u_history.append(u_prev_norm)
            self.last_hist_update_time = current_time_sec

        if current_time_sec < self.warmup_time:
            u_opt = np.zeros(self.control_dim)
            self.prev_u = u_opt
            
            out_payload = messaging.CmdTorqueBodyMsgPayload()
            out_payload.torqueRequestBody = u_opt.tolist()
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
                 horizon: int, 
                 u_min: float, 
                 u_max: float, 
                 Q_diag: list, 
                 R_diag: list,
                 F_diag: list,
                 device: str = "cuda"):
        
        self.model = model
        self.horizon = horizon
        self.device = device
        self.latent_dim = model.latent_dim
        self.control_dim = model.control_dim
        
        self.z = cp.Variable((horizon + 1, self.latent_dim))
        self.u = cp.Variable((horizon, self.control_dim))
        
        self.z_init = cp.Parameter(self.latent_dim)
        self.z_ref = cp.Parameter((horizon + 1, self.latent_dim))
        
        self.A_dyn = cp.Parameter((self.latent_dim, self.latent_dim)) 
        self.B_dyn = cp.Parameter((self.latent_dim, self.control_dim))

        self.prev_z_seq = None
        self.prev_u_seq = None
        self.last_solver_time = None
        
        Q_np = np.array(Q_diag)
        R_np = np.array(R_diag)
        F_np = np.array(F_diag)
        
        cost = 0
        state_err = self.z[:-1] - self.z_ref[:-1]
        cost += cp.sum(cp.multiply(Q_np, cp.square(state_err)))
        cost += cp.sum(cp.multiply(R_np, cp.square(self.u)))
        term_err = self.z[horizon] - self.z_ref[horizon]
        cost += cp.sum(cp.multiply(F_np, cp.square(term_err)))

        constraints = [self.z[0] == self.z_init]
        for k in range(horizon):
            constraints.append(self.z[k+1] == self.A_dyn @ self.z[k] + self.B_dyn @ self.u[k])
            constraints.append(self.u[k] >= u_min)
            constraints.append(self.u[k] <= u_max)
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def getControl(self, obs: dict):
        x_ref  = obs["x_ref"]
        x_curr = obs["x_curr"]
        x_hist = obs["x_history"]
        u_hist = obs["u_history"]
        
        with torch.no_grad():
            x_hist_t = torch.FloatTensor(np.array(x_hist)).unsqueeze(0).to(self.device)
            u_hist_t = torch.FloatTensor(np.array(u_hist)).unsqueeze(0).to(self.device)
            x_curr_t = torch.FloatTensor(np.array(x_curr)).unsqueeze(0).to(self.device)
            
            A_val = self.model.get_A(x_hist_t, u_hist_t).squeeze(0)
            B_val = self.model.get_B(x_hist_t, u_hist_t).squeeze(0)
            
            z_curr = self.model.encoder(x_curr_t).squeeze(0).cpu().numpy()
    
            x_ref_t = torch.FloatTensor(np.array(x_ref)).unsqueeze(0).to(self.device)
            z_ref_point = self.model.encoder(x_ref_t).squeeze(0).cpu().numpy()
            z_ref = np.tile(z_ref_point, (self.horizon + 1, 1))
            
        self.z_init.value = z_curr
        self.z_ref.value = z_ref
        
        A_val_np = A_val.cpu().numpy()
        self.A_dyn.value = spa.csc_matrix(A_val_np)
        self.B_dyn.value = B_val.cpu().numpy()

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
                polish=False
            )
            # Try to get solver time
            if hasattr(self.prob, 'solver_stats') and self.prob.solver_stats is not None:
                self.last_solver_time = self.prob.solver_stats.solve_time
            else:
                self.last_solver_time = None
                
        except cp.SolverError:
            log.error("Solver error in Koopman MPC: cp.SolverError")
            return np.zeros(self.control_dim)
            
        if self.u.value is None:
            log.error("Solver error in Koopman MPC: u.value is None")
            return np.zeros(self.control_dim)
        
        self.prev_z_seq = self.z.value
        self.prev_u_seq = self.u.value

        return self.u.value[0]
