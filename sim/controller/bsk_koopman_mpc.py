from typing import Dict
import logging
import time
import numpy as np
import torch
from collections import deque

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

from src.data.dataset import KoopmanDataProcessor
from sim.controller.koopman_mpc import KoopmanMPC
from sim.utils.load import load_model
from sim.utils.profiler import ControllerProfiler


class BskKoopmanMPC(sysModel.SysModel):
    def __init__(
        self,
        checkpoint_dir: str,
        mpc_params: Dict,
        ctrl_dt: float = 0.1,
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

        pred_dt = self.prev_cfg.data.pred_dt
        if pred_dt != ctrl_dt:
            self.bskLogger.bskLog(bskLogging.BSK_WARNING, f"pred_dt ({pred_dt}) != ctrl_dt ({ctrl_dt}). This will cause unexpected control behavior.")
        self.hist_len = self.prev_cfg.data.hist_len
        self.hist_dt = self.prev_cfg.data.hist_dt
        self.control_dim = self.prev_cfg.data.control_dim

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
        self.last_hist_update_time = -self.hist_dt
        
        self.profiler = ControllerProfiler(skip_first=True)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION, f"Basilisk Koopman MPC initialized successfully from {checkpoint_dir}")

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
        self.last_hist_update_time = -self.hist_dt
        self.last_ctrl_update_time = -self.ctrl_dt
        
        # Reset MPC Controller Cache
        self.controller.reset_cache()

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