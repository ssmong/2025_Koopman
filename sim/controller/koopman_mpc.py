from ast import Dict
import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as spa

from Basilisk.architecture import sysModel, messaging
from Basilisk.architecture import bskLogging

from sim.utils.load import load_model

class BskKoopmanMPC(sysModel.SysModel):
    def __init__(
        self,
        checkpoint_dir: str,
        mpc_params: Dict,
        device: str = "cuda"
        ):
        super().__init__()

        self.model, self.hist_len, self.stats_tensor = load_model(checkpoint_dir, device)
        self.device = device

        

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

    def get_control(self, obs: dict):
        x_ref  = obs["x_ref"]
        x_curr = obs["x_curr"]
        x_hist = obs["x_history"]
        u_hist = obs["u_history"]
        
        with torch.no_grad():
            x_hist_t = torch.FloatTensor(x_hist).unsqueeze(0).to(self.device)
            u_hist_t = torch.FloatTensor(u_hist).unsqueeze(0).to(self.device)
            x_curr_t = torch.FloatTensor(x_curr).unsqueeze(0).to(self.device)
            
            A_val = self.model.get_A(x_hist_t, u_hist_t).squeeze(0).cpu().numpy()
            B_val = self.model.get_B(x_hist_t, u_hist_t).squeeze(0).cpu().numpy()
            
            z_curr = self.model.encoder(x_curr_t).squeeze(0).cpu().numpy()
    
            x_ref_t = torch.FloatTensor(x_ref).unsqueeze(0).to(self.device)
            z_ref_point = self.model.encoder(x_ref_t).squeeze(0).cpu().numpy()
            z_ref = np.tile(z_ref_point, (self.horizon + 1, 1))
            
        self.z_init.value = z_curr
        self.z_ref.value = z_ref
        
        if not spa.issparse(A_val):
            A_val = spa.csc_matrix(A_val)
            
        self.A_dyn.value = A_val
        self.B_dyn.value = B_val 
        
        try:
            self.prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-3,
                eps_rel=1e-3,
                adaptive_rho=True,
                polish=False
            )
        except cp.SolverError:
            return np.zeros(self.control_dim)
            
        if self.u.value is None:
            return np.zeros(self.control_dim)
            
        return self.u.value[0]