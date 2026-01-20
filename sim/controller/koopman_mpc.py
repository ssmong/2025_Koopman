import logging
import torch
import numpy as np
import cvxpy as cp
import traceback

log = logging.getLogger(__name__)

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
                 Q_rest: float = 0.01,
                 F_rest: float = 0.01,
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
        
        # A_dyn is dense because CVXPY Parameter doesn't fully support time-varying sparse matrices well
        self.A_dyn = cp.Parameter((self.latent_dim, self.latent_dim))
        self.B_dyn = cp.Parameter((self.latent_dim, self.control_dim))

        self.prev_z_seq = None
        self.prev_u_seq = None
        self.last_solver_time = None
        
        self.past_key_values = None
        
        self.Q_np = np.array(Q_diag)
        self.R_np = np.array(R_diag)
        self.F_np = np.array(F_diag)
        
        cost = 0

        # Expand Q and F matrices if they don't match latent dimension
        if self.Q_np.shape[0] != self.latent_dim:
            log.warning(f"Q_diag dimension {self.Q_np.shape[0]} != latent_dim {self.latent_dim}. Padding with defaults.")
            Q_expanded = np.ones(self.latent_dim)
            Q_expanded[:self.Q_np.shape[0]] = self.Q_np
            Q_expanded[self.Q_np.shape[0]:] = Q_rest
            self.Q_np = Q_expanded

        if self.F_np.shape[0] != self.latent_dim:
             log.warning(f"F_diag dimension {self.F_np.shape[0]} != latent_dim {self.latent_dim}. Padding with defaults.")
             F_expanded = np.ones(self.latent_dim)
             F_expanded[:self.F_np.shape[0]] = self.F_np
             F_expanded[self.F_np.shape[0]:] = F_rest
             self.F_np = F_expanded
             
        # --- Prepare AFZ Constraints ---
        A_constr = None
        if self.constraint_cfg is not None:
            try:
                boresight_vec = np.array(self.constraint_cfg['boresight_vec'])
                afz_list = self.constraint_cfg.get('afz_list', [])
                
                M_list = []
                for item in afz_list:
                    theta = np.deg2rad(item['theta'])
                    afz_vec = np.array(item['afz_vec'])
                    M_prime = self._get_afz_matrix(theta, afz_vec, boresight_vec)
                    M_list.append(M_prime)
                
                if M_list:
                    M_total = np.vstack(M_list)  # Shape: (N_zones, 10)
                    
                    # Veronese part is at indices 7 to 17 (0-based) -> 7:17
                    # z = [q(4) | w(3) | veronese(10) | nn_feature(...)],
                    A_constr = np.zeros((M_total.shape[0], self.latent_dim))
                    A_constr[:, 7:17] = M_total
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

    def reset_cache(self):
        self.past_key_values = None

    def getControl(self, obs: dict):
        x_ref  = obs["x_ref"]
        x_curr = obs["x_curr"]
        x_hist = obs["x_history"]
        u_hist = obs["u_history"]
        
        with torch.no_grad():
            x_ref_t = torch.FloatTensor(np.array(x_ref)).unsqueeze(0).to(self.device)
            z_curr = self.model.encoder(torch.FloatTensor(np.array(x_curr)).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
            z_ref_point = self.model.encoder(x_ref_t).squeeze(0).cpu().numpy()
            
            z_ref = np.tile(z_ref_point, (self.horizon + 1, 1))

            # --- O(1) Update with Cache ---
            if self.past_key_values is None:
                # First step (or after reset): Process full history to warm up cache
                x_hist_t = torch.FloatTensor(np.array(x_hist)).unsqueeze(0).to(self.device)
                u_hist_t = torch.FloatTensor(np.array(u_hist)).unsqueeze(0).to(self.device)
                
                ctxt = torch.cat([x_hist_t, u_hist_t], dim=-1) # [1, T, Dim]
                h, self.past_key_values, _ = self.model.ctxt_encoder(ctxt, use_cache=True)
                
                # h is [1, Dim] (last step hidden state)
                A_val = self.model.get_A_dt(h).squeeze(0)
                B_flat = self.model._to_B(h)
                B_val = B_flat.view(h.size(0), self.latent_dim, self.control_dim).squeeze(0)
                
            else:
                x_new = torch.FloatTensor(np.array(x_hist[-1])).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, Dim]
                u_new = torch.FloatTensor(np.array(u_hist[-1])).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, Dim]
                
                A_dt, B_dt, self.past_key_values = self.model.get_dynamics_step(x_new, u_new, past_key_values=self.past_key_values)
                
                A_val = A_dt.squeeze(0)
                B_val = B_dt.squeeze(0)

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