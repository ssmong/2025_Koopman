import logging
import time
import numpy as np
import torch
import casadi as ca
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
                 solver_max_iter: int = 100,
                 device: str = "cuda"):
        
        self.model = model
        self.processor = processor
        self.horizon = horizon
        self.device = device
        self.latent_dim = model.latent_dim
        self.control_dim = model.control_dim
        self.constraint_cfg = constraint_cfg
        self.solver_max_iter = solver_max_iter
        
        u_min_phys = np.full(self.control_dim, u_min)
        u_max_phys = np.full(self.control_dim, u_max)
        
        self.u_min_norm = self.processor.normalize_control(torch.from_numpy(u_min_phys).float().to(self.processor.device)).cpu().numpy()
        self.u_max_norm = self.processor.normalize_control(torch.from_numpy(u_max_phys).float().to(self.processor.device)).cpu().numpy()
            
        self.prev_z_seq = None
        self.prev_u_seq = None
        self.last_solver_time = None
        self.past_key_values = None
        
        self.Q_np = np.array(Q_diag)
        self.R_np = np.array(R_diag)
        self.F_np = np.array(F_diag)
        
        if self.Q_np.shape[0] != self.latent_dim:
            Q_expanded = np.ones(self.latent_dim)
            Q_expanded[:self.Q_np.shape[0]] = self.Q_np
            Q_expanded[self.Q_np.shape[0]:] = Q_rest
            self.Q_np = Q_expanded

        if self.F_np.shape[0] != self.latent_dim:
            F_expanded = np.ones(self.latent_dim)
            F_expanded[:self.F_np.shape[0]] = self.F_np
            F_expanded[self.F_np.shape[0]:] = F_rest
            self.F_np = F_expanded
              
        self.A_constr = None
        self.A_constr_dm = None
        
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
                    M_total = np.vstack(M_list)
                    self.A_constr = np.zeros((M_total.shape[0], self.latent_dim))
                    self.A_constr[:, 7:17] = M_total
                    self.A_constr_dm = ca.DM(self.A_constr)
                    log.info(f"Initialized AFZ constraints with {len(M_list)} zones.")
            
            except Exception as e:
                log.critical(f"FATAL: Failed to initialize AFZ constraints: {e}")
                traceback.print_exc()
                raise e

        self.opti = ca.Opti()

        self.z = self.opti.variable(self.latent_dim, horizon + 1)
        self.u = self.opti.variable(self.control_dim, horizon)
        self.slack = self.opti.variable(horizon)

        self.z_init = self.opti.parameter(self.latent_dim, 1)
        self.z_ref = self.opti.parameter(self.latent_dim, horizon + 1)
        self.A_dyn = self.opti.parameter(self.latent_dim, self.latent_dim)
        self.B_dyn = self.opti.parameter(self.latent_dim, self.control_dim)

        self.Q_sqrt = ca.diag(ca.sqrt(ca.DM(self.Q_np)))
        self.R_sqrt = ca.diag(ca.sqrt(ca.DM(self.R_np)))
        self.F_sqrt = ca.diag(ca.sqrt(ca.DM(self.F_np)))

        self.u_min_norm_dm = ca.DM(self.u_min_norm).reshape((self.control_dim, 1))
        self.u_max_norm_dm = ca.DM(self.u_max_norm).reshape((self.control_dim, 1))

        state_err = self.z[:, :horizon] - self.z_ref[:, :horizon]
        cost = ca.sumsqr(ca.mtimes(self.Q_sqrt, state_err))
        cost += ca.sumsqr(ca.mtimes(self.R_sqrt, self.u))
        
        term_err = self.z[:, horizon] - self.z_ref[:, horizon]
        cost += ca.sumsqr(ca.mtimes(self.F_sqrt, term_err))
        
        SLACK_PENALTY = 1e5
        cost += SLACK_PENALTY * ca.sumsqr(self.slack)

        self.opti.minimize(cost)
     
        self.opti.subject_to(self.z[:, 0] == self.z_init)
        self.opti.subject_to(self.slack >= 0)
        
        for k in range(horizon):
            self.opti.subject_to(
                self.z[:, k + 1] == ca.mtimes(self.A_dyn, self.z[:, k]) + ca.mtimes(self.B_dyn, self.u[:, k])
            )
            self.opti.subject_to(self.u[:, k] >= self.u_min_norm_dm)
            self.opti.subject_to(self.u[:, k] <= self.u_max_norm_dm)

            if self.A_constr_dm is not None:
                self.opti.subject_to(ca.mtimes(self.A_constr_dm, self.z[:, k + 1]) <= 2.0 + self.slack[k])

        self.solver_name = "osqp"
        self.solver_opts = {
            "osqp": {
                "verbose": False,
                "eps_abs": 1e-4,
                "eps_rel": 1e-4,
                "max_iter": self.solver_max_iter,
                "warm_start": True
            },
            "print_time": 0,
            "error_on_fail": False
        }

        try:
            self.opti.solver(self.solver_name, self.solver_opts)
        except Exception as e:
            log.error(f"Failed to initialize CasADi solver '{self.solver_name}': {e}")
            raise

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

            if self.past_key_values is None:
                x_hist_t = torch.FloatTensor(np.array(x_hist)).unsqueeze(0).to(self.device)
                u_hist_t = torch.FloatTensor(np.array(u_hist)).unsqueeze(0).to(self.device)
                
                ctxt = torch.cat([x_hist_t, u_hist_t], dim=-1)
                h, self.past_key_values, _ = self.model.ctxt_encoder(ctxt, use_cache=True)
                
                A_val = self.model.get_A_dt(h).squeeze(0)
                B_flat = self.model._to_B(h)
                B_val = B_flat.view(h.size(0), self.latent_dim, self.control_dim).squeeze(0)
                
            else:
                x_new = torch.FloatTensor(np.array(x_hist[-1])).unsqueeze(0).unsqueeze(0).to(self.device)
                u_new = torch.FloatTensor(np.array(u_hist[-1])).unsqueeze(0).unsqueeze(0).to(self.device)
                
                A_dt, B_dt, self.past_key_values = self.model.get_dynamics_step(x_new, u_new, past_key_values=self.past_key_values)
                
                A_val = A_dt.squeeze(0)
                B_val = B_dt.squeeze(0)

        z_curr = np.array(z_curr).reshape(self.latent_dim)
        self.opti.set_value(self.z_init, z_curr.reshape(self.latent_dim, 1))
        self.opti.set_value(self.z_ref, z_ref.T)

        self.opti.set_value(self.A_dyn, A_val.float().cpu().numpy())
        self.opti.set_value(self.B_dyn, B_val.float().cpu().numpy())

        def _use_previous_plan():
            if self.prev_u_seq is None:
                return None
            u_next = self.prev_u_seq[1]
            self.prev_u_seq = np.vstack([self.prev_u_seq[1:], self.prev_u_seq[-1:]])
            if self.prev_z_seq is not None:
                self.prev_z_seq = np.vstack([self.prev_z_seq[1:], self.prev_z_seq[-1:]])
            return u_next

        if self.prev_z_seq is not None:
            z_guess = np.vstack([self.prev_z_seq[1:], self.prev_z_seq[-1:]])
            u_guess = np.vstack([self.prev_u_seq[1:], self.prev_u_seq[-1:]])
        else:
            z_guess = np.tile(z_curr, (self.horizon + 1, 1))
            u_guess = np.zeros((self.horizon, self.control_dim))

        self.opti.set_initial(self.z, z_guess.T)
        self.opti.set_initial(self.u, u_guess.T)
        self.opti.set_initial(self.slack, 0.0)

        self.last_solver_time = None
        try:
            t_start = time.perf_counter()
            sol = self.opti.solve()
            self.last_solver_time = time.perf_counter() - t_start

            stats = sol.stats()
            if stats.get("success") is False:
                log.warning(f"Solver status: {stats.get('return_status', 'unknown')}. Using fallback.")
                u_next = _use_previous_plan()
                if u_next is not None:
                    return u_next
                return np.zeros(self.control_dim)

        except Exception as e:
            log.error(f"Solver error in Koopman MPC: {e}")
            u_next = _use_previous_plan()
            if u_next is not None:
                return u_next
            return np.zeros(self.control_dim)

        u_opt = sol.value(self.u)
        z_opt = sol.value(self.z)

        if u_opt is None or z_opt is None:
            log.error("Solver error: solution is None")
            return np.zeros(self.control_dim)

        self.prev_u_seq = np.array(u_opt).T
        self.prev_z_seq = np.array(z_opt).T

        return self.prev_u_seq[0]

    def _get_afz_matrix(self, theta: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float).flatten()  
        y = np.array(y, dtype=float).flatten()  

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

        M_prime = np.array([
            M[0,0],                                     
            2*M[0,1], 2*M[0,2], 2*M[0,3],               
            M[1,1],                                     
            2*M[1,2], 2*M[1,3],                         
            M[2,2],                                     
            2*M[2,3],                                   
            M[3,3]                                      
        ])

        return M_prime.reshape(1, -1)