import logging
import time
import numpy as np
import casadi as ca
import traceback

log = logging.getLogger(__name__)

class NMPC:
    def __init__(self, 
                 inertia: list,
                 dt: float,
                 horizon: int, 
                 u_min: float, 
                 u_max: float, 
                 Q_diag: list, 
                 R_diag: list,
                 F_diag: list,
                 constraint_cfg: dict = None,
                 solver_max_iter: int = 100):
        
        self.dt = dt
        self.horizon = horizon
        self.constraint_cfg = constraint_cfg
        self.solver_max_iter = solver_max_iter
        
        # Inertia Matrix
        if len(inertia) == 9:
            self.J = np.array(inertia).reshape(3, 3)
        elif len(inertia) == 3:
            self.J = np.diag(inertia)
        else:
            raise ValueError("Inertia must be a list of length 3 (diagonal) or 9 (flattened matrix)")
        
        self.J_inv = np.linalg.inv(self.J)
        
        # Dimensions
        self.state_dim = 7  # q(4) + w(3)
        self.control_dim = 3
        
        # Control limits
        self.u_min = u_min
        self.u_max = u_max
        
        # Weights
        self.Q_np = np.array(Q_diag)
        self.R_np = np.array(R_diag)
        self.F_np = np.array(F_diag)
        
        # Ensure dimensions match
        if self.Q_np.shape[0] != self.state_dim:
            log.warning(f"Q_diag dimension {self.Q_np.shape[0]} does not match state dim {self.state_dim}. Padding/Truncating.")
            Q_new = np.zeros(self.state_dim)
            min_len = min(self.Q_np.shape[0], self.state_dim)
            Q_new[:min_len] = self.Q_np[:min_len]
            self.Q_np = Q_new

        if self.F_np.shape[0] != self.state_dim:
            log.warning(f"F_diag dimension {self.F_np.shape[0]} does not match state dim {self.state_dim}. Padding/Truncating.")
            F_new = np.zeros(self.state_dim)
            min_len = min(self.F_np.shape[0], self.state_dim)
            F_new[:min_len] = self.F_np[:min_len]
            self.F_np = F_new
            
        # AFZ Constraints Setup
        self.M_list = []
        if self.constraint_cfg is not None:
            try:
                boresight_vec = np.array(self.constraint_cfg['boresight_vec'])
                afz_list = self.constraint_cfg.get('afz_list', [])
                
                for item in afz_list:
                    theta = np.deg2rad(item['theta'])
                    afz_vec = np.array(item['afz_vec'])
                    M = self._get_afz_matrix_quad(theta, afz_vec, boresight_vec)
                    self.M_list.append(M)
                
                log.info(f"Initialized AFZ constraints with {len(self.M_list)} zones.")
            
            except Exception as e:
                log.critical(f"FATAL: Failed to initialize AFZ constraints: {e}")
                traceback.print_exc()
                raise e

        # --- CasADi Integrator Setup ---
        # Define symbolic variables for dynamics
        x_sym = ca.MX.sym('x', self.state_dim)
        u_sym = ca.MX.sym('u', self.control_dim)
        
        # Continuous dynamics function
        x_dot = self._dynamics(x_sym, u_sym)
        
        # Create integrator
        dae = {'x': x_sym, 'p': u_sym, 'ode': x_dot}
        opts = {'tf': self.dt} # Integrate over one time step
        # 'rk' is explicit Runge-Kutta 4
        self.integrator = ca.integrator('F', 'rk', dae, opts)

        # --- CasADi Optimization Setup ---
        self.opti = ca.Opti()

        # Variables
        self.X = self.opti.variable(self.state_dim, self.horizon + 1) # State
        self.U = self.opti.variable(self.control_dim, self.horizon)   # Control
        self.slack = self.opti.variable(self.horizon)                 # Slack

        # Parameters
        self.x_init = self.opti.parameter(self.state_dim)
        self.x_ref = self.opti.parameter(self.state_dim, self.horizon + 1)
        
        # Weight Matrices (Diagonal)
        Q_mat = ca.diag(self.Q_np)
        R_mat = ca.diag(self.R_np)
        F_mat = ca.diag(self.F_np)
        
        # Cost Function
        cost = 0
        
        # Running Cost
        for k in range(self.horizon):
            x_err = self.X[:, k] - self.x_ref[:, k]
            cost += ca.mtimes([x_err.T, Q_mat, x_err])
            
            u_val = self.U[:, k]
            cost += ca.mtimes([u_val.T, R_mat, u_val])
            
            # Slack Penalty
            cost += 1e5 * self.slack[k]**2
            
        # Terminal Cost
        x_err_term = self.X[:, self.horizon] - self.x_ref[:, self.horizon]
        cost += ca.mtimes([x_err_term.T, F_mat, x_err_term])
        
        self.opti.minimize(cost)
        
        # Constraints
        # 1. Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        self.opti.subject_to(self.slack >= 0)
        
        # 2. Dynamics & Input Constraints & AFZ
        for k in range(self.horizon):
            # Dynamics using Integrator
            # integrator returns a dict, 'xf' is the final state
            x_next = self.integrator(x0=self.X[:, k], p=self.U[:, k])['xf']
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            # Input Constraints
            self.opti.subject_to(self.U[:, k] >= self.u_min)
            self.opti.subject_to(self.U[:, k] <= self.u_max)
            
            # AFZ Constraints: q^T M q <= slack
            q_k = self.X[:4, k+1] 
            for M in self.M_list:
                quad_val = ca.mtimes([q_k.T, ca.DM(M), q_k])
                self.opti.subject_to(quad_val <= self.slack[k]) 

        # Solver Configuration (IPOPT for Nonlinear)
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': self.solver_max_iter,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        self.opti.solver('ipopt', opts)
        
        # Cache for warm start
        self.prev_u_seq = None
        self.prev_x_seq = None

    def _dynamics(self, x, u):
        """
        Continuous time dynamics: x_dot = f(x, u)
        x = [q0, q1, q2, q3, w1, w2, w3]
        """
        q = x[:4]
        w = x[4:]
        
        # Quaternion kinematics: q_dot = 0.5 * Xi(q) * w
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        # Matrix multiplication form
        q_dot = 0.5 * ca.vertcat(
            -q1*w[0] - q2*w[1] - q3*w[2],
             q0*w[0] - q3*w[1] + q2*w[2],
             q3*w[0] + q0*w[1] - q1*w[2],
            -q2*w[0] + q1*w[1] + q0*w[2]
        )
        
        # Angular velocity dynamics: J * w_dot = -w x (J * w) + u
        Jw = ca.mtimes(ca.DM(self.J), w)
        cross_term = ca.cross(w, Jw)
        w_dot = ca.mtimes(ca.DM(self.J_inv), -cross_term + u)
        
        return ca.vertcat(q_dot, w_dot)

    def _get_afz_matrix_quad(self, theta: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns Matrix M such that q^T M q <= 0 defines the allowed region (outside the cone).
        Constraint: angle(v_body, v_inertial) >= theta
        """
        x = np.array(x, dtype=float).flatten() # Inertial vector
        y = np.array(y, dtype=float).flatten() # Body vector
        
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)

        c = np.cos(theta)
        dot_xy = np.dot(x, y)
        
        # m00 = x^T y - c
        m00 = dot_xy - c
        
        # m0v = (x x y)^T
        m0v = np.cross(x, y)
        
        # mvv = x y^T + y x^T - (x^T y + c) I
        mvv = np.outer(x, y) + np.outer(y, x) - (dot_xy + c) * np.eye(3)
        
        M = np.zeros((4, 4))
        M[0, 0] = m00
        M[0, 1:] = m0v
        M[1:, 0] = m0v
        M[1:, 1:] = mvv
        
        return M

    def getControl(self, obs: dict):
        """
        Solve NMPC and return control input.
        obs: {
            "x_curr": [q, w],
            "x_ref": [q_ref, w_ref] (horizon+1, 7)
        }
        """
        x_curr = np.array(obs["x_curr"])
        x_ref = np.array(obs["x_ref"])
        
        self.opti.set_value(self.x_init, x_curr)
        
        if x_ref.shape[0] != self.state_dim:
            x_ref = x_ref.T
            
        self.opti.set_value(self.x_ref, x_ref)
        
        # Warm Start
        if self.prev_u_seq is not None:
            self.opti.set_initial(self.U, np.vstack([self.prev_u_seq[1:], self.prev_u_seq[-1:]]).T)
            self.opti.set_initial(self.X, np.vstack([self.prev_x_seq[1:], self.prev_x_seq[-1:]]).T)
        else:
            self.opti.set_initial(self.U, np.zeros((self.control_dim, self.horizon)))
            self.opti.set_initial(self.X, np.tile(x_curr, (self.horizon + 1, 1)).T)
            
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            
            self.prev_u_seq = u_opt.T
            self.prev_x_seq = x_opt.T
            
            return u_opt[:, 0]
            
        except Exception as e:
            log.error(f"NMPC Solver failed: {e}")
            if self.prev_u_seq is not None:
                 u_next = self.prev_u_seq[1]
                 self.prev_u_seq = np.vstack([self.prev_u_seq[1:], self.prev_u_seq[-1:]])
                 return u_next
            return np.zeros(self.control_dim)

    def reset_cache(self):
        self.prev_u_seq = None
        self.prev_x_seq = None
