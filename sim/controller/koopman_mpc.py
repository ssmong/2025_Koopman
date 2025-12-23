import torch
import cvxpy as cp
import numpy as np
import scipy.sparse as spa

class KoopmanMPC:
    def __init__(self, latent_dim, control_dim, horizon, u_min, u_max, Q_diag, R_diag):
        """
        Q_diag, R_diag: numpy array (vector), 대각 성분만 받음
        """
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.horizon = horizon
        
        # 1. Variables (Multiple Shooting)
        self.z = cp.Variable((horizon + 1, latent_dim))
        self.u = cp.Variable((horizon, control_dim))
        
        # 2. Parameters (Sparse A 지원을 위해 구조 잡기)
        self.z_init = cp.Parameter(latent_dim)
        self.z_ref = cp.Parameter((horizon + 1, latent_dim))
        
        # A와 B는 매번 바뀌는 LPV지만, 구조적 희소성을 알리기 위해 
        # (rows, cols) 속성을 가질 수 있음. 
        # 다만 CVXPY Parameter는 기본적으로 Dense 취급되므로, 
        # 아래 solve 단계에서 sparse matrix를 value로 넣는 것이 핵심입니다.
        self.A_dyn = cp.Parameter((latent_dim, latent_dim)) 
        self.B_dyn = cp.Parameter((latent_dim, control_dim))
        
        # 3. Objective (Vectorized)
        # Quad form 대신 sum_squares 사용 (OSQP 변환 시 더 효율적일 수 있음)
        cost = 0
        
        # Q, R이 대각 행렬이므로 벡터 곱으로 처리 -> 속도 향상
        for k in range(horizon):
            cost += cp.sum(cp.multiply(Q_diag, (self.z[k] - self.z_ref[k])**2))
            cost += cp.sum(cp.multiply(R_diag, self.u[k]**2))
            
        # Terminal Cost
        cost += cp.sum(cp.multiply(Q_diag * 10, (self.z[horizon] - self.z_ref[horizon])**2))
        
        # 4. Constraints
        constraints = [self.z[0] == self.z_init]
        for k in range(horizon):
            # Dynamics: z[k+1] == A @ z[k] + B @ u[k]
            # 여기에 Linear Constraints 추가 가능 (예: State bound)
            constraints.append(self.z[k+1] == self.A_dyn @ self.z[k] + self.B_dyn @ self.u[k])
            
            # Input Constraints
            constraints.append(self.u[k] >= u_min)
            constraints.append(self.u[k] <= u_max)
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, z0_val, z_ref_val, A_val, B_val):
        """
        A_val: scipy.sparse matrix 권장 (Block Diagonal 구조 활용)
        """
        self.z_init.value = z0_val
        self.z_ref.value = z_ref_val
        
        # [핵심] A 행렬을 Sparse 포맷으로 주입
        # CVXPY는 Parameter 값으로 Sparse Matrix를 받으면 내부 연산을 최적화함
        if not spa.issparse(A_val):
            A_val = spa.csc_matrix(A_val)
            
        self.A_dyn.value = A_val
        self.B_dyn.value = B_val # B는 보통 Dense하지만 Sparse라면 변환해서 넣음
        
        # 5. Solver Options (Realistic Tuning)
        try:
            self.prob.solve(
                solver=cp.OSQP,
                warm_start=True,        # [필수] 이전 solution 재사용
                eps_abs=1e-3,           # 제어용으로는 충분한 정밀도
                eps_rel=1e-3, 
                check_termination=25,   # 체크 주기 (너무 짧으면 오버헤드)
                polish=False,           # [중요] 실시간성 위해 끔 (정밀도 보정 기능)
                adaptive_rho=True
            )
        except cp.SolverError:
            return None
            
        return self.u.value[0]