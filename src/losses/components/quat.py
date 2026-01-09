import torch
from typing import List, Optional
from src.losses.components.base import BaseLoss

class QuatLoss(BaseLoss):
    def __init__(self, 
                 quat_indices: Optional[List[int]] = None,
                 quat_loss_type: str = 'geodesic',
                 quat_norm_weight: float = 0.0,
                 quat_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.quat_indices = quat_indices
        self.quat_loss_type = quat_loss_type
        self.quat_norm_weight = quat_norm_weight
        self.quat_weight = quat_weight

    def _compute_quat_loss(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        q_pred_norm = q_pred.norm(dim=-1, keepdim=True) + 1e-5
        q_pred_unit = q_pred / q_pred_norm
        
        q_target_norm = q_target.norm(dim=-1, keepdim=True) + 1e-5
        q_target_unit = q_target / q_target_norm

        if self.quat_loss_type == 'geodesic':
            # 1 - <q_a, q_b>^2 (Geodesic distance approximation)
            # Handles double cover automatically because of the square
            dot = (q_pred_unit * q_target_unit).sum(dim=-1)
            quat_loss = 1.0 - dot ** 2
            
        elif self.quat_loss_type == 'chordal':
            # Euclidean distance handling double cover: min(|q_a-q_b|^2, |q_a+q_b|^2)
            # Equivalent to 2 * (1 - |<q_a, q_b>|) for unit quaternions
            diff_plus = (q_pred_unit - q_target_unit).norm(dim=-1)
            diff_minus = (q_pred_unit + q_target_unit).norm(dim=-1)
            quat_loss = torch.min(diff_plus, diff_minus).square() # Use squared distance
        elif self.quat_loss_type == 'angle':
            safe_eps = 1e-6 
            cos_half_theta = (q_pred_unit * q_target_unit).sum(dim=-1)
            sin_half_theta = torch.sqrt(1.0 - cos_half_theta ** 2 + safe_eps)
            angle_diff = 2.0 * torch.atan2(sin_half_theta, cos_half_theta)
            
            charb_eps = 1e-3 
            quat_loss = torch.sqrt(angle_diff**2 + charb_eps**2)
        else:
            raise ValueError(f"Unknown quat_loss_type: {self.quat_loss_type}. Supported: ['geodesic', 'chordal']")

        # 2. Norm Regularization Loss
        if self.quat_norm_weight > 0:
            norm_loss = (q_pred_norm.squeeze(-1) - 1.0) ** 2
            return quat_loss + self.quat_norm_weight * norm_loss
            
        return quat_loss

    def forward(self, results: dict) -> torch.Tensor:
        pred, target = self.get_inputs(results)  # [B, T, D]

        # 1. If no quat indices, use general MSE
        if not self.quat_indices:
            loss = self.loss_fn(pred, target).mean(dim=-1)
        else:
            # 2. Create mask (Boolean Masking)
            D = pred.size(-1)
            is_quat = torch.zeros(D, dtype=torch.bool, device=pred.device)
            is_quat[self.quat_indices] = True

            # 3. Quat Loss (Single Quaternion)
            # [B, T, 4] -> [B, T]
            loss_q = self._compute_quat_loss(
                pred[..., self.quat_indices], 
                target[..., self.quat_indices]
            )

            # 4. Non-Quat Loss 
            loss_nq_sum = 0.0
            if (~is_quat).any():
                # [B, T, D_nq] -> [B, T] (Sum over D_nq)
                nq_errors = self.loss_fn(pred[..., ~is_quat], target[..., ~is_quat])
                loss_nq_sum = nq_errors.sum(dim=-1)

            # 5. Combine final loss
            # Apply additional weight to quaternion part
            loss = (self.quat_weight * loss_q + loss_nq_sum) / D

        loss = self.weight * self.apply_weight_decay(loss)
        return loss