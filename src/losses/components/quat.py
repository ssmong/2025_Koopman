import torch
from typing import List, Optional, Tuple
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
        self.additional_metrics = {}

    def _compute_parts(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pred_norm = q_pred.norm(dim=-1, keepdim=True) + 1e-5
        q_pred_unit = q_pred / q_pred_norm
        
        q_target_norm = q_target.norm(dim=-1, keepdim=True) + 1e-5
        q_target_unit = q_target / q_target_norm

        # 1. Angle/Geodesic Loss
        if self.quat_loss_type == 'geodesic':
            dot = (q_pred_unit * q_target_unit).sum(dim=-1)
            quat_loss = 1.0 - dot ** 2
            
        elif self.quat_loss_type == 'chordal':
            diff_plus = (q_pred_unit - q_target_unit).norm(dim=-1)
            diff_minus = (q_pred_unit + q_target_unit).norm(dim=-1)
            quat_loss = torch.min(diff_plus, diff_minus).square()
            
        elif self.quat_loss_type == 'angle':
            cos_half_theta = (q_pred_unit * q_target_unit).sum(dim=-1).abs()
            cos_half_theta = torch.clamp(cos_half_theta, min=0.0, max=1.0)
            sin_sq = 1.0 - cos_half_theta ** 2
            sin_sq = torch.clamp(sin_sq, min=1e-12)
            sin_half_theta = torch.sqrt(sin_sq)
            angle_diff = 2.0 * torch.atan2(sin_half_theta, cos_half_theta)
            
            charb_eps = 1e-8
            quat_loss = torch.sqrt(angle_diff ** 2 + charb_eps ** 2)
        else:
            raise ValueError(f"Unknown quat_loss_type: {self.quat_loss_type}")

        # 2. Norm Regularization Loss
        norm_loss = torch.zeros_like(quat_loss)
        if self.quat_norm_weight > 0:
            norm_loss = (q_pred_norm.squeeze(-1) - 1.0) ** 2
            
        return quat_loss, norm_loss

    def _compute_quat_loss(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """
        Wrapper to maintain compatibility with subclasses (e.g. IsometryLoss)
        """
        loss_angle, loss_norm = self._compute_parts(q_pred, q_target)
        return loss_angle + self.quat_norm_weight * loss_norm

    def forward(self, results: dict) -> torch.Tensor:
        pred, target = self.get_inputs(results)  # [B, T, D]
        self.additional_metrics = {}

        # 1. If no quat indices, use general MSE
        if not self.quat_indices:
            loss = self.loss_fn(pred, target).mean(dim=-1)
        else:
            # 2. Create mask
            D = pred.size(-1)
            is_quat = torch.zeros(D, dtype=torch.bool, device=pred.device)
            is_quat[self.quat_indices] = True

            # 3. Compute Parts
            q_p = pred[..., self.quat_indices]
            q_t = target[..., self.quat_indices]
            
            loss_angle, loss_norm = self._compute_parts(q_p, q_t)
            
            # Save metrics (raw mean before weight decay)
            self.additional_metrics['angle'] = loss_angle.mean().item()
            if self.quat_norm_weight > 0:
                self.additional_metrics['norm'] = loss_norm.mean().item()

            loss_q = loss_angle + self.quat_norm_weight * loss_norm

            # 4. Non-Quat Loss 
            loss_nq_sum = 0.0
            if (~is_quat).any():
                nq_errors = self.loss_fn(pred[..., ~is_quat], target[..., ~is_quat])
                loss_nq_sum = nq_errors.sum(dim=-1)
                
                # Save metric (average per dimension for comparability)
                n_nq = (~is_quat).sum().item()
                self.additional_metrics['omega'] = (loss_nq_sum / max(1, n_nq)).mean().item()

            # 5. Combine final loss
            loss = (self.quat_weight * loss_q + loss_nq_sum) / D

        loss = self.weight * self.apply_weight_decay(loss)
        return loss
