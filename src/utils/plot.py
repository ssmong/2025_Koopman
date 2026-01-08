import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def _compute_angle_error(q_a, q_b):
    q_a = q_a / (q_a.norm(dim=-1, keepdim=True) + 1e-5)
    q_b = q_b / (q_b.norm(dim=-1, keepdim=True) + 1e-5)
    
    dot = (q_a * q_b).sum(dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2 * torch.acos(dot)

def _unwrap_quaternion(q_seq):
    """
    q_seq: (T, 4) 
    If the quaternion at time t is opposite to the quaternion at time t-1, invert the sign to maintain continuity.
    """
    q_out = q_seq.clone()
    for t in range(1, q_out.shape[0]):
        dot = torch.sum(q_out[t] * q_out[t-1])
        if dot < 0:
            q_out[t] = -q_out[t]
    return q_out

def plot_trajectory(
    pred_dt, x_pred, x_gt, 
    angle_indices, quat_indices, 
    save_path=None, 
    mean=None, std=None
):
    x_pred = x_pred.detach()
    x_gt = x_gt.detach()

    if mean is not None and std is not None:
        device = x_pred.device
        mean = mean.to(device)
        std = std.to(device)

    plots = [] 
    angle_set = set(angle_indices)
    quat_set = set(quat_indices)
    
    current_col = 0
    raw_idx = 0
    
    while current_col < x_pred.shape[-1]:
        if raw_idx in quat_set:
            q_pred = x_pred[:, current_col:current_col+4]
            q_gt = x_gt[:, current_col:current_col+4]
            
            q_pred = _unwrap_quaternion(q_pred)
            q_gt = _unwrap_quaternion(q_gt)

            if torch.sum(q_pred[0] * q_gt[0]) < 0:
                q_pred = -q_pred

            err = torch.rad2deg(_compute_angle_error(q_pred, q_gt))
            plots.append((f"State {raw_idx} (Quat Error)", err, None, "Degree"))

            labels = ['0', '1', '2', '3']
            for i in range(4):
                val_p = q_pred[:, i]
                val_g = q_gt[:, i]
                plots.append((f"State {raw_idx + i} (q_{labels[i]})", val_p, val_g, "Value"))
            
            current_col += 4
            raw_idx += 4
            
        elif raw_idx in angle_set:
            cos_p, sin_p = x_pred[:, current_col], x_pred[:, current_col+1]
            cos_g, sin_g = x_gt[:, current_col], x_gt[:, current_col+1]
            
            ang_p = torch.atan2(sin_p, cos_p)
            ang_g = torch.atan2(sin_g, cos_g)
            
            plots.append((f"State {raw_idx} (Angle)", ang_p, ang_g, "Rad"))
            current_col += 2
            raw_idx += 1
            
        else:
            val_p = x_pred[:, current_col]
            val_g = x_gt[:, current_col]
            
            if mean is not None:
                val_p = val_p * std[current_col] + mean[current_col]
                val_g = val_g * std[current_col] + mean[current_col]

            plots.append((f"State {raw_idx}", val_p, val_g, "Value"))
            current_col += 1
            raw_idx += 1

    n_plots = len(plots)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]

    time_steps = np.arange(x_pred.shape[0]) * pred_dt

    for ax, (title, p_seq, g_seq, ylabel) in zip(axes, plots):
        if g_seq is not None:
            ax.plot(time_steps, g_seq.cpu().numpy(), 'k-', label='GT', alpha=0.6)
            ax.plot(time_steps, p_seq.cpu().numpy(), 'r--', label='Pred')
        else:
            ax.plot(time_steps, p_seq.cpu().numpy(), 'b-', label='Error')
            
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.xlabel("Time Step")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()