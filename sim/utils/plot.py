import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Basilisk.utilities import unitTestSupport, macros
from Basilisk.utilities import RigidBodyKinematics as rbk

def plot_quaternion_error(timeData, dataEP, position=(100, 100)):
    """Plot the attitude errors in Quaternion."""
    fig = plt.figure(1, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    labels = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    for idx in range(4):
        plt.plot(timeData, dataEP[:, idx],
                 color=unitTestSupport.getLineColor(idx, 4),
                 label=labels[idx])
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel(r'Attitude Error Quaternion $\beta_{B/R}$')
    plt.grid(True)
    plt.title('Attitude Error (Quaternion)')

def plot_angle_error(timeData, dataAngle, position=(500, 100)):
    """Plot the attitude angle error."""
    fig = plt.figure(6, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    plt.plot(timeData, dataAngle * macros.R2D, color='b', label=r'$\theta$')
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle Error [deg]')
    plt.grid(True)
    plt.title('Attitude Angle Error')

def plot_latent_error(timeData, dataLatentError, position=(500, 500)):
    """Plot the L2 norm of latent state error."""
    fig = plt.figure(7, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    plt.plot(timeData, dataLatentError, color='r', label=r'$||z - z_{ref}||_2$')
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Latent Error L2 Norm')
    plt.grid(True)
    plt.title('Latent Space Error')

def plot_rate_error(timeData, dataOmegaBR, position=(100, 600)):
    """Plot the body angular velocity rate tracking errors."""
    fig = plt.figure(2, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Rate Tracking Error (rad/s)')
    plt.grid(True)
    plt.title('Rate Tracking Error')

def plot_torque(timeData, dataTorque, position=(900, 100)):
    """Plot the control torque."""
    fig = plt.figure(3, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    for idx in range(3):
        plt.plot(timeData, dataTorque[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\tau_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Torque (Nm)')
    plt.grid(True)
    plt.title('Control Torque')

def plot_rw_motor_torque(timeData, dataUsReq, position=(100, 500)):
    """Plot the RW motor torques."""
    fig = plt.figure(4, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass

    for idx in range(3):
        plt.plot(timeData, dataUsReq[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('RW Motor Torque [Nm]')
    plt.grid(True)
    plt.title('RW Motor Torque')

def plot_rw_speeds(timeData, dataOmegaRW, position=(1100, 500)):
    """Plot the RW speeds."""
    fig = plt.figure(5, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass

    for idx in range(3):
        plt.plot(timeData, dataOmegaRW[:, idx] * macros.RPM,
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('RW Speed [RPM]')
    plt.grid(True)
    plt.title('RW Speeds')

# -----------------------------------------------------------------------------
# AFZ 3D Visualization Helper
# -----------------------------------------------------------------------------
def rotation_matrix_from_vectors(vec1, vec2):
    """
    Calculate the rotation matrix that aligns vec1 to vec2.
    Handling singularities when vectors are parallel or anti-parallel.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    
    # Check for parallel vectors (s ≈ 0, c ≈ 1)
    # Using dot product close to 1
    if np.isclose(np.dot(a, b), 1.0):
        return np.eye(3)
        
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.isclose(c, -1.0):
        # Vectors are opposite (Anti-parallel)
        # Choose an arbitrary orthogonal vector to rotate around 180 deg
        orthogonal_vector = np.array([1, 0, 0], dtype=float)
        if np.abs(np.dot(a, orthogonal_vector)) > 0.99: # If a is nearly x-axis
            orthogonal_vector = np.array([0, 1, 0], dtype=float)
            
        v = np.cross(a, orthogonal_vector)
        v = v / np.linalg.norm(v)
        
        # Rodrigues formula for 180 degree rotation: R = I + 2*K^2 = 2*v*v^T - I
        # where v is the rotation axis (normalized)
        R = -np.eye(3) + 2 * np.outer(v, v)
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        # R = I + K + K^2 * (1-c)/s^2
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R

def plot_cone(ax, vector, angle, color):
    """
    Plot a cone representing the forbidden zone.
    """
    vector = np.array(vector)
    vector = vector / np.linalg.norm(vector)

    slant_height = 1.0
    height = slant_height * np.cos(np.radians(angle))
    radius = slant_height * np.sin(np.radians(angle))

    # Base cone along Z-axis
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, height, 30)
    u, v = np.meshgrid(u, v)
    x = radius * (v / height) * np.cos(u)
    y = radius * (v / height) * np.sin(u)
    z = v

    # Rotate cone to align with target vector
    # Initial direction is Z-axis [0, 0, 1]
    R = rotation_matrix_from_vectors(np.array([0, 0, 1]), vector)
    
    # Apply rotation to all surface points
    xyz = np.dot(R, np.array([x.flatten(), y.flatten(), z.flatten()]))
    x_rot = xyz[0, :].reshape(x.shape)
    y_rot = xyz[1, :].reshape(y.shape)
    z_rot = xyz[2, :].reshape(z.shape)

    ax.plot_surface(x_rot, y_rot, z_rot, color=color, alpha=0.3)

def plot_afz_trajectory(dataEP, afz_list, boresight_vec=[1,0,0], position=(900, 500)):
    """
    Plot the trajectory of the boresight axis based on quaternion data in 3D sphere,
    and include the forbidden zones as cones.
    
    :param dataEP: Array of quaternions (Scalar First: [q0, q1, q2, q3])
    :param afz_list: List of dicts with 'afz_vec' and 'theta'
    :param boresight_vec: Body frame vector to track (default [1,0,0])
    :param position: Window position
    """
    fig = plt.figure(8, figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass

    # Extract boresight trajectory
    boresight_traj = []
    
    # Ensure boresight_vec is numpy array and normalized
    b_vec = np.array(boresight_vec, dtype=float)
    b_vec = b_vec / np.linalg.norm(b_vec)

    # Basilisk (rbk) uses Scalar First [q0, q1, q2, q3]
    for i in range(len(dataEP)):
        q = dataEP[i] 
        dcm_BN = rbk.EP2C(q) # DCM from Inertial (N) to Body (B)
        
        # We want vector in Inertial Frame: v_N = [BN]^T * v_B
        v_N = dcm_BN.T @ b_vec
        boresight_traj.append(v_N)

    boresight_traj = np.array(boresight_traj)

    # Plot trajectory
    ax.plot(boresight_traj[:, 0], boresight_traj[:, 1], boresight_traj[:, 2], 
            label='Boresight Trajectory', color='green', linestyle='-', linewidth=2)

    # Markers
    ax.scatter(boresight_traj[0, 0], boresight_traj[0, 1], boresight_traj[0, 2], 
               color='green', s=100, label='Start', marker='o')
    ax.scatter(boresight_traj[-1, 0], boresight_traj[-1, 1], boresight_traj[-1, 2], 
               color='green', s=100, label='End', marker='*')

    # Target: q_target = [1, 0, 0, 0] (Scalar First, Identity)
    # This implies Body Frame aligns with Inertial Frame.
    # So v_N_target = v_B = boresight_vec
    ax.scatter(b_vec[0], b_vec[1], b_vec[2], color='blue', s=100, label='Target Direction', marker='x')

    # Plot AFZ cones
    colors = ['red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    if afz_list:
        for i, item in enumerate(afz_list):
            afz_vec = item['afz_vec']
            theta = item['theta']
            color = colors[i % len(colors)]
            plot_cone(ax, afz_vec, theta, color=color)

    # Sphere Grid
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    
    ax.view_init(elev=30, azim=180)

    ax.legend(loc='upper right')
    plt.title('Attitude Trajectory & Forbidden Zones')
