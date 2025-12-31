import matplotlib.pyplot as plt
import numpy as np
from Basilisk.utilities import unitTestSupport, macros

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
