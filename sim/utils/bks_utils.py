import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk

def normalToDcmF0B(nHat_B):
    nHat_B = np.array(nHat_B) / np.linalg.norm(nHat_B)
    if np.allclose(nHat_B, [0, 1, 0]):
        return np.eye(3)
    elif np.allclose(nHat_B, [0, -1, 0]):
        return rbk.PRV2C(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        y_axis = nHat_B
        temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x_axis = temp - np.dot(temp, y_axis) * y_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis) / np.linalg.norm(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis]).T