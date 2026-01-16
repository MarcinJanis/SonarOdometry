import numpy as np
from enum import IntEnum


class projection_type(IntEnum):
    POLAR2CARTESIAN = 0
    CARTESIAN2POLAR = 1

def project_points(pts, mode:projection_type):
    '''

    :pt: numpy array with points
    if projection_type == POLAR2CARTESIAN:
        pt[:,0] -> r
        pt[:,1] -> theta
        pt[:,2] -> phi
    
    if projection_type == CARTESIAN2POLAR:
        pt[:,0] -> x
        pt[:,1] -> y
        pt[:,2] -> z

    :mode: projection_type
    '''

    if mode == projection_type.POLAR2CARTESIAN:
        r, theta, phi = pts[:,0], pts[:,1], pts[:,2]
        r_xy = r * np.cos(phi)
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        z = r * np.sin(phi) 
        return  np.column_stack((x, y, z))
    
    elif mode == projection_type.CARTESIAN2POLAR:
        x, y, z = pts[:,0], pts[:,1], pts[:,2] 

        # phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        # r = z / np.sin(phi)
        # theta = np.arctan2(y, z)

        r = np.sqrt(x**2 +  y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))

        return np.column_stack((r, theta, phi))


def affine_matrix(state, invert = False):
    """
    Generates a 4x4 homogeneous transformation matrix.
    state: (x, y, z, roll, pitch, yaw) in radians.

    if invert == True, returns also inverse affine matrix
    """
    x, y, z, roll, pitch, yaw = state

    # Precompute sines and cosines for efficiency
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    # Construct the matrix T (ZYX convention)
    T = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp,   cp*sr,            cp*cr,            z],
        [0,     0,                0,                1]
    ])

    if invert:
        R = T[:3, :3]
        t = T[:3, 3]

        # Invert rotation 
        R_inv = R.T
        
        # Invert translation: -R^T * t
        t_inv = -R_inv @ t

        # Assemble the inverted matrix
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T, T_inv
    else:
        return T

