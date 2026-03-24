
# Transform points between polar and carthesian coords system

class projection_type(IntEnum):
    POLAR2CARTESIAN = 0
    CARTESIAN2POLAR = 1

def transorm_points_coords(pts, mode:projection_type):
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
        r_xy = r * torch.cos(phi)
        x = r_xy * torch.cos(theta)
        y = r_xy * torch.sin(theta)
        z = r * torch.sin(phi) 
        return  torch.stack((x, y, z), dim=1)
    
    elif mode == projection_type.CARTESIAN2POLAR:
        x, y, z = pts[:,0], pts[:,1], pts[:,2] 

        # # x = torch.clamp(x, min=1e-5)
        # x = torch.where(x.abs() < 1e-5, torch.sign(x + 1e-9) * 1e-5, x) # safe (no exploding grad) but to not set grad to 0 for points with small x

        # r_sq_xy = torch.clamp(x**2 + y**2, min=1e-8)
        # r_sq = torch.clamp(x**2 + y**2 + z**2, min=1e-8)

        # r = torch.sqrt(r_sq)
        # theta = torch.atan2(y, x)
        # phi = torch.atan2(z, torch.sqrt(r_sq_xy))

        # return torch.stack((r, theta, phi), dim=1)
        
        r_sq_xy = x**2 + y**2
        r_xy = torch.sqrt(r_sq_xy + 1e-8)
        
        r_sq = r_sq_xy + z**2
        r = torch.sqrt(r_sq + 1e-8)

        # x_s = torch.where(r_sq_xy < 1e-10, x + 1e-5, x)
        x_s = torch.where(r_sq_xy < 1e-10, x + torch.sign(x + 1e-9) * 1e-5, x)
        
        theta = torch.atan2(y, x_s)
        phi = torch.atan2(z, r_xy)

        return torch.stack((r, theta, phi), dim=1)
    

# === Create Transform Matrix from quaterions=== 
def transform_matrix(state):

    # -- -Extrac shift and rotation components for each pose ---
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    qx, qy, qz, qw = state[:, 3], state[:, 4], state[:, 5], state[:, 6]

    # --- Pre-calculation --- 
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    # Construct rows 
    
    # Row 0: [R00, R01, R02, x]
    row0 = torch.stack([1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),      x], dim=1)
    
    # Row 1: [R10, R11, R12, y]
    row1 = torch.stack([2*(xy + wz), 1 - 2*(xx + zz),  2*(yz - wx), y], dim=1)
    
    # Row 2: [R20, R21, R22, z]
    row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy),  z], dim=1)
    
    # Row 3: [0, 0, 0, 1] 
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

    # --- Stack to build transfor matrix ---
    T = torch.stack([row0, row1, row2, row3], dim=1)
    
    return T # shape (N, 4, 4)