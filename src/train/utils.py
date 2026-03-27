import torch
import torch.nn.functional as F


def dist_err(x_pred, x_target):
    Lx = torch.linalg.norm(x_target-x_pred, dim=1)
    return Lx
 
def rot_err(q_pred, q_target):
    q_dist = torch.abs(torch.linalg.vecdot(q_pred, q_target, dim=1))
    return 2*torch.arccos(torch.clamp(q_dist, max = 1))
    
def pose_err(pred, target):
    
    b, n, _ = pred.shape
    target_act = target[:, :n, :]
    pred = pred.view(b*n, -1)
    target_act = target_act.view(b*n, -1)

    x_pred, x_target = pred[:, :3], target_act[:, :3]
    q_pred, q_target = pred[:, 3:7], target_act[:, 3:7]

    dist = dist_err(x_pred, x_target)
    rot = rot_err(q_pred, q_target)
    
    return torch.mean(dist, dim=-1), torch.mean(rot, dim=-1)



# ----------------------------------------------------------------------------- #
# x1 = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [3.0, 1.0, 5.0]])
# x2 = torch.Tensor([[0.0, 0.0, 3.0], [2.0, 0.0, 4.0], [6.0, 14.0,3.0]])

# print(dist_loss(x1, x2))
# # --- test ---
# import math

# # testcase 
# # u1 = F.normalize(torch.Tensor([0.23, 0.6, 0.3]), p=2, dim=0) # axis of rotation 1: y
# u1 = F.normalize(torch.Tensor([0.23, 0.6, 0.3]), p=2, dim=0) # axis of rotation 1: y
# ang1 = 60.0 * math.pi / 180.0 # angle of rotation 1: 60 deg

# # u2 = F.normalize(torch.Tensor([0.33, 0.7, 0.1]), p=2, dim=0) # axis of rotation 1: x
# u2 = F.normalize(torch.Tensor([0.23, 0.6, 0.3]), p=2, dim=0) # axis of rotation 1: y
# ang2 = 60.0 * math.pi /180.0 # angle of rotation 1: 30.0 deg
# print(f'rotations:\n1) axis = {u1}, angle = {ang1} rad\n2) axis = {u2}, angle = {ang2} rad')

# q1 = torch.cat([u1*math.sin(ang1/2), torch.Tensor([math.cos(ang1/2)])])
# q2 = torch.cat([u2*math.sin(ang2/2), torch.Tensor([math.cos(ang2/2)])])
# q1 = F.normalize(q1, p=2, dim=0)
# q2 = F.normalize(q2, p=2, dim=0)

# dist  = rot_loss(q1, q2)
# print(f'distance {dist*180/torch.pi}')