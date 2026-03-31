#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top    = tanHalfFovY * znear
    bottom = -top
    right  = tanHalfFovX * znear
    left   = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# ── Quaternion helpers (used by 4DGaussians deformation network) ─────────────

def apply_rotation(q1, q2):
    """Apply rotation q2 to quaternion q1 (single quaternions as 1-D tensors)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q3 = torch.tensor([w3, x3, y3, z3])
    return q3 / torch.norm(q3)


def batch_quaternion_multiply(q1, q2):
    """Multiply batches of quaternions.  q1, q2: (N, 4).  Returns (N, 4)."""
    w = q1[:, 0]*q2[:, 0] - q1[:, 1]*q2[:, 1] - q1[:, 2]*q2[:, 2] - q1[:, 3]*q2[:, 3]
    x = q1[:, 0]*q2[:, 1] + q1[:, 1]*q2[:, 0] + q1[:, 2]*q2[:, 3] - q1[:, 3]*q2[:, 2]
    y = q1[:, 0]*q2[:, 2] - q1[:, 1]*q2[:, 3] + q1[:, 2]*q2[:, 0] + q1[:, 3]*q2[:, 1]
    z = q1[:, 0]*q2[:, 3] + q1[:, 1]*q2[:, 2] - q1[:, 2]*q2[:, 1] + q1[:, 3]*q2[:, 0]
    q3 = torch.stack((w, x, y, z), dim=1)
    return q3 / torch.norm(q3, dim=1, keepdim=True)
