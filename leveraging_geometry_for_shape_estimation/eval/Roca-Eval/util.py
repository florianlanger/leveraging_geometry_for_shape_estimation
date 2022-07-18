import numpy as np
import quaternion
import torch


# See https://github.com/skanti/Scan2CAD/blob/master/Routines/Script/EvaluateBenchmark.py
def calc_rotation_diff(q: np.quaternion, q00: np.quaternion) -> float:
    np.seterr(all='raise')
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def rotation_diff(q: np.quaternion, q_gt: np.quaternion, sym='') -> float:
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q, q_gt)
    return error_rotation


def scale_ratio(pred_scale: torch.Tensor, gt_scale: torch.Tensor) -> float:
    # return torch.abs(torch.mean(pred_scale / gt_scale) - 1).item()
    return np.abs(np.mean(pred_scale.numpy() / gt_scale.numpy()) - 1).item()


def translation_diff(pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> float:
    # return torch.norm(pred_trans - gt_trans)
    return np.linalg.norm(pred_trans.numpy() - gt_trans.numpy(), ord=2)


import numpy as np
import quaternion
import torch

from pytorch3d.structures import Meshes


def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def decompose_mat4(M: np.ndarray) -> tuple:
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    q = quaternion.as_float_array(quaternion.from_rotation_matrix(R[0:3, 0:3]))
    # q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s
