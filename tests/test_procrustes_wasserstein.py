import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from cryo_challenge.map_to_map.procrustes_wasserstein.procrustes_wasserstein import (
    procrustes_wasserstein,
)


def generate_data(
    n,
    m,
    d,
    point_noise_level,
    marginal_noise_level,
    marginal_diff_level,
    angle_deg,
    flip_Y,
    X_close_to_Y=True,
    p_close_to_q=True,
):
    """Helper function to generate data for testing procrustes_wasserstein"""
    X = torch.rand(n, d)

    if X_close_to_Y or p_close_to_q:
        assert n == m

    if X_close_to_Y:
        assert n == m
        Y = X + point_noise_level * torch.randn(m, d)
    else:
        Y = torch.rand(m, d)

    rotation = torch.from_numpy(
        R.from_euler("zyx", [angle_deg, 0, 0], degrees=True).as_matrix()[:d, :d]
    ).to(X.dtype)
    Y = Y @ rotation
    if flip_Y:
        Y[:, 0] *= -1
    marginal_noise_level = 0
    p = torch.softmax(marginal_noise_level * torch.randn(n), dim=0)

    if p_close_to_q:
        q = p + marginal_diff_level * torch.softmax(torch.randn(m), dim=0)
    else:
        q = torch.softmax(marginal_noise_level * torch.randn(m), dim=0)

    q = torch.abs(q)
    q /= q.sum()
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    return X, Y, p, q, rotation


def test_procrustes_wasserstein_recovers():
    """Test that point cloud is recovered when it is slighgly preterbed and rotated 45 degrees"""
    for d in [2, 3]:
        n = m = 40
        point_noise_level = 0.01
        marginal_noise_level = 00.01 / n
        marginal_diff_level = 0.01 / n

        torch.manual_seed(0)
        angle_deg = 45
        X, Y, p, q, rotation_gt = generate_data(
            n,
            m,
            d,
            point_noise_level,
            marginal_noise_level,
            marginal_diff_level,
            angle_deg,
            flip_Y=False,
            X_close_to_Y=True,
            p_close_to_q=True,
        )

        transportation_plan, rotation_estimated, logs = procrustes_wasserstein(
            X, Y, p, q, max_iter=10, verbose_log=True
        )

        torch.allclose(rotation_estimated, rotation_gt.T, atol=1e-2)
        np.allclose(transportation_plan, np.eye(n) / n, atol=1e-3)


def test_procrustes_wasserstein_nonsquare():
    """Test that procrustes_wasserstein works with non-square matrices"""
    d = 3
    n = 40
    m = 50
    point_noise_level = marginal_noise_level = marginal_diff_level = None  # not used
    torch.manual_seed(0)
    angle_deg = 1
    X, Y, p, q, _ = generate_data(
        n,
        m,
        d,
        point_noise_level,
        marginal_noise_level,
        marginal_diff_level,
        angle_deg,
        flip_Y=False,
        X_close_to_Y=False,
        p_close_to_q=False,
    )

    for verbose_log in [True, False]:
        _, _, __build_class__ = procrustes_wasserstein(
            X, Y, p, q, max_iter=2, verbose_log=verbose_log
        )
