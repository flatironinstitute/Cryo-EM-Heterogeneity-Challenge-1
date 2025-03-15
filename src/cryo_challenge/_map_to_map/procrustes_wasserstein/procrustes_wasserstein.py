import torch
import ot
import numpy as np
from typing import Tuple


def procrustes_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-10,
    verbose_log: bool = False,
) -> Tuple[np.ndarray, torch.Tensor, list]:
    """Solves the Procrustes-Wasserstein problem.

    Iteratively optimizes the rotation and point correspondence between two point sets X and Y, with weights p and q, respectively.

    Parameters:
    -----------
    X : A matrix of size n x d, where n is the number of points and d is the dimensionality of the points.
    Y : A matrix of size m x d, where m is the number of points and d is the dimensionality of the points.
    p : A vector of size n, representing the weights of the points in X.
    q : A vector of size m, representing the weights of the points in Y.
    max_iter : Maximum number of iterations.
    tol : Tolerance for convergence.

    Notes:
    ------

    transport_plan does not need to be square. It is a matrix of size n x m, where n is the number of points in X and m is the number of points in Y.

    the ensure_determinant_one is a trick to ensure that the determinant of the rotation matrix is 1. This is important to avoid flipping. See ref [3] for details.

    Implements Algorithm 1 from [1]
    See [2] for details on ensure_determinant_one

    [1] Adamo, D., Corneli, M., Vuillien, M., Vila, E., Adamo, D., Corneli, M., … Vila, E. (2025).
    An in depth look at the Procrustes-Wasserstein distance: properties and barycenters, 0–21.

    [2] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An analysis of SVD for deep rotation estimation. Advances in Neural Information Processing Systems, 2020-Decem(3), 1–18.

    [3] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An Analysis of SVD for Deep Rotation Estimation. Proceedings of the 34th International Conference on Neural Information Processing Systems, (3), 1–12.
    http://doi.org/10.5555/3495724.3497615
    """
    n, dx = X.shape
    m, dy = Y.shape
    assert dx == dy, "X and Y must have the same number of columns"
    d = dx

    rotation = torch.eye(d).to(X.dtype)

    logs = []
    for idx in range(max_iter):
        YR = Y @ rotation
        C = torch.cdist(X, YR, p=2) ** 2

        # Solve optimal transport problem using EMD
        transport_plan, log = ot.emd(p.numpy(), q.numpy(), C.numpy(), log=True)
        log["R"] = rotation
        log["YR"] = YR
        if verbose_log:
            log["transport_plan"] = transport_plan
        else:
            del log["u"]  # free up space
            del log["v"]
        logs.append(log)

        # Update P using SVD
        U, _, Vh = torch.linalg.svd(Y.T @ transport_plan.T @ X, full_matrices=True)
        ensure_determinant_one = torch.ones(d).to(X.dtype)
        ensure_determinant_one[-1] = torch.det(U @ Vh)  # ensure no flipping. see
        rotation_new = U @ torch.diag(ensure_determinant_one) @ Vh

        if len(logs) > 1:
            if np.linalg.norm(log["cost"] - logs[-2]["cost"]) < tol:
                break

        rotation = rotation_new

    return transport_plan, rotation, logs
