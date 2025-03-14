import torch
import ot
import numpy as np


def procrustes_wasserstein(X, Y, p, q, max_iter=10, tol=1e-10):
    """Solves the Procrustes-Wasserstein problem.

    Notes:
    ------

    Implements Algorithm 1 from [1]
    See [2] for details on ensure_determinant_one

    [1] Adamo, D., Corneli, M., Vuillien, M., Vila, E., Adamo, D., Corneli, M., … Vila, E. (2025).
    An in depth look at the Procrustes-Wasserstein distance: properties and barycenters, 0–21.

    [2] Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A., Rostamizadeh, A., & Makadia, A. (2020).
    An analysis of SVD for deep rotation estimation. Advances in Neural Information Processing Systems, 2020-Decem(3), 1–18.
    """
    n, d = X.shape
    m, _ = Y.shape

    rotation = torch.eye(d).to(X.dtype)

    logs = []
    for idx in range(max_iter):
        YR = Y @ rotation
        C = torch.cdist(X, YR, p=2) ** 2

        # Solve optimal transport problem using EMD
        transport_plan, log = ot.emd(p.numpy(), q.numpy(), C.numpy(), log=True)
        log["transport_plan"] = transport_plan
        log["R"] = rotation
        log["YR"] = YR
        logs.append(log)

        # Update P using SVD
        U, _, V = torch.svd(Y.T @ transport_plan.T @ X)
        ensure_determinant_one = torch.ones(d).to(X.dtype)
        ensure_determinant_one[-1] = torch.det(U @ V.T)  # ensure no flipping. see
        rotation_new = U @ torch.diag(ensure_determinant_one) @ V.T

        if len(logs) > 1:
            if np.linalg.norm(log["cost"] - logs[-2]["cost"]) < tol:
                break

        rotation = rotation_new

    return transport_plan, rotation, logs
