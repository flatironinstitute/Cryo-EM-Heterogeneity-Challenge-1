import numpy as np
import ot
from collections import defaultdict


def objective_no_constant_term(X, Y, Gamma, L):
    """Compute 2 * X * Gamma * Y^T"""
    Z = 2 * X @ Gamma @ Y.T
    frob_sq = np.sum(Z**2)  # ||·||_F^2
    inner_product = np.sum(L * Gamma)  # <L, Gamma>
    return -frob_sq - inner_product


def gw_objective_cost(Cx, Cy, Gamma):
    """
    Compute the Gromov-Wasserstein objective given distance matrices Cx, Cy and transport plan Gamma.
    Args:
        Cx: np.ndarray, shape (n, n), pairwise distance matrix for X
        Cy: np.ndarray, shape (m, m), pairwise distance matrix for Y
        Gamma: np.ndarray, shape (n, m), transport plan
    Returns:
        obj: float, GW objective value
    """
    term = (
        (Cx[:, :, None, None] - Cy[None, None, :, :]) ** 2
        * Gamma[:, None, :, None]
        * Gamma[None, :, None, :]
    )
    return np.sum(term)


def frank_wolfe_emd(X, Y, Gamma0, mu_x, mu_y, num_iters, Gamma_atol=1e-6):
    """
    Frank-Wolfe algorithm for Gromov-wasserstein optimal transport using the Earth Mover's Distance (EMD).

    Objective extended from https://openreview.net/forum?id=l9MbuqzlZt to include non-uniform marginals mu_x and mu_y.

    """
    lx, nx = X.shape
    ly, ny = Y.shape
    assert nx == ny
    XtX = X.T @ X
    YtY = Y.T @ Y

    mx = (np.linalg.norm(X, axis=0) ** 2).reshape(-1, 1)
    my = (np.linalg.norm(Y, axis=0) ** 2).reshape(-1, 1)
    vec_1xy = np.ones_like(mx)

    L = 2 * mu_y.dot(vec_1xy) * np.outer(mx, my)
    L -= 4 * np.outer(mx, mu_y) @ Y.T @ Y
    L -= 4 * X.T @ X @ np.outer(mu_x, my)

    assert Gamma0.shape == L.shape

    # Initial coupling: uniform transport plan
    Gamma_t_plus_1 = Gamma0

    # log
    log = defaultdict(list)
    log["objective"].append(objective_no_constant_term(X, Y, Gamma0, L))
    log["Gamma"].append(Gamma0)
    for t in range(num_iters):
        print("iter", t)
        # Gradient
        Gamma_t = Gamma_t_plus_1
        grad = -8 * XtX @ Gamma_t @ YtY - L

        # Linear oracle: solve transport problem using EMD
        C = grad  # cost matrix = gradient
        S = ot.emd(mu_x, mu_y, C)  # optimal coupling (transport plan)

        # Line search
        diff = S - Gamma_t
        numerator = -8 * np.sum((XtX @ diff) * (Gamma_t @ YtY)) - np.sum(L * diff)
        denominator = 8 * np.sum((XtX @ diff) * (diff @ YtY))
        eta = np.clip(numerator / denominator, 0, 1) if denominator > 1e-12 else 0

        # Update
        Gamma_t_plus_1 = eta * Gamma_t + (1 - eta) * S

        # Update Gamma_t
        log["objective"].append(objective_no_constant_term(X, Y, Gamma_t_plus_1, L))
        log["Gamma"].append(Gamma_t_plus_1)
        log["eta"].append(0)
        log["Gamma_t"].append(Gamma_t_plus_1)
        log["S"].append(Gamma_t_plus_1)
        log["diff"].append(np.zeros_like(Gamma_t_plus_1))
        log["numerator"].append(0)
        log["denominator"].append(0)
        log["Gamma_t_plus_1"].append(Gamma_t_plus_1)

        if np.linalg.norm(Gamma_t_plus_1 - Gamma_t) < Gamma_atol:
            print("Converged")
            break

    return Gamma_t_plus_1, log
