import numpy as np
import ot
from collections import defaultdict


def objective(X, Y, Gamma, L):
    # Compute 2 * X * Gamma * Y^T
    Z = 2 * X @ Gamma @ Y.T
    frob_sq = np.sum(Z**2)  # ||·||_F^2
    inner_product = np.sum(L * Gamma)  # <L, Gamma>
    return -frob_sq - inner_product


def frank_wolfe_emd(X, Y, Gamma0, mu_x, mu_y, num_iters, Gamma_atol=1e-6):
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
    log["objective"].append(objective(X, Y, Gamma0, L))
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
        numerator = -8 * np.trace(diff.T @ XtX @ Gamma_t @ YtY) - np.trace(L.T @ diff)
        denominator = 8 * np.trace(diff.T @ XtX @ diff @ YtY)
        eta = np.clip(numerator / denominator, 0, 1) if denominator > 1e-12 else 0

        # print('eta', eta)
        # print('Gamma_t', Gamma_t)
        # print('S', S)

        # Update
        Gamma_t_plus_1 = eta * Gamma_t + (1 - eta) * S
        # print('Gamma_t+1:', Gamma_t_plus_1)
        # print('Gamma_t - Gamma_t+1:', Gamma_t_plus_1 - Gamma_t)
        # print('||Gamma_t - Gamma_t+1||_F:', np.linalg.norm(Gamma_t_plus_1 - Gamma_t))
        # print('objective:', objective(X, Y, Gamma_t_plus_1, L))

        # Update Gamma_t
        log["objective"].append(objective(X, Y, Gamma_t_plus_1, L))
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


def main():
    # Example usage
    X = np.random.rand(10, 5)
    noise = 0
    Y = X + noise
    mu_x = np.ones(X.shape[1]) / X.shape[1]
    mu_y = np.ones(Y.shape[1]) / Y.shape[1]
    Gamma0 = np.outer(mu_x, mu_y)
    num_iters = 100

    Gamma, log = frank_wolfe_emd(X, Y, Gamma0, mu_x, mu_y, num_iters)
    print("Final coupling:", Gamma)
