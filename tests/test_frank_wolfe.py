import numpy as np
from cryo_challenge.map_to_map.gromov_wasserstein.frank_wolfe import frank_wolfe_emd
import ot


def test_frank_wolfe_emd_1d():
    np.random.seed(0)
    n = 100

    for d in [1, 2, 3]:
        X = np.random.rand(d, n)
        scale_noise = 0.1
        noise = scale_noise * np.random.randn(*X.shape)
        Y = X + noise
        mu_x = np.ones(X.shape[1]) / X.shape[1]
        mu_y = np.ones(Y.shape[1]) / Y.shape[1]
        Gamma0 = np.outer(mu_x, mu_y)
        num_iters = 100

        Gamma, log = frank_wolfe_emd(
            X, Y, Gamma0, mu_x, mu_y, num_iters, Gamma_atol=1e-8
        )

        if d == 1:
            gamma_gt = ot.emd_1d(X.flatten(), Y.flatten(), mu_x, mu_y)
        else:
            Cx = ot.dist(X.T, X.T)
            Cy = ot.dist(Y.T, Y.T)
            gamma_gt = ot.gromov_wasserstein(Cx, Cy, mu_x, mu_y)

        assert np.allclose(
            Gamma.flatten(), gamma_gt.flatten(), atol=1e-3
        ), "Gamma does not match the expected result."
