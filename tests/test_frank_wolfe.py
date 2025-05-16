import ot
import numpy as np
from cryo_challenge.map_to_map.gromov_wasserstein.frank_wolfe import (
    frank_wolfe_emd,
    gw_objective_cost,
)


def test_frank_wolfe_emd():
    np.random.seed(0)
    n = 100

    for d in [1, 2, 3]:
        X = np.random.rand(d, n)
        scale_noise = 0.1
        noise = scale_noise * np.random.randn(*X.shape)
        Y = X + noise

        for non_uniform_factor in [0, 0.5 / n]:
            mu_x = np.ones(X.shape[1]) + np.arange(X.shape[1]) * non_uniform_factor
            mu_x = mu_x / np.sum(mu_x)
            mu_y = np.ones(Y.shape[1]) - np.arange(Y.shape[1]) * non_uniform_factor
            mu_y = mu_y / np.sum(mu_y)
            assert np.all(mu_x >= 0)
            assert np.all(mu_y >= 0)
            Gamma0 = np.outer(mu_x, mu_y)
            num_iters = 100

            Gamma, log = frank_wolfe_emd(
                X, Y, Gamma0, mu_x, mu_y, num_iters, Gamma_atol=1e-8
            )

            Cx = ot.utils.euclidean_distances(X.T, X.T, squared=True)
            Cy = ot.utils.euclidean_distances(Y.T, Y.T, squared=True)
            gw_frank_wolfe = gw_objective_cost(Cx, Cy, Gamma)

            if d == 1:
                gamma_pot, log_pot = ot.emd_1d(
                    X.flatten(), Y.flatten(), mu_x, mu_y, log=True
                )
            else:
                Cx = ot.dist(X.T, X.T)
                Cy = ot.dist(Y.T, Y.T)
                gamma_pot, log_pot = ot.gromov_wasserstein(
                    Cx, Cy, mu_x, mu_y, loss_fun="square_loss", log=True
                )
                assert np.isclose(
                    log_pot["gw_dist"], gw_frank_wolfe, atol=1e-3
                ), "GW distance does not match the expected result."

            assert np.allclose(
                Gamma.flatten(), gamma_pot.flatten(), atol=1e-3
            ), "Gamma does not match the expected result."


if __name__ == "__main__":
    test_frank_wolfe_emd()
    print("All tests passed.")
