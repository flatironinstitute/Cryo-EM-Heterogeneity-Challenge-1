import os
import numpy as np
from gurobipy import Model, GRB
from cryo_challenge._map_to_map.gromov_wasserstein.gw_weighted_voxels import (
    parse_args,
    setup_volume_and_distance,
    precision,
)


def solve_gromov_wasserstein(mu, nu, Cx, Cy, factor_out_marginals):
    """
    Solves the Gromov-Wasserstein quadratic program using Gurobi.

    Parameters:
        mu (numpy.ndarray): Source distribution, shape (n,).
        nu (numpy.ndarray): Target distribution, shape (m,).
        Cx (numpy.ndarray): Quadratic cost matrix, shape (n, n).
        Cy (numpy.ndarray): Quadratic cost matrix, shape (m, m).
    Returns:
        numpy.ndarray: Optimal transport matrix T, shape (n, m).
    """
    # Dimensions
    n = len(mu)
    m = len(nu)

    # Create a Gurobi model
    model = Model("Gromov-Wasserstein")

    # Add variables for T (n x m), flattened as a single vector
    transport_plan = model.addVars(n, m, lb=0, vtype=GRB.CONTINUOUS, name="T")

    # Flatten T for indexing consistency

    # Add marginal constraints
    # Row marginal: sum_j T[i, j] = mu[i]
    for i in range(n):
        model.addConstr(sum(transport_plan[i, j] for j in range(m)) == mu[i])

    # Column marginal: sum_i T[i, j] = nu[j]
    for j in range(m):
        model.addConstr(sum(transport_plan[i, j] for i in range(n)) == nu[j])

    obj = 0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for el in range(m):
                    if factor_out_marginals:
                        obj += (
                            -2
                            * (Cx[i, j] * Cy[k, el])
                            * transport_plan[i, k]
                            * transport_plan[j, el]
                        )
                    else:
                        obj += (
                            (Cx[i, j] - Cy[k, el]) ** 2
                            * transport_plan[i, k]
                            * transport_plan[j, el]
                        )
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Extract solution
    if model.Status == GRB.OPTIMAL:
        transport_plan_opt = np.array(
            [[transport_plan[i, j].X for j in range(m)] for i in range(n)]
        )
        return transport_plan_opt, model
    else:
        print("Optimization failed!")
        return None, None


def get_distance_matrix(
    volumes_i,
    volumes_j,
    pairwise_distances_i,
    pairwise_distances_j,
    marginals_i,
    marginals_j,
    factor_out_marginals,
):
    gw_distances = np.zeros((len(volumes_i), len(volumes_j)))
    m = len(marginals_i[0])
    n = len(marginals_j[0])
    transport_plan_optimals = np.zeros((len(volumes_i), len(volumes_j), m, n))
    for idx_i in range(len(volumes_i)):
        for idx_j in range(len(volumes_j)):
            if idx_i > idx_j:
                print(f"Computing GW distance between {idx_i} and {idx_j}...")

                Cx = pairwise_distances_i[idx_i]
                Cy = pairwise_distances_j[idx_j]

                # Ensure the distance matrices are symmetric and non-negative
                Cx = 0.5 * (Cx + Cx.T)
                Cy = 0.5 * (Cy + Cy.T)
                Cx[Cx < 0] = 0
                Cy[Cy < 0] = 0
                assert np.allclose(Cx, Cx.T)
                assert np.allclose(Cy, Cy.T)
                assert np.all(Cx >= 0)
                assert np.all(Cy >= 0)

                mu = marginals_i[idx_i]
                nu = marginals_j[idx_j]
                transport_plan_optimal, model = solve_gromov_wasserstein(
                    mu, nu, Cx, Cy, factor_out_marginals
                )
                transport_plan_optimals[idx_i, idx_j] = transport_plan_optimal
                if model is not None:
                    optimum = model.ObjVal
                    if factor_out_marginals:
                        optimum += (np.outer(mu, mu) * Cx**2).sum() + (
                            np.outer(nu, nu) * Cy**2
                        ).sum()
                    gw_distances[idx_i, idx_j] = gw_distances[idx_j, idx_i] = optimum
                else:
                    gw_distances[idx_i, idx_j] = gw_distances[idx_j, idx_i] = np.nan
                    Warning(f"GW distance between {idx_i} and {idx_j} failed.")

    return gw_distances, transport_plan_optimals


def main(args):
    n_i = args.n_i
    n_j = args.n_j
    n_downsample_pix = args.n_downsample_pix
    top_k = args.top_k
    exponent = args.exponent
    cost_scale_factor = args.cost_scale_factor
    normalize = not args.skip_normalize
    factor_out_marginals = True

    (
        marginals_i,
        marginals_j,
        pairwise_distances_i,
        pairwise_distances_j,
        _,
        _,
        volumes_i,
        volumes_j,
    ) = setup_volume_and_distance(
        n_i,
        n_j,
        n_downsample_pix,
        top_k,
        exponent,
        cost_scale_factor,
        normalize=normalize,
    )

    gw_distances, transport_plan_optimals = get_distance_matrix(
        volumes_i,
        volumes_j,
        pairwise_distances_i,
        pairwise_distances_j,
        marginals_i,
        marginals_j,
        factor_out_marginals,
    )

    np.save(
        os.path.join(
            args.outdir,
            f"qp_factoroutmarginals{factor_out_marginals}_topk{top_k}_ds{n_downsample_pix}_float{precision}_costscalefactor{cost_scale_factor}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
        ),
        gw_distances,
    )
    return gw_distances, transport_plan_optimals


if __name__ == "__main__":
    args = parse_args()
    gw_distance, transport_plan_optimals = main(args)
