import os
import numpy as np
from gurobipy import Model, GRB
from cryo_challenge._map_to_map.gromov_wasserstein.gw_weighted_voxels import (
    parse_args,
    setup_volume_and_distance,
    precision,
)


def solve_gromov_wasserstein(mu, nu, quadratic_matrix):
    """
    Solves the Gromov-Wasserstein quadratic program using Gurobi.

    Parameters:
        mu (numpy.ndarray): Source distribution, shape (n,).
        nu (numpy.ndarray): Target distribution, shape (m,).
        quadratic_matrix (numpy.ndarray): Quadratic cost matrix, shape (nm, nm).

    Returns:
        numpy.ndarray: Optimal transport matrix T, shape (n, m).
    """
    # Dimensions
    n = len(mu)
    m = len(nu)

    # Create a Gurobi model
    model = Model("Gromov-Wasserstein")

    # Add variables for T (n x m), flattened as a single vector
    T = model.addVars(n, m, lb=0, vtype=GRB.CONTINUOUS, name="T")

    # Flatten T for indexing consistency
    T_vec = np.array([[T[i, j] for j in range(m)] for i in range(n)]).flatten()

    # Add marginal constraints
    # Row marginal: sum_j T[i, j] = mu[i]
    for i in range(n):
        model.addConstr(sum(T[i, j] for j in range(m)) == mu[i])

    # Column marginal: sum_i T[i, j] = nu[j]
    for j in range(m):
        model.addConstr(sum(T[i, j] for i in range(n)) == nu[j])

    # Objective function: 0.5 * T_vec' * G * T_vec
    obj = 0.5 * sum(
        quadratic_matrix[i, j] * T_vec[i] * T_vec[j]
        for i in range(n * m)
        for j in range(n * m)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Extract solution
    if model.Status == GRB.OPTIMAL:
        T_opt = np.array([[T[i, j].X for j in range(m)] for i in range(n)])
        return T_opt, model
    else:
        print("Optimization failed!")
        return None, None


def generate_quadratic_matrix(Cx, Cy):
    """
    Generate the quadratic cost matrix Q for the Gromov-Wasserstein problem as QP: x^T Q x

    Parameters:
        Cx (numpy.ndarray): Distance matrix for the source space, shape (n, n).
        Cy (numpy.ndarray): Distance matrix for the target space, shape (m, m).

    Returns:
        numpy.ndarray: Quadratic cost matrix G, shape (nm, nm).
    """
    n, m = Cx.shape[0], Cy.shape[0]

    # Initialize the G matrix with zeros
    quadratic_matrix = np.zeros((n * m, n * m))

    # Fill in G using the formula
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for el in range(m):
                    idx1 = i * m + j  # Flattened index for (i, j)
                    idx2 = k * m + el  # Flattened index for (k, l)
                    quadratic_matrix[idx1, idx2] = (Cx[i, j] - Cy[k, el]) ** 2

    return quadratic_matrix


def get_distance_matrix(
    volumes_i,
    volumes_j,
    pairwise_distances_i,
    pairwise_distances_j,
    marginals_i,
    marginals_j,
):
    gw_distance = np.zeros((len(volumes_i), len(volumes_j)))
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

                quadratic_matrix = generate_quadratic_matrix(Cx, Cy)
                mu = marginals_i[idx_i]
                nu = marginals_j[idx_j]
                T_optimal, model = solve_gromov_wasserstein(mu, nu, quadratic_matrix)
                if model is not None:
                    gw_distance[idx_i, idx_j] = gw_distance[idx_j, idx_i] = model.ObjVal
                else:
                    gw_distance[idx_i, idx_j] = gw_distance[idx_j, idx_i] = np.nan
                    Warning(f"GW distance between {idx_i} and {idx_j} failed.")
    return gw_distance


def main(args):
    n_i = args.n_i
    n_j = args.n_j
    n_downsample_pix = args.n_downsample_pix
    top_k = args.top_k
    exponent = args.exponent
    cost_scale_factor = args.cost_scale_factor

    (
        marginals_i,
        marginals_j,
        pairwise_distances_i,
        pairwise_distances_j,
        volumes_i,
        volumes_j,
    ) = setup_volume_and_distance(
        n_i, n_j, n_downsample_pix, top_k, exponent, cost_scale_factor
    )

    gw_distance = get_distance_matrix(
        volumes_i,
        volumes_j,
        pairwise_distances_i,
        pairwise_distances_j,
        marginals_i,
        marginals_j,
    )

    np.save(
        os.path.join(
            args.outdir,
            f"qp_topk{top_k}_ds{n_downsample_pix}_float{precision}_costscalefactor{cost_scale_factor}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
        ),
        gw_distance,
    )
    return gw_distance


if __name__ == "__main__":
    args = parse_args()
    gw_distance = main(args)
