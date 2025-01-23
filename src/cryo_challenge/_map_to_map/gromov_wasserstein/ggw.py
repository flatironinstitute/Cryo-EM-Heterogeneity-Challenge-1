"""https://github.com/thaning0/Global-Gromov-Wasserstein (MIT liscence)"""

import numpy as np
import ot
from itertools import product
import time
import os
from scipy.sparse import csr_matrix, bmat
import itertools as it


from cryo_challenge._map_to_map.gromov_wasserstein.gw_weighted_voxels import (
    parse_args,
    setup_volume_and_distance,
    precision,
)


def compute_gw(transport_plan, C1, C2, method="vectorized"):
    n = len(C1)
    assert len(C1) == len(C2)
    if method == "uniform-einsum":
        return ((np.ones((n, n)) / n**2) * (C1**2 + C2**2)).sum() - 2 * np.einsum(
            "ij,kl,ik,jl->", C1, C2, transport_plan, transport_plan
        )
    elif method == "vectorized":
        mu_x = transport_plan.sum(1)
        mu_x_out = np.outer(mu_x, mu_x)
        mu_y = transport_plan.sum(0)
        mu_y_out = np.outer(mu_y, mu_y)
        diag = (mu_x_out * C1**2).sum() + (mu_y_out * C2**2).sum()
        cross = -2 * np.sum((C1 @ transport_plan) * (transport_plan @ C2))
        return diag + cross

    elif method == "naive-loop":
        obj = 0
        for i, j, k, el in it.product(range(n), range(n), range(n), range(n)):
            obj += (
                (C1[i, j] - C2[k, el]) ** 2
                * transport_plan[i, k]
                * transport_plan[j, el]
            )
        return obj
    else:
        raise ValueError("method not recognized")


def init_extreme_points(lower_bounds, upper_bounds):
    """
    Initiate the extreme points for constraints: lower_bounds[i]<=wi<=upper_bounds[i]
    Input
    lower_bounds: ndarray(n,)
    upper_bounds: ndarray(n,)

    Return
    E: Extreme points ndarray(m,d)
    B: Bool matrix ndarray(n,m) with binary elements in which the element B_ij= 1 if extreme point j satisfies constraint i with equality
    D: Bool matrix ndarray(m,m) with binary elements in which the element D_ij= 1 if extreme point i and j are adjacent, i.e. extreme points that satisfies the same r − 1 constraints with equality, where r is the rank of the
    """
    n = len(lower_bounds)

    A = np.r_[np.eye(n), -np.eye(n)]
    b = np.r_[upper_bounds, -lower_bounds]

    bounds = [(lo, u) for lo, u in zip(lower_bounds, upper_bounds)]

    E = np.array(list(product(*bounds)))

    # Binary matrix B indicating if a constraint is met with equality
    B = A @ E.T == b[:, None] @ np.ones([1, len(E)])

    # Rank of the problem, r
    # r = np.linalg.matrix_rank(A)
    r = E.shape[1]

    # Adjacency matrix D
    D = (B.astype(int)).T @ B == r - 1

    return E, B, D


def extreme_points_update(E, B, D, A, b, do_optimize_with_sparse):
    """Add a new constraint A @ x \le b to the problem and update the extreme points and the corresponding matrix B, D.
    Input
    E: Extreme points ndarray(n,d)
    B: Bool matrix with binary elements in which the element B_ij= 1 if extreme point j satisfies constraint i with equality
    D: Bool matrix with binary elements in which the element D_ij= 1 if extreme point i and j are adjacent, i.e. extreme points that satisfies the same r − 1 constraints with equality, where r is the rank of the problem.
    A: ndarray(1,d)
    b: ndarray(1,1)

    Output
    E: New extreme points ndarray(m,d)
    B: Constraint bool matrix
    D: Adjacent bool matrix
    """

    new_constraint_values = A @ E.T - b[:, None] @ np.ones([1, len(E)])
    infeasible_indices = np.where(np.squeeze(new_constraint_values > 0))[0]

    # Collect feasible adjacent extreme points
    new_extreme_points = []
    C = []
    O_m = []

    if do_optimize_with_sparse:
        D_sparse = D
    for i in infeasible_indices:
        if do_optimize_with_sparse:
            feasible_adjacent_indices_sparse = D_sparse[i].indices
            # assert np.allclose(feasible_adjacent_indices, feasible_adjacent_indices_sparse)
            feasible_adjacent_indices = feasible_adjacent_indices_sparse
        else:
            feasible_adjacent_indices = np.where(D[i] == 1)[0]

        for j in feasible_adjacent_indices:
            if j not in infeasible_indices:
                lamb = (b - A @ E[j].T) / (A @ (E[i] - E[j]).T)
                new_point = (1 - lamb) * E[j] + lamb * E[i]
                new_extreme_points.append(new_point)
                C.append(B[:, i] & B[:, j])
                new_O = np.zeros(len(E), dtype=bool)
                new_O[j] = True
                new_O[i] = True
                O_m.append(new_O)

    # Convert new_extreme_points to a numpy array and append to E
    if new_extreme_points:
        mask = np.ones(len(E), dtype=bool)
        mask[infeasible_indices] = False

        E = E[mask]
        B = B[:, mask]

        P = np.array(new_extreme_points)
        E = np.r_[E, P]
        b_row = np.zeros((1, B.shape[1])).astype(bool)
        b_row = np.column_stack(
            (b_row, np.ones((1, len(new_extreme_points))).astype(bool))
        )
        C = np.array(C).T
        B = np.c_[B, C]
        B = np.r_[B, b_row]

        # Update adjacency matrix D
        r = E.shape[1]
        # N_old = ((C.astype(int)).T @ C == r-2) | (np.eye(len(C.T),dtype=bool)) # slow
        # Convert C to integer for consistent behavior
        C_int = C.astype(int)  # Dense

        if do_optimize_with_sparse:
            C_sparse_int = csr_matrix(C_int)
            CC_intermediate = C_sparse_int.T @ C_sparse_int  # slow
            ### method 1
            # N_sparse = CC_intermediate == r - 2 # slow

            ### method 2
            # Assuming C_sparse_int is a sparse matrix
            r_minus_2 = r - 2
            # Perform the sparse multiplication
            # CC_intermediate = C_sparse_int.T @ C_sparse_int
            # Compare only non-zero elements to r-2, keeping it sparse
            r_mask = CC_intermediate.data == r_minus_2
            CC_intermediate.data[~r_mask] = 0  # Set non-matching entries to zero
            CC_intermediate.eliminate_zeros()  # Remove zero entries efficiently
            N_sparse = CC_intermediate.astype(bool)  # Convert to boolean
            # assert np.allclose(result_sparse.toarray(), N_sparse.toarray())

            N_sparse.setdiag(True)  # Set diagonal to True

            # assert np.allclose(N_sparse.toarray(), N_dense)
            # N = N_sparse
        else:
            N_dense = C_int.T @ C_int == r - 2
            np.fill_diagonal(N_dense, True)
            # N = N_dense

        O_m = np.array(O_m).T

        # Convert D to sparse matrix if it's sparse
        if do_optimize_with_sparse:
            # Apply mask to rows and columns efficiently
            D_sparse = D_sparse[mask, :][:, mask]  # Slicing in sparse format
            # D = D_sparse  # Convert back to dense if needed
            # assert np.allclose(D_sparse, D_dense)
        else:
            D_dense = D[np.ix_(mask, mask)]  # D[:,mask][mask] # slow

        O_dense = O_m[mask, :]  # slow

        if do_optimize_with_sparse:
            O_sparse = csr_matrix(O_m)  # slow
            O_sparse = O_sparse[mask, :]
            # assert np.allclose(O_sparse.toarray(), O_dense)
            D_sparse = bmat(
                [[D_sparse, O_sparse], [O_sparse.T, N_sparse]], format="csr"
            )
            # assert np.allclose(D_sparse.toarray(), D_dense)
            D = D_sparse
        else:
            D_dense = np.block([[D_dense, O_dense], [O_dense.T, N_dense]])  # slow
            D = D_dense

    return E, B, D


def gw_obj(Cx, Cy, Pi):
    """
    Input:
    Cx: ndarray(nx,nx) Cx = \{\|x_i-x_j\|^2\}
    Cy: ndarray(ny,ny) Cy = \{\|y_i-y_j\|^2\}
    Pi: ndarray(nx,ny), permutation matrix sum(Pi,axis=0) = [1,1,...,1],sum(Pi,axis=1) = [1,1,...,1]

    Return:
    GW value
    """
    return np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum((Cx @ Pi) * (Pi @ Cy))


def optimal_extreme_point(E):
    """
    \min_i -\|W_i\|^2-w_i+c

    Input:
    E: extreme points ndarray(n,d), where Ei = [w_i,vec(W_i)]

    Return:
    Ei: Minimizer
    Fi: Minimum value
    """
    F = -E[:, 0] - np.sum(E[:, 1:] ** 2, axis=1)
    index = np.argmin(F)
    return E[index], F[index]


def gw_global(
    X,
    Y,
    epsilon=1e-6,
    IterMax=100,
    verbose=False,
    log=False,
    mu_x=None,
    mu_y=None,
    do_optimize_with_sparse=True,
):
    """
    Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].

    Input:
    X: ndarray(nx,lx)
    Y: ndarray(ny,ly)
    epsilon: Stop threshold on bound gap
    IterMax: Max number of iterations
    verbose: Print information along iterations
    log: Record log if True

    Return:
    Pi: ndarray(nx,ny) Global optimal solution
    logs:  log dictionary return only if log==True in parameters

    [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
    """

    print("do_optimize_with_sparse", do_optimize_with_sparse)

    start_time = time.time()

    if mu_x is not None or mu_y is not None:
        non_uniform = True
        assert len(X) == len(Y)
        assert len(X.T) == len(Y.T)

    else:
        non_uniform = False

    X = X.T
    Y = Y.T

    lx, nx = X.shape
    ly, ny = Y.shape
    l_bound = -np.inf
    u_bound = np.inf

    mx = (np.linalg.norm(X, axis=0) ** 2).reshape(-1, 1)
    my = (np.linalg.norm(Y, axis=0) ** 2).reshape(-1, 1)
    vec_1x = np.ones_like(mx)
    vec_1y = np.ones_like(my)

    if non_uniform:
        # L = 2 * mx @ mu_y.T @ vec_1y @ my.T
        L = 2 * mu_y.dot(vec_1y) * np.outer(mx, my)
        L -= 4 * np.outer(mx, mu_y) @ Y.T @ Y
        L -= 4 * X.T @ X @ np.outer(mu_x, my)
    else:
        L = (
            (nx + ny) * mx @ my.T
            - 4 * mx @ vec_1y.T @ Y.T @ Y
            - 4 * X.T @ X @ vec_1x @ my.T
        )

    Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
    Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
    if non_uniform:
        c0 = (
            (np.outer(mu_x, mu_y) * Cx**2).sum()
            + (np.outer(mu_x, mu_y) * Cy**2).sum()
            - 2 * mu_y.dot(my) * mu_x.dot(mx)
        )
    else:
        c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)

    W_lower = np.zeros([lx, ly])
    W_upper = np.zeros([lx, ly])

    if non_uniform:
        a = mu_x
        b = mu_y
    else:
        a = np.ones(nx)
        b = np.ones(ny)

    if verbose:
        print("Iter |Bound gap" + "\n" + "-" * 22)

    if log:
        E_cache = []
        E0_cache = []
        gap_cache = []
        Pi_cache = []
        l_cache = []
        u_cache = []
        c_cache = []
        time_cache = []
        obj_cache = []

    w_lower, w_upper = ot.emd2(a, b, L), -ot.emd2(a, b, -L)

    for i in range(lx):
        for j in range(ly):
            M = (
                2 * X[i, None].T @ Y[j, None]
            )  # TODO: 2(X Gamma Y')_ij. Gamma is missing
            W_lower[i, j] = ot.emd2(a, b, M)
            W_upper[i, j] = -ot.emd2(a, b, -M)

    lower_bounds = np.r_[np.array([w_lower]), W_lower.reshape(-1)]
    upper_bounds = np.r_[np.array([w_upper]), W_upper.reshape(-1)]

    E, B, D = init_extreme_points(
        lower_bounds, upper_bounds
    )  # TODO: bouding box changes?
    if do_optimize_with_sparse:
        D = csr_matrix(D)

    end_time = time.time()
    initialization_time = end_time - start_time
    A_ = b_ = iteration_time = (
        None  # TODO: need to pass linter. code will work find if this line is commented out
    )
    for niter in range(IterMax):
        start_time = time.time()

        wW_cache, l_bound = optimal_extreme_point(E)  # TODO: changes?
        l_bound = (l_bound + c0).item()
        # print('bound',l_bound, type(l_bound))
        Wn = wW_cache[1:].reshape(lx, ly)

        M = 4 * X.T @ Wn @ Y + L

        Pi = ot.emd(a, b, -M)
        bound = (-np.sum((2 * X @ Pi @ Y.T) ** 2) - np.sum(L * Pi) + c0).item()
        # print('bound',bound, bound.item(), type(bound), bound.shape)
        u_bound = np.min([u_bound, bound])

        if log:
            E_cache.append(E)
            gap_cache.append(u_bound - l_bound)
            l_cache.append(l_bound)
            u_cache.append(u_bound)
            Pi_cache.append(Pi)
            E0_cache.append(wW_cache)
            obj_cache.append(compute_gw(Pi, Cx, Cy, "vectorized"))
            if niter == 0:
                c_cache.append((lower_bounds, upper_bounds))
                time_cache.append(initialization_time)
            else:
                c_cache.append((A_, b_))
                time_cache.append(iteration_time)

        if u_bound - l_bound < epsilon:
            if verbose:
                print(f"{niter:5d}|{(u_bound-l_bound):8e}")
            break

        if verbose:
            if niter % 1 == 0:
                print(f"{niter:5d}|{(u_bound-l_bound):8e}")

        Zn = 2 * Wn.reshape(-1)
        alphan = 1

        betan = np.sum(M * Pi)

        A_ = np.r_[np.array([alphan]), Zn]
        b_ = np.array([betan])

        E, B, D = extreme_points_update(
            E, B, D, A_, b_, do_optimize_with_sparse
        )  # TODO: changes?

        end_time = time.time()
        iteration_time = end_time - start_time

    if u_bound - l_bound > epsilon:
        print("Warning: algorithm does not converge. Try larget IterMax.")

    transport_plan_normalized = Pi / Pi.sum()
    if log:
        cum_time = np.cumsum(time_cache)
        logs = {
            "niter": niter + 1,
            "obj_cache": obj_cache,
            "E_cache": E_cache,
            "gap_cache": gap_cache,
            "Pi_cache": Pi_cache,
            "E0_cache": E0_cache,
            "l_cache": l_cache,
            "u_cache": u_cache,
            "c_cache": c_cache,
            "time_cache": time_cache,
            "cum_time": cum_time,
            "L": L,
        }
        return transport_plan_normalized, logs
    else:
        return transport_plan_normalized


def get_distance_matrix(
    volumes_i,
    volumes_j,
    points_i,
    points_j,
    pairwise_distances_i,
    pairwise_distances_j,
    marginals_i,
    marginals_j,
    IterMax,
    epsilon,
    do_optimize_with_sparse,
):
    gw_distances = np.zeros((len(volumes_i), len(volumes_j)))
    m = len(marginals_i[0])
    n = len(marginals_j[0])
    transport_plan_optimals = np.zeros((len(volumes_i), len(volumes_j), m, n))
    for idx_i in range(len(volumes_i)):
        for idx_j in range(len(volumes_j)):
            if idx_i < idx_j:
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
                X = points_i[idx_i]
                Y = points_j[idx_j]
                transport_plan_optimal, logs = gw_global(
                    X,
                    Y,
                    epsilon=epsilon,
                    IterMax=IterMax,
                    verbose=True,
                    log=True,
                    mu_x=mu,
                    mu_y=nu,
                    do_optimize_with_sparse=do_optimize_with_sparse,
                )
                print(logs["cum_time"], logs["time_cache"])
                optimum = compute_gw(
                    transport_plan_optimal, Cx, Cy, method="vectorized"
                )
                if len(volumes_j) == len(volumes_i):
                    gw_distances[idx_i, idx_j] = gw_distances[idx_j, idx_i] = optimum
                else:
                    gw_distances[idx_i, idx_j] = optimum

    return gw_distances, transport_plan_optimals


def main(args):
    n_i = args.n_i
    n_j = args.n_j
    n_downsample_pix = args.n_downsample_pix
    top_k = args.top_k
    exponent = args.exponent
    cost_scale_factor = args.cost_scale_factor
    normalize = not args.skip_normalize
    IterMax = 50
    epsilon = 1e-2
    do_optimize_with_sparse = True

    (
        marginals_i,
        marginals_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i,
        pairwise_distances_j,
        volumes_i,
        volumes_j,
    ) = setup_volume_and_distance(
        n_i, n_j, n_downsample_pix, top_k, exponent, cost_scale_factor, normalize
    )

    gw_distances, transport_plan_optimals = get_distance_matrix(
        volumes_i,
        volumes_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i,
        pairwise_distances_j,
        marginals_i,
        marginals_j,
        IterMax,
        epsilon,
        do_optimize_with_sparse,
    )

    np.save(
        os.path.join(
            args.outdir,
            f"ggw_topk{top_k}_ds{n_downsample_pix}_float{precision}_costscalefactor{cost_scale_factor}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
        ),
        gw_distances,
    )
    return gw_distances, transport_plan_optimals


if __name__ == "__main__":
    args = parse_args()
    gw_distance, transport_plan_optimals = main(args)
