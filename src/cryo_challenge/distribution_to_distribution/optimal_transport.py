import torch
import ot
import numpy as np
import cvxpy as cp
import time


def precompute_cost_flattened_3d_voxel_distance(box_size, mask=None):
    arr_1d = torch.arange(box_size)
    x, y, z = torch.meshgrid(arr_1d, arr_1d, arr_1d, indexing="ij")
    x = x[mask]
    y = y[mask]
    z = z[mask]
    cost_distance = (
        (x.flatten()[..., None] - x.flatten()[None, ...]).pow(2)
        + (y.flatten()[..., None] - y.flatten()[None, ...]).pow(2)
        + (z.flatten()[..., None] - z.flatten()[None, ...]).pow(2)
    ).sqrt()
    return cost_distance


def compute_cost_ot(map_1, map_2, cost, method="emd2", numItermax=10**6):
    if method == "emd2":
        return ot.emd2(
            map_1.flatten(), map_2.flatten(), cost, numItermax=numItermax
        ).item()
    elif method == "emd":
        T = ot.emd(map_1.flatten(), map_2.flatten(), cost, numItermax=numItermax)
        W = (T * cost).sum().item()
        return W, T
    elif method == "partial_wasserstein2":
        return ot.partial.partial_wasserstein2(
            map_1.flatten(), map_2.flatten(), cost, numItermax=numItermax
        ).item()
    elif method == "gromov_wasserstein2":
        method
        return ot.gromov.gromov_wasserstein2(
            cost.numpy(),
            cost.numpy(),
            map_1.flatten().numpy(),
            map_2.flatten().numpy(),
            max_iter=numItermax,
        )
    else:
        assert False, "method not recognized"


def make_mock_maps(box_size):
    torch.random.manual_seed(0)
    map1 = torch.rand(box_size, box_size, box_size)
    map2 = torch.rand(box_size, box_size, box_size)
    a = map1.flatten()
    a /= a.sum()
    b = map2.flatten()
    b /= b.sum()
    return a, b


def normalize(x):
    x -= x.min()
    return x / x.sum()  # TODO: std, max, sum, ???


def threshold(vol, thresh):
    vol[vol < thresh] = 0
    return vol


def preprocess_mask_thresh_norm(dist1, dist2, cost, mask, thresh):
    cost_masked = cost[mask][:, mask]
    dist1_masked = dist1[mask]
    dist2_masked = dist2[mask]
    dist1_masked = threshold(dist1_masked, thresh)
    dist2_masked = threshold(dist2_masked, thresh)
    dist1_masked = normalize(dist1_masked)
    dist2_masked = normalize(dist2_masked)
    return cost_masked, dist1_masked, dist2_masked


def compute_cost_ot_wrapper(
    map1, map2, method="emd2", mask=None, numItermax=10**6, thresh=-float("inf")
):
    assert map1.ndim == 3, "map must be 3d"
    box_size = len(map1)
    cost_distance = precompute_cost_flattened_3d_voxel_distance(box_size)
    (
        cost_distance_masked,
        map1_flat_masked,
        map2_flat_masked,
    ) = preprocess_mask_thresh_norm(
        map1.flatten(), map2.flatten(), cost_distance, mask, thresh
    )
    return compute_cost_ot(
        map1_flat_masked,
        map2_flat_masked,
        cost_distance_masked,
        method=method,
        numItermax=numItermax,
    )


def optimal_q_emd(p, cost, constraints=None, **kwargs):
    """
    References:
    ----------
    https://en.wikipedia.org/wiki/Minimum-cost_flow_problem
    """
    R, L = cost.shape

    flow = cp.Variable(L + R + L * R)
    u = np.zeros(L + R + L * R)
    u[L:-R] = cost.flatten()

    def make_constraints(flow, p, L, R):
        constraints = []
        sum_i_fsi = 0
        for i in range(L):
            sum_j_fij = 0
            fsi = flow[i]
            sum_i_fsi += fsi
            for j in range(R):
                fij = flow[L + i + j * L]
                fjt = flow[-j]
                sum_j_fij += fij
            constraints.append(fsi - sum_j_fij == 0)  # constraint
        constraints.append(sum_i_fsi - 1 == 0)

        for idx in range(L + R + L * R):
            constraints.append(flow[idx] >= 0)  # constraint

        sum_j_fjt = 0
        for j in range(R):
            sum_i_fij = 0
            fjt = flow[L + L * R + j]
            sum_j_fjt += fjt
            for i in range(L):
                fsi = flow[i]
                fij = flow[L + i + j * L]
                sum_i_fij += fij
            constraints.append(sum_i_fij - fjt == 0)  # constraint
        constraints.append(sum_j_fjt - 1 == 0)

        for j in range(R):
            fjt = flow[L + L * R + j]
            pj = p[j]
            constraints.append(fjt - pj == 0)

        return constraints

    if constraints is None:
        constraints = make_constraints(flow, p, L, R)

    prob = cp.Problem(cp.Minimize(u.flatten().T @ flow), constraints)
    start = time.time()
    prob.solve(**kwargs)
    end = time.time()
    runtime = end - start
    T = flow[L:-R].value.reshape(cost.shape)
    q_opt = T.sum(0)

    return q_opt, T, flow, prob, runtime


def optimal_q_emd_vec(p, cost, constraints=None, **kwargs):
    R, L = cost.shape

    flow = cp.Variable(L + L * R)
    u = np.zeros(L + L * R)
    u[L:] = cost.flatten()

    def make_constraints(flow, p, L, R):
        q = flow[:L]
        return [
            cp.sum(flow[L:].reshape((L, R)), axis=1) == q,
            cp.sum(flow[L:].reshape((L, R)), axis=0) == p,
            cp.sum(q) == 1,
            flow >= 0,
        ]

    constraints = make_constraints(flow, p, L, R)
    prob = cp.Problem(cp.Minimize(u.flatten().T @ flow), constraints)
    start = time.time()
    prob.solve(**kwargs)
    end = time.time()
    runtime = end - start
    T = flow[L:].value.reshape(cost.shape)
    q_opt = T.sum(0)

    return q_opt, T, flow, prob, runtime


def optimal_q_emd_vec_regularized(
    p, q_sub, cost, self_cost, regularization_dict, constraints=None, **kwargs
):
    R, L = cost.shape
    flow = cp.Variable(L + L * R)
    q = flow[:L]

    # split_q_in_half = True
    R_self, L_self = self_cost.shape
    assert R_self == L_self, "self_cost must be square"
    assert L == L_self, "cost and self_cost must share a marginal"
    # if split_q_in_half:
    #     idx_half_set_1 = np.arange(0, L_self, 2)
    #     idx_half_set_2 = idx_half_set_1 + 1
    #     self_cost_subset = self_cost[idx_half_set_1][:, idx_half_set_2]
    #     self_cost = self_cost_subset
    #     L_self = R_self = len(idx_half_set_1)
    #     q = flow[:L]
    #     q_row = q[idx_half_set_1]
    #     q_col = q[idx_half_set_2]
    # else:
    #     q_row = q_col = flow[:L]

    transport_plan_self = cp.Variable(L_self * R_self)
    u = np.zeros(L + L * R)
    u_self = np.zeros(L_self * R_self)
    u[L:] = cost.flatten()
    u_self[:] = self_cost.flatten()

    def make_constraints(flow, p, L, R):
        q = flow[:L]
        return [
            cp.sum(flow[L:].reshape((L, R)), axis=1) == q,
            cp.sum(flow[L:].reshape((L, R)), axis=0) == p,
            cp.sum(q) == 1,
            flow >= 0,
        ]

    def make_constraints_self(
        transport_plan_self, q_to_opt, q_ref, self_transport_fix_zero
    ):
        L = q_to_opt.size
        R = len(q_ref)
        assert L == R, "self_cost must be square"
        constraints = [
            cp.sum(transport_plan_self.reshape((L, R)), axis=1) == q_to_opt,
            cp.sum(transport_plan_self.reshape((L, R)), axis=0) == q_ref,
            transport_plan_self >= 0,
        ]

        if self_transport_fix_zero:
            transport_plan_self_asmatrix = cp.reshape(transport_plan_self, (L, R))
            diag_elements = cp.diag(transport_plan_self_asmatrix)
            constraints.append(diag_elements == 0)
        return constraints

    constraints = make_constraints(flow, p, L, R)

    constraints_self = make_constraints_self(transport_plan_self, q, q_sub, True)
    flow_term_cross = u.flatten().T @ flow
    flow_term_self = u_self.flatten().T @ transport_plan_self
    eps = regularization_dict["entropy_epsilon"]
    entropy_q = -cp.sum(cp.entr(q + eps))
    prob = cp.Problem(
        cp.Minimize(
            flow_term_cross
            + regularization_dict["scalar_hyperparam_self_emd"] * flow_term_self
            + regularization_dict["scalar_hyperparam_self_entropy_q"] * entropy_q
        ),
        constraints + constraints_self,
    )
    start = time.time()
    prob.solve(**kwargs, max_iters=1000)
    end = time.time()
    runtime = end - start

    T = flow[L:].value.reshape(cost.shape)
    q_opt = T.sum(0)

    T_self = None  # transport_plan_self.value.reshape(self_cost.shape)
    # if not split_q_in_half:
    #     assert np.allclose(q_opt, T_self.sum(0))

    return (
        q_opt,
        T,
        T_self,
        flow,
        prob,
        runtime,
    )


def main():
    p = np.array([0.25, 0.25, 0.5])
    cost = np.array(
        [
            [0, 1],
            [1 / 2, 0],
            [1 / 2, 0],
        ]
    )
    q_opt, T, _, _, _ = optimal_q_emd_vec(p, cost, solver=cp.CVXOPT, verbose=True)

    cost = np.array(
        [
            [0, 1],
            [1 / 2, 0],
            [1 / 2, 0],
        ]
    )
    q_opt_reg, T_reg, _, _, _ = optimal_q_emd_vec_regularized(
        p, cost, self_cost=-np.eye(2), solver=cp.CVXOPT, verbose=True
    )

    print("q_opt", q_opt)
    print("q_opt_reg", q_opt_reg)
    print("T", T)
    print("T_reg", T_reg)


if __name__ == "__main__":
    main()
