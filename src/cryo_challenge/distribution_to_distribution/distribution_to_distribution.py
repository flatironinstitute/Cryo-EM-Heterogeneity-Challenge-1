import pandas as pd
import numpy as np
import pickle
import torch
import ot
from scipy.stats import rankdata

from .optimal_transport import optimal_q_emd_vec, optimal_q_emd_vec_regularized
from ..config_validation._distribution_to_disribution_validation import (
    DistToDistResultsValidator,
)


def sort_by_transport(cost):
    m, n = cost.shape
    _, transport = compute_wasserstein_between_distributions_from_weights_and_cost(
        np.ones(m) / m, np.ones(n) / n, cost
    )
    indices = np.argsort((transport * np.arange(m)[..., None]).sum(0))
    return cost[:, indices], indices, transport


def compute_wasserstein_between_distributions_from_weights_and_cost(
    weights_a, weights_b, cost, numItermax=1000000
):
    weights_a = weights_a.numpy() if isinstance(weights_a, torch.Tensor) else weights_a
    weights_b = weights_b.numpy() if isinstance(weights_b, torch.Tensor) else weights_b
    cost = cost.numpy() if isinstance(cost, torch.Tensor) else cost
    transport = ot.emd(weights_a, weights_b, cost, numItermax=numItermax)
    wassertstein = (transport * cost).sum()
    return wassertstein, transport


def map_distributions(cost_matrix, weights_b):
    """
    Map sampels and weights from user distribution to gt distribution using cost_matrix

    Parameters:
    cost_matrix: shape: (n_gt_maps, n_user_maps)
    weights_b: shape (n_user_maps,)

    Returns:
    weights_a_user: shape (n_gt_maps,)
    """
    correspondence = cost_matrix.argmin(axis=1)
    weights_a_user = weights_b[correspondence]
    weights_a_user /= weights_a_user.sum()
    return weights_a_user


def compute_kl_between_distributions(weights_a, weights_a_user):
    kl_pq = (weights_a * np.log(weights_a / weights_a_user)).sum()
    kl_qp = (weights_a_user * np.log(weights_a_user / weights_a)).sum()
    return kl_pq, kl_qp


def make_assignment_matrix(cost_matrix):
    A = np.zeros_like(cost_matrix)
    assignment = cost_matrix.argmin(axis=1)
    A[np.arange(len(A)), assignment] = 1
    A /= A.sum(0)
    A = np.nan_to_num(A, nan=0)
    A[:, np.isclose(A.sum(0), 0)] = 1 / len(A)  # all zero columns sum to 1
    return A


def run(config):
    metadata_df = pd.read_csv(config["path_to_ground_truth_metadata"])
    metadata_df.sort_values("pc1", inplace=True)

    with open(config["path_to_map_to_map_results"], "rb") as f:
        data = pickle.load(f)

    user_submitted_populations = data["user_submitted_populations"]  # .numpy()
    id = torch.load(
        data["config"]["data_params"]["submission_params"]["path_to_submission_file"],
        weights_only=False,
    )["id"]

    results_dict = {}
    results_dict["config"] = config
    results_dict["user_submitted_populations"] = torch.tensor(
        user_submitted_populations
    )
    results_dict["id"] = id

    assert np.isclose(user_submitted_populations.sum(), 1)

    for metric in config["metrics"].keys():
        results_dict[metric] = {}
        results_dict[metric]["replicates"] = {}
        cost_matrix_df = data[metric]["cost_matrix"]
        cost_matrix_df = cost_matrix_df.iloc[
            metadata_df.index.tolist()
        ]  # ordering along het-pc for windowing
        m = len(cost_matrix_df)
        cost_matrix = cost_matrix_df.values

        ## self for regularization
        cost_self = data[metric]["cost_matrix_self"].values
        if config["metrics"][metric]["apply_rank_normalization"] is not None:
            cost_self_rank = np.apply_along_axis(rankdata, 1, cost_self)
            W_distance_self = cost_self_rank
        else:
            W_distance_self = cost_self

        results_dict[metric]["cost_self"] = W_distance_self

        n = cost_matrix.shape[1]

        n_pool_ground_truth_microstates = config["replicate_params"][
            "n_pool_ground_truth_microstates"
        ]
        n_replicates = config["replicate_params"]["n_replicates"]

        for replicate_idx in range(n_replicates):
            replicate_fraction = config["replicate_params"]["replicate_fraction"]
            m_replicate = int(replicate_fraction * m)
            results_dict[metric]["replicates"][replicate_idx] = {}

            # make windowed cost matrix and gt prob
            ## cost matrix
            prob_gt = metadata_df.populations.values
            np.random.seed(replicate_idx)
            idxs = np.random.choice(
                np.arange(m), size=m_replicate, replace=False
            )  # p=prob_gt?
            idxs.sort()
            prob_gt_reduced = prob_gt[idxs] / prob_gt[idxs].sum()
            m_reduce = m_replicate // n_pool_ground_truth_microstates
            Window = np.zeros((m_reduce, m_replicate))
            for i, e in enumerate(np.array_split(np.arange(m_replicate), m_reduce)):
                Window[i, e] = 1  # TODO: soft windowing

            cost = cost_matrix[idxs]

            if config["metrics"][metric]["apply_rank_normalization"]:
                cost_rank = np.apply_along_axis(rankdata, 1, cost)
                W_distance = Window @ cost_rank
            else:
                W_distance = Window @ cost

            ## gt prob
            Wp = Window @ prob_gt_reduced

            # EMD
            ## opt
            q_opt, T, flow, prob, runtime = optimal_q_emd_vec(
                Wp, W_distance, cvxpy_solve_kwargs=config["cvxpy_solve_kwargs"]
            )
            q_opt_reg, T_reg, T_self, flow_reg, prob_reg, runtime_reg = (
                optimal_q_emd_vec_regularized(
                    Wp,
                    user_submitted_populations,
                    W_distance,
                    W_distance_self,
                    config["emd_regularization"],
                )
            )

            results_dict[metric]["replicates"][replicate_idx]["EMD"] = {
                "q_opt": q_opt,
                "EMD_opt": prob.value,
                "transport_plan_opt": T,
                "flow_opt": flow,
                "prob_opt": prob,
                "runtime_opt": runtime,
                "q_opt_reg": q_opt_reg,
                "EMD_opt_reg": prob_reg.value,
                "transport_plan_opt_reg": T_reg,
                "transport_plan_opt_self": T_self,
                "flow_opt_reg": flow_reg,
                "prob_opt_reg": prob_reg,
                "runtime_opt_reg": runtime_reg,
            }
            ## submission
            (
                wasserstein,
                transport,
            ) = compute_wasserstein_between_distributions_from_weights_and_cost(
                Wp, user_submitted_populations, W_distance
            )
            results_dict[metric]["replicates"][replicate_idx]["EMD"].update(
                {"EMD_submitted": wasserstein, "transport_plan_submitted": transport}
            )

            # KL

            A = make_assignment_matrix(cost_matrix=W_distance)

            def optimal_q_kl(n_iter, x_start, A, Window, prob_gt, break_atol):
                n = A.shape[1]
                xs = np.zeros((n_iter + 1, n))
                ones = np.ones(n)
                xs[0] = x_ = x_start
                zeros = np.zeros(n)
                WA = A
                Wp = Window @ prob_gt
                objective = np.zeros(n_iter)
                for iter in range(n_iter):
                    objective[iter] = (Wp * np.log(WA.dot(x_))).sum()
                    gradf = ((Wp.reshape(-1, 1) * WA) / WA.dot(x_).reshape(-1, 1)).sum(
                        0
                    )
                    xs[iter + 1] = gradf * xs[iter]
                    eps = np.linalg.norm(xs[iter + 1] - xs[iter])
                    x_ = xs[iter + 1]

                    if np.logical_or(
                        np.isclose(gradf, ones, atol=break_atol),
                        np.isclose(gradf, zeros, atol=break_atol),
                    ).all():
                        break
                q_opt = x_
                iter_stop = iter
                eps_stop = eps
                klpq, klqp = compute_kl_between_distributions(Wp, WA.dot(q_opt))
                return q_opt, objective[:iter_stop], klpq, klqp, iter_stop, eps_stop

            ## opt
            q_opt, objective, klpq, klqp, iter_stop, eps_stop = optimal_q_kl(
                n_iter=config["optimal_q_kl_params"]["n_iter"],
                x_start=np.ones(n) / n,
                A=A,
                Window=Window,
                prob_gt=prob_gt_reduced,
                break_atol=config["optimal_q_kl_params"]["break_atol"],
            )
            results_dict[metric]["replicates"][replicate_idx]["KL"] = {
                "q_opt": q_opt,
                "klpq_opt": klpq,
                "klqp_opt": klqp,
                "A": A,
                "iter_stop": iter_stop,
                "eps_stop": eps_stop,
                "objective": objective,
            }
            ## submission
            klpq, klqp = compute_kl_between_distributions(
                Wp, A.dot(user_submitted_populations)
            )
            results_dict[metric]["replicates"][replicate_idx]["KL"].update(
                {"klpq_submitted": klpq, "klqp_submitted": klqp}
            )

    print(results_dict.keys())
    results_dict = dict(DistToDistResultsValidator(**results_dict).model_dump())
    with open(config["path_to_output_file"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
