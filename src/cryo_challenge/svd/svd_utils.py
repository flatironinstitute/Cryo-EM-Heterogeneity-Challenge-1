import torch
from typing import Optional, Dict

from ..utils._distance_matrix_manipulation import sort_distance_matrix_with_ref_label
from ..map_to_map.map_to_map_distance import fourier_shell_correlation

### Compare subspaces for each submission ###


def compute_captured_variance(V, U, S):
    """
    Compute the captured variance of the subspace with eigenvectors U and eigenvalues S^2
    by the subspace with eigenvectors V.
    """
    US = U @ torch.diag(S)
    return torch.norm(torch.adjoint(V) @ US) ** 2 / torch.sum(S**2)


def compute_cap_var_distance_matrix(
    submissions_svd: Dict[str, torch.Tensor],
    gt_svd: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    n_subs = len(submissions_svd)
    labels = list(submissions_svd.keys())
    dtype = submissions_svd[labels[0]]["eigenvectors"].dtype

    if gt_svd is not None:
        dist_matrix = torch.ones((n_subs + 1, n_subs + 1), dtype=dtype)
    else:
        dist_matrix = torch.ones((n_subs, n_subs), dtype=dtype)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            dist_matrix[i, j] = compute_captured_variance(
                submissions_svd[label2]["eigenvectors"],
                submissions_svd[label1]["eigenvectors"],
                submissions_svd[label1]["singular_values"],
            )

    if gt_svd is not None:
        for i, label in enumerate(labels):
            dist_matrix[-1, i] = compute_captured_variance(
                submissions_svd[label]["eigenvectors"],
                gt_svd["eigenvectors"],
                gt_svd["singular_values"],
            )
            dist_matrix[i, -1] = compute_captured_variance(
                gt_svd["eigenvectors"],
                submissions_svd[label]["eigenvectors"],
                submissions_svd[label]["singular_values"],
            )

        labels.append("Ground Truth")

        dist_matrix, labels = sort_distance_matrix_with_ref_label(
            dist_matrix, labels, ref_label="Ground Truth"
        )

    results = {"dist_matrix": dist_matrix, "labels": labels}
    return results


### FSC Distance
def compute_fsc_matrix_first_eigvecs(
    submissions_svd: Dict[str, torch.Tensor],
    gt_svd: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    def get_fsc_metric(vol1, vol2):
        fsc = fourier_shell_correlation(x=vol1, y=vol2, normalize=True)
        return torch.abs(fsc)

    vols_for_comp = [
        submissions_svd[sub]["eigenvectors"][:, 0] for sub in submissions_svd
    ]
    labels = [label for label in submissions_svd]

    if gt_svd is not None:
        vols_for_comp.append(gt_svd["eigenvectors"][:, 0])
        labels.append("Ground Truth")

    vols_for_comp = torch.stack(vols_for_comp)

    distance_matrix_func = torch.vmap(
        torch.vmap(get_fsc_metric, in_dims=(None, 0)), in_dims=(0, None)
    )

    distance_matrix = distance_matrix_func(vols_for_comp, vols_for_comp)

    if gt_svd is not None:
        distance_matrix, labels = sort_distance_matrix_with_ref_label(
            distance_matrix, labels, ref_label="Ground Truth"
        )

    results = {
        "dist_matrix": distance_matrix,
        "labels": labels,
    }

    return results


### Compute common embedding ###
def compute_common_embedding(
    submissions_svd: Dict[str, torch.Tensor],
    gt_svd: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    labels = list(submissions_svd.keys())
    n_subs = len(labels)
    shape_per_sub = submissions_svd[labels[0]]["eigenvectors"].T.shape
    dtype = submissions_svd[labels[0]]["eigenvectors"].dtype
    eigenvectors = torch.zeros(
        (n_subs * shape_per_sub[0], shape_per_sub[1]), dtype=dtype
    )

    for i, label in enumerate(labels):
        eigenvectors[i * shape_per_sub[0] : (i + 1) * shape_per_sub[0], :] = (
            submissions_svd[label]["eigenvectors"].T
        ) * submissions_svd[label]["singular_values"][:, None]

    U, S, V = torch.linalg.svd(eigenvectors, full_matrices=False)

    Z_common = (U @ torch.diag(S)).reshape(n_subs, shape_per_sub[0], -1)
    embeddings = {}

    for i, label in enumerate(labels):
        Z_i = submissions_svd[label]["u_matrices"] @ torch.diag(
            submissions_svd[label]["singular_values"]
        )
        Z_i_common = torch.einsum("ij, jk -> ik", Z_i, Z_common[i])
        embeddings[labels[i]] = Z_i_common

    results = {
        "common_embedding": embeddings,
        "singular_values": S,
        "common_eigenvectors": V,
    }

    if gt_svd is not None:
        gt_proj = gt_svd["u_matrices"] @ torch.diag(gt_svd["singular_values"])
        gt_proj = gt_proj @ (gt_svd["eigenvectors"].T @ V.T)

        results["gt_embedding"] = gt_proj

    return results


### Project to GT embedding ###
def project_to_gt_embedding(submissions_svd, gt_svd):
    embedding_in_gt = {}

    for label, submission in submissions_svd.items():
        projection = (
            submission["u_matrices"]
            @ torch.diag(submission["singular_values"])
            @ (submission["eigenvectors"].T @ gt_svd["eigenvectors"])
        )
        embedding_in_gt[label] = projection

    gt_coords = gt_svd["u_matrices"] @ torch.diag(gt_svd["singular_values"])

    results = {"submission_embedding": embedding_in_gt, "gt_embedding": gt_coords}

    return results
