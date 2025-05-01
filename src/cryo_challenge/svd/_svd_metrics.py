import torch
from torch import Tensor
from typing import Optional, Dict
from tqdm import tqdm

# from ._svd_utils import sort_distance_matrix_with_ref_label
# from ..map_to_map.map_to_map_distance import fourier_shell_correlation

### Compare subspaces for each submission ###


def compute_captured_variance(
    eigvecs_test_subspace: Tensor,
    eigvecs_ref_subspace: Tensor,
    singvals_ref_subspace: Tensor,
) -> Tensor:
    """
    Compute how much variance of a subspace is captured by another subspace. The subspaces are identified
    with their eigenvectors. The variance is computed as:

    $$||V^* U S||^2_F / ||S||^2_F$$

    where ||.||_F is the Frobenius norm. $V$ corresponds to the eigenvectors of the covariance matrix
    obtained from the test subspace (`eigevts_test_subspace`), and $U$ corresponds to the eigenvectors of the covariance matrix
    obtained from the reference subspace (`eigevts_ref_subspace`). $S$ are the singular values associated with $U$.

    **Arguments:**
        eigvecs_test_subspace: Tensor[n_eigv, n_eigv]
            Eigenvectors of the covariance matrix of the test subspace.
        eigvecs_ref_subspace: Tensor[n_eigv, n_eigv]
            Eigenvectors of the covariance matrix of the reference subpspace.
        singvals_ref_subspace: Tensor[n_eigv]
            Singular values associated to the eigenvectors of the cov. matrix of the ref supspace.
    **Returns:**
        Tensor[float]
            The captured variance of the subspace with eigenvectors U and eigenvalues S^2
            by the subspace with eigenvectors V.
    """
    US = eigvecs_ref_subspace @ torch.diag(singvals_ref_subspace)
    return torch.norm(torch.adjoint(eigvecs_test_subspace) @ US) ** 2 / torch.sum(
        singvals_ref_subspace**2
    )


def compute_pcv_matrix(submissions_svd: Dict, gt_svd: Optional[Dict] = None) -> Dict:
    """
    Compute the captured variance (PCV) matrix for the given submissions and ground truth.
    The PCV matrix is a square matrix where each element (i, j) represents the captured variance
    of the subspace of submission i by the subspace of submission j. The diagonal elements
    represent the captured variance of each submission by itself, which is always 1.
    The last row and column of the matrix correspond to the ground truth (if provided).

    !!!Info
    The eigenvectors must have the same format as the one obtained by running
    `torch.svd_lowrank`. That is, if the data matrix has shape (n, d) and has rank r, then
    the eigenvectors will have shape (d, r) and the singular values will have shape (r,).

    **Arguments:**
        submissions_svd (Dict): A dictionary containing the SVD results for each submission.
            The keys are the submission IDs and the values are dictionaries with the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the submission.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.
        gt_svd (Optional[Dict]): A dictionary containing the SVD results for the ground truth.
            The dictionary must have the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the ground truth.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.

    **Returns:**
        results (Dict): A dictionary containing the PCV matrix and the labels for each submission.
            The keys are:
            - "pcv_matrix": The PCV matrix.
            - "labels": The labels for each submission.
    """
    n_subs = len(submissions_svd)
    labels = list(submissions_svd.keys())

    if gt_svd is not None:
        pcv_matrix = torch.ones((n_subs + 1, n_subs + 1))

    else:
        pcv_matrix = torch.ones((n_subs, n_subs))

    total = n_subs * (n_subs - 1) // 2
    with tqdm(total=total, desc="PCV sub vs sub") as pbar:
        for i in tqdm(range(n_subs)):
            for j in range(i + 1, n_subs):
                pcv_j_on_i = compute_captured_variance(
                    submissions_svd[labels[i]]["eigenvectors"],
                    submissions_svd[labels[j]]["eigenvectors"],
                    submissions_svd[labels[i]]["singular_values"],
                )
                pcv_i_on_j = compute_captured_variance(
                    submissions_svd[labels[j]]["eigenvectors"],
                    submissions_svd[labels[i]]["eigenvectors"],
                    submissions_svd[labels[j]]["singular_values"],
                )
                pcv_matrix[i, j] = pcv_j_on_i
                pcv_matrix[j, i] = pcv_i_on_j
                pbar.update(1)

    if gt_svd is not None:
        for i in tqdm(range(n_subs), desc="PCV sub vs GT"):
            pcv_matrix[i, -1] = compute_captured_variance(
                gt_svd["eigenvectors"],
                submissions_svd[labels[i]]["eigenvectors"],
                submissions_svd[labels[i]]["singular_values"],
            )
            pcv_matrix[-1, i] = compute_captured_variance(
                submissions_svd[labels[i]]["eigenvectors"],
                gt_svd["eigenvectors"],
                gt_svd["singular_values"],
            )

        labels.append("Ground Truth")

    results = {
        "pcv_matrix": pcv_matrix,
        "labels": labels,
    }
    return results


### FSC Distance
# def compute_fsc_matrix_first_eigvecs(
#     submissions_svd: Dict[str, Tensor],
#     gt_svd: Optional[Dict[str, Tensor]] = None,
# ) -> Dict:
#     def get_fsc_metric(vol1, vol2):
#         fsc = fourier_shell_correlation(x=vol1, y=vol2, normalize=True)
#         return torch.abs(fsc)

#     vols_for_comp = [
#         submissions_svd[sub]["eigenvectors"][:, 0] for sub in submissions_svd
#     ]
#     labels = [label for label in submissions_svd]

#     if gt_svd is not None:
#         vols_for_comp.append(gt_svd["eigenvectors"][:, 0])
#         labels.append("Ground Truth")

#     vols_for_comp = torch.stack(vols_for_comp)

#     distance_matrix_func = torch.vmap(
#         torch.vmap(get_fsc_metric, in_dims=(None, 0)), in_dims=(0, None)
#     )

#     distance_matrix = distance_matrix_func(vols_for_comp, vols_for_comp)

#     if gt_svd is not None:
#         distance_matrix, labels = sort_distance_matrix_with_ref_label(
#             distance_matrix, labels, ref_label="Ground Truth"
#         )

#     results = {
#         "dist_matrix": distance_matrix,
#         "labels": labels,
#     }

#     return results


## Compute common embedding ###
def compute_common_embedding(
    submissions_svd: Dict[str, Tensor],
    gt_svd: Optional[Dict[str, Tensor]] = None,
) -> Dict:
    """
    Compute the common embedding for the given submissions and ground truth. The common embedding
    is computed by concatenating the eigenvectors obtained from the SVD of each submission,
    weighting them by their singular values, and computing the SVD of the resulting matrix.

    Each submission is then projected to the common embedding space. The ground truth is also projected
    to the common embedding space if provided. The ground truth is not used to compute the common embedding,

    **Arguments:**
        submissions_svd (Dict[str, Tensor]): A dictionary containing the SVD results for each submission.
            The keys are the submission IDs and the values are dictionaries with the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the submission.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.
        gt_svd (Optional[Dict[str, Tensor]]): A dictionary containing the SVD results for the ground truth.
            The dictionary must have the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the ground truth.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.

    **Returns:**
        results (Dict): A dictionary containing the common embedding and the labels for each submission.
            The keys are:
            - "common_embedding": A dictionary containing the common embedding for each submission.
            - "singular_values": The singular values associated with the common embedding.
            - "common_eigenvectors": The eigenvectors of the common embedding.
            - "gt_embedding": The ground truth embedding (if provided).

    """

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
        Z_i_common = torch.einsum(
            "ij, jk -> ik", submissions_svd[label]["u_matrices"], Z_common[i]
        )
        embeddings[labels[i]] = Z_i_common

    results = {
        "common_embedding": embeddings,
        "singular_values": S,
        "common_eigenvectors": V,
    }

    if gt_svd is not None:
        gt_proj = gt_svd["u_matrices"] @ torch.diag(gt_svd["singular_values"])
        gt_proj = gt_proj @ (
            gt_svd["eigenvectors"].T @ V.T
        )  # (U_gt S_gt V_gt^T) V_common

        results["gt_embedding"] = gt_proj

    return results


## Project to GT embedding ###
def project_to_gt_embedding(submissions_svd: Dict, gt_svd: Dict) -> Dict:
    """
    Project the submissions to the subspace defined by the eigenvectors of the ground truth.

    **Arguments:**
        submissions_svd (Dict): A dictionary containing the SVD results for each submission.
            The keys are the submission IDs and the values are dictionaries with the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the submission.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.
        gt_svd (Dict): A dictionary containing the SVD results for the ground truth.
            The dictionary must have the following keys
            - "eigenvectors": The eigenvectors of the covariance matrix of the ground truth.
            - "singular_values": The singular values associated with the eigenvectors.
            - "u_matrices": The right singular vectors of the SVD.
    **Returns:**
        results (Dict): A dictionary containing the projection of each submission to the ground truth embedding.
            The keys are:
            - "submission_embedding": A dictionary containing the projection of each submission to the ground truth embedding.
            - "gt_embedding": The ground truth embedding.

    """
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
