import torch
import numpy as np


### Compare subspaces for each submission ###


def captured_variance(V, U, S):
    US = U @ torch.diag(S)
    return torch.norm(torch.adjoint(V) @ US) / torch.norm(torch.adjoint(U) @ US)


# V US


def svd_metric(V1, V2, S1, S2):
    return 0.5 * (captured_variance(V1, V2, S2) + captured_variance(V2, V1, S1))


def sort_matrix_using_gt(dist_matrix: torch.Tensor, labels: np.ndarray):
    sort_idx = torch.argsort(dist_matrix[:, -1])
    dist_matrix = dist_matrix[:, sort_idx][sort_idx]
    labels = labels[sort_idx.numpy()]

    dist_matrix = torch.flip(dist_matrix, dims=(0,))
    dist_matrix = torch.flip(dist_matrix, dims=(1,))
    labels = np.flip(labels)

    return dist_matrix, labels


def sort_matrix(dist_matrix, labels):
    dist_matrix = dist_matrix.clone()
    labels = labels.copy()

    # Sort by sum of rows
    row_sum = torch.sum(dist_matrix, dim=0)
    sort_idx = torch.argsort(row_sum, descending=True)
    dist_matrix = dist_matrix[:, sort_idx][sort_idx]
    labels = labels[sort_idx.numpy()]

    # Sort the first row
    sort_idx = torch.argsort(dist_matrix[:, 0], descending=True)
    dist_matrix = dist_matrix[:, sort_idx][sort_idx]
    labels = labels[sort_idx.numpy()]

    return dist_matrix, labels


def compute_distance_matrix(submissions_data, gt_data=None):
    n_subs = len(list(submissions_data.keys()))
    labels = list(submissions_data.keys())
    dtype = submissions_data[labels[0]]["eigenvectors"].dtype

    if gt_data is not None:
        dist_matrix = torch.ones((n_subs + 1, n_subs + 1), dtype=dtype)
    else:
        dist_matrix = torch.ones((n_subs, n_subs), dtype=dtype)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels[i:]):
            dist_matrix[i, j + i] = svd_metric(
                submissions_data[label1]["eigenvectors"],
                submissions_data[label2]["eigenvectors"],
                submissions_data[label1]["singular_values"],
                submissions_data[label2]["singular_values"],
            )
            dist_matrix[j + i, i] = dist_matrix[i, j + i]

    if gt_data is not None:
        for i, label in enumerate(labels):
            dist_matrix[i, n_subs] = svd_metric(
                submissions_data[label]["eigenvectors"],
                gt_data["eigenvectors"],
                submissions_data[label]["singular_values"],
                gt_data["singular_values"],
            )
            dist_matrix[n_subs, i] = dist_matrix[i, n_subs]

        labels.append("Ground Truth")
        labels = np.array(labels)

        dist_matrix, labels = sort_matrix_using_gt(dist_matrix, labels)

    else:
        labels = np.array(labels)
        dist_matrix, labels = sort_matrix(dist_matrix, labels)

    results = {"dist_matrix": dist_matrix, "labels": labels}
    return results


### Compute common embedding ###


def compute_common_embedding(submissions_data, gt_data=None):
    labels = list(submissions_data.keys())
    n_subs = len(labels)
    shape_per_sub = submissions_data[labels[0]]["eigenvectors"].T.shape
    dtype = submissions_data[labels[0]]["eigenvectors"].dtype
    eigenvectors = torch.zeros(
        (n_subs * shape_per_sub[0], shape_per_sub[1]), dtype=dtype
    )

    for i, label in enumerate(labels):
        eigenvectors[i * shape_per_sub[0] : (i + 1) * shape_per_sub[0], :] = (
            submissions_data[label]["eigenvectors"].T
        ) * submissions_data[label]["singular_values"][:, None]

    U, S, V = torch.linalg.svd(eigenvectors, full_matrices=False)

    Z_common = (U @ torch.diag(S)).reshape(n_subs, shape_per_sub[0], -1)
    embeddings = {}

    for i, label in enumerate(labels):
        Z_i = submissions_data[label]["u_matrices"]  # @ torch.diag(
        # submissions_data[label]["singular_values"]
        # )
        Z_i_common = torch.einsum("ij, jk -> ik", Z_i, Z_common[i])
        embeddings[labels[i]] = Z_i_common

    results = {
        "common_embedding": embeddings,
        "singular_values": S,
        "common_eigenvectors": V,
    }

    if gt_data is not None:
        gt_proj = gt_data["u_matrices"] @ torch.diag(gt_data["singular_values"])
        gt_proj = gt_proj @ (gt_data["eigenvectors"].T @ V.T)

        results["gt_embedding"] = gt_proj

    return results


### Project to GT embedding ###
def project_to_gt_embedding(submissions_data, gt_data):
    embedding_in_gt = {}

    for label, submission in submissions_data.items():
        projection = (
            submission["u_matrices"]
            @ torch.diag(submission["singular_values"])
            @ (submission["eigenvectors"].T @ gt_data["eigenvectors"])
        )
        embedding_in_gt[label] = projection

    gt_coords = gt_data["u_matrices"] @ torch.diag(gt_data["singular_values"])

    results = {"submission_embedding": embedding_in_gt, "gt_embedding": gt_coords}

    return results
