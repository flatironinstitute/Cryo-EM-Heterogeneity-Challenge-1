"""pytorch impelementation of https://dadapy.readthedocs.io/en/latest/_modules/metric_comparisons.html#MetricComparisons.return_information_imbalace"""

import torch


def _return_ranks(dist_indices_1, dist_indices_2, k=1):
    """Finds all the ranks according to distance 2 of the neighbours according to distance 1.
       Neighbours in distance 1 are considered up to order k.

    Args:
        dist_indices_1 (np.ndarray(int)): N x maxk matrix, nearest neighbours according to distance 1
        dist_indices_2 (np.ndarray(int))): N x maxk_2 matrix, nearest neighbours according to distance 2
        k (int): order of nearest neighbour considered for the calculation of the conditional ranks, default is 1

    Returns:
        conditional_ranks (np.ndarray(int)): N x k matrix, ranks according to distance 2 of the neighbours in distance 1

    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]

    conditional_ranks = torch.zeros(N, k)

    for i in range(N):
        idx_k_d1 = dist_indices_1[i, 1 : k + 1]

        wr = [
            torch.where(idx_k_d1[k_neighbor] == dist_indices_2[i])[0]
            for k_neighbor in range(k)
        ]

        for k_neighbor in range(k):
            conditional_ranks[i, k_neighbor] = wr[k_neighbor][0]

    return conditional_ranks


def _return_imbal(nearest_neighbors_i, nearest_neighbors_j, k):
    N = len(nearest_neighbors_i)
    ranks = _return_ranks(nearest_neighbors_i, nearest_neighbors_j, k=k)
    return 2 * ranks.mean() / N


def return_information_imbalace(self_distance_matrix_i, self_distance_matrix_j, k):
    nearest_neighbors_i = torch.sort(self_distance_matrix_i, dim=1).indices

    nearest_neighbors_j = torch.sort(self_distance_matrix_j, dim=1).indices

    ii_ij = _return_imbal(nearest_neighbors_i, nearest_neighbors_j, k)
    ii_ji = _return_imbal(nearest_neighbors_j, nearest_neighbors_i, k)

    return ii_ij, ii_ji


def main():
    """Demonstrate the use of the function"""
    # Parameters
    n_points = 10  # Number of points
    point_dimentions = 5  # Number of dimensions
    noise_level = 0.5  # Standard deviation of the noise

    # Generate n random points in k dimensions
    # set random seed
    torch.manual_seed(0)
    points = torch.randn(n_points, point_dimentions)

    # Create a noise-perturbed version of the points
    noise = torch.randn_like(points) * noise_level
    perturbed_points = points + noise

    self_distance_matrix_i = torch.cdist(points, points, p=2)
    self_distance_matrix_j = torch.cdist(perturbed_points, perturbed_points, p=2)
    number_of_nearest_neighbors = 1
    return return_information_imbalace(
        self_distance_matrix_i, self_distance_matrix_j, k=number_of_nearest_neighbors
    )


if __name__ == "__main__":
    main()
