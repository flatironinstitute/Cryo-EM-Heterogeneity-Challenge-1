import torch
from dadapy.metric_comparisons import MetricComparisons


def main():
    # Parameters
    n = 10  # Number of points
    k = 5  # Number of dimensions
    noise_level = 0.5  # Standard deviation of the noise

    # Generate n random points in k dimensions
    # set random seed
    torch.manual_seed(0)
    points = torch.randn(n, k)

    print(points)

    # Create a noise-perturbed version of the points
    noise = torch.randn(n, k) * noise_level
    perturbed_points = points + noise
    d = MetricComparisons(points.numpy(), maxk=n - 1)
    imbalances = d.return_information_imbalace(perturbed_points.numpy(), k=1)
    return imbalances


if __name__ == "__main__":
    main()
