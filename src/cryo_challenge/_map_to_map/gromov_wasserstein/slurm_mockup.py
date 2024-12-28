from distributed import Client
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import numpy as np
from time import time


def main():
    N = 100
    F = 1_000

    large_matrix = np.arange(N * F).reshape(N, F)

    @delayed
    def mm(i, N):
        # sleep(0.1)
        row_of_dot_products = []
        for j in range(N):
            row_of_dot_products.append(np.dot(large_matrix[i], large_matrix[j]))
        return row_of_dot_products

    start = time()
    tasks = [mm(i, N) for i in range(N)]

    with ProgressBar():
        results = compute(tasks)
        # results = tasks
    time_interval = time() - start

    formatted_results = np.array(results).reshape(N, N)
    return formatted_results, time_interval


if __name__ == "__main__":
    client = Client()
    # cluster = LocalCluster(n_workers=n_workers, processes=True)
    n_workers = -1
    formatted_results, time_interval = main()
    print(formatted_results)
    print(f"n_workers={n_workers} | Time (s): {time_interval}")
