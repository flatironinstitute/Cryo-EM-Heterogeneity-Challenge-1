from distributed import Client
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import numpy as np
from time import time


def main():
    N_I = 4000
    N_J = 8
    N_PIX3 = 20**3

    large_matrix_I = np.random.randn(N_I * N_PIX3).reshape(N_I, N_PIX3)
    large_matrix_J = np.random.randn(N_J * N_PIX3).reshape(N_J, N_PIX3)

    row_wise = True
    start = time()

    if row_wise:

        @delayed
        def mm_rows(large_matrix_i, N_J):
            row_of_dot_products = []
            for j in range(N_J):
                # sleep(1 / N_J)
                row_of_dot_products.append(np.dot(large_matrix_i, large_matrix_J[j]))
            return row_of_dot_products

        tasks = [mm_rows(large_matrix_I[i], N_J) for i in range(N_I)]
    else:

        @delayed
        def mm(large_matrix_i, large_matrix_j):
            return np.dot(large_matrix_i, large_matrix_j)

        tasks = [
            mm(large_matrix_I[i], large_matrix_J[j])
            for j in range(N_J)
            for i in range(N_I)
        ]

    with ProgressBar():
        results = compute(tasks)
        # results = tasks
    time_interval = time() - start

    formatted_results = np.array(results).reshape(N_I, N_J)
    return formatted_results, time_interval


if __name__ == "__main__":
    n_workers = -1
    # cluster = LocalCluster(n_workers=n_workers, processes=True)
    client = Client()
    formatted_results, time_interval = main()
    print(formatted_results)
    print(f"n_workers={n_workers} | Time (s): {time_interval}")
