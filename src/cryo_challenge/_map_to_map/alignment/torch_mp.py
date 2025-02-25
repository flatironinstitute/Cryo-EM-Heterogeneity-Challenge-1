import torch
import torch.multiprocessing as mp
import time
import logging
from omegaconf import DictConfig
import hydra
from time import sleep

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def align_and_distance(a, b, rotation):
    b = b + rotation.max()
    sleep(1)  # for alignment, and perhaps more expensive distance computation
    return torch.norm(a - b)


def pairwise_norm(a_block, b, rotations_block):
    """Compute pairwise norm between a_block (nn, k) and b (m, k) with a naive for loop over rows of a_block."""
    results = []
    for idx in range(len(a_block)):  # Explicit naive loop over rows
        a_row = a_block[idx]  # Shape: (k,)
        rotations_row = rotations_block[idx]  # Shape: (m, 3, 3)
        # Vectorize the custom distance function using torch.vmap
        batched_custom_distance = torch.vmap(align_and_distance, in_dims=(None, 0, 0))
        # Compute the pairwise distances using the vectorized custom distance function
        row_of_results = batched_custom_distance(a_row, b, rotations_row)  # Shape: (n,)
        results.append(row_of_results)
    return torch.stack(results)  # Returns (nn, m)


def process_batch(args):
    """Process a batch of size nn starting at batch_idx."""
    batch_idx, a_block, b, rotations_block = args
    # start_time = time.time()
    result = pairwise_norm(a_block, b, rotations_block)
    # logging.info(f"Processed batch starting at index {batch_idx} in {time.time() - start_time:.4f} seconds")
    return result


def compute_pairwise_distances(a, b, rotations, nn=10, num_workers=4):
    """Compute pairwise distances between matrices a and b using multiprocessing."""
    n = a.shape[0]
    mp.set_start_method("spawn", force=True)  # Ensures compatibility
    pool = mp.Pool(processes=num_workers)
    batch_indices = list(range(0, n, nn))

    # logging.info("Starting multiprocessing...")
    start_time = time.time()
    results = list(
        pool.imap_unordered(
            process_batch,
            [
                (idx, a[idx : min(n, idx + nn)], b, rotations[idx : min(n, idx + nn)])
                for idx in batch_indices
            ],
        )
    )
    pool.close()
    pool.join()
    logging.info(f"Completed multiprocessing in {time.time() - start_time:.4f} seconds")

    # Concatenate results into final (n, m) matrix
    final_result = torch.cat(results, dim=0)
    # logging.info(f"Final result shape: {final_result.shape}")
    return final_result


@hydra.main(version_base=None, config_name=None)
def main(cfg: DictConfig):
    logging.info(f"torch.get_num_threads()={torch.get_num_threads()}")
    logging.info("Generating random data...")
    start_time = time.time()
    n, m, k = 3600, 80, cfg.n_pix**3
    a = torch.randn(n, k)
    b = torch.randn(m, k)
    rotations = torch.randn(n, m, 3, 3)
    logging.info(f"Random data generated in {time.time() - start_time:.4f} seconds")

    result = compute_pairwise_distances(
        a, b, rotations, nn=cfg.nn, num_workers=cfg.num_workers
    )
    return result


if __name__ == "__main__":
    main()
