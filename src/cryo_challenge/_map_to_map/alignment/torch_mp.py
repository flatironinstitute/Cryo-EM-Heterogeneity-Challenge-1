import torch
import torch.multiprocessing as mp
import time
import logging
from omegaconf import DictConfig
import hydra

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def pairwise_norm(a_block, b):
    """Compute pairwise norm between a_block (nn, k) and b (m, k) with a naive for loop over rows of a_block."""
    results = []
    for a_row in a_block:  # Explicit naive loop over rows
        b_row = b + torch.rand_like(b)
        results.append(torch.norm(a_row - b_row, dim=-1))  # Broadcasting over m (vmap)
    return torch.stack(results)  # Returns (nn, m)


def process_batch(args):
    """Process a batch of size nn starting at batch_idx."""
    batch_idx, a_block, b = args
    # start_time = time.time()
    result = pairwise_norm(a_block, b)
    # logging.info(f"Processed batch starting at index {batch_idx} in {time.time() - start_time:.4f} seconds")
    return result


def compute_pairwise_distances(a, b, nn=10, num_workers=4):
    """Compute pairwise distances between matrices a and b using multiprocessing."""
    n = a.shape[0]
    mp.set_start_method("spawn", force=True)  # Ensures compatibility
    pool = mp.Pool(processes=num_workers)
    batch_indices = list(range(0, n, nn))

    # logging.info("Starting multiprocessing...")
    start_time = time.time()
    results = list(
        pool.imap_unordered(
            process_batch, [(idx, a[idx : idx + nn], b) for idx in batch_indices]
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
    logging.info("Generating random data...")
    start_time = time.time()
    n, m, k = 3600, 80, cfg.n_pix**3
    a = torch.randn(n, k)
    b = torch.randn(m, k)
    logging.info(f"Random data generated in {time.time() - start_time:.4f} seconds")

    result = compute_pairwise_distances(a, b, nn=cfg.nn, num_workers=cfg.num_workers)
    print(result.shape)  # Should be (n, m)


if __name__ == "__main__":
    main()
