from dask.distributed import Client
from dask_hpc_runner import SlurmRunner
import os
from slurm_mockup import main

# Only process ID 1 will execute the contents of the context manager
# the rest will start the Dask cluster components instead
print(os.environ["SLURM_NTASKS"])
print(os.environ["SLURM_PROCID"])
job_id = os.environ["SLURM_JOB_ID"]
print(job_id)

with SlurmRunner(
    scheduler_file=f"/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/scheduler-{job_id}.json"
) as runner:
    # The runner object contains the scheduler address and can be passed directly to a client
    with Client(runner) as client:
        # We can wait for all the workers to be ready before continuing
        # client.wait_for_workers(runner.n_workers)

        # Then we can submit some work to the cluster
        assert client.submit(lambda x: x + 1, 10).result() == 11
        assert client.submit(lambda x: x + 1, 20, workers=2).result() == 21

        formatted_results, time_interval = main()

print("you passed the test")
print(f"Time (s): {time_interval}")
print(formatted_results)
