"""for set 2 mock ground truth"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Average volumes and save new versions with different numbers of final maps."
    )
    parser.add_argument(
        "--path_to_submission_file",
        type=str,
        required=False,
        help="Path to input .pt file. For mode divisible_average: requires preprocessing format with 'volumes' and 'populations' as keys.",
    )
    parser.add_argument(
        "--gt_path_to_volumes",
        type=str,
        required=False,
        help="For mode gt_equally_spaced_samples and gt_equally_spaced_averaged_disjoint.",
    )
    parser.add_argument(
        "--gt_path_to_metadata",
        type=str,
        required=False,
        help="For mode gt_equally_spaced_samples and gt_equally_spaced_averaged_disjoint.",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        required=True,
        help="ice cream flavour with all lower cases and spaces as underscores.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--n_final_maps",
        type=int,
        nargs="+",
        required=True,
        help="List of final map counts (e.g. 8 10 16). Should be a factor of the number of maps in the input file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="gt_equally_spaced_samples (re-orders indices by metadata pc1; splits into 80 chunks; selects on index in each chunk); gt_equally_spaced_averaged_disjoint (re-orders indices by metadata pc1; splits into 80 chunks; averges over indeces in each chunk); divisible_average (averages across neighbouring indices).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    assert (
        args.mode
        in [
            "divisible_average",
            "gt_equally_spaced_samples",
            "gt_equally_spaced_averaged_disjoint",
        ]
    ), "Invalid mode. Choose from 'divisible_average', 'gt_equally_spaced_samples', or 'gt_equally_spaced_averaged_disjoint'."

    if args.mode == "divisible_average":
        ground_truth = torch.load(args.path_to_submission_file, weights_only=False)
        assert (
            "volumes" in ground_truth.keys()
        ), "Input file must contain 'volumes' key."
        assert (
            "populations" in ground_truth.keys()
        ), "Input file must contain 'populations' key."
        n_pix = ground_truth["volumes"].shape[-1]

        for n in args.n_final_maps:
            output_fname = os.path.join(
                args.output_dir, f"submission_{args.output_label}_{n}.pt"
            )
            new_shape = (n, -1, n_pix, n_pix, n_pix)

            new_averaged_volumes = (
                ground_truth["volumes"].reshape(new_shape).mean(axis=1)
            )
            new_averaged_populations = (
                ground_truth["populations"].reshape(n, -1).sum(axis=1)
            )  # sum not mean since population needs to sum to 1

            id_label_titleized = args.output_label.replace("_", " ").title()
            torch.save(
                {
                    "volumes": new_averaged_volumes,
                    "populations": new_averaged_populations,
                    "id": f"{id_label_titleized} {n}",
                },
                output_fname,
            )

            plt.plot(new_averaged_populations)
            plt.title(f"Populations for {n} Final Maps")
            plt.xlabel("Index")
            plt.ylabel("Population")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    args.output_dir, f"{args.output_label}_{n}_populations.png"
                )
            )
            plt.close()
    elif args.mode in [
        "gt_equally_spaced_averaged_disjoint",
        "gt_equally_spaced_samples",
    ]:
        torch.manual_seed(args.seed)
        gt_volumes = torch.load(args.gt_path_to_volumes, weights_only=False)
        n_pix = int(round(gt_volumes.shape[-1] ** (1 / 3)))
        assert gt_volumes.shape[-1] == n_pix**3, "Input file must be a cube."

        gt_metadata = pd.read_csv(args.gt_path_to_metadata)
        gt_metadata.sort_values(by=["pc1"], ascending=True, inplace=True)
        gt_volumes_reordered = gt_volumes[gt_metadata.index]

        for n in args.n_final_maps:
            new_volumes = torch.zeros((n, n_pix, n_pix, n_pix), dtype=gt_volumes.dtype)
            new_averaged_populations = torch.zeros(n)
            indices = torch.linspace(0, len(gt_volumes_reordered), n + 1).long()
            for i in range(n):
                if args.mode == "gt_equally_spaced_averaged_disjoint":
                    first_index = indices[i].item()
                    last_index = indices[i + 1].item()
                    new_volumes[i] = (
                        gt_volumes_reordered[first_index:last_index]
                        .mean(axis=0)
                        .reshape(n_pix, n_pix, n_pix)
                    )
                    summed_population = (
                        gt_metadata["populations"]
                        .iloc[first_index:last_index]
                        .sum(axis=0)
                    )  # sum not mean since population needs to sum to 1
                elif args.mode == "gt_equally_spaced_samples":
                    first_index = indices[i].item()
                    last_index = indices[i + 1].item()
                    random_index = torch.randint(first_index, last_index, (1,)).item()
                    new_volumes[i] = gt_volumes_reordered[random_index].reshape(
                        n_pix, n_pix, n_pix
                    )
                    summed_population = gt_metadata["populations"].iloc[random_index]

                new_averaged_populations[i] = (
                    summed_population  # have to renormalize later
                )
            new_averaged_populations /= new_averaged_populations.sum()

            output_fname = os.path.join(
                args.output_dir, f"submission_{args.output_label}_{n}.pt"
            )

            id_label_titelized = args.output_label.replace("_", " ").title()
            torch.save(
                {
                    "volumes": new_volumes,
                    "populations": new_averaged_populations,
                    "id": f"{id_label_titelized} {n}",
                },
                output_fname,
            )


if __name__ == "__main__":
    main()
