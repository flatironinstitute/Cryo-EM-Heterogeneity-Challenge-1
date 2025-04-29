"""for set 2 mock ground truth"""

import argparse
import os
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Average volumes and save new versions with different numbers of final maps."
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="Path to input .pt file. Maps need to be in the order you want to average them.",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        required=True,
        help="Path to input .pt file. Maps need to be in the order you want to average them.",
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
    args = parser.parse_args()

    ground_truth = torch.load(args.fname, weights_only=False)
    n_pix = ground_truth["volumes"].shape[-1]

    for n in args.n_final_maps:
        output_fname = os.path.join(
            args.output_dir, f"submission_{args.output_label}_{n}.pt"
        )
        new_shape = (n, -1, n_pix, n_pix, n_pix)

        new_averaged_volumes = ground_truth["volumes"].reshape(new_shape).mean(axis=1)
        new_averaged_populations = (
            ground_truth["populations"].reshape(n, -1).sum(axis=1)
        )  # sum not mean since population needs to sum to 1

        id_label_titelized = args.output_label.replace("_", " ").title()
        torch.save(
            {
                "volumes": new_averaged_volumes,
                "populations": new_averaged_populations,
                "id": f"{id_label_titelized} {n}",
            },
            output_fname,
        )

        plt.plot(new_averaged_populations)
        plt.title(f"Populations for {n} Final Maps")
        plt.xlabel("Index")
        plt.ylabel("Population")
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"{args.output_label}_{n}_populations.png")
        )
        plt.close()


if __name__ == "__main__":
    main()
