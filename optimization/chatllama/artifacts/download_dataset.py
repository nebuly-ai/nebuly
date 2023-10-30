import argparse
import os

from chatllama.rlhf.dataset import AnthropicRLHF, StanfordNLPSHPDataset, StanfordNLPSHPRewardDataset


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(
        prog="generate_rewards.py",
        description="Generate rewards using LangChain and LLMs",
    )

    parser.add_argument(
        "dataset_name",
        help="dataset name it can be. SSHP: stanfordnlp/SHP or ",
        choices=["SHP", "ARLHF", "SHPReward"],
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Specify the path for the dataset",
        default="./datasets",
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        help="Specify the number of samples for the reward dataset",
        default=200,
    )

    args = parser.parse_args()
    if os.path.exists(args.path) is False:
        os.mkdir(args.path)

    try:
        n_samples = int(args.number_of_samples)
    except ValueError:
        raise ValueError("Number of samples should be an integer")

    if args.dataset_name == "SHP":
        dataset = StanfordNLPSHPDataset()
        dataset.save_dataset(args.path, n_samples)

    elif args.dataset_name == "ARLHF":
        dataset = AnthropicRLHF()
        dataset.save_dataset(
            args.path,
            n_samples,
        )
    
    if args.dataset_name == "SHPReward":
        dataset = StanfordNLPSHPRewardDataset()
        dataset.save_dataset(args.path, n_samples)
