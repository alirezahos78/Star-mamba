import argparse

import torch

from starmamba.experiments import EXPERIMENTS, get_experiment
from starmamba.training import build_model, train_classifier
from starmamba.utils import Logger, release_cuda_memory, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Star-Mamba experiments.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="Experiment configuration to run.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override loader batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument(
        "--optimizer",
        choices=("adamw-safe", "adamw"),
        default=None,
        help="Override optimizer. adamw-safe disables PyTorch fused/foreach AdamW paths.",
    )
    parser.add_argument(
        "--empty-cache-each-epoch",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each epoch. Useful for long runs with native CUDA extensions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to a log file.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = get_experiment(args.experiment)
    train_config = exp["train"]
    loader_kwargs = dict(exp["loader_kwargs"])

    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        loader_kwargs["batch_size"] = args.batch_size
    if args.workers is not None:
        loader_kwargs["num_workers"] = args.workers
    if args.optimizer is not None:
        train_config.optimizer = args.optimizer
    if args.empty_cache_each_epoch:
        train_config.empty_cache_each_epoch = True

    logger = None
    original_stdout = None
    if args.log_file:
        import sys

        original_stdout = sys.stdout
        logger = Logger(args.log_file)
        sys.stdout = logger

    try:
        print(f"Experiment: {args.experiment}")
        print(f"Device: {device}")
        print(f"Model config: {exp['model']}")
        print(f"Train config: {train_config}")
        print(f"Loader kwargs: {loader_kwargs}")

        train_loader, val_loader = exp["loader"](**loader_kwargs)
        model = build_model(exp["model"])
        result = train_classifier(
            model,
            train_loader,
            val_loader,
            device,
            config=train_config,
            metric_name=exp["metric_name"],
        )

        print("Training finished.")
        print(f"Best accuracy: {result['best_acc']:.2f}%")
        print(f"Best checkpoint: {result['best_checkpoint']}")
    finally:
        release_cuda_memory(device, empty_cache=True)
        if logger is not None:
            import sys

            sys.stdout = original_stdout
            logger.close()


if __name__ == "__main__":
    main()
