import copy
from dataclasses import replace

from .data import cifar10_loaders, cifar100_loaders, day2night_loaders, fashionmnist_loaders, tiny_imagenet_loaders
from .models import (
    NoEWStarMambaBlock,
    NoGlobalStarMambaBlock,
    NoNSStarMambaBlock,
    OnlyGlobalMambaBlock,
    StarMambaBlock,
)
from .training import ModelConfig, TrainConfig


def _with_patch_size(base_experiment, patch_size):
    model = replace(base_experiment["model"], patch_size=patch_size)
    train = replace(
        base_experiment["train"],
        checkpoint_prefix=f"{base_experiment['train'].checkpoint_prefix}_patch{patch_size}",
    )
    return {
        **base_experiment,
        "model": model,
        "train": train,
    }


EXPERIMENTS = {
    "cifar10": {
        "model": ModelConfig(
            img_size=32,
            patch_size=2,
            in_channels=3,
            dim=112,
            depth=6,
            num_classes=10,
            block_type=StarMambaBlock,
        ),
        "train": TrainConfig(warmup_epochs=10, checkpoint_prefix="best_starmamba_cifar10"),
        "loader": cifar10_loaders,
        "loader_kwargs": {"batch_size": 128, "num_workers": 0, "repeated_aug": True},
        "metric_name": "Test Acc",
    },
    "cifar100": {
        "model": ModelConfig(
            img_size=32,
            patch_size=2,
            in_channels=3,
            dim=112,
            depth=6,
            num_classes=100,
            block_type=StarMambaBlock,
        ),
        "train": TrainConfig(warmup_epochs=10, checkpoint_prefix="best_starmamba_cifar100"),
        "loader": cifar100_loaders,
        "loader_kwargs": {"batch_size": 128, "num_workers": 0, "repeated_aug": True},
        "metric_name": "Test Acc",
    },
    "fashionmnist": {
        "model": ModelConfig(
            img_size=28,
            patch_size=2,
            in_channels=1,
            dim=112,
            depth=6,
            num_classes=10,
            block_type=StarMambaBlock,
        ),
        "train": TrainConfig(warmup_epochs=10, checkpoint_prefix="best_starmamba_fashionmnist"),
        "loader": fashionmnist_loaders,
        "loader_kwargs": {"batch_size": 128, "num_workers": 0, "repeated_aug": True},
        "metric_name": "Test Acc",
    },
    "day2night": {
        "model": ModelConfig(
            img_size=64,
            patch_size=4,
            in_channels=3,
            dim=112,
            depth=6,
            num_classes=2,
            block_type=StarMambaBlock,
        ),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_day2night", mixup_alpha=0.0),
        "loader": day2night_loaders,
        "loader_kwargs": {"batch_size": 32, "num_workers": 0, "img_size": 64},
        "metric_name": "Test Acc",
    },
    "tiny-imagenet": {
        "model": ModelConfig(
            img_size=64,
            patch_size=4,
            in_channels=3,
            dim=112,
            depth=6,
            num_classes=200,
            block_type=StarMambaBlock,
        ),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_tiny_imagenet"),
        "loader": tiny_imagenet_loaders,
        "loader_kwargs": {"batch_size": 64, "num_workers": 0, "img_size": 64},
        "metric_name": "Val Acc",
    },
    "tiny-imagenet-no-global": {
        "model": ModelConfig(img_size=64, patch_size=4, dim=112, depth=6, num_classes=200, block_type=NoGlobalStarMambaBlock),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_tiny_imagenet_ablation_no_global"),
        "loader": tiny_imagenet_loaders,
        "loader_kwargs": {"batch_size": 64, "num_workers": 0, "img_size": 64},
        "metric_name": "Val Acc",
    },
    "tiny-imagenet-no-ew": {
        "model": ModelConfig(img_size=64, patch_size=4, dim=112, depth=6, num_classes=200, block_type=NoEWStarMambaBlock),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_tiny_imagenet_ablation_no_ew"),
        "loader": tiny_imagenet_loaders,
        "loader_kwargs": {"batch_size": 64, "num_workers": 0, "img_size": 64},
        "metric_name": "Val Acc",
    },
    "tiny-imagenet-no-ns": {
        "model": ModelConfig(img_size=64, patch_size=4, dim=112, depth=6, num_classes=200, block_type=NoNSStarMambaBlock),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_tiny_imagenet_ablation_no_ns"),
        "loader": tiny_imagenet_loaders,
        "loader_kwargs": {"batch_size": 64, "num_workers": 0, "img_size": 64},
        "metric_name": "Val Acc",
    },
    "tiny-imagenet-only-global": {
        "model": ModelConfig(img_size=64, patch_size=4, dim=112, depth=6, num_classes=200, block_type=OnlyGlobalMambaBlock),
        "train": TrainConfig(warmup_epochs=5, checkpoint_prefix="best_starmamba_tiny_imagenet_ablation_only_global"),
        "loader": tiny_imagenet_loaders,
        "loader_kwargs": {"batch_size": 64, "num_workers": 0, "img_size": 64},
        "metric_name": "Val Acc",
    },
}

PATCH_SIZE_ABLATIONS = {
    "cifar10-patch4": ("cifar10", 4),
    "cifar10-patch8": ("cifar10", 8),
    "cifar100-patch4": ("cifar100", 4),
    "cifar100-patch8": ("cifar100", 8),
    "fashionmnist-patch4": ("fashionmnist", 4),
    "fashionmnist-patch8": ("fashionmnist", 8),
    "tiny-imagenet-patch8": ("tiny-imagenet", 8),
    "tiny-imagenet-patch16": ("tiny-imagenet", 16),
}

for experiment_name, (base_name, patch_size) in PATCH_SIZE_ABLATIONS.items():
    EXPERIMENTS[experiment_name] = _with_patch_size(EXPERIMENTS[base_name], patch_size)


def get_experiment(name):
    if name not in EXPERIMENTS:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment '{name}'. Available experiments: {choices}")
    return copy.deepcopy(EXPERIMENTS[name])
