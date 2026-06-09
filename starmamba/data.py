import os
import shutil
import urllib.request
import zipfile

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from .augmentations import Cutout, RABatchSampler


CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
CIFAR100_STATS = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
FASHIONMNIST_STATS = ((0.2860,), (0.3530,))
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_dataloader_kwargs(num_workers, pin_memory=None, persistent_workers=False, prefetch_factor=2):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def make_repeated_aug_loader(
    dataset,
    batch_size,
    num_workers,
    repeats=5,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    sampler = RABatchSampler(len(dataset), batch_size=batch_size, repeats=repeats)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )


def cifar10_loaders(
    root="./data",
    batch_size=128,
    num_workers=0,
    repeated_aug=True,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    mean, std = CIFAR10_STATS
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
            Cutout(n_holes=1, length=16),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if repeated_aug:
        train_loader = make_repeated_aug_loader(
            trainset, batch_size, num_workers, repeats=5, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
    else:
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
        )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    return train_loader, test_loader


def cifar100_loaders(
    root="./data",
    batch_size=128,
    num_workers=0,
    repeated_aug=True,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    mean, std = CIFAR100_STATS
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
            Cutout(n_holes=1, length=16),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if repeated_aug:
        train_loader = make_repeated_aug_loader(
            trainset, batch_size, num_workers, repeats=5, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
    else:
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
        )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    return train_loader, test_loader


def fashionmnist_loaders(
    root="./data",
    batch_size=128,
    num_workers=0,
    repeated_aug=True,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    mean, std = FASHIONMNIST_STATS
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
            Cutout(n_holes=1, length=8),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)

    if repeated_aug:
        train_loader = make_repeated_aug_loader(
            trainset, batch_size, num_workers, repeats=5, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
    else:
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
        )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    return train_loader, test_loader


class Day2NightDataset(Dataset):
    """Binary image dataset with trainA/trainB and testA/testB folders."""

    def __init__(self, root, split, transform=None):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = root
        self.split = split
        self.transform = transform
        self.classes = ["A", "B"]
        self.class_to_idx = {"A": 0, "B": 1}
        self.samples = []

        for class_name in self.classes:
            folder = os.path.join(root, f"{split}{class_name}")
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected Day2Night folder not found: {folder}")
            for dirpath, _dirnames, filenames in os.walk(folder):
                for filename in sorted(filenames):
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        self.samples.append((os.path.join(dirpath, filename), self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found for Day2Night split '{split}' under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def day2night_loaders(
    root="./data",
    batch_size=32,
    num_workers=0,
    img_size=64,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    dataset_dir = os.path.join(root, "day2night")
    mean, std = IMAGENET_STATS
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = Day2NightDataset(dataset_dir, "train", transform=transform)
    testset = Day2NightDataset(dataset_dir, "test", transform=transform)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    return train_loader, test_loader


def prepare_tiny_imagenet(root="./data"):
    dataset_dir = os.path.join(root, "tiny-imagenet-200")
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not os.path.exists(dataset_dir):
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)

    val_dir = os.path.join(dataset_dir, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_images_dir) and os.path.exists(val_annotations):
        with open(val_annotations) as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name, class_name = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_images_dir, image_name)
                dst = os.path.join(class_dir, image_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
        if os.path.exists(val_images_dir) and not os.listdir(val_images_dir):
            shutil.rmtree(val_images_dir)

    return dataset_dir


def tiny_imagenet_loaders(
    root="./data",
    batch_size=64,
    num_workers=0,
    img_size=64,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
):
    tiny_dir = prepare_tiny_imagenet(root)
    mean, std = IMAGENET_STATS

    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
            Cutout(n_holes=1, length=32),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(os.path.join(tiny_dir, "train"), transform=transform_train)
    valset = torchvision.datasets.ImageFolder(os.path.join(tiny_dir, "val"), transform=transform_val)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        **get_dataloader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor),
    )
    return train_loader, val_loader
