"""
data.py
=======
CIFAR-10 dataset download and loading (shared between training and CKA).
"""

from __future__ import annotations

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)


def get_transform() -> transforms.Compose:
    """Standard transform: ToTensor + CIFAR-10 normalization.

    Deliberately identical between training and CKA: intermediate
    activations directly depend on input scaling, so any normalization
    mismatch would invalidate the CKA comparison.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ]
    )


def download_cifar10(data_root: str | Path, url: str) -> None:
    """Download and extract CIFAR-10 if the folder doesn't already exist."""
    data_root = Path(data_root)
    if data_root.exists():
        logger.info("Dataset already present in '%s', skipping download.", data_root)
        return

    archive_name = data_root.parent / (data_root.name + ".tgz")
    logger.info("Downloading CIFAR-10 from %s ...", url)
    urllib.request.urlretrieve(url, archive_name)

    logger.info("Extracting archive...")
    with tarfile.open(archive_name, "r:gz") as tar:
        tar.extractall(path=data_root.parent)

    os.remove(archive_name)
    logger.info("Dataset ready in '%s'.", data_root)


def load_cifar10(
    data_root: str | Path,
    batch_size: int,
    shuffle_train: bool = True,
    shuffle_test: bool = True,
    drop_last_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, ImageFolder, ImageFolder]:
    """Load CIFAR-10 train/test DataLoaders with the standard normalization.

    Returns:
        (train_loader, test_loader, train_dataset, test_dataset)
    """
    data_root = Path(data_root)
    transform = get_transform()

    train_dataset = ImageFolder(root=str(data_root / "train"), transform=transform)
    test_dataset = ImageFolder(root=str(data_root / "test"), transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last_train,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
    )

    logger.info("CIFAR-10 loaded: %d train / %d test", len(train_dataset), len(test_dataset))
    return train_loader, test_loader, train_dataset, test_dataset


def set_seed(seed: int) -> None:
    """Fix all sources of randomness for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)