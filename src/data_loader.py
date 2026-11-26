"""Reusable data loading utilities for the CIFAR-10 dataset.

Provides train, validation, and test PyTorch DataLoaders with standard
CIFAR-10 normalization and reproducible train/validation splitting.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Channel-wise mean and standard deviation computed over the CIFAR-10 training set
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
DATA_ROOT: str = "./data/CIFAR10"


def get_cifar10_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 and return train, validation, and test DataLoaders.

    Downloads the dataset into ``./data/CIFAR10`` if not already present,
    converts images to tensors, applies standard CIFAR-10 normalization, and
    splits the 50,000 training images into training and validation subsets
    using a fixed random generator seed for reproducibility. The 10,000 test
    images are kept completely separate.

    Args:
        batch_size: Number of samples per batch for all loaders.
        val_split: Fraction of the 50,000 training images to allocate to the
            validation set (e.g., 0.1 yields 45,000 train and 5,000 val).
        seed: Random seed used for ``torch.Generator`` during splitting.
        augment: If True, applies random horizontal flips and random crops
            (with 4-pixel reflective padding) to the training set for regularization.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)`` configured with
        ``num_workers=2`` and ``pin_memory=True``.

    Raises:
        ValueError: If ``val_split`` is not strictly between 0 and 1.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}.")

    # 1. Define evaluation transforms (deterministic normalization)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # 2. Define training transforms (with optional data augmentation)
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    else:
        train_transform = eval_transform

    # 3. Load CIFAR-10 datasets
    full_train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_val_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # 4. Compute split indices reproducibly
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_train, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_val_dataset, val_indices)

    # 5. Build DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_cifar10_loaders()
    images, labels = next(iter(train_dl))
    print(f"Train samples: {len(train_dl.dataset)}, batches: {len(train_dl)}")
    print(f"Val samples:   {len(val_dl.dataset)}, batches: {len(val_dl)}")
    print(f"Test samples:  {len(test_dl.dataset)}, batches: {len(test_dl)}")
    print(f"Batch image shape: {images.shape}, dtype: {images.dtype}")
    print(f"Batch label shape: {labels.shape}, dtype: {labels.dtype}")
