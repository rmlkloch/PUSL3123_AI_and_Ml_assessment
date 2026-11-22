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

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)`` configured with
        ``num_workers=2`` and ``pin_memory=True``.

    Raises:
        ValueError: If ``val_split`` is not strictly between 0 and 1.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}.")

    # 1. Define preprocessing: convert PIL images to [0, 1] tensors and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # 2. Load the full 50,000-image training set and separate 10,000-image test set
    full_train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    # 3. Compute split sizes (45,000 train / 5,000 val for default val_split=0.1)
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    # 4. Perform reproducible split with a fixed generator seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # 5. Build DataLoaders with num_workers=2 and pin_memory=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
