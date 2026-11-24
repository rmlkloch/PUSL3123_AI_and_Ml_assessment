"""Dataset setup and verification utility for CIFAR-10 (Assessment 1).

Ensures the CIFAR-10 dataset is downloaded and verifies the train (45,000),
validation (5,000), and test (10,000) partition sizes and tensor dimensions.
"""

from data_loader import get_cifar10_loaders


def verify_cifar10_setup() -> None:
    """Download (if needed) and verify CIFAR-10 DataLoaders."""
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=64,
        val_split=0.1,
        seed=42,
    )
    images, labels = next(iter(train_loader))
    print("=== CIFAR-10 Dataset Verification ===")
    print(f"Training partition:   {len(train_loader.dataset):,} images ({len(train_loader)} batches)")
    print(f"Validation partition: {len(val_loader.dataset):,} images ({len(val_loader)} batches)")
    print(f"Test partition:       {len(test_loader.dataset):,} images ({len(test_loader)} batches)")
    print(f"Sample batch tensor:  shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"Sample label tensor:  shape={tuple(labels.shape)}, dtype={labels.dtype}")


if __name__ == "__main__":
    verify_cifar10_setup()
