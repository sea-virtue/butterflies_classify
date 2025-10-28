from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import ExperimentConfig


def _resolve_data_dir(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    project_root = Path(__file__).resolve().parents[1]
    return project_root / raw_path


def _build_transforms(img_size: Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }


def create_dataloaders(config: ExperimentConfig) -> Tuple[Dict[str, DataLoader], List[str]]:
    """Create training and validation dataloaders along with discovered class names."""
    train_dir = _resolve_data_dir(config.dataset.train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. "
            "Run the dataset preparation script to create train/test splits."
        )

    transforms_map = _build_transforms(
        config.dataset.img_size,
        config.normalize_mean,
        config.normalize_std,
    )

    base_dataset = datasets.ImageFolder(train_dir)
    num_samples = len(base_dataset)
    if num_samples == 0:
        raise ValueError(f"No samples found in training directory: {train_dir}")

    val_split = max(0.0, min(config.dataset.val_split, 0.5))
    val_count = int(val_split * num_samples)
    val_count = min(val_count, max(num_samples - 1, 0))

    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(num_samples, generator=generator)
    if val_count > 0:
        val_indices = permutation[:val_count].tolist()
        train_indices = permutation[val_count:].tolist()
    else:
        val_indices = []
        train_indices = permutation.tolist()

    train_dataset_raw = datasets.ImageFolder(train_dir, transform=transforms_map["train"])
    train_dataset = Subset(train_dataset_raw, train_indices)

    val_dataset = None
    if val_indices:
        val_dataset_raw = datasets.ImageFolder(train_dir, transform=transforms_map["val"])
        val_dataset = Subset(val_dataset_raw, val_indices)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    if val_dataset is not None:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_dir = _resolve_data_dir(config.dataset.test_dir)
    if test_dir.exists():
        test_dataset = datasets.ImageFolder(test_dir, transform=transforms_map["val"])
        if len(test_dataset) == 0:
            raise ValueError(f"No samples found in test directory: {test_dir}")
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders, base_dataset.classes
