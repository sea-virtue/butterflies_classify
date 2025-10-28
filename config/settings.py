from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    base_dir: Path = Path("ButterflyClassificationDataset")
    train_dir: Path = Path("ButterflyClassificationDataset") / "train"
    test_dir: Path = Path("ButterflyClassificationDataset") / "test"
    img_size: Tuple[int, int] = (128, 128)
    val_split: float = 0.1
    test_split: float = 0.2
    batch_size: int = 32
    num_workers: int = 0  # use 0 on Windows unless spawn-safe guard is present


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.001
    num_classes: int = 50
    checkpoint_name: str = "custom_cnn_best.pt"
    history_plot_name: str = "pytorch_cnn_training_history.png"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    artifacts_dir: Path = Path("artifacts")


CONFIG = ExperimentConfig()
