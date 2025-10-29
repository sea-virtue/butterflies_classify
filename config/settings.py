from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    base_dir: Path = Path("ButterflyClassificationDataset")
    processed_dir: Path = Path("dataset")
    train_dir: Path = processed_dir / "train"
    test_dir: Path = processed_dir / "test"
    img_size: Tuple[int, int] = (192, 192)
    val_split: float = 0.1
    test_split: float = 0.2
    batch_size: int = 32
    num_workers: int = 0  # use 0 on Windows unless spawn-safe guard is present
    augmentations_per_image: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 5e-4
    num_classes: int = 50
    checkpoint_name: str = "custom_cnn_best.pt"
    history_plot_name: str = "pytorch_cnn_training_history.png"


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "plateau"  # options: "none", "cosine", "step", "plateau"
    step_size: int = 10
    gamma: float = 0.1
    t_max: Optional[int] = None
    eta_min: float = 1e-5
    factor: float = 0.5
    patience: int = 4
    min_lr: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    artifacts_dir: Path = Path("artifacts")


CONFIG = ExperimentConfig()
