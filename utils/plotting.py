from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], output_path: Path) -> Path:
    """Plot training history and persist the figure to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.get("train_acc", []), label="Train Accuracy")
    plt.plot(history.get("val_acc", []), label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
