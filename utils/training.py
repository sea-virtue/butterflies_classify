from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    save_path: Optional[Path] = None,
    *,
    start_epoch: int = 0,
    total_epochs: Optional[int] = None,
    history: Optional[Dict[str, List[float]]] = None,
    best_val_acc: float = 0.0,
    best_weights: Optional[Dict[str, torch.Tensor]] = None,
    scheduler: Optional[Any] = None,
    scheduler_name: str = "none",
) -> Dict[str, List[float]]:
    """Train the model and return epoch-wise metrics."""
    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    else:
        history = {key: list(value) for key, value in history.items()}
        for required_key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            history.setdefault(required_key, [])
        history.setdefault("lr", [])

    if best_weights is None:
        best_weights = deepcopy(model.state_dict())
    else:
        best_weights = deepcopy(best_weights)

    current_best_acc = best_val_acc

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    target_epochs = total_epochs if total_epochs is not None else start_epoch + epochs

    for epoch_idx in range(start_epoch, start_epoch + epochs):
        print(f"Epoch {epoch_idx + 1}/{target_epochs}")
        print("-" * 10)

        val_metric: Optional[float] = None
        train_metric: Optional[float] = None

        for phase in ("train", "val"):
            if phase not in dataloaders:
                continue
            is_training = phase == "train"
            model.train(mode=is_training)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc.item() >= current_best_acc:
                current_best_acc = epoch_acc.item()
                best_weights = deepcopy(model.state_dict())

            if phase == "val":
                val_metric = epoch_loss
            elif phase == "train":
                train_metric = epoch_loss

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau) or scheduler_name.lower() == "plateau":
                metric = val_metric if val_metric is not None else train_metric
                if metric is not None:
                    scheduler.step(metric)
            else:
                scheduler.step()

        history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])

        if save_path is not None:
            checkpoint = {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": current_best_acc,
                "best_model": best_weights,
                "history": history,
                "scheduler_name": scheduler_name,
            }
            if scheduler is not None:
                checkpoint["scheduler"] = scheduler.state_dict()
            torch.save(checkpoint, save_path)
            print(f"[INFO] Saved checkpoint to {save_path}")

    if "val" not in dataloaders:
        best_weights = deepcopy(model.state_dict())
        current_best_acc = history["train_acc"][-1] if history["train_acc"] else 0.0

    if best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_val_acc"] = [current_best_acc]

    return history
