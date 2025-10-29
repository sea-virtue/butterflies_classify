from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from config import CONFIG
from utils import (
    build_model,
    create_dataloaders,
    plot_history,
    set_global_seed,
    train_model,
)


def _create_scheduler(optimizer: optim.Optimizer, total_epochs: int):
    cfg = CONFIG.scheduler
    name = cfg.name.lower()

    if name in {"none", ""}:
        return None, "none"

    if name == "cosine":
        t_max = cfg.t_max or max(total_epochs, 1)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cfg.eta_min), name

    if name == "step":
        step_size = max(cfg.step_size, 1)
        return StepLR(optimizer, step_size=step_size, gamma=cfg.gamma), name

    if name == "plateau":
        return (
            ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr,
            ),
            name,
        )

    raise ValueError(f"Unsupported scheduler type: {cfg.name}")


def main() -> None:
    set_global_seed(CONFIG.seed)

    dataloaders, class_names = create_dataloaders(CONFIG)
    print(f"检测到的类别 (共 {len(class_names)} 类): {class_names[:5]}...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将使用 {device} 设备进行训练。")

    model = build_model(CONFIG.training.num_classes, CONFIG.dataset.img_size).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)

    checkpoint_path = CONFIG.artifacts_dir / CONFIG.training.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    history_state = None
    best_val_acc = 0.0
    best_weights = None

    scheduler, scheduler_name = _create_scheduler(optimizer, CONFIG.training.epochs)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model" in checkpoint and "optimizer" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", -1) + 1
            history_state = checkpoint.get("history")
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            best_weights = checkpoint.get("best_model")
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"[INFO] 从 checkpoint 恢复，起始 epoch: {start_epoch}")
        else:
            print("[WARN] 检测到旧格式权重文件，将重新开始训练并覆盖。")
            if isinstance(checkpoint, dict):
                try:
                    model.load_state_dict(checkpoint)
                    best_weights = checkpoint
                except RuntimeError as exc:
                    print(f"[WARN] 无法加载旧权重: {exc}")
                    best_weights = None
            else:
                best_weights = None

    total_epochs = CONFIG.training.epochs
    remaining_epochs = max(total_epochs - start_epoch, 0)

    if remaining_epochs == 0:
        print("[INFO] 目标 epoch 已完成，跳过训练。")
        history = history_state or {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "best_val_acc": [best_val_acc],
        }
        if best_weights is not None:
            model.load_state_dict(best_weights)
    else:
        history = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=remaining_epochs,
            save_path=checkpoint_path,
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            history=history_state,
            best_val_acc=best_val_acc,
            best_weights=best_weights,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
        )

    best_val_history = history.get("best_val_acc", [])
    best_val = best_val_history[-1] if best_val_history else max(history.get("val_acc", []) or [0.0])
    print(f"\n[INFO] 最佳验证集准确率: {best_val:.4f}")

    plot_path = plot_history(
        history,
        CONFIG.artifacts_dir / CONFIG.training.history_plot_name,
    )
    print(f"[INFO] 训练历史曲线图已保存为 {plot_path}")


if __name__ == "__main__":
    main()
