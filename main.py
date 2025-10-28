from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import CONFIG
from utils import (
    build_model,
    create_dataloaders,
    plot_history,
    set_global_seed,
    train_model,
)


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
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=CONFIG.training.epochs,
        save_path=checkpoint_path,
    )

    best_val = max(history.get("val_acc", []) or [0.0])
    print(f"\n[INFO] 最佳验证集准确率: {best_val:.4f}")

    plot_path = plot_history(
        history,
        CONFIG.artifacts_dir / CONFIG.training.history_plot_name,
    )
    print(f"[INFO] 训练历史曲线图已保存为 {plot_path}")


if __name__ == "__main__":
    main()
