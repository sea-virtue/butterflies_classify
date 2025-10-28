import torch

from config import CONFIG
from utils import build_model, create_dataloaders, evaluate_model, set_global_seed


def main() -> None:
    set_global_seed(CONFIG.seed)

    dataloaders, class_names = create_dataloaders(CONFIG)
    if "test" not in dataloaders:
        raise ValueError(
            "Test dataloader not available. Run the dataset preparation script to "
            "create train/test splits before testing."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将使用 {device} 设备进行测试。")

    model = build_model(CONFIG.training.num_classes, CONFIG.dataset.img_size).to(device)

    checkpoint_path = CONFIG.artifacts_dir / CONFIG.training.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run main.py to train and save the model first."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    evaluation = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        class_names=class_names,
    )

    print("\n--- 测试集评估结果 ---")
    print(f"测试集准确率 (Accuracy): {evaluation['accuracy']:.4f}")
    print("\n--- 分类报告 ---")
    print(evaluation["report"])

    print("\n--- 混淆矩阵 (Confusion Matrix) ---")
    print(evaluation["confusion_matrix"])


if __name__ == "__main__":
    main()
