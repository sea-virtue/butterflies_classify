from typing import Dict, List

import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, object]:
    """Run inference on the validation dataloader and gather metrics."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    true_tensor = torch.tensor(y_true)
    pred_tensor = torch.tensor(y_pred)
    accuracy = (pred_tensor == true_tensor).double().mean().item()

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": matrix,
        "targets": y_true,
        "predictions": y_pred,
    }
