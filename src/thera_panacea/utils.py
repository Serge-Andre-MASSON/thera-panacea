import torch
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_class_weights(df, device):
    class_weight_ = compute_class_weight(
        "balanced", classes=[0, 1], y=df["label"])
    return torch.FloatTensor(class_weight_).to(device)


def get_results(preds, labels):
    _, fp, fn, _ = confusion_matrix(labels, preds, normalize="true").ravel()
    hter = (fp + fn) / 2

    return {
        "hter": hter,
        "accuracy": accuracy_score(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(labels, preds)
    }


def display_results(results):
    for k, v in results.items():
        print(k)
        print(v)
        print("\n")
