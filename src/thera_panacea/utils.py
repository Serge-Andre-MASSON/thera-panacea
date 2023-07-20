import torch
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_class_weights(df, device):
    """Return the classe weight in order to handle unbalancing."""
    class_weight_ = compute_class_weight(
        "balanced", classes=[0, 1], y=df["label"])
    return torch.FloatTensor(class_weight_).to(device)


def get_results(preds, labels):
    """Return a dict containing a metrics summary of the results of a model."""
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
