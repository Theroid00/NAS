"""Sample-weighted classification summaries from a streaming confusion matrix."""
def classification_metrics(matrix):
    import numpy as np
    counts = np.asarray(matrix, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or (counts < 0).any():
        raise ValueError("Expected a square nonnegative confusion matrix")
    support = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    true_positive = counts.diagonal()
    recall = np.divide(true_positive, support, out=np.zeros(len(support)), where=support != 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(len(support)), where=predicted != 0)
    f1 = np.divide(2 * true_positive, support + predicted, out=np.zeros(len(support)), where=(support + predicted) != 0)
    present = support > 0
    if not present.any():
        raise ValueError("Cannot score empty predictions")
    return {"balanced_accuracy": float(recall[present].mean()), "macro_f1": float(f1.mean()),
            "per_class_recall": recall.tolist(), "per_class_precision": precision.tolist(),
            "per_class_f1": f1.tolist(), "class_support": support.tolist(),
            "confusion_matrix": counts.tolist(), "class_ids": list(range(len(support)))}
