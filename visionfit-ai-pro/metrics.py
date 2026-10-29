"""Dependency-free classification reports: counts are from a held-out set."""
import numpy as np


def classification_report(y_true, y_pred, labels):
    n = len(labels)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError('y_true/y_pred must be same-length 1D arrays')
    if ((y_true < 0) | (y_true >= n) | (y_pred < 0) | (y_pred >= n)).any():
        raise ValueError('Labels out of range')
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    scores = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i]); support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {'precision': round(precision, 4), 'recall': round(recall, 4),
                         'f1': round(f1, 4), 'support': support}
    return {'accuracy': round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
            'macro_f1': round(float(np.mean([v['f1'] for v in scores.values()])), 4),
            'confusion_matrix': matrix.tolist(), 'matrix_rows': 'ground_truth',
            'matrix_columns': 'prediction', 'classes': labels, 'per_class': scores,
            'samples': len(y_true)}
