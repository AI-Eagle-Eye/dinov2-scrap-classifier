from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

_CLASS_NAMES = ["cut", "danger", "excluded"]


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """분류 지표 계산. 핵심 지표: danger_as_safe_rate, safe_precision.

    Class labels: cut(0), danger(1), excluded(2)
    """
    yt = np.array(y_true)
    yp = np.array(y_pred)

    prec_per: np.ndarray = precision_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
    rec_per: np.ndarray = recall_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per: np.ndarray = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
    f1_macro = float(f1_score(yt, yp, average="macro", zero_division=0))
    f2_macro = float(fbeta_score(yt, yp, beta=2, average="macro", zero_division=0))

    safe_prec = float(precision_score(yt, yp, labels=[0], average="micro", zero_division=0))

    danger_mask = yt == 1
    danger_as_safe = (
        float(((yp == 0) & danger_mask).sum() / danger_mask.sum())
        if danger_mask.sum() > 0 else 0.0
    )

    return {
        "precision_cut":      float(prec_per[0]),
        "precision_danger":   float(prec_per[1]),
        "precision_excluded": float(prec_per[2]),
        "recall_cut":         float(rec_per[0]),
        "recall_danger":      float(rec_per[1]),
        "recall_excluded":    float(rec_per[2]),
        "f1_cut":             float(f1_per[0]),
        "f1_danger":          float(f1_per[1]),
        "f1_excluded":        float(f1_per[2]),
        "f1_macro":           f1_macro,
        "f2_macro":           f2_macro,
        "safe_precision":     safe_prec,
        "danger_as_safe_rate": danger_as_safe,
        "accuracy":           float((yt == yp).mean()),
    }


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path,
) -> np.ndarray:
    """Save confusion matrix as CSV; returns the 3×3 array."""
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + _CLASS_NAMES)
        for name, row in zip(_CLASS_NAMES, cm):
            writer.writerow([name] + row.tolist())
    return cm
