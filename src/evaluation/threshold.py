from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..data.dataset import CUT, DANGER, EXCLUDED
from .evaluator import compute_metrics


def apply_danger_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """danger prob >= threshold → danger(1); 그 외 argmax. 평가 전반의 단일 결정 규칙."""
    preds = probs.argmax(axis=1)
    preds = np.where(probs[:, DANGER] >= threshold, DANGER, preds)
    return preds.astype(int)


def apply_margin_rule(probs: np.ndarray, danger_thr: float, margin: float) -> np.ndarray:
    """(danger_thr × margin) 2D decision layer — 모호 danger 후보를 excluded로 라우팅."""
    p_cut = probs[:, CUT]
    p_danger = probs[:, DANGER]
    p_excluded = probs[:, EXCLUDED]

    is_candidate = p_danger >= danger_thr
    is_confident = (p_danger - p_cut) >= margin

    preds = np.where(p_cut >= p_excluded, CUT, EXCLUDED)
    preds = np.where(is_candidate & ~is_confident, EXCLUDED, preds)
    preds = np.where(is_candidate & is_confident, DANGER, preds)
    return preds.astype(int)


def select_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Iterable[float],
    das_limit: float = 0.15,
    primary_metric: str = "precision_danger",
) -> float:
    """danger threshold sweep에서 das <= limit 제약 하 primary_metric 최대인 thr 선택.

    제약을 만족하는 thr이 없으면 danger_as_safe_rate가 가장 낮은 thr로 대체한다.
    평가/튜닝 스크립트의 단일 threshold 선택 규칙 (no-leakage: val에서만 호출).
    """
    y_true = labels.tolist()
    best_thr: float | None = None
    best_score = -1.0
    fallback_thr: float | None = None
    fallback_das = float("inf")
    for thr in thresholds:
        thr = float(thr)
        preds = apply_danger_threshold(probs, thr)
        m = compute_metrics(y_true, preds.tolist())
        das = m["danger_as_safe_rate"]
        if das < fallback_das:
            fallback_das, fallback_thr = das, thr
        if das <= das_limit and m[primary_metric] > best_score:
            best_score, best_thr = m[primary_metric], thr
    if best_thr is not None:
        return best_thr
    return float(fallback_thr) if fallback_thr is not None else 0.0
