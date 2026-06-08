from __future__ import annotations

import numpy as np
import torch

from .evaluator import compute_metrics


def decide(
    probs: list[float] | torch.Tensor | np.ndarray,
    safe_thr: float = 0.70,
    danger_thr: float = 0.60,
    margin: float = 0.20,
) -> str:
    """단일 샘플 threshold 기반 decision layer (SPEC.md 정의).

    Class labels: cut(0), danger(1), excluded(2)
    MVP에서는 safe_thr만 sweep하고 danger_thr, margin은 고정.
    cut 판정은 고신뢰일 때만 허용 — 위험→cut 오류 비용이 가장 크다.
    """
    if isinstance(probs, (torch.Tensor, np.ndarray)):
        probs = probs.tolist()

    p_safe = float(probs[0])    # label 0 = cut
    p_danger = float(probs[1])  # label 1 = danger
    second = sorted(probs)[-2]
    safe_margin = p_safe - second

    if p_safe >= safe_thr and safe_margin >= margin:
        return "cut"
    if p_danger >= danger_thr:
        return "danger"
    return "excluded"


def sweep_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    start: float = 0.30,
    end: float = 0.90,
    step: float = 0.05,
    danger_thr: float = 0.60,
    margin: float = 0.20,
) -> list[dict[str, float]]:
    """safe_threshold를 sweep해 각 thr에서의 지표를 반환.

    Args:
        probs: [N, 3] softmax 확률
        labels: [N] 정수 레이블
        start/end/step: safe_thr 탐색 범위
    Returns:
        각 thr에서 {safe_thr, danger_as_safe_rate, safe_precision, coverage} 딕셔너리 목록
    """
    thresholds = np.arange(start, end + 1e-9, step)
    results = []

    for thr in thresholds:
        preds = [
            _label_index(decide(p, safe_thr=float(thr), danger_thr=danger_thr, margin=margin))
            for p in probs
        ]
        # coverage = 1 - 제외/재검토 비율
        coverage = float(np.mean(np.array(preds) != 2))
        metrics = compute_metrics(labels.tolist(), preds)
        results.append({
            "safe_thr": float(thr),
            "danger_as_safe_rate": metrics["danger_as_safe_rate"],
            "safe_precision": metrics["safe_precision"],
            "f1_macro": metrics["f1_macro"],
            "coverage": coverage,
        })

    return results


def _label_index(decision: str) -> int:
    mapping = {"cut": 0, "danger": 1, "excluded": 2}
    return mapping[decision]


def select_best_threshold(
    sweep_results: list[dict[str, float]],
    primary_metric: str = "safe_precision",
    danger_as_safe_limit: float = 0.05,
) -> dict[str, float]:
    """danger_as_safe_rate 제약 하에서 safe_precision을 최대화하는 thr 선택."""
    candidates = [r for r in sweep_results if r["danger_as_safe_rate"] <= danger_as_safe_limit]
    if not candidates:
        # 제약을 만족하는 thr이 없으면 danger_as_safe_rate가 가장 낮은 것 선택
        candidates = sorted(sweep_results, key=lambda r: r["danger_as_safe_rate"])[:1]
    return max(candidates, key=lambda r: r[primary_metric])
