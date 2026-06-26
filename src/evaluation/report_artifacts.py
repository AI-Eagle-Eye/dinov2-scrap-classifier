"""평가 산출물 writer/plotter 모음.

evaluate_test.py가 타겟별로 호출한다. 코드 내부 지표 키는 그대로 두고, 표시명 변환이
필요한 출력(metrics/summary/merged)에서만 호출 측이 넘긴 display_map을 적용한다.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.dataset import CLASS_NAME_LIST, CUT, DANGER, EXCLUDED
from .calibration import _DEFAULT_ECE_BINS, TemperatureScaler, compute_ece, confidence_bin_masks
from .evaluator import compute_metrics, save_confusion_matrix
from .threshold import apply_danger_threshold, apply_margin_rule

_CLASS_NAMES: list[str] = CLASS_NAME_LIST
CLASS_COLORS: list[str] = ["#2ecc71", "#e74c3c", "#3498db"]
_EXCLUDED_IDX: int = EXCLUDED
_RELIABILITY_BINS: int = _DEFAULT_ECE_BINS

# selective-prediction sweep 격자 (confidence = max softmax prob)
RISK_COVERAGE_THRS: np.ndarray = np.round(np.linspace(0.0, 1.0, 101), 4)

# Cost-weighted 기대비용 placeholder (행=true, 열=pred). 인자로 주입 가능.
# danger→cut(치명)=100, cut→danger(오경보)=1, excluded→*(보수적/무해)=0, 정답=0.
DEFAULT_COST_MATRIX: list[list[float]] = [
    [0.0, 1.0, 0.0],    # true=cut
    [100.0, 0.0, 0.0],  # true=danger
    [0.0, 0.0, 0.0],    # true=excluded
]
_BOOTSTRAP_SEED: int = 42

# 'all' 타겟 등에서 confusion/sweep을 돌릴 때 쓰는 표준 sweep 격자
DANGER_THRS: list[float] = [round(0.30 + 0.05 * i, 2) for i in range(7)]   # 0.30..0.60
MARGINS: list[float] = [round(0.05 + 0.05 * i, 2) for i in range(6)]        # 0.05..0.30


def coverage_of(preds: np.ndarray) -> float:
    """자동판정 비율 = mean(pred != excluded)."""
    return float(np.mean(np.asarray(preds) != _EXCLUDED_IDX))


def support_counts(labels: np.ndarray) -> dict[str, int]:
    """클래스별 정답 개수 {cut, danger, excluded}."""
    counts = Counter(int(x) for x in labels)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(_CLASS_NAMES)}


# ── predictions / metrics ──────────────────────────────────────────────────

def write_predictions_csv(
    out_path: Path,
    paths: list[Path],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
    crop_pcts: list[int],
    applied_thr: float,
    applied_thr_source: str,
) -> None:
    """샘플별 확률/logit/crop_pct/적용 threshold(+선택 출처) 기록."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path", "true_label", "pred_label",
        "prob_cut", "prob_danger", "prob_excluded",
        "logit_cut", "logit_danger", "logit_excluded",
        "crop_pct", "applied_thr", "applied_thr_source",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(paths)):
            w.writerow({
                "image_path": str(paths[i]),
                "true_label": _CLASS_NAMES[int(labels[i])],
                "pred_label": _CLASS_NAMES[int(preds[i])],
                "prob_cut": round(float(probs[i, 0]), 6),
                "prob_danger": round(float(probs[i, 1]), 6),
                "prob_excluded": round(float(probs[i, 2]), 6),
                "logit_cut": round(float(logits[i, 0]), 6),
                "logit_danger": round(float(logits[i, 1]), 6),
                "logit_excluded": round(float(logits[i, 2]), 6),
                "crop_pct": crop_pcts[i],
                "applied_thr": round(float(applied_thr), 4),
                "applied_thr_source": applied_thr_source,
            })


def write_metrics_csv(
    out_path: Path,
    metrics: dict[str, float],
    support: dict[str, int],
    coverage: float,
    display_map: dict[str, str],
) -> None:
    """per-class(precision/recall/f1 + support) + 전역 지표를 표시명으로 기록."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "n_support"])
        for cls in _CLASS_NAMES:
            w.writerow([
                cls,
                round(metrics[f"precision_{cls}"], 6),
                round(metrics[f"recall_{cls}"], 6),
                round(metrics[f"f1_{cls}"], 6),
                support[cls],
            ])
        w.writerow([])
        w.writerow(["metric", "value"])
        global_keys = ["accuracy", "f1_macro", "f2_macro", "danger_as_safe_rate"]
        for key in global_keys:
            w.writerow([display_map.get(key, key), round(metrics[key], 6)])
        w.writerow([display_map.get("coverage", "coverage"), round(coverage, 6)])


# ── threshold / margin sweep ────────────────────────────────────────────────

def write_threshold_sweep_csv(
    out_path: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    """thr별 danger precision/recall/miss-rate + macro F1/F2 + coverage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = labels.tolist()
    fields = ["thr", "precision_danger", "recall_danger", "miss_rate_danger",
              "f1_macro", "f2_macro", "coverage"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for thr in thresholds:
            preds = apply_danger_threshold(probs, float(thr))
            m = compute_metrics(y_true, preds.tolist())
            w.writerow({
                "thr": round(float(thr), 4),
                "precision_danger": round(m["precision_danger"], 6),
                "recall_danger": round(m["recall_danger"], 6),
                "miss_rate_danger": round(m["danger_as_safe_rate"], 6),
                "f1_macro": round(m["f1_macro"], 6),
                "f2_macro": round(m["f2_macro"], 6),
                "coverage": round(coverage_of(preds), 6),
            })


MARGIN_SWEEP_FIELDS: list[str] = [
    "danger_thr", "margin", "precision_danger", "recall_danger",
    "miss_rate_danger", "coverage", "f1_macro", "accuracy",
]


def margin_sweep_rows(
    probs: np.ndarray,
    labels: np.ndarray,
    danger_thrs: list[float] = DANGER_THRS,
    margins: list[float] = MARGINS,
) -> list[dict[str, float]]:
    """(danger_thr × margin) 2D decision layer sweep 결과 행 목록 (CSV writer와 공유)."""
    y_true = labels.tolist()
    rows: list[dict[str, float]] = []
    for danger_thr in danger_thrs:
        for margin in margins:
            preds = apply_margin_rule(probs, danger_thr, margin)
            m = compute_metrics(y_true, preds.tolist())
            rows.append({
                "danger_thr": round(danger_thr, 6),
                "margin": round(margin, 6),
                "precision_danger": round(m["precision_danger"], 6),
                "recall_danger": round(m["recall_danger"], 6),
                "miss_rate_danger": round(m["danger_as_safe_rate"], 6),
                "coverage": round(coverage_of(preds), 6),
                "f1_macro": round(m["f1_macro"], 6),
                "accuracy": round(m["accuracy"], 6),
            })
    return rows


def write_margin_sweep_csv(
    out_path: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    danger_thrs: list[float] = DANGER_THRS,
    margins: list[float] = MARGINS,
) -> None:
    """(danger_thr × margin) 2D decision layer sweep CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = margin_sweep_rows(probs, labels, danger_thrs, margins)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MARGIN_SWEEP_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ── confusion matrix / curves ───────────────────────────────────────────────

def write_confusion_matrix(
    out_csv: Path,
    out_png: Path,
    labels: np.ndarray,
    preds: np.ndarray,
) -> None:
    """raw count CSV + 3×3 heatmap PNG."""
    cm = save_confusion_matrix(labels.tolist(), preds.tolist(), out_csv)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [f"pred_{c}" for c in _CLASS_NAMES])
    ax.set_yticks(range(3), [f"true_{c}" for c in _CLASS_NAMES])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    ax.set_title("Confusion Matrix (count)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_pr_curve(out_png: Path, probs: np.ndarray, labels: np.ndarray) -> None:
    """3-class overlay PR curve."""
    from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: PLC0415
    from sklearn.preprocessing import label_binarize  # noqa: PLC0415

    y_bin = label_binarize(labels, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for idx, (name, color) in enumerate(zip(_CLASS_NAMES, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, idx], probs[:, idx])
        ap = average_precision_score(y_bin[:, idx], probs[:, idx])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name}  AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_roc_curve(out_png: Path, probs: np.ndarray, labels: np.ndarray) -> None:
    """3-class overlay ROC curve (+ micro/macro avg)."""
    from sklearn.metrics import auc, roc_auc_score, roc_curve  # noqa: PLC0415
    from sklearn.preprocessing import label_binarize  # noqa: PLC0415

    y_bin = label_binarize(labels, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for idx, (name, color) in enumerate(zip(_CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], probs[:, idx])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc(fpr, tpr):.3f}")
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    ax.plot(fpr_micro, tpr_micro, "k--", lw=1.5,
            label=f"micro-avg  AUC={auc(fpr_micro, tpr_micro):.3f}")
    auc_macro = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    ax.plot([], [], "k:", lw=1.5, label=f"macro-avg  AUC={auc_macro:.3f}")
    ax.plot([0, 1], [0, 1], "gray", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── calibration ─────────────────────────────────────────────────────────────

def _reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin_center, bin_accuracy, bin_count). 빈 bin은 accuracy NaN."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges, masks = confidence_bin_masks(conf, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    count = np.array([int(m.sum()) for m in masks])
    for b, m in enumerate(masks):
        if count[b] > 0:
            acc[b] = float(correct[m].mean())
    return centers, acc, count


def write_calibration(
    out_json: Path,
    out_png: Path,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    target_logits: np.ndarray,
    target_probs: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, float]:
    """val에서 temperature T를 적합(데이터 누수 방지)하고 타겟의 ECE 전/후 + reliability 저장."""
    scaler = TemperatureScaler()
    temperature = scaler.fit(torch.from_numpy(val_logits).float(),
                             torch.from_numpy(val_labels).long())
    calib_probs = scaler.calibrate(torch.from_numpy(target_logits).float()).numpy()

    ece_before = compute_ece(target_probs, target_labels, _RELIABILITY_BINS)
    ece_after = compute_ece(calib_probs, target_labels, _RELIABILITY_BINS)
    payload = {
        "temperature": round(temperature, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "n_bins": _RELIABILITY_BINS,
        "temperature_fit_on": "val",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    centers, acc_before, _ = _reliability_bins(target_probs, target_labels, _RELIABILITY_BINS)
    _, acc_after, _ = _reliability_bins(calib_probs, target_labels, _RELIABILITY_BINS)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "gray", linestyle="--", lw=1, label="perfect")
    ax.plot(centers, acc_before, "o-", color="#e67e22", lw=2,
            label=f"before  ECE={ece_before:.3f}")
    ax.plot(centers, acc_after, "s-", color="#2980b9", lw=2,
            label=f"after (T={temperature:.2f})  ECE={ece_after:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return payload


# ── excluded → danger 병합 ───────────────────────────────────────────────────

def write_merged_metrics_csv(
    out_path: Path,
    labels: np.ndarray,
    preds: np.ndarray,
    display_map: dict[str, str],
) -> None:
    """excluded(2)→danger(1) 보수 병합 전/후 핵심 지표 비교."""
    before = compute_metrics(labels.tolist(), preds.tolist())
    merged_labels = np.where(labels == _EXCLUDED_IDX, 1, labels)
    merged_preds = np.where(np.asarray(preds) == _EXCLUDED_IDX, 1, np.asarray(preds))
    after = compute_metrics(merged_labels.tolist(), merged_preds.tolist())

    keys = ["accuracy", "f1_macro", "f2_macro", "danger_as_safe_rate",
            "precision_danger", "recall_danger"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before_merge", "after_merge"])
        for key in keys:
            w.writerow([display_map.get(key, key),
                        round(before[key], 6), round(after[key], 6)])


# ── 실무/납품 지표 ────────────────────────────────────────────────────────────

def _danger_prec_miss(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """compute_metrics와 동일 정의의 (precision_danger, danger_as_safe_rate) 직접 계산."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    pred_d = preds == DANGER
    true_d = labels == DANGER
    prec = float((pred_d & true_d).sum() / pred_d.sum()) if pred_d.sum() > 0 else 0.0
    miss = float((true_d & (preds == CUT)).sum() / true_d.sum()) if true_d.sum() > 0 else 0.0
    return prec, miss


def _risk_coverage_rows(probs: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    """danger prob >= thr → danger로 보는 이진(danger vs 그 외) threshold sweep.

    coverage         = danger로 판정한 비율 (전체 샘플 중)
    precision_danger = 판정된 danger 중 실제 danger 비율
    miss_rate        = 실제 danger 중 놓친 비율 (= 1 - recall)
    """
    p_danger = probs[:, DANGER]
    true_d = np.asarray(labels) == DANGER
    n = len(labels)
    n_true_d = int(true_d.sum())
    rows: list[dict[str, float]] = []
    for thr in RISK_COVERAGE_THRS:
        pred_d = p_danger >= thr
        n_pred = int(pred_d.sum())
        tp = int((pred_d & true_d).sum())
        prec = tp / n_pred if n_pred > 0 else 0.0
        miss = (n_true_d - tp) / n_true_d if n_true_d > 0 else 0.0
        rows.append({
            "threshold": round(float(thr), 4),
            "coverage": round(n_pred / n, 6),
            "precision_danger": round(prec, 6),
            "miss_rate": round(miss, 6),
        })
    return rows


def write_risk_coverage_curve(
    out_csv: Path,
    out_png: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    applied_thr: float | None = None,
) -> None:
    """danger 판정 비율(coverage) ↔ Precision(Danger)/Miss Rate(Danger) tradeoff 곡선.

    applied_thr: 현재 운영 threshold. 주어지면 해당 coverage 위치에 수직선을 그린다.
    """
    rows = _risk_coverage_rows(probs, labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["threshold", "coverage", "precision_danger", "miss_rate"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ordered = sorted(rows, key=lambda r: r["coverage"])  # x축(coverage) 단조 증가로 정렬
    cov = [r["coverage"] for r in ordered]
    prec = [r["precision_danger"] for r in ordered]
    miss = [r["miss_rate"] for r in ordered]

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()
    (l1,) = ax1.plot(cov, prec, "-", color="#2ecc71", lw=2, label="Precision (Danger)")
    (l2,) = ax2.plot(cov, miss, "--", color="#e74c3c", lw=2, label="Miss Rate (Danger)")
    ax1.set_xlabel("Coverage (danger-predicted rate)")
    ax1.set_ylabel("Precision (Danger)", color="#2ecc71")
    ax2.set_ylabel("Miss Rate (Danger)", color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    handles = [l1, l2]
    if applied_thr is not None:
        cov_at = float((probs[:, DANGER] >= applied_thr).mean())
        handles.append(ax1.axvline(
            cov_at, color="gray", ls=":", lw=1.5,
            label=f"op thr={applied_thr:.2f} (cov={cov_at:.2f})"))
    ax1.set_title("Risk-Coverage Curve", fontsize=12, fontweight="bold")
    ax1.legend(handles=handles, fontsize=9, loc="lower center")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def precision_recall_crossing(
    thr: np.ndarray, prec: np.ndarray, rec: np.ndarray, f1: np.ndarray,
) -> tuple[float, str]:
    """precision==recall 교차 threshold를 선형보간으로 탐지; 없으면 best-F1로 폴백.

    skew된 타겟에서 best-F1이 무의미한 가장자리(thr≈0)에 박히는 문제를 피하려고,
    sweep 곡선이 교차하는 균형 운영점을 우선 보고한다(시각화 모듈 공유용 pure 함수).
    """
    thr = np.asarray(thr, dtype=float)
    diff = np.asarray(prec, dtype=float) - np.asarray(rec, dtype=float)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) > 0:
        i = int(sign_change[0])
        t0, t1, d0, d1 = thr[i], thr[i + 1], diff[i], diff[i + 1]
        t_cross = t0 + (t1 - t0) * (-d0) / (d1 - d0) if d1 != d0 else t0
        return float(t_cross), f"precision=recall 교차점 thr={t_cross:.2f}"
    best_t = float(thr[int(np.argmax(f1))])
    return best_t, f"best-F1 thr={best_t:.2f} (교차점 없음)"


def danger_risk_coverage_arrays(
    probs: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """danger threshold sweep → (coverage, precision_danger, miss_rate), coverage 오름차순.

    write_risk_coverage_curve와 동일한 _risk_coverage_rows 정의를 재사용한다(시각화 모듈 공유용).
    """
    rows = [r for r in _risk_coverage_rows(probs, labels) if r["coverage"] > 0]
    rows.sort(key=lambda r: r["coverage"])
    cov = np.array([r["coverage"] for r in rows])
    prec = np.array([r["precision_danger"] for r in rows])
    miss = np.array([r["miss_rate"] for r in rows])
    return cov, prec, miss


def _select_operating_point(
    rows: list[dict[str, float]], constraint_key: str, op: str, limit: float,
    objective_key: str,
) -> dict[str, float] | None:
    """제약(op: 'le'면 <=limit, 'ge'면 >=limit) 만족 행 중 objective_key 최대 선택."""
    feasible = [
        r for r in rows
        if (r[constraint_key] <= limit if op == "le" else r[constraint_key] >= limit)
    ]
    return max(feasible, key=lambda r: r[objective_key]) if feasible else None


def write_operating_points(out_csv: Path, probs: np.ndarray, labels: np.ndarray) -> None:
    """운영점 3종: (a) PrecD>=95% max cov, (b) PrecD>=90% max cov, (c) Miss<=10% max PrecD."""
    rows = _risk_coverage_rows(probs, labels)
    specs = [
        ("precision_danger>=95%", "precision_danger", "ge", 0.95, "coverage"),
        ("precision_danger>=90%", "precision_danger", "ge", 0.90, "coverage"),
        ("miss_rate<=10%", "miss_rate", "le", 0.10, "precision_danger"),
    ]
    fields = ["operating_point", "threshold", "coverage", "precision_danger", "miss_rate"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, ckey, op, limit, objective in specs:
            r = _select_operating_point(rows, ckey, op, limit, objective)
            if r is None:
                w.writerow({"operating_point": name})
                continue
            w.writerow({
                "operating_point": name,
                "threshold": r["threshold"],
                "coverage": r["coverage"],
                "precision_danger": r["precision_danger"],
                "miss_rate": r["miss_rate"],
            })


def write_cost_matrix(
    out_csv: Path,
    labels: np.ndarray,
    preds: np.ndarray,
    cost_matrix: list[list[float]] | np.ndarray | None = None,
) -> None:
    """Cost-weighted 기대비용 = Σ(confusion_count × cost_matrix) + per-image 정규화."""
    cost = np.asarray(cost_matrix if cost_matrix is not None else DEFAULT_COST_MATRIX, dtype=float)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    cmat = np.zeros((3, 3), dtype=int)
    for t in range(3):
        for p in range(3):
            cmat[t, p] = int(((labels == t) & (preds == p)).sum())
    n = int(len(labels))
    total_cost = float((cmat * cost).sum())
    per_image = total_cost / n if n > 0 else 0.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cost_matrix (true\\pred)", *_CLASS_NAMES])
        for i, name in enumerate(_CLASS_NAMES):
            w.writerow([name, *[cost[i, j] for j in range(3)]])
        w.writerow([])
        w.writerow(["confusion_count (true\\pred)", *_CLASS_NAMES])
        for i, name in enumerate(_CLASS_NAMES):
            w.writerow([name, *[int(cmat[i, j]) for j in range(3)]])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["total_expected_cost", round(total_cost, 6)])
        w.writerow(["per_image_expected_cost", round(per_image, 6)])
        w.writerow(["n_samples", n])


def write_bootstrap_ci(
    metrics_path: Path,
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Precision(Danger)/Miss Rate의 95% bootstrap CI를 metrics.csv metric 섹션에 append."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    precs = np.empty(n_bootstrap)
    misses = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        precs[b], misses[b] = _danger_prec_miss(labels[idx], preds[idx])
    ci = {
        "precision_danger_ci_low": float(np.percentile(precs, 2.5)),
        "precision_danger_ci_high": float(np.percentile(precs, 97.5)),
        "miss_rate_ci_low": float(np.percentile(misses, 2.5)),
        "miss_rate_ci_high": float(np.percentile(misses, 97.5)),
    }
    # write_metrics_csv가 만든 metric 섹션 끝에 행 추가 (시그니처 불변 유지)
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for key, value in ci.items():
            w.writerow([key, round(value, 6)])
    return ci


def write_confidence_histogram(out_png: Path, probs: np.ndarray, labels: np.ndarray) -> None:
    """max softmax prob 분포를 클래스(3) × 정답/오답(2) 3×2 subplot으로 시각화."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    labels = np.asarray(labels)
    correct = pred == labels
    bins = np.linspace(0.0, 1.0, 21)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=150, sharex=True)
    for i, name in enumerate(_CLASS_NAMES):
        cls_mask = labels == i
        for j, (sub, title, color) in enumerate([
            (cls_mask & correct, "correct", "#2ecc71"),
            (cls_mask & ~correct, "incorrect", "#e74c3c"),
        ]):
            ax = axes[i, j]
            ax.hist(conf[sub], bins=bins, color=color, alpha=0.8)
            ax.set_title(f"{name} — {title} (n={int(sub.sum())})", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel("Max softmax prob (confidence)")
            if j == 0:
                ax.set_ylabel("Count")
    fig.suptitle("Confidence Distribution (class × correctness)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── summary ──────────────────────────────────────────────────────────────────

SUMMARY_FIELDS: list[str] = [
    "exp_name", "eval_target", "resolution", "checkpoint",
    "accuracy", "precision_danger", "recall_danger", "miss_rate_danger",
    "precision_cut", "recall_cut", "f1_macro", "f2_macro", "coverage",
    "n_cut", "n_danger", "n_excluded", "applied_thr",
]


def build_summary_row(
    exp_name: str,
    eval_target: str,
    resolution: int,
    checkpoint: str,
    metrics: dict[str, float],
    coverage: float,
    support: dict[str, int],
    applied_thr: float,
) -> dict[str, Any]:
    return {
        "exp_name": exp_name,
        "eval_target": eval_target,
        "resolution": resolution,
        "checkpoint": checkpoint,
        "accuracy": round(metrics["accuracy"], 6),
        "precision_danger": round(metrics["precision_danger"], 6),
        "recall_danger": round(metrics["recall_danger"], 6),
        "miss_rate_danger": round(metrics["danger_as_safe_rate"], 6),
        "precision_cut": round(metrics["precision_cut"], 6),
        "recall_cut": round(metrics["recall_cut"], 6),
        "f1_macro": round(metrics["f1_macro"], 6),
        "f2_macro": round(metrics["f2_macro"], 6),
        "coverage": round(coverage, 6),
        "n_cut": support["cut"],
        "n_danger": support["danger"],
        "n_excluded": support["excluded"],
        "applied_thr": round(applied_thr, 4),
    }


def write_summary_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _read_applied_thr(predictions_path: Path, fallback: float | None) -> float:
    """predictions.csv 첫 행의 applied_thr; 없으면 fallback(또는 0.0)."""
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                return float(row["applied_thr"])
    return float(fallback) if fallback is not None else 0.0


def read_summary_row_from_metrics(
    target_dir: Path,
    exp_name: str,
    eval_target: str,
    resolution: int,
    checkpoint: str,
    display_map: dict[str, str],
    applied_thr_fallback: float | None = None,
) -> dict[str, Any]:
    """스킵된 타겟의 기존 metrics.csv(+predictions.csv)로 summary row 재구성."""
    with (target_dir / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    per_class: dict[str, dict[str, float]] = {}
    glob_by_display: dict[str, float] = {}
    section = "class"
    for row in rows:
        if not row:
            continue
        if row[0] == "class":
            continue
        if row[0] == "metric":
            section = "metric"
            continue
        if section == "class":
            per_class[row[0]] = {
                "precision": float(row[1]),
                "recall": float(row[2]),
                "n_support": int(float(row[4])),
            }
        else:
            glob_by_display[row[0]] = float(row[1])

    def g(key: str) -> float:
        return glob_by_display[display_map.get(key, key)]

    applied_thr = _read_applied_thr(target_dir / "predictions.csv", applied_thr_fallback)
    return {
        "exp_name": exp_name,
        "eval_target": eval_target,
        "resolution": resolution,
        "checkpoint": checkpoint,
        "accuracy": round(g("accuracy"), 6),
        "precision_danger": per_class["danger"]["precision"],
        "recall_danger": per_class["danger"]["recall"],
        "miss_rate_danger": round(g("danger_as_safe_rate"), 6),
        "precision_cut": per_class["cut"]["precision"],
        "recall_cut": per_class["cut"]["recall"],
        "f1_macro": round(g("f1_macro"), 6),
        "f2_macro": round(g("f2_macro"), 6),
        "coverage": round(g("coverage"), 6),
        "n_cut": per_class["cut"]["n_support"],
        "n_danger": per_class["danger"]["n_support"],
        "n_excluded": per_class["excluded"]["n_support"],
        "applied_thr": round(applied_thr, 4),
    }
