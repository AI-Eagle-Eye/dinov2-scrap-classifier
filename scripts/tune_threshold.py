"""Threshold tuning for best_model.pth on the val set.

Usage:
    python scripts/tune_threshold.py \\
        --config configs/exp_a_mlp.yaml \\
        --checkpoint experiments/exp_a_mlp/checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    apply_threshold,
    build_model,
    collect_probs,
    dataset_kwargs,
    detect_device,
    load_config,
)
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import compute_metrics
from src.evaluation.threshold import select_best_threshold

_DAS_LIMIT: float = 0.10  # danger_as_safe < 10%


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Val-set threshold tuning for danger class")
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    p.add_argument("--checkpoint", type=Path, required=True, help="best_model.pth path")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    start: float = 0.10,
    end: float = 0.90,
    step: float = 0.05,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for thr in np.arange(start, end + 1e-9, step):
        preds = apply_threshold(probs, float(thr))
        m = compute_metrics(labels.tolist(), preds)
        results.append({
            "threshold":        float(thr),
            "accuracy":         m["accuracy"],
            "danger_precision": m["precision_danger"],
            "danger_recall":    m["recall_danger"],
            "danger_as_safe":   m["danger_as_safe_rate"],
            "cut_precision":    m["precision_cut"],
        })
    return results


def _print_table(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'thresh':>6} | {'acc':>5} | {'danger_prec':>11} |"
        f" {'danger_recall':>13} | {'danger_as_safe':>14} | {'cut_prec':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        flag = " *" if r["danger_as_safe"] < _DAS_LIMIT else ""
        print(
            f"{r['threshold']:>6.2f} | {r['accuracy']:>5.3f} |"
            f" {r['danger_precision'] * 100:>10.2f}% |"
            f" {r['danger_recall'] * 100:>12.2f}% |"
            f" {r['danger_as_safe'] * 100:>13.2f}% |"
            f" {r['cut_precision'] * 100:>7.2f}%{flag}"
        )
    print(sep)
    print("  * danger_as_safe < 10% 조건 만족")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")

    device = detect_device(args.device)
    print(f"[tune] Device     : {device}")
    print(f"[tune] Config     : {args.config}")
    print(f"[tune] Checkpoint : {args.checkpoint}")

    print("[tune] Building model …")
    model = build_model(cfg, args.checkpoint, device)

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    padding_color: str = d.get("padding_color", "black")
    ds_kwargs = dataset_kwargs(d, use_eval_label_col=True)
    val_ds = HazardDataset("val", transform=get_val_transforms(image_size, padding_color), **ds_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    print(f"[tune] Val samples: {len(val_ds)}")

    print("[tune] Running inference …")
    probs, labels = collect_probs(model, val_loader, device)
    print(f"[tune] Collected probs: {probs.shape}, labels: {labels.shape}\n")

    print("[tune] Threshold sweep  (danger_prob >= threshold → danger, else argmax)\n")
    rows = _sweep(probs, labels)
    _print_table(rows)

    candidates = [r for r in rows if r["danger_as_safe"] < _DAS_LIMIT]
    print(f"\n[tune] danger_as_safe < {_DAS_LIMIT * 100:.0f}% 만족 threshold 목록:")
    for r in candidates:
        print(
            f"  thr={r['threshold']:.2f}  "
            f"danger_prec={r['danger_precision'] * 100:.2f}%  "
            f"danger_recall={r['danger_recall'] * 100:.2f}%  "
            f"danger_as_safe={r['danger_as_safe'] * 100:.2f}%"
        )
    if not candidates:
        print("  (조건 만족 threshold 없음 — danger_as_safe 최솟값 기준으로 대체 추천)")

    # 추천 thr: 평가 파이프라인과 동일한 단일 규칙 (das <= limit 하 danger precision 최대)
    best_thr = select_best_threshold(probs, labels, [r["threshold"] for r in rows], _DAS_LIMIT)
    best = next(r for r in rows if abs(r["threshold"] - best_thr) < 1e-9)
    print(
        f"\n[tune] 추천 threshold : {best['threshold']:.2f}"
        f"\n         danger_prec   = {best['danger_precision'] * 100:.2f}%"
        f"\n         danger_recall = {best['danger_recall'] * 100:.2f}%"
        f"\n         danger_as_safe= {best['danger_as_safe'] * 100:.2f}%"
        f"\n         accuracy      = {best['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
