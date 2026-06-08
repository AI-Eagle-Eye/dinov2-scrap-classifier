"""Final test-set evaluation with val comparison.

Usage:
    python scripts/evaluate_test.py \\
        --config configs/exp_f_learnable_w.yaml \\
        --checkpoint experiments/exp_f_learnable_w/checkpoints/best_model.pth \\
        --threshold 0.30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    apply_threshold,
    build_model,
    collect_probs,
    detect_device,
    load_config,
)
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import compute_metrics

_CLASS_NAMES: list[str] = ["cut", "danger", "excluded"]
_GAP_WARN: float = 0.05  # val vs test 절대 차이 > 5%p 이면 경고


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final test-set evaluation")
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    p.add_argument("--checkpoint", type=Path, required=True, help="best_model.pth path")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Danger probability threshold applied to both val and test (default: 0.30)")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _print_metrics(metrics: dict[str, float], split: str) -> None:
    print(f"\n[{split}] Accuracy  : {metrics['accuracy']:.4f}")
    print(f"[{split}] F1 macro  : {metrics['f1_macro']:.4f}")
    print(f"[{split}] F2 macro  : {metrics['f2_macro']:.4f}")
    col = 12
    print(f"\n[{split}] Per-class:")
    print(f"  {'class':>{col}} | {'precision':>9} | {'recall':>6} | {'f1':>6}")
    print("  " + "-" * (col + 33))
    for cls in _CLASS_NAMES:
        p = metrics[f"precision_{cls}"]
        r = metrics[f"recall_{cls}"]
        f = metrics[f"f1_{cls}"]
        print(f"  {cls:>{col}} | {p * 100:>8.2f}% | {r * 100:>5.2f}% | {f * 100:>5.2f}%")
    print(f"\n[{split}] Danger-as-Safe Rate : {metrics['danger_as_safe_rate'] * 100:.2f}%")
    print(f"[{split}] Safe(cut) Precision  : {metrics['safe_precision'] * 100:.2f}%")


def _print_confusion_matrix(y_true: list[int], y_pred: list[int], split: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    col_w = 14
    label_w = 16
    print(f"\n[{split}] Confusion Matrix:")
    print(" " * label_w + "".join(f"pred_{c}".rjust(col_w) for c in _CLASS_NAMES))
    for i, name in enumerate(_CLASS_NAMES):
        row_label = f"actual_{name}".ljust(label_w)
        row_vals = "".join(str(cm[i, j]).rjust(col_w) for j in range(3))
        print(row_label + row_vals)


def _print_comparison(
    val_m: dict[str, float],
    test_m: dict[str, float],
    threshold: float,
) -> None:
    rows: list[tuple[str, str]] = [
        ("accuracy",            "Accuracy"),
        ("f1_macro",            "F1 macro"),
        ("safe_precision",      "Safe(cut) precision"),
        ("danger_as_safe_rate", "Danger-as-Safe Rate"),
        ("precision_danger",    "Danger precision"),
        ("recall_danger",       "Danger recall"),
    ]
    thr_label = f"{threshold:.2f}"
    print(f"\n{'Metric':<24} | {'val (thr=' + thr_label + ')':>17} | {'test (thr=' + thr_label + ')':>18} |")
    print("-" * 68)
    for key, label in rows:
        v = val_m[key]
        t = test_m[key]
        flag = "  <-- gap > 5%p" if abs(v - t) > _GAP_WARN else ""
        print(f"{label:<24} | {v * 100:>16.2f}% | {t * 100:>17.2f}% |{flag}")
    print("-" * 68)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")

    device = detect_device(args.device)
    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    num_workers: int = d.get("num_workers", 4)
    unk_label: str = d.get("unk_label", "excluded")

    print("=" * 68)
    print("  evaluate_test.py")
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {device}")
    print("=" * 68)

    print("\n[eval] Building model …")
    model = build_model(cfg, args.checkpoint, device)

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    # --- val 추론 (비교 기준) ---
    val_ds = HazardDataset("val", transform=get_val_transforms(image_size), unk_label=unk_label)
    val_loader: DataLoader = DataLoader(val_ds, **loader_kwargs)
    print(f"[eval] Val  samples : {len(val_ds)}")
    val_probs, val_labels = collect_probs(model, val_loader, device)
    val_preds = apply_threshold(val_probs, args.threshold)
    val_metrics = compute_metrics(val_labels.tolist(), val_preds)

    # --- test 추론 (최종 평가) ---
    test_ds = HazardDataset("test", transform=get_val_transforms(image_size), unk_label=unk_label)
    test_loader: DataLoader = DataLoader(test_ds, **loader_kwargs)
    print(f"[eval] Test samples : {len(test_ds)}")
    test_probs, test_labels = collect_probs(model, test_loader, device)
    test_preds = apply_threshold(test_probs, args.threshold)
    test_metrics = compute_metrics(test_labels.tolist(), test_preds)

    # ── val 결과 ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  VAL RESULTS  (참고용 — threshold 선택 기준)")
    print("=" * 68)
    _print_metrics(val_metrics, "val")
    _print_confusion_matrix(val_labels.tolist(), val_preds, "val")

    # ── test 결과 ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  TEST RESULTS  (최종 평가 — 배포 보고 지표)")
    print("=" * 68)
    _print_metrics(test_metrics, "test")
    _print_confusion_matrix(test_labels.tolist(), test_preds, "test")

    # ── val vs test 비교 ──────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  VAL vs TEST COMPARISON")
    print("=" * 68)
    _print_comparison(val_metrics, test_metrics, args.threshold)

    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
