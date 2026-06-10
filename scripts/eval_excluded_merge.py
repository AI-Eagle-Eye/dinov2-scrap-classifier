"""Exp T 체크포인트로 excluded→danger 병합 후처리 전/후 test 지표 비교 (재학습 없음).

추론은 한 번만 수행하고, 동일한 예측에 두 가지 후처리를 적용해 비교한다.
  1. 3-class: cut/danger/excluded 그대로
  2. merged : 예측이 excluded면 danger로 변경 (모델 출력·threshold 불변)

Usage:
    python scripts/eval_excluded_merge.py \\
        --config configs/exp_t_eslfix.yaml \\
        --checkpoint experiments/exp_t_eslfix/checkpoints/best_val_loss_ep005_0.2865.ckpt \\
        --threshold 0.30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

_EXCLUDED_IDX: int = 2
_DANGER_IDX: int = 1

# 비교표에 출력할 지표: (metric_key, 표시 라벨)
_REPORT_METRICS: list[tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("precision_danger", "Danger precision"),
    ("recall_danger", "Danger recall"),
    ("danger_as_safe_rate", "Danger-as-Safe"),
    ("safe_precision", "Cut precision"),
    ("f1_macro", "F1 macro"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="excluded→danger 병합 후처리 전/후 비교")
    p.add_argument("--config", type=Path, default=Path("configs/exp_t_eslfix.yaml"))
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="미지정 시 experiments/<name>/checkpoints 에서 best_val_loss_*.ckpt 자동 탐색",
    )
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _resolve_checkpoint(cfg: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"[ERROR] checkpoint not found: {explicit}")
        return explicit
    ckpt_dir = _PROJECT_ROOT / "experiments" / cfg["experiment"]["name"] / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best_val_loss_*.ckpt"))
    if not candidates:
        sys.exit(f"[ERROR] best_val_loss_*.ckpt not found in {ckpt_dir}")
    return candidates[-1]


def _merge_excluded_to_danger(preds: list[int]) -> list[int]:
    """예측 라벨 중 excluded(2)를 danger(1)로 치환 (보수적 병합)."""
    return [_DANGER_IDX if p == _EXCLUDED_IDX else p for p in preds]


def _print_comparison(
    base: dict[str, float], merged: dict[str, float], threshold: float
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  TEST 비교: 3-class vs excluded→danger 병합  (threshold={threshold:.2f})")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<18} | {'3-class':>10} | {'merged':>10} | {'Δ':>9}")
    print("  " + "-" * 56)
    for key, label in _REPORT_METRICS:
        b = base[key] * 100
        m = merged[key] * 100
        print(f"  {label:<18} | {b:>9.2f}% | {m:>9.2f}% | {m - b:>+8.2f}%")
    print("  " + "-" * 56)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = _resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print("=" * 72)
    print("  eval_excluded_merge.py")
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {ckpt.name}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {device}")
    print("=" * 72)

    print("\n[eval] Building model …")
    model = build_model(cfg, ckpt, device)

    d = cfg["data"]
    ds_kwargs: dict = {
        "unk_label": d.get("unk_label", "excluded"),
        "label_col": d.get("label_col", "confirmed_label"),
        "split_col": d.get("split_col", "split"),
    }
    if "csv_path" in d:
        ds_kwargs["csv_path"] = Path(d["csv_path"])
    ds = HazardDataset(
        "test",
        transform=get_val_transforms(d.get("image_size", 336)),
        **ds_kwargs,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    print(f"[eval] Inferring test (n={len(ds)}) …")
    probs, labels = collect_probs(model, loader, device)
    y_true = labels.tolist()

    preds_3class = apply_threshold(probs, args.threshold)
    preds_merged = _merge_excluded_to_danger(preds_3class)

    metrics_3class = compute_metrics(y_true, preds_3class)
    metrics_merged = compute_metrics(y_true, preds_merged)

    _print_comparison(metrics_3class, metrics_merged, args.threshold)
    print("[eval] Done.")


if __name__ == "__main__":
    main()
