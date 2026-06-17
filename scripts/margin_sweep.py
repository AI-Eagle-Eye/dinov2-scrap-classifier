"""실험 1-b: (danger_thr, margin) 2D sweep — 모호 danger 후보를 excluded로 라우팅.

결정 규칙 (per sample, p = softmax; cut=0, danger=1, excluded=2):
    if p_danger >= danger_thr:
        if (p_danger - p_cut) >= margin:  -> danger
        else:                              -> excluded   (모호 → 재검토)
    else:
        -> cut if p_cut >= p_excluded else excluded       (danger 후보 아님)

coverage = 자동판정 비율 = mean(pred != excluded)

Usage:
    python scripts/margin_sweep.py \\
        --config configs/exp_f_learnable_w.yaml \\
        --checkpoint experiments/exp_f_learnable_w/checkpoints/best_val_loss_ep007_0.1866.ckpt \\
        --split val \\
        --output experiments/exp_f_learnable_w/logs/margin_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    build_model,
    collect_probs,
    dataset_kwargs,
    detect_device,
    load_config,
)
from src.data.transforms import get_val_transforms
from src.data.dataset import HazardDataset
from src.evaluation.report_artifacts import (
    DANGER_THRS,
    MARGIN_SWEEP_FIELDS,
    MARGINS,
    margin_sweep_rows,
)

_DAS_LIMIT: float = 0.15


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="margin rule sweep (danger_thr x margin)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")
    device = detect_device(args.device)

    print("=" * 72)
    print("  margin_sweep.py  (실험 1-b)")
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint.name}")
    print(f"  Split      : {args.split}  (threshold 선택은 val에서 수행 권장)")
    print(f"  Device     : {device}")
    print("=" * 72)

    model = build_model(cfg, args.checkpoint, device)

    d = cfg["data"]
    ds = HazardDataset(
        args.split,
        transform=get_val_transforms(d.get("image_size", 336)),
        **dataset_kwargs(d),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda",
    )
    print(f"[sweep] {args.split} samples: {len(ds)}  추론 중 …")
    probs, labels = collect_probs(model, loader, device)

    rows = margin_sweep_rows(probs, labels, DANGER_THRS, MARGINS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MARGIN_SWEEP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] 저장: {args.output}  ({len(rows)} combos)")

    # 콘솔 표
    print(f"\n  {'d_thr':>5} | {'margin':>6} | {'d_prec':>7} | {'d_recall':>8} |"
          f" {'d_as_safe':>9} | {'coverage':>8}")
    print("  " + "-" * 60)
    for r in rows:
        flag = " *" if r["miss_rate_danger"] < _DAS_LIMIT else ""
        print(f"  {r['danger_thr']:>5.2f} | {r['margin']:>6.2f} |"
              f" {r['precision_danger']*100:>6.2f}% | {r['recall_danger']*100:>7.2f}% |"
              f" {r['miss_rate_danger']*100:>8.2f}% | {r['coverage']*100:>7.2f}%{flag}")
    print("  " + "-" * 60)
    print(f"  * danger_as_safe < {_DAS_LIMIT*100:.0f}% 만족")

    # 최적 조합: das<0.15 제약 하 danger_precision 최대
    feasible = [r for r in rows if r["miss_rate_danger"] < _DAS_LIMIT]
    if feasible:
        best = max(feasible, key=lambda r: r["precision_danger"])
        print(f"\n[sweep] 최적 조합 (das<{_DAS_LIMIT:.2f} 하 danger_precision 최대):")
        print(f"  danger_thr     = {best['danger_thr']:.2f}")
        print(f"  margin         = {best['margin']:.2f}")
        print(f"  danger_precision = {best['precision_danger']*100:.2f}%")
        print(f"  danger_recall    = {best['recall_danger']*100:.2f}%")
        print(f"  danger_as_safe   = {best['miss_rate_danger']*100:.2f}%")
        print(f"  coverage         = {best['coverage']*100:.2f}%")
        print(f"  f1_macro         = {best['f1_macro']*100:.2f}%  accuracy={best['accuracy']*100:.2f}%")
    else:
        print(f"\n[sweep] das<{_DAS_LIMIT:.2f}를 만족하는 조합 없음.")


if __name__ == "__main__":
    main()
