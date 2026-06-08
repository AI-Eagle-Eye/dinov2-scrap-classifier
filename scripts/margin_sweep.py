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
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import collect_probs, detect_device, load_config
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import compute_metrics
from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig

_CUT, _DANGER, _EXCLUDED = 0, 1, 2
_DAS_LIMIT: float = 0.15

_DANGER_THRS: list[float] = [round(0.30 + 0.05 * i, 2) for i in range(7)]   # 0.30..0.60
_MARGINS: list[float] = [round(0.05 + 0.05 * i, 2) for i in range(6)]        # 0.05..0.30


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="margin rule sweep (danger_thr x margin)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _build_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> HazardModel:
    m = cfg["model"]
    vpt_raw = m.get("vpt", {})
    model_cfg = ModelConfig(
        backbone_name=m.get("backbone_name", "dinov2_vitb14"),
        head_type=m.get("head_type", "mlp"),
        vpt=VPTConfig(
            enabled=vpt_raw.get("enabled", False),
            num_tokens=vpt_raw.get("num_tokens", 10),
            insert_from_layer=vpt_raw.get("insert_from_layer", 0),
        ),
        dropout=m.get("dropout", 0.3),
        num_classes=m.get("num_classes", 3),
        use_grad_checkpoint=m.get("use_grad_checkpoint", True),
        class_aware_init_weights=m.get("class_aware_init_weights", None),
    )
    model = HazardModel(model_cfg)
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _apply_margin_rule(probs: np.ndarray, danger_thr: float, margin: float) -> np.ndarray:
    p_cut = probs[:, _CUT]
    p_danger = probs[:, _DANGER]
    p_excluded = probs[:, _EXCLUDED]

    is_candidate = p_danger >= danger_thr
    is_confident = (p_danger - p_cut) >= margin

    preds = np.where(p_cut >= p_excluded, _CUT, _EXCLUDED)  # 기본: danger 후보 아님
    preds = np.where(is_candidate & ~is_confident, _EXCLUDED, preds)  # 모호 → excluded
    preds = np.where(is_candidate & is_confident, _DANGER, preds)     # 확신 → danger
    return preds


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

    model = _build_model(cfg, args.checkpoint, device)

    d = cfg["data"]
    ds = HazardDataset(
        args.split,
        transform=get_val_transforms(d.get("image_size", 336)),
        unk_label=d.get("unk_label", "excluded"),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda",
    )
    print(f"[sweep] {args.split} samples: {len(ds)}  추론 중 …")
    probs, labels = collect_probs(model, loader, device)
    labels_list = labels.tolist()

    rows: list[dict[str, float]] = []
    for danger_thr in _DANGER_THRS:
        for margin in _MARGINS:
            preds = _apply_margin_rule(probs, danger_thr, margin)
            m = compute_metrics(labels_list, preds.tolist())
            coverage = float(np.mean(preds != _EXCLUDED))
            rows.append({
                "danger_thr": danger_thr,
                "margin": margin,
                "danger_precision": m["precision_danger"],
                "danger_recall": m["recall_danger"],
                "danger_as_safe": m["danger_as_safe_rate"],
                "coverage": coverage,
                "f1_macro": m["f1_macro"],
                "accuracy": m["accuracy"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["danger_thr", "margin", "danger_precision", "danger_recall",
              "danger_as_safe", "coverage", "f1_macro", "accuracy"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(r[k], 6) for k in fields})
    print(f"[sweep] 저장: {args.output}  ({len(rows)} combos)")

    # 콘솔 표
    print(f"\n  {'d_thr':>5} | {'margin':>6} | {'d_prec':>7} | {'d_recall':>8} |"
          f" {'d_as_safe':>9} | {'coverage':>8}")
    print("  " + "-" * 60)
    for r in rows:
        flag = " *" if r["danger_as_safe"] < _DAS_LIMIT else ""
        print(f"  {r['danger_thr']:>5.2f} | {r['margin']:>6.2f} |"
              f" {r['danger_precision']*100:>6.2f}% | {r['danger_recall']*100:>7.2f}% |"
              f" {r['danger_as_safe']*100:>8.2f}% | {r['coverage']*100:>7.2f}%{flag}")
    print("  " + "-" * 60)
    print(f"  * danger_as_safe < {_DAS_LIMIT*100:.0f}% 만족")

    # 최적 조합: das<0.15 제약 하 danger_precision 최대
    feasible = [r for r in rows if r["danger_as_safe"] < _DAS_LIMIT]
    if feasible:
        best = max(feasible, key=lambda r: r["danger_precision"])
        print(f"\n[sweep] 최적 조합 (das<{_DAS_LIMIT:.2f} 하 danger_precision 최대):")
        print(f"  danger_thr     = {best['danger_thr']:.2f}")
        print(f"  margin         = {best['margin']:.2f}")
        print(f"  danger_precision = {best['danger_precision']*100:.2f}%")
        print(f"  danger_recall    = {best['danger_recall']*100:.2f}%")
        print(f"  danger_as_safe   = {best['danger_as_safe']*100:.2f}%")
        print(f"  coverage         = {best['coverage']*100:.2f}%")
        print(f"  f1_macro         = {best['f1_macro']*100:.2f}%  accuracy={best['accuracy']*100:.2f}%")
    else:
        print(f"\n[sweep] das<{_DAS_LIMIT:.2f}를 만족하는 조합 없음.")


if __name__ == "__main__":
    main()
