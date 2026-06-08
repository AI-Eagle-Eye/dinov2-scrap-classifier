"""실험 1-a: 임의 체크포인트(.ckpt/.pth) 재평가 + JSON 저장 + 문서 baseline 비교.

Usage:
    python scripts/eval_checkpoint.py \\
        --config configs/exp_f_learnable_w.yaml \\
        --checkpoint experiments/exp_f_learnable_w/checkpoints/best_val_loss_ep007_0.1866.ckpt \\
        --threshold 0.30 \\
        --output experiments/exp_f_learnable_w/logs/test_results_ep_best_valloss.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import apply_threshold, collect_probs, detect_device, load_config
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import compute_metrics
from src.models.hazard_model import HazardModel, ModelConfig, VPTConfig

_CLASS_NAMES: list[str] = ["cut", "danger", "excluded"]

# 문서 기준 baseline: 기존 best_model.pth (ep31) @ test, threshold=0.30
_DOC_BASELINE: dict[str, float] = {
    "accuracy": 0.6359,
    "danger_as_safe_rate": 0.0662,
    "precision_danger": 0.5192,
    "recall_danger": 0.8252,
    "safe_precision": 0.8954,
    "f1_macro": 0.6291,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Checkpoint re-evaluation (val+test)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _build_model_from_ckpt(
    cfg: dict[str, Any], ckpt_path: Path, device: torch.device
) -> tuple[HazardModel, int | None]:
    """전체 ckpt(dict) 또는 bare state_dict(.pth) 모두 로드."""
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
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        epoch = obj.get("epoch")
    else:
        state = obj
        epoch = None
    model.load_state_dict(state)
    model.to(device).eval()
    return model, epoch


def _eval_split(
    split: str, model: HazardModel, cfg: dict[str, Any],
    device: torch.device, batch_size: int, threshold: float,
) -> dict[str, float]:
    d = cfg["data"]
    ds = HazardDataset(
        split,
        transform=get_val_transforms(d.get("image_size", 336)),
        unk_label=d.get("unk_label", "excluded"),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda",
    )
    probs, labels = collect_probs(model, loader, device)
    preds = apply_threshold(probs, threshold)
    metrics = compute_metrics(labels.tolist(), preds)
    metrics["n_samples"] = float(len(ds))
    return metrics


def _print_comparison(test_m: dict[str, float], threshold: float) -> None:
    rows: list[tuple[str, str]] = [
        ("accuracy", "Accuracy"),
        ("danger_as_safe_rate", "Danger-as-Safe"),
        ("precision_danger", "Danger precision"),
        ("recall_danger", "Danger recall"),
        ("safe_precision", "Safe(cut) precision"),
        ("f1_macro", "F1 macro"),
    ]
    print(f"\n{'='*72}")
    print(f"  TEST 비교  (threshold={threshold:.2f})")
    print(f"{'='*72}")
    print(f"  {'Metric':<22} | {'ep31 (문서)':>12} | {'ep7 best_valloss':>17} | {'Δ':>8}")
    print("  " + "-" * 66)
    for key, label in rows:
        base = _DOC_BASELINE[key] * 100
        new = test_m[key] * 100
        delta = new - base
        print(f"  {label:<22} | {base:>11.2f}% | {new:>16.2f}% | {delta:>+7.2f}%")
    print("  " + "-" * 66)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")
    device = detect_device(args.device)

    print("=" * 72)
    print("  eval_checkpoint.py  (실험 1-a)")
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {args.checkpoint.name}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {device}")
    print("=" * 72)

    print("\n[eval] Building model …")
    model, epoch = _build_model_from_ckpt(cfg, args.checkpoint, device)
    print(f"[eval] Loaded checkpoint epoch = {epoch}")

    print("[eval] Evaluating val …")
    val_m = _eval_split("val", model, cfg, device, args.batch_size, args.threshold)
    print("[eval] Evaluating test …")
    test_m = _eval_split("test", model, cfg, device, args.batch_size, args.threshold)

    for split, mm in (("VAL", val_m), ("TEST", test_m)):
        print(f"\n[{split}] n={int(mm['n_samples'])} | acc={mm['accuracy']*100:.2f}% "
              f"| danger_as_safe={mm['danger_as_safe_rate']*100:.2f}% "
              f"| danger_prec={mm['precision_danger']*100:.2f}% "
              f"| danger_recall={mm['recall_danger']*100:.2f}% "
              f"| cut_prec={mm['safe_precision']*100:.2f}% "
              f"| f1_macro={mm['f1_macro']*100:.2f}%")

    _print_comparison(test_m, args.threshold)

    payload = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_epoch": epoch,
        "threshold": args.threshold,
        "decision_rule": "danger_prob >= threshold -> danger, else argmax",
        "val": val_m,
        "test": test_m,
        "baseline_ep31_doc": _DOC_BASELINE,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] 저장: {args.output}")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
