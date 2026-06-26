"""PR curve / ROC curve / threshold sweep 그래프 저장.

Usage:
    python scripts/visualize_curves.py \
        --config configs/exp_v_bicubic.yaml \
        --checkpoint experiments/exp_v_bicubic/checkpoints/best_val_loss_ep010_0.312.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    resolve_checkpoint,
)
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation import report_artifacts as ra
from src.evaluation.evaluator import compute_metrics

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"

_DAS_LIMIT = 0.15


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PR curve / ROC curve / threshold sweep")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--thr-min", type=float, default=0.30)
    p.add_argument("--thr-max", type=float, default=0.65)
    p.add_argument("--thr-steps", type=int, default=36)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(f"experiments/{_DEFAULT_EXP}/visualizations"),
    )
    return p.parse_args()


def _plot_threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    thr_min: float,
    thr_max: float,
    thr_steps: int,
    out_path: Path,
) -> None:
    thresholds = np.linspace(thr_min, thr_max, thr_steps)
    y_true = labels.tolist()

    danger_prec: list[float] = []
    danger_rec: list[float] = []
    das_rates: list[float] = []
    f1_macros: list[float] = []

    for thr in thresholds:
        preds = apply_threshold(probs, float(thr))
        m = compute_metrics(y_true, preds)
        danger_prec.append(m["precision_danger"])
        danger_rec.append(m["recall_danger"])
        das_rates.append(m["danger_as_safe_rate"])
        f1_macros.append(m["f1_macro"])

    prec_arr, rec_arr = np.asarray(danger_prec), np.asarray(danger_rec)
    denom = prec_arr + rec_arr
    danger_f1 = np.divide(2 * prec_arr * rec_arr, denom, out=np.zeros_like(denom), where=denom > 0)
    mark_t, mark_label = ra.precision_recall_crossing(thresholds, prec_arr, rec_arr, danger_f1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(thresholds, danger_prec, color="#e74c3c", lw=2, label="danger precision")
    ax.plot(thresholds, danger_rec,  color="#e67e22", lw=2, label="danger recall",   linestyle="--")
    ax.plot(thresholds, das_rates,   color="#8e44ad", lw=2, label="danger-as-safe",  linestyle=":")
    ax.plot(thresholds, f1_macros,   color="#2980b9", lw=2, label="f1 macro",        linestyle="-.")
    ax.axhline(y=_DAS_LIMIT, color="gray", lw=1.2, linestyle="--",
               label=f"DAS limit={_DAS_LIMIT}")
    ax.axvline(x=mark_t, color="black", lw=1.4, linestyle="-", alpha=0.7, label=mark_label)

    ax.set_xlabel("Threshold (danger score)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Threshold Sweep (test set)", fontsize=12, fontweight="bold")
    ax.set_xlim(thr_min, thr_max)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[curves] config     : {args.config}")
    print(f"[curves] checkpoint : {ckpt.name}")
    print(f"[curves] device     : {device}")

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    model = build_model(cfg, ckpt, device)
    ds = HazardDataset("test", transform=get_val_transforms(image_size), **dataset_kwargs(d))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[curves] test n={len(ds)}, running inference …")

    probs, labels = collect_probs(model, loader, device)

    print("[curves] plotting PR curve …")
    ra.write_pr_curve(args.output_dir / "pr_curve.png", probs, labels)

    print("[curves] plotting ROC curve …")
    ra.write_roc_curve(args.output_dir / "roc_curve.png", probs, labels)

    print("[curves] plotting threshold sweep …")
    _plot_threshold_sweep(probs, labels, args.thr_min, args.thr_max,
                          args.thr_steps, args.output_dir / "threshold_sweep.png")

    print("[curves] done.")


if __name__ == "__main__":
    main()
