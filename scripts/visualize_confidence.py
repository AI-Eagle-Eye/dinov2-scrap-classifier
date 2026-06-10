"""Confidence 분포 히스토그램: 클래스별 danger 확률 분포 + 오분류 확신도 분포.

Usage:
    python scripts/visualize_confidence.py \
        --config configs/exp_v_bicubic.yaml \
        --checkpoint experiments/exp_v_bicubic/checkpoints/best_val_loss_ep010_0.312.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import apply_threshold, build_model, collect_probs, detect_device, load_config
from src.data.dataset import HazardDataset
from src.data.transforms import get_val_transforms

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"

_CLASS_NAMES = ["cut", "danger", "excluded"]
_CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db"]
_AMBIGUOUS_LO = 0.4
_AMBIGUOUS_HI = 0.6
_CONFIDENT_THR = 0.8

CUT, DANGER, EXCLUDED = 0, 1, 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confidence distribution visualization")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(f"experiments/{_DEFAULT_EXP}/visualizations"),
    )
    return p.parse_args()


def _resolve_checkpoint(cfg: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"[ERROR] checkpoint not found: {explicit}")
        return explicit
    ckpt_dir = _PROJECT_ROOT / "experiments" / cfg["experiment"]["name"] / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best_val_loss_*.ckpt"))
    if not candidates:
        sys.exit(f"[ERROR] best_val_loss_*.ckpt not found in {ckpt_dir}")
    return candidates[-1]


def _dataset_kwargs(d: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "unk_label": d.get("unk_label", "excluded"),
        "label_col": d.get("label_col", "confirmed_label"),
        "split_col": d.get("split_col", "split"),
    }
    if "csv_path" in d:
        kw["csv_path"] = Path(d["csv_path"])
    return kw


def _plot_class_danger_hist(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins: int,
    out_path: Path,
) -> None:
    """클래스별 danger 확률 분포 — 정답/오답 분리 (threshold 기반 판정)."""
    preds = np.array(apply_threshold(probs, threshold))
    danger_probs = probs[:, DANGER]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, sharey=False)
    fig.suptitle(
        f"Danger Score Distribution by True Class (correct vs incorrect, thr={threshold:.2f})",
        fontsize=12, fontweight="bold",
    )

    for cls_idx, (name, color) in enumerate(zip(_CLASS_NAMES, _CLASS_COLORS)):
        ax = axes[cls_idx]
        mask = labels == cls_idx
        correct_mask = mask & (preds == cls_idx)
        wrong_mask = mask & (preds != cls_idx)

        d_correct = danger_probs[correct_mask]
        d_wrong = danger_probs[wrong_mask]

        ax.hist(d_correct, bins=bins, range=(0, 1), color=color, alpha=0.7,
                label=f"correct (n={len(d_correct)})", density=False)
        ax.hist(d_wrong, bins=bins, range=(0, 1), color="gray", alpha=0.6,
                label=f"wrong   (n={len(d_wrong)})", density=False)
        ax.axvline(x=threshold, color="black", lw=1.5, linestyle="--",
                   alpha=0.8, label=f"thr={threshold:.2f}")
        ax.set_title(f"True class: {name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("danger score", fontsize=9)
        ax.set_ylabel("sample count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def _plot_misclassification_confidence(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins: int,
    out_path: Path,
) -> None:
    """오분류 샘플의 예측 확신도(max prob) 분포: 모호 vs 확신 비율 표시 (threshold 기반 판정)."""
    preds = np.array(apply_threshold(probs, threshold))
    wrong_mask = preds != labels
    wrong_probs = probs[wrong_mask]
    if len(wrong_probs) == 0:
        print("  skip confidence_misclassified: no wrong samples")
        return

    max_conf = wrong_probs.max(axis=1)  # 예측 클래스의 확률
    n_total = len(max_conf)
    n_ambig = ((max_conf >= _AMBIGUOUS_LO) & (max_conf <= _AMBIGUOUS_HI)).sum()
    n_conf = (max_conf > _CONFIDENT_THR).sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.suptitle("Misclassification Confidence Analysis", fontsize=12, fontweight="bold")

    # 왼쪽: max confidence 히스토그램
    ax = axes[0]
    ax.hist(max_conf, bins=bins, range=(0, 1), color="#e74c3c", alpha=0.75)
    ax.axvspan(_AMBIGUOUS_LO, _AMBIGUOUS_HI, color="orange", alpha=0.2,
               label=f"ambiguous [{_AMBIGUOUS_LO},{_AMBIGUOUS_HI}]  n={n_ambig} ({n_ambig/n_total*100:.1f}%)")
    ax.axvline(x=_CONFIDENT_THR, color="purple", lw=1.5, linestyle="--",
               label=f"confident >{_CONFIDENT_THR}  n={n_conf} ({n_conf/n_total*100:.1f}%)")
    ax.set_xlabel("Max prediction probability", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Wrong predictions confidence  (n={n_total})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 오른쪽: true class별 오분류 danger score 비교
    ax2 = axes[1]
    data_by_cls = [
        wrong_probs[labels[wrong_mask] == cls_idx, DANGER]
        for cls_idx in range(3)
    ]
    counts = [len(d) for d in data_by_cls]
    labels_bp = [f"{n}\n(n={c})" for n, c in zip(_CLASS_NAMES, counts)]

    bp = ax2.boxplot(
        [d for d in data_by_cls if len(d) > 0],
        tick_labels=[l for l, d in zip(labels_bp, data_by_cls) if len(d) > 0],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], [c for c, d in zip(_CLASS_COLORS, data_by_cls) if len(d) > 0]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.axhline(y=0.5, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Danger score", fontsize=10)
    ax2.set_title("Danger score of wrong samples\nby true class", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = _resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[confidence] config     : {args.config}")
    print(f"[confidence] checkpoint : {ckpt.name}")
    print(f"[confidence] threshold  : {args.threshold}")
    print(f"[confidence] device     : {device}")

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    model = build_model(cfg, ckpt, device)
    ds = HazardDataset("test", transform=get_val_transforms(image_size), **_dataset_kwargs(d))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[confidence] test n={len(ds)}, running inference …")

    probs, labels = collect_probs(model, loader, device)

    print("[confidence] plotting class danger distribution …")
    _plot_class_danger_hist(probs, labels, args.threshold, args.bins,
                            args.output_dir / "confidence_hist.png")

    print("[confidence] plotting misclassification confidence …")
    _plot_misclassification_confidence(probs, labels, args.threshold, args.bins,
                                       args.output_dir / "confidence_misclassified.png")

    print("[confidence] done.")


if __name__ == "__main__":
    main()
