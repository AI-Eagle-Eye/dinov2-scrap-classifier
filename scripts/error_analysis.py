"""Error analysis: danger↔cut 오분류 시각화.

Usage:
    python scripts/error_analysis.py \\
        --config configs/exp_f_learnable_w.yaml \\
        --checkpoint experiments/exp_f_learnable_w/checkpoints/best_val_loss_ep007_0.1866.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rc("font", family="NanumGothic")
matplotlib.rc("axes", unicode_minus=False)
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import build_model, collect_probs, detect_device, load_config
from src.data.dataset import CLASS_NAMES, CUT, DANGER, EXCLUDED, HazardDataset
from src.data.transforms import get_val_transforms

_GRID_COLS = 4
_GRID_ROWS = 5
_MAX_SAMPLES = _GRID_COLS * _GRID_ROWS  # 20
_DPI = 150
_TILE_SIZE = 2.4  # inches per tile


class ErrorSample(NamedTuple):
    img_path: Path
    true_idx: int
    pred_idx: int
    probs: np.ndarray  # shape [3] — cut, danger, excluded
    margin: float      # prob[pred] - prob[true]; 작을수록 모호


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error analysis: danger↔cut 오분류 시각화")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/exp_f_learnable_w/error_analysis"),
    )
    return p.parse_args()


def _collect_errors(
    dataset: HazardDataset,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    true_cls: int,
    pred_cls: int,
) -> list[ErrorSample]:
    """true_cls → pred_cls 오분류 샘플 수집 후 margin 오름차순 정렬."""
    samples: list[ErrorSample] = []
    labels = dataset.labels
    paths = [p for p, _ in dataset._samples]

    for i, (true, pred) in enumerate(zip(labels, all_preds)):
        if true == true_cls and pred == pred_cls:
            probs = all_probs[i]
            margin = float(probs[pred_cls] - probs[true_cls])
            samples.append(ErrorSample(paths[i], true, pred, probs, margin))

    samples.sort(key=lambda s: s.margin)
    return samples[:_MAX_SAMPLES]


def _render_grid(
    samples: list[ErrorSample],
    case_label: str,
    out_path: Path,
) -> None:
    """20개 타일 그리드 PNG 저장."""
    n = len(samples)
    rows = min(_GRID_ROWS, (n + _GRID_COLS - 1) // _GRID_COLS)
    fig_w = _TILE_SIZE * _GRID_COLS
    fig_h = _TILE_SIZE * rows + 0.6  # 상단 타이틀 여백
    fig, axes = plt.subplots(rows, _GRID_COLS, figsize=(fig_w, fig_h), dpi=_DPI)

    # axes를 항상 2D 배열로 처리
    if rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    fig.suptitle(case_label, fontsize=11, fontweight="bold", y=0.995)

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n:
            ax.axis("off")
            continue
        s = samples[ax_idx]
        try:
            img = Image.open(s.img_path).convert("RGB")
        except OSError:
            ax.axis("off")
            ax.set_title("load error", fontsize=6)
            continue

        ax.imshow(img)
        ax.axis("off")

        cut_p, dan_p, exc_p = s.probs
        true_name = CLASS_NAMES[s.true_idx]
        pred_name = CLASS_NAMES[s.pred_idx]
        fname_short = s.img_path.name[:28] + ("…" if len(s.img_path.name) > 28 else "")
        title_lines = [
            fname_short,
            f"true={true_name}  pred={pred_name}",
            f"cut={cut_p:.2f}  dan={dan_p:.2f}  exc={exc_p:.2f}",
        ]
        ax.set_title("\n".join(title_lines), fontsize=5.5, pad=2, linespacing=1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  ({n} samples)")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")

    device = detect_device(args.device)
    print(f"[analysis] device     : {device}")
    print(f"[analysis] checkpoint : {args.checkpoint}")

    print("[analysis] building model …")
    model = build_model(cfg, args.checkpoint, device)

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    unk_label: str = d.get("unk_label", "excluded")

    test_ds = HazardDataset("test", transform=get_val_transforms(image_size), unk_label=unk_label)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    print(f"[analysis] test samples: {len(test_ds)}")

    print("[analysis] running inference …")
    all_probs, _ = collect_probs(model, loader, device)
    all_preds = all_probs.argmax(axis=1)

    cases: list[tuple[str, str, int, int]] = [
        ("case_a_danger_as_safe.png",    "Case A: 실제 danger → 예측 cut  (가장 위험)",     DANGER,   CUT),
        ("case_b_cut_as_danger.png",     "Case B: 실제 cut → 예측 danger  (FP, precision killer)", CUT, DANGER),
        ("case_c_danger_as_excluded.png","Case C: 실제 danger → 예측 excluded",              DANGER,   EXCLUDED),
        ("case_d_excluded_as_danger.png","Case D: 실제 excluded → 예측 danger",              EXCLUDED, DANGER),
    ]

    print(f"\n[analysis] output dir : {args.output_dir}")
    for fname, label, true_cls, pred_cls in cases:
        samples = _collect_errors(test_ds, all_probs, all_preds, true_cls, pred_cls)
        print(f"  {label[:60]:<60}  n={len(samples):>3}", end="")
        if not samples:
            print("  (skip — no samples)")
            continue
        print()
        _render_grid(samples, label, args.output_dir / fname)

    print("\n[analysis] done.")


if __name__ == "__main__":
    main()
