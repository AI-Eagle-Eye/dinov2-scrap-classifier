"""DINOv2 fine-tune 전후 CLS token 피처 분포 UMAP 비교.

before: 사전학습 DINOv2 그대로 (체크포인트 미적용)
after:  체크포인트에서 로드한 backbone CLS token

Usage:
    python scripts/visualize_umap.py \
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
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    build_model,
    dataset_kwargs,
    detect_device,
    load_config,
    resolve_checkpoint,
)
from src.data.dataset import CLASS_NAME_LIST, HazardDataset
from src.data.transforms import get_val_transforms
from src.evaluation.report_artifacts import CLASS_COLORS
from src.models.hazard_model import HazardModel

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"

_CLASS_COLORS = CLASS_COLORS
_CLASS_LABELS = CLASS_NAME_LIST


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UMAP before/after fine-tuning comparison")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(f"experiments/{_DEFAULT_EXP}/visualizations"),
    )
    return p.parse_args()


@torch.inference_mode()
def _extract_cls_tokens(
    model: HazardModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """backbone CLS token 추출 → (features [N, D], labels [N])."""
    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []
    for images, labels in loader:
        cls_token, _ = model.backbone(images.to(device))
        all_feats.append(cls_token.cpu().numpy())
        all_labels.extend(labels.tolist())
    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=int)


def _run_umap(feats: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        sys.exit("[ERROR] umap-learn 미설치. pip install umap-learn 실행 후 재시도.")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(feats)


def _scatter(ax: plt.Axes, emb: np.ndarray, labels: np.ndarray, title: str) -> None:
    for cls_idx, (color, label) in enumerate(zip(_CLASS_COLORS, _CLASS_LABELS)):
        mask = labels == cls_idx
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, label=f"{label} (n={mask.sum()})",
            s=8, alpha=0.6, linewidths=0,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[umap_viz] config     : {args.config}")
    print(f"[umap_viz] checkpoint : {ckpt.name}")
    print(f"[umap_viz] device     : {device}")

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    transform = get_val_transforms(image_size)
    ds = HazardDataset("test", transform=transform, **dataset_kwargs(d))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[umap_viz] test n={len(ds)}, extracting features …")

    print("[umap_viz] [1/2] pretrained (before) …")
    model_before = build_model(cfg, None, device)
    feats_before, labels = _extract_cls_tokens(model_before, loader, device)
    del model_before

    print("[umap_viz] [2/2] fine-tuned (after) …")
    model_after = build_model(cfg, ckpt, device)
    feats_after, _ = _extract_cls_tokens(model_after, loader, device)
    del model_after

    print("[umap_viz] running UMAP (before) …")
    emb_before = _run_umap(feats_before, args.umap_neighbors, args.umap_min_dist)
    print("[umap_viz] running UMAP (after) …")
    emb_after = _run_umap(feats_after, args.umap_neighbors, args.umap_min_dist)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    _scatter(ax1, emb_before, labels, "Before fine-tuning (pretrained DINOv2)")
    _scatter(ax2, emb_after, labels, f"After fine-tuning ({ckpt.name[:30]})")
    fig.suptitle("DINOv2 CLS Token Feature Distribution (UMAP)", fontsize=13)
    fig.tight_layout()

    out_path = args.output_dir / "umap_before_after.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[umap_viz] saved → {out_path}")
    print("[umap_viz] done.")


if __name__ == "__main__":
    main()
