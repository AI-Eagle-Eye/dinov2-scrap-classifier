"""Attention map overlay: TP danger / FN (danger→cut) / FP (cut→danger) 8장씩 저장.

Usage:
    python scripts/visualize_attention.py \
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
try:
    matplotlib.rc("font", family="NanumGothic")
except Exception:
    matplotlib.rc("font", family="DejaVu Sans")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import (
    build_model,
    collect_cases,
    collect_probs,
    dataset_kwargs,
    detect_device,
    load_config,
    resolve_checkpoint,
)
from src.data.dataset import CLASS_NAMES, CUT, DANGER, HazardDataset
from src.data.transforms import get_val_transforms

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"
_SAMPLES_PER_CASE = 8
_COLS = 4
_DPI = 150
_ALPHA = 0.5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention map overlay visualization")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.30, help="파일명 태깅용 thr")
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(f"experiments/{_DEFAULT_EXP}/visualizations/attention_maps"),
    )
    return p.parse_args()


@torch.inference_mode()
def _get_attention_map(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    patch_grid: int,
) -> np.ndarray:
    """AttentionHead MHA의 attention weight를 공간 히트맵으로 반환 [H, W]."""
    captured: list[torch.Tensor] = []

    def _hook(_module: Any, _input: Any, output: Any) -> None:
        # output = (attn_output [1,N,D], attn_weights [1,N,N]) — need_weights=True default
        if output[1] is not None:
            captured.append(output[1].cpu())

    hook = model.head.attn.register_forward_hook(_hook)
    try:
        model(img_tensor.unsqueeze(0).to(device))
    finally:
        hook.remove()

    if not captured:
        return np.zeros((patch_grid, patch_grid), dtype=np.float32)

    attn = captured[0][0]  # [N, N]
    score = attn.mean(dim=0).numpy()  # mean over query → [N]
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score.reshape(patch_grid, patch_grid)


def _overlay(img: Image.Image, heat: np.ndarray, alpha: float = _ALPHA) -> np.ndarray:
    """PIL 이미지에 히트맵을 overlay한 numpy array 반환."""
    h, w = img.size[1], img.size[0]
    heat_pil = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat_rgb = np.array(cm.jet(np.array(heat_pil) / 255.0))[:, :, :3]
    img_arr = np.array(img).astype(np.float32) / 255.0
    blended = (1 - alpha) * img_arr + alpha * heat_rgb
    return np.clip(blended, 0, 1)


def _save_grid(
    cases: list[tuple[Path, int, int, np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    patch_grid: int,
    transform: Any,
    title: str,
    out_path: Path,
) -> None:
    n = len(cases)
    if n == 0:
        print(f"  skip (no samples): {out_path.name}")
        return
    rows = (n + _COLS - 1) // _COLS
    fig, axes = plt.subplots(rows, _COLS, figsize=(_COLS * 2.8, rows * 3.0), dpi=_DPI)
    if rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue
        img_path, true_idx, pred_idx, prob_vec = cases[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            ax.axis("off")
            continue
        img_tensor = transform(img)
        heat = _get_attention_map(model, img_tensor, device, patch_grid)
        overlay = _overlay(img.resize((image_size, image_size)), heat)
        ax.imshow(overlay)
        ax.axis("off")
        cut_p, dan_p, exc_p = prob_vec
        ax.set_title(
            f"T={CLASS_NAMES[true_idx]} P={CLASS_NAMES[pred_idx]}\n"
            f"cut={cut_p:.2f} dan={dan_p:.2f} exc={exc_p:.2f}",
            fontsize=6, pad=2,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  ({n} samples)")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[attn_viz] config     : {args.config}")
    print(f"[attn_viz] checkpoint : {ckpt.name}")
    print(f"[attn_viz] device     : {device}")

    d = cfg["data"]
    image_size: int = d.get("image_size", 336)
    backbone_name: str = cfg["model"].get("backbone_name", "dinov2_vitb14")
    patch_size = 14 if "14" in backbone_name else 16
    patch_grid = image_size // patch_size

    model = build_model(cfg, ckpt, device)
    transform = get_val_transforms(image_size)

    ds = HazardDataset("test", transform=transform, **dataset_kwargs(d))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=d.get("num_workers", 4), pin_memory=device.type == "cuda")
    print(f"[attn_viz] test n={len(ds)}")

    probs, _ = collect_probs(model, loader, device)

    exp_name: str = cfg["experiment"]["name"]
    tag = f"{exp_name}_thr{args.threshold:.2f}_{image_size}"
    cases_spec: list[tuple[str, str, int, int]] = [
        ("tp_danger",   "TP danger (정답 danger, 예측 danger)",  DANGER, DANGER),
        ("fn_danger2cut","FN danger→cut (danger 놓침, 최위험)", DANGER, CUT),
        ("fp_cut2danger","FP cut→danger (오경보)",              CUT,    DANGER),
    ]

    for kind, title, true_cls, pred_cls in cases_spec:
        cases = collect_cases(ds, probs, true_cls, pred_cls, _SAMPLES_PER_CASE)
        _save_grid(cases, model, device, image_size, patch_grid, transform,
                   title, args.output_dir / f"{kind}_{tag}.png")

    print("[attn_viz] done.")


if __name__ == "__main__":
    main()
