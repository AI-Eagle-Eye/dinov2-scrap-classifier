"""AR(448)+TTA 모델의 attention map을 predictions.csv 기준으로 오버레이 저장.

predictions.csv(이미 TTA로 산출된 결정)에서 케이스별 top-confidence 샘플을 골라,
AttentionHead MHA의 attention을 TTA(원본+hflip 평균)로 추출해 jet heatmap으로 오버레이한다.
샘플 선택/표시 지표는 predictions.csv를 그대로 재사용하고(이미지는 image_path 기준 읽기 전용),
attention map만 모델로 새로 계산한다.

Usage:
    python scripts/visualize_attention_tta.py \
        --config configs/exp_ar_448.yaml \
        --checkpoint experiments/exp_ar_448/checkpoints/best_f1_ep014_0.8198.ckpt \
        --predictions results/exp_ar_448_tta/crops_25pct/predictions.csv \
        --output-dir results/exp_ar_448_tta/attention
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
try:
    matplotlib.rc("font", family="NanumGothic")
except Exception:
    matplotlib.rc("font", family="DejaVu Sans")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._eval_common import build_model, detect_device, load_config
from scripts.visualize_attention import _overlay
from src.data.transforms import get_val_transforms

_SAMPLES_PER_CASE = 10
_COLS = 5
_DPI = 150

# (파일명 stem, 그리드 제목, true_label, pred_label, 정렬 기준 prob 컬럼)
_CASES: list[tuple[str, str, str, str, str]] = [
    ("correct_danger", "Correct danger (T=danger, P=danger) — top confidence",
     "danger", "danger", "prob_danger"),
    ("missed_danger", "Missed danger→cut (T=danger, P=cut, 치명) — top confidence",
     "danger", "cut", "prob_cut"),
    ("correct_cut", "Correct cut (T=cut, P=cut) — top confidence",
     "cut", "cut", "prob_cut"),
]


@torch.inference_mode()
def _attn_score(
    model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device, patch_grid: int,
) -> np.ndarray:
    """AttentionHead MHA의 per-key importance를 [patch_grid, patch_grid]로 반환 (CLS 키 제거)."""
    captured: list[torch.Tensor] = []

    def _hook(_m: object, _i: object, output: object) -> None:
        if output[1] is not None:  # (attn_output, attn_weights [1, T, T])
            captured.append(output[1].cpu())

    hook = model.head.attn.register_forward_hook(_hook)
    try:
        model(img_tensor.unsqueeze(0).to(device))
    finally:
        hook.remove()

    n_patches = patch_grid * patch_grid
    if not captured:
        return np.zeros((patch_grid, patch_grid), dtype=np.float32)
    attn = captured[0][0]                  # [T, T], T = 1+N(use_cls) or N
    score = attn.mean(dim=0).numpy()       # 각 key가 받은 평균 attention [T]
    if score.shape[0] == n_patches + 1:    # use_cls 헤드: 맨 앞 CLS 키 제거
        score = score[1:]
    return score[:n_patches].reshape(patch_grid, patch_grid)


def _attn_map_tta(
    model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device, patch_grid: int,
) -> np.ndarray:
    """원본 + hflip attention 평균(heatmap을 되뒤집어 정렬) 후 0~1 정규화."""
    heat = _attn_score(model, img_tensor, device, patch_grid)
    heat_flip = _attn_score(model, torch.flip(img_tensor, dims=[2]), device, patch_grid)
    heat = (heat + np.fliplr(heat_flip)) / 2.0
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)


def _select(df: pd.DataFrame, true_label: str, pred_label: str, sort_col: str) -> pd.DataFrame:
    """predictions.csv에서 (true,pred) 케이스를 sort_col 내림차순 top-N으로 선택."""
    sub = df[(df["true_label"] == true_label) & (df["pred_label"] == pred_label)]
    return sub.sort_values(sort_col, ascending=False).head(_SAMPLES_PER_CASE)


def _save_grid(
    rows: pd.DataFrame, model: torch.nn.Module, device: torch.device,
    image_size: int, patch_grid: int, transform: object, title: str, out_path: Path,
) -> int:
    n = len(rows)
    if n == 0:
        print(f"  skip (no samples): {out_path.name}")
        return 0
    grid_rows = (n + _COLS - 1) // _COLS
    fig, axes = plt.subplots(grid_rows, _COLS, figsize=(_COLS * 2.8, grid_rows * 3.0), dpi=_DPI)
    axes_flat = np.atleast_1d(axes).flatten()
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue
        row = rows.iloc[idx]
        img_path = _PROJECT_ROOT / str(row["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            ax.axis("off")
            continue
        heat = _attn_map_tta(model, transform(img), device, patch_grid)
        ax.imshow(_overlay(img.resize((image_size, image_size)), heat))
        ax.axis("off")
        # predictions.csv의 결정/확률을 그대로 표시 (TTA 반영된 값)
        conf = float(row[f"prob_{row['pred_label']}"])
        ax.set_title(
            f"T={row['true_label']}  P={row['pred_label']}\nconf={conf:.2f}",
            fontsize=7, pad=2,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  ({n} samples)")
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR+TTA attention map overlay (predictions.csv 기준)")
    p.add_argument("--config", type=Path, default=Path("configs/exp_ar_448.yaml"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("experiments/exp_ar_448/checkpoints/best_f1_ep014_0.8198.ckpt"))
    p.add_argument("--predictions", type=Path,
                   default=Path("results/exp_ar_448_tta/crops_25pct/predictions.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exp_ar_448_tta/attention"))
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] checkpoint not found: {args.checkpoint}")
    if not args.predictions.exists():
        sys.exit(f"[ERROR] predictions.csv not found: {args.predictions}")

    device = detect_device(args.device)
    d = cfg["data"]
    image_size: int = d.get("image_size", 448)
    padding_color: str = d.get("padding_color", "mean")
    backbone_name: str = cfg["model"].get("backbone_name", "dinov2_vitb14")
    patch_grid = image_size // (14 if "14" in backbone_name else 16)

    print(f"[attn_tta] config     : {args.config}")
    print(f"[attn_tta] checkpoint : {args.checkpoint.name}")
    print(f"[attn_tta] predictions: {args.predictions}")
    print(f"[attn_tta] device={device}  res={image_size}  patch_grid={patch_grid}  TTA=True")

    model = build_model(cfg, args.checkpoint, device)
    transform = get_val_transforms(image_size, padding_color)
    df = pd.read_csv(args.predictions)

    total = 0
    for stem, title, true_label, pred_label, sort_col in _CASES:
        rows = _select(df, true_label, pred_label, sort_col)
        total += _save_grid(rows, model, device, image_size, patch_grid, transform,
                            title, args.output_dir / f"{stem}_grid.png")
    print(f"[attn_tta] done. ({total} overlays across {len(_CASES)} grids)")


if __name__ == "__main__":
    main()
