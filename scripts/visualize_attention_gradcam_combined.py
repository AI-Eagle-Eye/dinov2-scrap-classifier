"""같은 이미지에 [원본 | attention | Grad-CAM] 을 나란히 비교하는 통합 시각화.

기존 visualize_attention.py / visualize_gradcam.py 는 그대로 두고(별도 PNG),
이 스크립트는 두 맵을 한 그림에서 직접 비교한다. heat-map 계산은 기존 함수를
재사용한다(_get_attention_map, _gradcam) — 중복 구현 없음.

케이스 선택:
  - 기본: 표준 3그룹(TP danger / FN danger→cut / FP cut→danger) 각 top-N
  - --filenames a.jpg,b.jpg : 해당 파일명만으로 그리드 구성(클래스 무관).
    steel의 name.jpg@crops_Npct 형식도 받지만, hazard는 단일 crops 폴더라
    @crop 접미사는 무시하고 파일명(basename)으로만 매칭한다.

Grad-CAM 타깃은 각 이미지의 예측 클래스(argmax)다("왜 이렇게 예측했나").

Usage:
    python scripts/visualize_attention_gradcam_combined.py \
        --config configs/exp_v_bicubic.yaml \
        --checkpoint experiments/exp_v_bicubic/checkpoints/best_val_loss_ep010_0.312.ckpt
    python scripts/visualize_attention_gradcam_combined.py \
        --config configs/exp_v_bicubic.yaml --filenames danger_001.jpg,cut_042.jpg
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
from scripts.visualize_attention import _get_attention_map  # heat-map 재사용 (중복 구현 없음)
from scripts.visualize_gradcam import _gradcam, _overlay  # heat-map/overlay 재사용 (중복 구현 없음)
from src.data.dataset import CLASS_NAMES, CUT, DANGER, HazardDataset
from src.data.transforms import get_val_transforms

_DEFAULT_CONFIG = Path("configs/exp_v_bicubic.yaml")
_DEFAULT_EXP = "exp_v_bicubic"
_PER_CASE = 6
_DPI = 150
_ATTN_ALPHA = 0.5
_GC_ALPHA = 0.55
_DANGER_COLOR = "#d62728"

# (파일 접두, 제목, true_cls, pred_cls)
_GROUPS: list[tuple[str, str, int, int]] = [
    ("tp_danger", "TP danger (T=danger, P=danger)", DANGER, DANGER),
    ("fn_danger2cut", "FN danger→cut (T=danger, P=cut)", DANGER, CUT),
    ("fp_cut2danger", "FP cut→danger (T=cut, P=danger)", CUT, DANGER),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="attention·Grad-CAM 나란히 비교")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.30, help="파일명 태깅용 thr")
    p.add_argument("--per-case", type=int, default=_PER_CASE, help="그룹별 표시 샘플 수")
    p.add_argument("--filenames", default=None,
                   help="쉼표 구분 파일명(예: 'danger_001.jpg,cut_042.jpg'). 지정 시 "
                        "표준 그룹 대신 해당 파일만으로 그리드 구성. name.jpg@crops_Npct "
                        "형식도 받지만 @crop 접미사는 무시한다(단일 crops 폴더).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(f"experiments/{_DEFAULT_EXP}/visualizations/attn_gradcam_combined"))
    return p.parse_args()


def _parse_filenames(raw: str) -> list[str]:
    """'a.jpg,b.jpg@crops_25pct' → ['a.jpg', 'b.jpg'] (@crop 접미사 제거)."""
    names: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            names.append(tok.partition("@")[0].strip())
    return names


def _render_combined(
    cases: list[tuple[Path, int, int, np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    patch_grid: int,
    transform: Any,
    title: str,
    out_path: Path,
) -> Path | None:
    """행=샘플, 열=[원본 | attention | Grad-CAM] 비교 그리드."""
    rows: list[tuple[Image.Image, np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for img_path, true_idx, pred_idx, prob_vec in cases:
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            continue
        img_tensor = transform(img)
        attn = _get_attention_map(model, img_tensor, device, patch_grid)
        gcam = _gradcam(model, img_tensor, pred_idx, device, patch_grid)  # 타깃 = 예측 클래스
        rows.append((img, attn, gcam, true_idx, pred_idx, prob_vec))
    if not rows:
        print(f"  skip (no images): {out_path.name}")
        return None

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 2.9, n * 3.1), dpi=_DPI, squeeze=False)
    fig.suptitle(f"{title}  —  원본 | attention | Grad-CAM(pred)", fontsize=11, fontweight="bold")
    col_titles = ["원본", "attention", "Grad-CAM"]
    for r, (img, attn, gcam, true_idx, pred_idx, prob_vec) in enumerate(rows):
        base = np.array(img.resize((image_size, image_size))).astype(np.float32) / 255.0
        panels = [base, _overlay(img, attn, image_size, _ATTN_ALPHA),
                  _overlay(img, gcam, image_size, _GC_ALPHA)]
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
        cut_p, dan_p, exc_p = prob_vec
        wrong = true_idx != pred_idx
        axes[r][0].set_ylabel(
            f"T={CLASS_NAMES[true_idx]} P={CLASS_NAMES[pred_idx]}\n"
            f"cut={cut_p:.2f} dan={dan_p:.2f} exc={exc_p:.2f}",
            fontsize=7, color=_DANGER_COLOR if wrong else "black")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}  ({n} samples)")
    return out_path


def _cases_from_filenames(
    ds: HazardDataset, probs: np.ndarray, names: list[str],
) -> list[tuple[Path, int, int, np.ndarray]]:
    """파일명(basename) 매칭으로 케이스 구성. 못 찾은 파일은 경고 후 건너뜀."""
    by_name: dict[str, int] = {p.name: i for i, (p, _) in enumerate(ds._samples)}
    labels = ds.labels
    cases: list[tuple[Path, int, int, np.ndarray]] = []
    for name in names:
        idx = by_name.get(name)
        if idx is None:
            print(f"[warn] test split에서 파일을 찾지 못함: {name}")
            continue
        path = ds._samples[idx][0]
        cases.append((path, int(labels[idx]), int(probs[idx].argmax()), probs[idx]))
    return cases


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    device = detect_device(args.device)

    print(f"[combined] config     : {args.config}")
    print(f"[combined] checkpoint : {ckpt.name}")
    print(f"[combined] device     : {device}")

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
    print(f"[combined] test n={len(ds)}")
    probs, _ = collect_probs(model, loader, device)

    exp_name: str = cfg["experiment"]["name"]
    tag = f"{exp_name}_thr{args.threshold:.2f}_{image_size}"

    if args.filenames:
        names = _parse_filenames(args.filenames)
        cases = _cases_from_filenames(ds, probs, names)
        if not cases:
            sys.exit("[ERROR] --filenames 중 test split에서 찾은 파일이 없습니다.")
        _render_combined(cases, model, device, image_size, patch_grid, transform,
                         "선택 이미지 비교", args.output_dir / f"filenames_{tag}.png")
    else:
        for kind, title, true_cls, pred_cls in _GROUPS:
            cases = collect_cases(ds, probs, true_cls, pred_cls, args.per_case)
            _render_combined(cases, model, device, image_size, patch_grid, transform,
                             title, args.output_dir / f"{kind}_{tag}.png")

    print("[combined] done.")


if __name__ == "__main__":
    main()
