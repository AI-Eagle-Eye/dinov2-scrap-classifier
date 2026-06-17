"""오분류 갤러리: predictions.csv 기준 치명(danger→cut)/오경보(cut→danger) 케이스를 원본으로 저장.

predictions.csv(이미 산출된 결정)에서 케이스별 top-confidence 샘플을 골라 원본 이미지를
grid로 나열한다. 추론을 재실행하지 않고, 이미지는 image_path 기준 읽기 전용으로만 연다.

Usage:
    python scripts/visualize_misclassified.py \
        --predictions results/exp_ar_448_tta/crops_25pct/predictions.csv \
        --output-dir results/exp_ar_448_tta/gallery
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
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLES_PER_CASE = 20
_COLS = 5
_DPI = 150
_DISPLAY_SIZE = 256

# (파일명 stem, 그리드 제목, true_label, pred_label, 정렬 기준 prob 컬럼)
_CASES: list[tuple[str, str, str, str, str]] = [
    ("missed_danger", "Missed danger→cut (T=danger, P=cut, 치명) — top confidence",
     "danger", "cut", "prob_cut"),
    ("false_alarm", "False alarm cut→danger (T=cut, P=danger, 오경보) — top confidence",
     "cut", "danger", "prob_danger"),
]


def _select(df: pd.DataFrame, true_label: str, pred_label: str, sort_col: str) -> pd.DataFrame:
    sub = df[(df["true_label"] == true_label) & (df["pred_label"] == pred_label)]
    return sub.sort_values(sort_col, ascending=False).head(_SAMPLES_PER_CASE)


def _save_grid(rows: pd.DataFrame, title: str, out_path: Path) -> int:
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
            img = Image.open(img_path).convert("RGB").resize((_DISPLAY_SIZE, _DISPLAY_SIZE))
        except OSError:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.axis("off")
        conf = float(row[f"prob_{row['pred_label']}"])  # 예측 클래스의 확률 = confidence
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
    p = argparse.ArgumentParser(description="Misclassification gallery (predictions.csv 기준)")
    p.add_argument("--predictions", type=Path,
                   default=Path("results/exp_ar_448_tta/crops_25pct/predictions.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exp_ar_448_tta/gallery"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.predictions.exists():
        sys.exit(f"[ERROR] predictions.csv not found: {args.predictions}")

    df = pd.read_csv(args.predictions)
    print(f"[gallery] predictions: {args.predictions}  (n={len(df)})")

    total = 0
    for stem, title, true_label, pred_label, sort_col in _CASES:
        rows = _select(df, true_label, pred_label, sort_col)
        total += _save_grid(rows, title, args.output_dir / f"{stem}_gallery.png")
    print(f"[gallery] done. ({total} images across {len(_CASES)} grids)")


if __name__ == "__main__":
    main()
