"""Step 2-A: Deep resolution analysis — class × padding, input size recommendation, aspect ratio."""
from __future__ import annotations

import json
import os
from typing import TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── paths (never printed) ──────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_PATH: str = os.path.join(_BASE, "dataset", "classification", "annotations.json")
FIGURES_DIR: str = os.path.join(_BASE, "reports", "eda", "figures")
EDA_JSON_PATH: str = os.path.join(_BASE, "reports", "eda", "eda_results.json")

# ── constants ──────────────────────────────────────────────────────────────────
PADDINGS: dict[str, float] = {
    "crops_0pct": 0.0,
    "crops_10pct": 0.10,
    "crops_25pct": 0.25,
}
CAT_ID_TO_NAME: dict[int, str] = {2: "cut", 3: "danger", 4: "excluded"}
CLASS_COLORS: dict[str, str] = {
    "cut": "#2196F3",
    "danger": "#F44336",
    "excluded": "#9E9E9E",
}
PERCENTILES: list[int] = [10, 25, 50, 75, 90, 95]
THRESHOLD_224 = 224
THRESHOLD_336 = 336

CropDims: TypeAlias = dict[str, dict[str, list[float]]]  # pad -> class -> [w or h]


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[dict[int, tuple[int, int]], list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    img_dims: dict[int, tuple[int, int]] = {
        img["id"]: (img["width"], img["height"]) for img in raw["images"]
    }
    anns = [a for a in raw["annotations"] if a["category_id"] in CAT_ID_TO_NAME]
    return img_dims, anns


def crop_size(bbox: list[float], img_w: int, img_h: int, pad: float) -> tuple[float, float]:
    x, y, w, h = bbox
    pw, ph = w * pad, h * pad
    x1 = max(0.0, x - pw)
    y1 = max(0.0, y - ph)
    x2 = min(float(img_w), x + w + pw)
    y2 = min(float(img_h), y + h + ph)
    return x2 - x1, y2 - y1


def build_dims(
    img_dims: dict[int, tuple[int, int]], anns: list[dict]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Returns {pad_key -> {class -> {'w': [...], 'h': [...]}}}"""
    result: dict[str, dict[str, dict[str, list[float]]]] = {
        p: {c: {"w": [], "h": []} for c in CAT_ID_TO_NAME.values()}
        for p in PADDINGS
    }
    for ann in anns:
        cls = CAT_ID_TO_NAME[ann["category_id"]]
        iw, ih = img_dims[ann["image_id"]]
        for pad_key, pad_val in PADDINGS.items():
            cw, ch = crop_size(ann["bbox"], iw, ih, pad_val)
            result[pad_key][cls]["w"].append(cw)
            result[pad_key][cls]["h"].append(ch)
    return result


# ── analysis 1-1: percentile table ────────────────────────────────────────────

def percentile_table(
    dims: dict[str, dict[str, dict[str, list[float]]]]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Returns {pad -> class -> {axis -> {pXX: value}}}"""
    out: dict = {}
    for pad_key, cls_data in dims.items():
        out[pad_key] = {}
        for cls, axes in cls_data.items():
            out[pad_key][cls] = {}
            for axis, vals in axes.items():
                arr = np.array(vals)
                out[pad_key][cls][axis] = {
                    f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES
                }
    return out


def print_percentile_table(ptable: dict) -> None:
    header_classes = list(CAT_ID_TO_NAME.values())
    for pad_key in PADDINGS:
        print(f"\n{'─'*70}")
        print(f"  {pad_key}  — 백분위수 (width / height, px)")
        print(f"{'─'*70}")
        for axis in ("w", "h"):
            label = "Width" if axis == "w" else "Height"
            print(f"\n  [{label}]")
            header = f"  {'클래스':>10s} | " + " | ".join(f"p{p:>2d}" for p in PERCENTILES)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for cls in header_classes:
                pvals = ptable[pad_key][cls][axis]
                row = " | ".join(f"{pvals[f'p{p}']:6.0f}" for p in PERCENTILES)
                print(f"  {cls:>10s} | {row}")


# ── analysis 1-2: boxplot ─────────────────────────────────────────────────────

def plot_resolution_boxplot(
    dims: dict[str, dict[str, dict[str, list[float]]]]
) -> None:
    pad_key = "crops_25pct"
    classes = list(CAT_ID_TO_NAME.values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Crop Size Distribution by Class (crops_25pct)", fontsize=12, fontweight="bold")

    for ax, axis, label in zip(axes, ("w", "h"), ("Width (px)", "Height (px)")):
        data = [dims[pad_key][cls][axis] for cls in classes]
        colors = [CLASS_COLORS[cls] for cls in classes]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(THRESHOLD_224, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.8, label="224 px")
        ax.axhline(THRESHOLD_336, color="orange", linestyle=":", linewidth=1.2,
                   alpha=0.8, label="336 px")
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"{c}\n(n={len(dims[pad_key][c][axis]):,})" for c in classes],
                           fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=c, alpha=0.7) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "resolution_boxplot_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── analysis 1-3: <224 ratio by class × padding ───────────────────────────────

def under_threshold_ratios(
    dims: dict[str, dict[str, dict[str, list[float]]]],
    threshold: int,
) -> dict[str, dict[str, float]]:
    """Returns {pad -> {cls -> ratio}}"""
    result: dict[str, dict[str, float]] = {}
    for pad_key, cls_data in dims.items():
        result[pad_key] = {}
        for cls, axes in cls_data.items():
            ws = np.array(axes["w"])
            hs = np.array(axes["h"])
            ratio = float(np.mean((ws < threshold) & (hs < threshold)))
            result[pad_key][cls] = ratio
    return result


def print_under224_table(ratios: dict[str, dict[str, float]]) -> None:
    pads = list(PADDINGS.keys())
    classes = list(CAT_ID_TO_NAME.values())
    col_w = 10

    border_top = "┌" + "──────────┬" + ("─" * (col_w + 2) + "┬") * (len(pads) - 1) + "─" * (col_w + 2) + "┐"
    border_mid = "├" + "──────────┼" + ("─" * (col_w + 2) + "┼") * (len(pads) - 1) + "─" * (col_w + 2) + "┤"
    border_bot = "└" + "──────────┴" + ("─" * (col_w + 2) + "┴") * (len(pads) - 1) + "─" * (col_w + 2) + "┘"

    pad_labels = ["  0pct  ", " 10pct  ", " 25pct  "]
    header = "│  클래스  │" + "│".join(f" {lbl:^{col_w}s} " for lbl in pad_labels) + "│"

    print(border_top)
    print(header)
    print(border_mid)
    for cls in classes:
        row = "│" + f" {cls:^8s} │"
        for pad_key in pads:
            r = ratios[pad_key][cls]
            row += f" {r:>8.1%}   │"
        print(row)
    print(border_bot)


# ── analysis 2-1: size-band distribution ──────────────────────────────────────

def size_band_counts(
    dims: dict[str, dict[str, dict[str, list[float]]]],
    pad_key: str = "crops_25pct",
) -> dict[str, dict[str, int]]:
    """For a given pad, count per class how many fall in each size band (short side)."""
    bands = ["<112", "112–224", "224–336", ">336"]
    result: dict[str, dict[str, int]] = {cls: {b: 0 for b in bands} for cls in CAT_ID_TO_NAME.values()}
    result["ALL"] = {b: 0 for b in bands}

    for cls, axes in dims[pad_key].items():
        for w, h in zip(axes["w"], axes["h"]):
            short = min(w, h)
            if short < 112:
                band = "<112"
            elif short < 224:
                band = "112–224"
            elif short < 336:
                band = "224–336"
            else:
                band = ">336"
            result[cls][band] += 1
            result["ALL"][band] += 1
    return result


def print_size_band_table(band_counts: dict[str, dict[str, int]]) -> None:
    classes_order = [*CAT_ID_TO_NAME.values(), "ALL"]
    bands = ["<112", "112–224", "224–336", ">336"]
    print("\n  구간별 분포 (crops_25pct, 단변 기준)")
    print(f"  {'클래스':>10s} | " +
          " | ".join(f"{b:>9s}" for b in bands) + " | TOTAL")
    print("  " + "-" * 70)
    for cls in classes_order:
        total = sum(band_counts[cls].values())
        row = " | ".join(
            f"{band_counts[cls][b]:5d}({band_counts[cls][b]/total:4.0%})"
            for b in bands
        )
        print(f"  {cls:>10s} | {row} | {total:,d}")


# ── analysis 2-2: <336 ratios ─────────────────────────────────────────────────

def print_under336_table(
    ratios224: dict[str, dict[str, float]],
    ratios336: dict[str, dict[str, float]],
    dims: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    classes = list(CAT_ID_TO_NAME.values())
    print("\n  <336 비율 (W<336 AND H<336) vs <224 비율 (crops_25pct)")
    print(f"  {'클래스':>10s} | {'<224':>8s} | {'<336':>8s} | {'추가 upscale':>12s}")
    print("  " + "-" * 50)
    for cls in classes:
        r224 = ratios224["crops_25pct"][cls]
        r336 = ratios336["crops_25pct"][cls]
        print(f"  {cls:>10s} | {r224:>8.1%} | {r336:>8.1%} | +{r336 - r224:>10.1%}")
    pad_data = dims["crops_25pct"]
    all_ws = np.array(sum([pad_data[c]["w"] for c in pad_data], []))
    all_hs = np.array(sum([pad_data[c]["h"] for c in pad_data], []))
    r224_all = float(np.mean((all_ws < 224) & (all_hs < 224)))
    r336_all = float(np.mean((all_ws < 336) & (all_hs < 336)))
    print(f"  {'ALL':>10s} | {r224_all:>8.1%} | {r336_all:>8.1%} | +{r336_all - r224_all:>10.1%}")


# ── analysis 2-3: recommendation ──────────────────────────────────────────────

def input_size_recommendation(
    dims: dict[str, dict[str, dict[str, list[float]]]]
) -> dict[str, object]:
    danger_w = np.median(dims["crops_25pct"]["danger"]["w"])
    danger_h = np.median(dims["crops_25pct"]["danger"]["h"])
    if danger_w >= THRESHOLD_224 and danger_h >= THRESHOLD_224:
        rec = "336"
        reason = (
            f"danger 중앙값 W={danger_w:.0f} H={danger_h:.0f} — "
            "224 이상이므로 336 입력 시 다운스케일 없이 디테일 보존 가능. "
            "224→336 업그레이드로 추가 upscaling 없이 feature map 해상도 향상."
        )
    else:
        rec = "224"
        reason = (
            f"danger 중앙값 W={danger_w:.0f} H={danger_h:.0f} — "
            "224 미만이므로 336 입력 시 오히려 upscaling 비율 증가. "
            "224 유지가 현실적."
        )
    return {
        "recommended": rec,
        "reason": reason,
        "danger_median_w": float(danger_w),
        "danger_median_h": float(danger_h),
    }


# ── analysis 3: aspect ratio ──────────────────────────────────────────────────

def plot_aspect_ratio(
    dims: dict[str, dict[str, dict[str, list[float]]]]
) -> None:
    pad_key = "crops_25pct"
    classes = list(CAT_ID_TO_NAME.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0.2, 4.0, 80)

    for cls in classes:
        ws = np.array(dims[pad_key][cls]["w"])
        hs = np.array(dims[pad_key][cls]["h"])
        ratios = ws / np.maximum(hs, 1e-6)
        ax.hist(ratios, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                label=f"{cls} (n={len(ratios):,})", histtype="stepfilled")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="1:1 (square)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="2:1 portrait")
    ax.axvline(2.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="1:2 landscape")
    ax.set_xlabel("Aspect Ratio (width / height)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Aspect Ratio Distribution by Class (crops_25pct)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "aspect_ratio_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── JSON output ───────────────────────────────────────────────────────────────

def build_eda_json(
    ptable: dict,
    ratios224: dict[str, dict[str, float]],
    rec: dict[str, object],
) -> dict:
    by_pad_cls: dict = {}
    for pad_key in PADDINGS:
        by_pad_cls[pad_key] = {}
        for cls in CAT_ID_TO_NAME.values():
            by_pad_cls[pad_key][cls] = {
                "p50_w": ptable[pad_key][cls]["w"]["p50"],
                "p50_h": ptable[pad_key][cls]["h"]["p50"],
                "under_224_ratio": ratios224[pad_key][cls],
            }
    return {
        "resolution_detail": {
            "by_padding_class": by_pad_cls,
            "input_size_recommendation": rec,
        }
    }


def save_eda_json(update: dict) -> None:
    os.makedirs(os.path.dirname(EDA_JSON_PATH), exist_ok=True)
    existing: dict = {}
    if os.path.exists(EDA_JSON_PATH):
        with open(EDA_JSON_PATH) as f:
            existing = json.load(f)
    existing.update(update)
    with open(EDA_JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def main() -> None:
    img_dims, anns = load_data(ANNOTATIONS_PATH)
    dims = build_dims(img_dims, anns)

    # ── 분석 1-1: 백분위수 표
    print("\n" + "=" * 70)
    print("  [분석 1-1] 백분위수 표")
    print("=" * 70)
    ptable = percentile_table(dims)
    print_percentile_table(ptable)

    # ── 분석 1-2: boxplot
    plot_resolution_boxplot(dims)
    print("\n  [분석 1-2] boxplot 저장 완료: figures/resolution_boxplot_by_class.png")

    # ── 분석 1-3: <224 비율 클래스별 × 패딩별
    print("\n" + "=" * 70)
    print("  [분석 1-3] 224px 미만 비율 (W<224 AND H<224)")
    print("=" * 70)
    ratios224 = under_threshold_ratios(dims, THRESHOLD_224)
    print_under224_table(ratios224)

    # ── 분석 2-1: 구간별 분포
    print("\n" + "=" * 70)
    print("  [분석 2-1] 구간별 분포 (crops_25pct, 단변 min(W,H) 기준)")
    print("=" * 70)
    band_counts = size_band_counts(dims, "crops_25pct")
    print_size_band_table(band_counts)

    # ── 분석 2-2: <336 비율
    print("\n" + "=" * 70)
    print("  [분석 2-2] 224 vs 336 upscaling 비율 비교")
    print("=" * 70)
    ratios336 = under_threshold_ratios(dims, THRESHOLD_336)
    print_under336_table(ratios224, ratios336, dims)

    # ── 분석 2-3: 권고
    print("\n" + "=" * 70)
    print("  [분석 2-3] 입력 크기 권고")
    print("=" * 70)
    rec = input_size_recommendation(dims)
    print(f"\n  권장 입력 크기: {rec['recommended']}×{rec['recommended']}")
    print(f"  근거: {rec['reason']}")

    # ── 분석 3: aspect ratio
    plot_aspect_ratio(dims)
    print("\n" + "=" * 70)
    print("  [분석 3] aspect ratio 히스토그램 저장 완료: figures/aspect_ratio_by_class.png")
    print("=" * 70)

    # aspect ratio 요약 콘솔 출력
    pad_key = "crops_25pct"
    print(f"\n  aspect ratio 요약 (crops_25pct, width/height)")
    print(f"  {'클래스':>10s} | {'중앙값':>8s} | {'p25':>8s} | {'p75':>8s} | {'<0.7(세로)':>10s} | {'>1.4(가로)':>10s}")
    print("  " + "-" * 72)
    for cls in CAT_ID_TO_NAME.values():
        ws = np.array(dims[pad_key][cls]["w"])
        hs = np.array(dims[pad_key][cls]["h"])
        ar = ws / np.maximum(hs, 1e-6)
        portrait = float(np.mean(ar < 0.7))
        landscape = float(np.mean(ar > 1.4))
        print(f"  {cls:>10s} | {np.median(ar):>8.3f} | {np.percentile(ar,25):>8.3f} | "
              f"{np.percentile(ar,75):>8.3f} | {portrait:>10.1%} | {landscape:>10.1%}")

    # ── JSON 저장
    eda_payload = build_eda_json(ptable, ratios224, rec)
    save_eda_json(eda_payload)
    print("\n  eda_results.json 저장 완료: reports/eda/eda_results.json")


if __name__ == "__main__":
    main()
