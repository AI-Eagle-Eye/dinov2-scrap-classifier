"""Step 2-B: Small object deep analysis — crops_25pct, W<224 AND H<224."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths (never printed) ──────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_PATH: str = os.path.join(_BASE, "dataset", "classification", "annotations.json")
FIGURES_DIR: str = os.path.join(_BASE, "reports", "eda", "figures")
EDA_JSON_PATH: str = os.path.join(_BASE, "reports", "eda", "eda_results.json")

# ── constants ──────────────────────────────────────────────────────────────────
PAD_25 = 0.25
SMALL_THR = 224
TINY_THR = 112
RESIZE_TARGET = 336
BOUNDARY_MARGIN = 0.02

CAT_ID_TO_NAME: dict[int, str] = {2: "cut", 3: "danger", 4: "excluded"}
CLASSES: list[str] = ["cut", "danger", "excluded"]
CLASS_COLORS: dict[str, str] = {
    "cut": "#2196F3",
    "danger": "#F44336",
    "excluded": "#9E9E9E",
}

ArrF: TypeAlias = np.ndarray


# ── data structures ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AnnRecord:
    cls: str
    crop_w: float
    crop_h: float
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float
    img_w: int
    img_h: int

    @property
    def is_small(self) -> bool:
        return self.crop_w < SMALL_THR and self.crop_h < SMALL_THR

    @property
    def is_tiny(self) -> bool:
        return self.crop_w < TINY_THR and self.crop_h < TINY_THR

    @property
    def aspect_ratio(self) -> float:
        return self.crop_w / max(self.crop_h, 1e-6)

    @property
    def scale_factor_336(self) -> float:
        return RESIZE_TARGET / max(self.crop_w, self.crop_h, 1e-6)

    @property
    def is_boundary_touching(self) -> bool:
        margin_left = self.raw_x / self.img_w
        margin_top = self.raw_y / self.img_h
        margin_right = (self.img_w - (self.raw_x + self.raw_w)) / self.img_w
        margin_bottom = (self.img_h - (self.raw_y + self.raw_h)) / self.img_h
        return min(margin_left, margin_top, margin_right, margin_bottom) < BOUNDARY_MARGIN


# ── data loading ───────────────────────────────────────────────────────────────

def load_records(path: str) -> list[AnnRecord]:
    with open(path) as f:
        raw = json.load(f)
    img_dims: dict[int, tuple[int, int]] = {
        img["id"]: (img["width"], img["height"]) for img in raw["images"]
    }
    records: list[AnnRecord] = []
    for ann in raw["annotations"]:
        if ann["category_id"] not in CAT_ID_TO_NAME:
            continue
        cls = CAT_ID_TO_NAME[ann["category_id"]]
        iw, ih = img_dims[ann["image_id"]]
        rx, ry, rw, rh = ann["bbox"]
        pw, ph = rw * PAD_25, rh * PAD_25
        x1 = max(0.0, rx - pw)
        y1 = max(0.0, ry - ph)
        x2 = min(float(iw), rx + rw + pw)
        y2 = min(float(ih), ry + rh + ph)
        records.append(AnnRecord(
            cls=cls,
            crop_w=x2 - x1, crop_h=y2 - y1,
            raw_x=rx, raw_y=ry, raw_w=rw, raw_h=rh,
            img_w=iw, img_h=ih,
        ))
    return records


def by_class(records: list[AnnRecord]) -> dict[str, list[AnnRecord]]:
    result: dict[str, list[AnnRecord]] = {c: [] for c in CLASSES}
    for r in records:
        result[r.cls].append(r)
    return result


# ── analysis 1 ────────────────────────────────────────────────────────────────

def analysis_1(records: list[AnnRecord], cls_map: dict[str, list[AnnRecord]]) -> None:
    print("\n" + "=" * 68)
    print("  [분석 1-1] 소형 vs 일반 클래스별 카운트")
    print("=" * 68)
    bdr = "┌──────────┬──────────┬──────────┬──────────┐"
    hdr = "│  클래스  │ 소형(<224)│ 일반(≥224)│  비율    │"
    mdr = "├──────────┼──────────┼──────────┼──────────┤"
    ftr = "└──────────┴──────────┴──────────┴──────────┘"
    print(bdr)
    print(hdr)
    print(mdr)
    for cls in CLASSES:
        recs = cls_map[cls]
        small = sum(1 for r in recs if r.is_small)
        normal = len(recs) - small
        ratio = small / len(recs) if recs else 0.0
        print(f"│ {cls:^8s} │ {small:^9,d}│ {normal:^9,d}│ {ratio:^8.1%} │")
    total_small = sum(1 for r in records if r.is_small)
    total = len(records)
    print(mdr.replace("├", "├").replace("┤", "┤"))
    print(f"│ {'ALL':^8s} │ {total_small:^9,d}│ {total-total_small:^9,d}│ {total_small/total:^8.1%} │")
    print(ftr)

    print("\n" + "=" * 68)
    print("  [분석 1-2] 소형 샘플 내 백분위수 (crops_25pct 기준, px)")
    print("=" * 68)
    pcts = [10, 25, 50, 75, 90]
    for axis_name, attr in (("Width", "crop_w"), ("Height", "crop_h")):
        print(f"\n  [{axis_name}]")
        print(f"  {'클래스':>10s} |" + "".join(f" p{p:>2d}  |" for p in pcts))
        print("  " + "-" * 54)
        for cls in CLASSES:
            small_recs = [r for r in cls_map[cls] if r.is_small]
            if not small_recs:
                continue
            arr = np.array([getattr(r, attr) for r in small_recs])
            vals = " | ".join(f"{np.percentile(arr, p):5.0f}" for p in pcts)
            print(f"  {cls:>10s} | {vals}")

    print("\n" + "=" * 68)
    print("  [분석 1-3] 극소형 비율 (W<112 AND H<112)")
    print("=" * 68)
    print(f"  {'클래스':>10s} | {'극소형 수':>9s} | {'클래스 내 비율':>13s} | {'소형 내 비율':>12s}")
    print("  " + "-" * 56)
    for cls in CLASSES:
        recs = cls_map[cls]
        tiny = sum(1 for r in recs if r.is_tiny)
        small = sum(1 for r in recs if r.is_small)
        ratio_cls = tiny / len(recs) if recs else 0.0
        ratio_small = tiny / small if small else 0.0
        print(f"  {cls:>10s} | {tiny:>9,d} | {ratio_cls:>13.1%} | {ratio_small:>12.1%}")
    all_tiny = sum(1 for r in records if r.is_tiny)
    all_small = sum(1 for r in records if r.is_small)
    print(f"  {'ALL':>10s} | {all_tiny:>9,d} | {all_tiny/total:>13.1%} | {all_tiny/all_small:>12.1%}")


# ── analysis 2: aspect ratio for small samples ────────────────────────────────

def plot_aspect_ratio_small(cls_map: dict[str, list[AnnRecord]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0.2, 4.0, 80)

    for ax, subset_label, is_small_filter in [
        (axes[0], "All samples", False),
        (axes[1], "Small only (W<224 AND H<224)", True),
    ]:
        for cls in CLASSES:
            recs = [r for r in cls_map[cls] if (r.is_small == is_small_filter or not is_small_filter)]
            if is_small_filter:
                recs = [r for r in cls_map[cls] if r.is_small]
            else:
                recs = cls_map[cls]
            ars = np.array([r.aspect_ratio for r in recs])
            ax.hist(ars, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                    label=f"{cls} (n={len(ars):,})", histtype="stepfilled")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="1:1")
        ax.axvline(0.7, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axvline(1.4, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Aspect Ratio (W/H)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Aspect Ratio — {subset_label}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "aspect_ratio_small_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # console summary
    print("\n" + "=" * 68)
    print("  [분석 2] 소형 샘플 aspect ratio 요약")
    print("=" * 68)
    print(f"  {'':>10s} | {'전체 중앙값':>11s} | {'소형 중앙값':>11s} | {'소형 세로<0.7':>13s} | {'소형 가로>1.4':>13s}")
    print("  " + "-" * 68)
    for cls in CLASSES:
        all_ar = np.array([r.aspect_ratio for r in cls_map[cls]])
        small_recs = [r for r in cls_map[cls] if r.is_small]
        small_ar = np.array([r.aspect_ratio for r in small_recs])
        portrait = float(np.mean(small_ar < 0.7)) if len(small_ar) else 0.0
        landscape = float(np.mean(small_ar > 1.4)) if len(small_ar) else 0.0
        print(f"  {cls:>10s} | {np.median(all_ar):>11.3f} | {np.median(small_ar):>11.3f} | "
              f"{portrait:>13.1%} | {landscape:>13.1%}")


# ── analysis 3: class ratio by size band ──────────────────────────────────────

def plot_class_ratio_by_size(cls_map: dict[str, list[AnnRecord]]) -> None:
    bands = ["<112", "112–224", "224–336", "≥336"]

    def get_band(r: AnnRecord) -> str:
        w = r.crop_w
        if w < 112:
            return "<112"
        elif w < 224:
            return "112–224"
        elif w < 336:
            return "224–336"
        return "≥336"

    # count per band × class
    band_cls: dict[str, dict[str, int]] = {b: {c: 0 for c in CLASSES} for b in bands}
    for cls in CLASSES:
        for r in cls_map[cls]:
            band_cls[get_band(r)][cls] += 1

    band_totals = {b: sum(band_cls[b].values()) for b in bands}

    # console output
    print("\n" + "=" * 68)
    print("  [분석 3] 크기 구간별 클래스 비율 (width 기준)")
    print("=" * 68)
    print(f"  {'구간':>9s} | {'cut':>6s} | {'danger':>6s} | {'excluded':>8s} | TOTAL")
    print("  " + "-" * 50)
    for b in bands:
        tot = band_totals[b]
        if tot == 0:
            continue
        row = " | ".join(f"{band_cls[b][c]:5d}({band_cls[b][c]/tot:3.0%})" for c in CLASSES)
        print(f"  {b:>9s} | {row} | {tot:,d}")

    # stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bands))
    width = 0.55
    bottoms = np.zeros(len(bands))
    for cls in CLASSES:
        heights = np.array([
            band_cls[b][cls] / band_totals[b] if band_totals[b] else 0.0
            for b in bands
        ])
        ax.bar(x, heights, width, bottom=bottoms, color=CLASS_COLORS[cls],
               label=cls, alpha=0.85)
        for i, (h, bot) in enumerate(zip(heights, bottoms)):
            if h > 0.04:
                ax.text(x[i], bot + h / 2, f"{h:.0%}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += heights

    band_labels = [f"{b}\n(n={band_totals[b]:,})" for b in bands]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel("Class Ratio", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Class Ratio by Size Band (width, crops_25pct)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "class_ratio_by_size.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── analysis 4: boundary touching ─────────────────────────────────────────────

def analysis_4(records: list[AnnRecord], cls_map: dict[str, list[AnnRecord]]) -> None:
    print("\n" + "=" * 68)
    print("  [분석 4] bbox 경계 근접 비율 (원본 bbox, margin < 2%)")
    print("=" * 68)
    print(f"  {'클래스':>10s} | {'소형 중 경계(%)':>14s} | {'전체 중 경계(%)':>14s} | {'소형 n':>7s} | {'전체 n':>7s}")
    print("  " + "-" * 64)
    for cls in CLASSES:
        recs = cls_map[cls]
        small_recs = [r for r in recs if r.is_small]
        bt_small = sum(1 for r in small_recs if r.is_boundary_touching)
        bt_all = sum(1 for r in recs if r.is_boundary_touching)
        r_small = bt_small / len(small_recs) if small_recs else 0.0
        r_all = bt_all / len(recs) if recs else 0.0
        print(f"  {cls:>10s} | {r_small:>14.1%} | {r_all:>14.1%} | {len(small_recs):>7,d} | {len(recs):>7,d}")
    # ALL
    all_small = [r for r in records if r.is_small]
    bt_small_all = sum(1 for r in all_small if r.is_boundary_touching)
    bt_all_all = sum(1 for r in records if r.is_boundary_touching)
    print(f"  {'ALL':>10s} | {bt_small_all/len(all_small):>14.1%} | {bt_all_all/len(records):>14.1%} | "
          f"{len(all_small):>7,d} | {len(records):>7,d}")


# ── analysis 5: upscaling factor ──────────────────────────────────────────────

def plot_upscaling_factor(records: list[AnnRecord], cls_map: dict[str, list[AnnRecord]]) -> None:
    BANDS = [(0.0, 1.0, "downscale"), (1.0, 1.5, "약(×1.0~1.5)"),
             (1.5, 3.0, "중(×1.5~3.0)"), (3.0, 99.0, "강(×3.0+)")]

    print("\n" + "=" * 68)
    print("  [분석 5] 336 resize 시 scale_factor 분포 (전체 샘플)")
    print("=" * 68)
    print(f"  {'클래스':>10s} | {'downscale':>10s} | {'×1.0~1.5':>10s} | {'×1.5~3.0':>10s} | {'×3.0+':>10s}")
    print("  " + "-" * 60)
    for cls in CLASSES + ["ALL"]:
        recs = records if cls == "ALL" else cls_map[cls]
        sfs = np.array([r.scale_factor_336 for r in recs])
        counts = [
            int(np.sum((sfs >= lo) & (sfs < hi))) for lo, hi, _ in BANDS
        ]
        n = len(sfs)
        row = " | ".join(f"{c:5d}({c/n:3.0%})" for c in counts)
        print(f"  {cls:>10s} | {row}")

    # figure: histogram
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle("Scale Factor Distribution (336 / max(W,H), crops_25pct)",
                 fontsize=11, fontweight="bold")

    for ax, cls in zip(axes, CLASSES):
        recs = cls_map[cls]
        sfs = np.array([r.scale_factor_336 for r in recs])
        small_sfs = np.array([r.scale_factor_336 for r in recs if r.is_small])
        normal_sfs = np.array([r.scale_factor_336 for r in recs if not r.is_small])
        bins = np.linspace(0.1, 8.0, 80)

        ax.hist(normal_sfs, bins=bins, color=CLASS_COLORS[cls], alpha=0.45,
                label="normal (>=224)", histtype="stepfilled")
        ax.hist(small_sfs, bins=bins, color=CLASS_COLORS[cls], alpha=0.75,
                label="small (<224)", histtype="step", linewidth=1.5)

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="×1.0")
        ax.axvline(3.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="×3.0")
        ax.set_xlabel("Scale Factor (336 / max side)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{cls} (n={len(recs):,})", fontsize=10)
        ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "upscaling_factor_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── JSON payload ──────────────────────────────────────────────────────────────

def build_json_payload(
    records: list[AnnRecord],
    cls_map: dict[str, list[AnnRecord]],
) -> dict:
    all_sfs = np.array([r.scale_factor_336 for r in records])
    high_upscaling_ratio = float(np.mean(all_sfs >= 3.0))

    by_class: dict[str, dict] = {}
    for cls in CLASSES:
        recs = cls_map[cls]
        small_recs = [r for r in recs if r.is_small]
        tiny_recs = [r for r in recs if r.is_tiny]
        bt_small = sum(1 for r in small_recs if r.is_boundary_touching)
        sfs = np.array([r.scale_factor_336 for r in recs])
        by_class[cls] = {
            "small_count": len(small_recs),
            "small_ratio": len(small_recs) / len(recs) if recs else 0.0,
            "tiny_count": len(tiny_recs),
            "tiny_ratio": len(tiny_recs) / len(recs) if recs else 0.0,
            "median_w": float(np.median([r.crop_w for r in small_recs])) if small_recs else 0.0,
            "median_h": float(np.median([r.crop_h for r in small_recs])) if small_recs else 0.0,
            "boundary_touching_ratio": bt_small / len(small_recs) if small_recs else 0.0,
            "scale_factor_over_3_ratio": float(np.mean(sfs >= 3.0)) if len(sfs) else 0.0,
        }

    note = (
        f"전체 중 scale_factor ≥ 3.0 비율 = {high_upscaling_ratio:.1%}. "
        "이 샘플들은 336 resize 시 3배 이상 upscaling 발생 → "
        "DINOv2 patch feature 품질 저하 위험 있음. "
        "학습 시 WeightedRandomSampler 또는 별도 augmentation 검토 필요."
    )
    return {
        "small_object_analysis": {
            "by_class": by_class,
            "risk_assessment": {
                "high_upscaling_risk_ratio": high_upscaling_ratio,
                "note": note,
            },
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    records = load_records(ANNOTATIONS_PATH)
    cls_map = by_class(records)

    analysis_1(records, cls_map)
    plot_aspect_ratio_small(cls_map)
    plot_class_ratio_by_size(cls_map)
    analysis_4(records, cls_map)
    plot_upscaling_factor(records, cls_map)

    payload = build_json_payload(records, cls_map)
    save_eda_json(payload)

    print("\n" + "=" * 68)
    print("  저장 완료")
    print("=" * 68)
    print("  figures/aspect_ratio_small_by_class.png")
    print("  figures/class_ratio_by_size.png")
    print("  figures/upscaling_factor_by_class.png")
    print("  reports/eda/eda_results.json  (small_object_analysis 추가)")


if __name__ == "__main__":
    main()
