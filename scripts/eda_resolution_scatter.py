"""Resolution scatter plot: crop dimensions per padding level, computed from annotations.json bbox."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "classification", "annotations.json"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "reports", "eda", "figures", "resolution_scatter.png"
)

PADDINGS: dict[str, float] = {
    "crops_0pct": 0.0,
    "crops_10pct": 0.10,
    "crops_25pct": 0.25,
}

CLASS_COLORS: dict[str, str] = {
    "cut": "#2196F3",       # blue
    "danger": "#F44336",    # red
    "excluded": "#9E9E9E",  # grey
}

CAT_ID_TO_NAME: dict[int, str] = {
    2: "cut",
    3: "danger",
    4: "excluded",
}

THRESHOLD = 224


def load_annotations(path: str) -> tuple[dict[int, tuple[int, int]], list[dict]]:
    with open(path) as f:
        data = json.load(f)
    img_dims: dict[int, tuple[int, int]] = {
        img["id"]: (img["width"], img["height"])
        for img in data["images"]
    }
    annotations: list[dict] = [
        ann for ann in data["annotations"]
        if ann["category_id"] in CAT_ID_TO_NAME
    ]
    return img_dims, annotations


def compute_crop_size(
    bbox: list[float],
    img_w: int,
    img_h: int,
    padding: float,
) -> tuple[float, float]:
    x, y, w, h = bbox
    pad_w = w * padding
    pad_h = h * padding
    x1 = max(0.0, x - pad_w)
    y1 = max(0.0, y - pad_h)
    x2 = min(float(img_w), x + w + pad_w)
    y2 = min(float(img_h), y + h + pad_h)
    return x2 - x1, y2 - y1


def build_dataframe(
    img_dims: dict[int, tuple[int, int]],
    annotations: list[dict],
    padding: float,
) -> tuple[list[float], list[float], list[str]]:
    widths: list[float] = []
    heights: list[float] = []
    labels: list[str] = []
    for ann in annotations:
        img_w, img_h = img_dims[ann["image_id"]]
        cw, ch = compute_crop_size(ann["bbox"], img_w, img_h, padding)
        widths.append(cw)
        heights.append(ch)
        labels.append(CAT_ID_TO_NAME[ann["category_id"]])
    return widths, heights, labels


def below_threshold_ratio(widths: list[float], heights: list[float]) -> float:
    n = len(widths)
    if n == 0:
        return 0.0
    count = sum(1 for w, h in zip(widths, heights) if w < THRESHOLD and h < THRESHOLD)
    return count / n


def plot_jointplot(
    ax_joint: plt.Axes,
    ax_margx: plt.Axes,
    ax_margy: plt.Axes,
    widths: list[float],
    heights: list[float],
    labels: list[str],
    title: str,
) -> None:
    classes = list(CLASS_COLORS.keys())

    # scatter by class
    for cls in classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        xs = [widths[i] for i in idx]
        ys = [heights[i] for i in idx]
        ax_joint.scatter(xs, ys, c=CLASS_COLORS[cls], alpha=0.25, s=4, label=cls, rasterized=True)

    # 224×224 reference lines
    ax_joint.axvline(THRESHOLD, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_joint.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

    # shade W<224 AND H<224 region
    xmax = max(widths) * 1.02
    ymax = max(heights) * 1.02
    ax_joint.add_patch(
        mpatches.Rectangle(
            (0, 0), THRESHOLD, THRESHOLD,
            facecolor="red", alpha=0.07, zorder=0,
        )
    )

    ax_joint.set_xlim(left=0, right=xmax)
    ax_joint.set_ylim(bottom=0, top=ymax)
    ax_joint.set_xlabel("Width (px)", fontsize=9)
    ax_joint.set_ylabel("Height (px)", fontsize=9)
    ax_joint.set_title(title, fontsize=10, fontweight="bold")

    # marginal histograms
    bins = 60
    for cls in classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        xs = [widths[i] for i in idx]
        ys = [heights[i] for i in idx]
        ax_margx.hist(xs, bins=bins, color=CLASS_COLORS[cls], alpha=0.5, histtype="stepfilled")
        ax_margy.hist(ys, bins=bins, color=CLASS_COLORS[cls], alpha=0.5,
                      histtype="stepfilled", orientation="horizontal")

    ax_margx.axvline(THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_margy.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    ax_margx.set_xlim(ax_joint.get_xlim())
    ax_margy.set_ylim(ax_joint.get_ylim())
    ax_margx.axis("off")
    ax_margy.axis("off")


def main() -> None:
    img_dims, annotations = load_annotations(ANNOTATIONS_PATH)

    padding_keys = list(PADDINGS.keys())
    n_pads = len(padding_keys)

    fig_w = 6.0 * n_pads
    fig = plt.figure(figsize=(fig_w, 7))

    # grid: 2 rows × (3 * n_pads) cols  — [marg_x | joint | marg_y] per padding
    COLS_PER = 12
    MARG_RATIO = 2
    JOINT_RATIO = 9

    gs = fig.add_gridspec(
        2, COLS_PER * n_pads,
        height_ratios=[MARG_RATIO, JOINT_RATIO],
        hspace=0.05, wspace=0.05,
        left=0.05, right=0.97, top=0.90, bottom=0.08,
    )

    print(f"\n{'패딩':15s} | {'224미만 비율':>12s} | {'224미만 건수':>10s} | 총계")
    print("-" * 60)

    for col_idx, pad_key in enumerate(padding_keys):
        padding = PADDINGS[pad_key]
        widths, heights, labels = build_dataframe(img_dims, annotations, padding)

        ratio = below_threshold_ratio(widths, heights)
        n_below = sum(1 for w, h in zip(widths, heights) if w < THRESHOLD and h < THRESHOLD)
        print(f"{pad_key:15s} | {ratio:11.2%} | {n_below:10,d} | {len(widths):,d}")

        base_col = col_idx * COLS_PER
        ax_margx = fig.add_subplot(gs[0, base_col: base_col + JOINT_RATIO])
        ax_joint = fig.add_subplot(gs[1, base_col: base_col + JOINT_RATIO])
        ax_margy = fig.add_subplot(gs[1, base_col + JOINT_RATIO: base_col + COLS_PER])

        plot_jointplot(ax_joint, ax_margx, ax_margy, widths, heights, labels, pad_key)

    # legend — one shared legend at the top
    handles = [
        mpatches.Patch(color=COLOR, label=cls)
        for cls, COLOR in CLASS_COLORS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="224 px")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
    )

    fig.suptitle("Crop Resolution Distribution (by padding level)", fontsize=12, y=1.0)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n저장 완료: reports/eda/figures/resolution_scatter.png")


if __name__ == "__main__":
    main()
