"""Step 2-C: Color analysis — class color bias, padding shift, small vs normal, custom normalization."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image

# ── paths (never printed) ──────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CLS = os.path.join(_BASE, "dataset", "classification")
ANNOTATIONS_PATH = os.path.join(DATASET_CLS, "annotations.json")
FIGURES_DIR = os.path.join(_BASE, "reports", "eda", "figures")
EDA_JSON_PATH = os.path.join(_BASE, "reports", "eda", "eda_results.json")

# ── constants ──────────────────────────────────────────────────────────────────
CLASSES: list[str] = ["cut", "danger", "excluded"]
CLASS_COLORS: dict[str, str] = {
    "cut": "#2196F3",
    "danger": "#F44336",
    "excluded": "#9E9E9E",
}
PAD_KEYS: list[str] = ["crops_0pct", "crops_10pct", "crops_25pct"]
PAD_VALS: dict[str, float] = {"crops_0pct": 0.0, "crops_10pct": 0.10, "crops_25pct": 0.25}
CAT_ID_TO_NAME: dict[int, str] = {2: "cut", 3: "danger", 4: "excluded"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
THUMB = 32          # thumbnail size for pixel stats
MAX_WORKERS = min(8, (os.cpu_count() or 4))
COLOR_BIAS_THR = 0.05
PADDING_SHIFT_THR = 0.05
SMALL_THR = 224

PixelStats: TypeAlias = dict[str, float]  # r_mean, g_mean, b_mean, brightness, h_mean


# ── image reading ──────────────────────────────────────────────────────────────

def _read_one(filepath: str) -> PixelStats:
    img = Image.open(filepath)
    img.draft("RGB", (THUMB, THUMB))
    img.load()
    img = img.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (THUMB, THUMB, 3)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    hsv = rgb_to_hsv(arr)
    # h channel: suppress low-saturation pixels (grey → hue meaningless)
    sat_mask = hsv[:, :, 1] > 0.1
    h_vals = hsv[:, :, 0][sat_mask]
    h_mean = float(h_vals.mean()) if h_vals.size > 0 else float(hsv[:, :, 0].mean())
    # also collect per-pixel H for histogram (sampled, not all)
    return {
        "r_mean": float(r.mean()),
        "g_mean": float(g.mean()),
        "b_mean": float(b.mean()),
        "r_sq": float((r ** 2).mean()),
        "g_sq": float((g ** 2).mean()),
        "b_sq": float((b ** 2).mean()),
        "brightness": float(brightness.mean()),
        "h_mean": h_mean,
        # flatten pixels for class-level H histogram (take every 4th pixel to save memory)
        "h_pixels": hsv[::2, ::2, 0][hsv[::2, ::2, 1] > 0.1].tolist(),
    }


@dataclass(slots=True)
class ClassStats:
    cls: str
    r_means: list[float] = field(default_factory=list)
    g_means: list[float] = field(default_factory=list)
    b_means: list[float] = field(default_factory=list)
    r_sqs: list[float] = field(default_factory=list)
    g_sqs: list[float] = field(default_factory=list)
    b_sqs: list[float] = field(default_factory=list)
    brightness_vals: list[float] = field(default_factory=list)
    h_means: list[float] = field(default_factory=list)
    h_pixels: list[float] = field(default_factory=list)


def read_class_dir(class_dir: str, cls: str) -> ClassStats:
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".jpg")]
    stats = ClassStats(cls=cls)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_read_one, fp): fp for fp in files}
        for future in as_completed(futures):
            ps = future.result()
            stats.r_means.append(ps["r_mean"])
            stats.g_means.append(ps["g_mean"])
            stats.b_means.append(ps["b_mean"])
            stats.r_sqs.append(ps["r_sq"])
            stats.g_sqs.append(ps["g_sq"])
            stats.b_sqs.append(ps["b_sq"])
            stats.brightness_vals.append(ps["brightness"])
            stats.h_means.append(ps["h_mean"])
            stats.h_pixels.extend(ps["h_pixels"])
    return stats


def pooled_std(means: list[float], sqs: list[float]) -> float:
    """Pooled pixel-level std from per-image mean and mean-of-squares."""
    mu = float(np.mean(means))
    mu_sq = float(np.mean(sqs))
    var = max(0.0, mu_sq - mu ** 2)
    return float(np.sqrt(var))


def class_color_summary(s: ClassStats) -> dict[str, float]:
    return {
        "R_mean": float(np.mean(s.r_means)),
        "G_mean": float(np.mean(s.g_means)),
        "B_mean": float(np.mean(s.b_means)),
        "R_std": pooled_std(s.r_means, s.r_sqs),
        "G_std": pooled_std(s.g_means, s.g_sqs),
        "B_std": pooled_std(s.b_means, s.b_sqs),
    }


def load_small_ann_ids() -> frozenset[int]:
    """Return annotation IDs where padded crop (25pct) is W<224 AND H<224."""
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    img_dims = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    small_ids: list[int] = []
    for ann in data["annotations"]:
        if ann["category_id"] not in CAT_ID_TO_NAME:
            continue
        iw, ih = img_dims[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        pw, ph = w * 0.25, h * 0.25
        cw = min(float(iw), x + w + pw) - max(0.0, x - pw)
        ch = min(float(ih), y + h + ph) - max(0.0, y - ph)
        if cw < SMALL_THR and ch < SMALL_THR:
            small_ids.append(ann["id"])
    return frozenset(small_ids)


def ann_id_from_filename(filename: str) -> int | None:
    m = re.match(r"ann(\d+)_", filename)
    return int(m.group(1)) if m else None


# ── analysis 1 ────────────────────────────────────────────────────────────────

def print_color_table(by_cls: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 70)
    print("  [분석 1-1] 채널별 mean / std (crops_25pct, 0-1 scale)")
    print("=" * 70)
    print(f"  {'클래스':>10s} | {'R_μ':>6s} | {'G_μ':>6s} | {'B_μ':>6s} | "
          f"{'R_σ':>6s} | {'G_σ':>6s} | {'B_σ':>6s}")
    print("  " + "-" * 60)
    for cls in CLASSES:
        c = by_cls[cls]
        print(f"  {cls:>10s} | {c['R_mean']:6.4f} | {c['G_mean']:6.4f} | {c['B_mean']:6.4f} | "
              f"{c['R_std']:6.4f} | {c['G_std']:6.4f} | {c['B_std']:6.4f}")


def print_color_distances(by_cls: dict[str, dict[str, float]]) -> dict[str, float]:
    pairs = [("danger", "cut"), ("danger", "excluded"), ("cut", "excluded")]
    dists: dict[str, float] = {}
    print("\n  클래스 간 색상 L2 거리 (RGB 평균 벡터 기준)")
    print(f"  {'비교':>22s} | {'L2 거리':>9s} | {'편향 위험':>10s}")
    print("  " + "-" * 48)
    for a, b in pairs:
        v_a = np.array([by_cls[a]["R_mean"], by_cls[a]["G_mean"], by_cls[a]["B_mean"]])
        v_b = np.array([by_cls[b]["R_mean"], by_cls[b]["G_mean"], by_cls[b]["B_mean"]])
        d = float(np.linalg.norm(v_a - v_b))
        key = f"{a}_vs_{b}"
        dists[key] = d
        flag = "⚠ YES" if d > COLOR_BIAS_THR else "ok"
        print(f"  {a:>10s} vs {b:<10s} | {d:9.4f} | {flag:>10s}")
    return dists


def check_imagenet_diff(global_mean: np.ndarray, global_std: np.ndarray) -> np.ndarray:
    diff = np.abs(global_mean - IMAGENET_MEAN)
    print("\n  ImageNet 정규화값 비교")
    print(f"  {'':>12s} | {'R':>8s} | {'G':>8s} | {'B':>8s}")
    print("  " + "-" * 40)
    print(f"  {'dataset mean':>12s} | {global_mean[0]:8.4f} | {global_mean[1]:8.4f} | {global_mean[2]:8.4f}")
    print(f"  {'ImageNet mean':>12s} | {IMAGENET_MEAN[0]:8.4f} | {IMAGENET_MEAN[1]:8.4f} | {IMAGENET_MEAN[2]:8.4f}")
    print(f"  {'|diff|':>12s} | {diff[0]:8.4f} | {diff[1]:8.4f} | {diff[2]:8.4f}")
    std_diff = np.abs(global_std - IMAGENET_STD)
    print(f"  {'dataset std':>12s} | {global_std[0]:8.4f} | {global_std[1]:8.4f} | {global_std[2]:8.4f}")
    print(f"  {'ImageNet std':>12s} | {IMAGENET_STD[0]:8.4f} | {IMAGENET_STD[1]:8.4f} | {IMAGENET_STD[2]:8.4f}")
    print(f"  {'|diff| std':>12s} | {std_diff[0]:8.4f} | {std_diff[1]:8.4f} | {std_diff[2]:8.4f}")
    needs_custom = bool(np.any(diff > 0.05) or np.any(std_diff > 0.05))
    print(f"\n  커스텀 정규화 필요: {'YES ⚠' if needs_custom else 'no (ImageNet 사용 가능)'}")
    return diff


def plot_brightness(stats_25: dict[str, ClassStats]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 60)
    for cls in CLASSES:
        arr = np.array(stats_25[cls].brightness_vals)
        ax.hist(arr, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                label=f"{cls} (n={len(arr):,}  med={np.median(arr):.3f})",
                histtype="stepfilled")
    ax.set_xlabel("Per-image Luminance Mean (0–1)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Brightness Distribution by Class (crops_25pct)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    _save_fig(fig, "brightness_by_class.png")


def plot_hsv(stats_25: dict[str, ClassStats]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # H channel (per-image mean)
    ax = axes[0]
    bins_h = np.linspace(0, 1, 60)
    for cls in CLASSES:
        arr = np.array(stats_25[cls].h_means)
        ax.hist(arr, bins=bins_h, color=CLASS_COLORS[cls], alpha=0.55,
                label=f"{cls} (med={np.median(arr):.3f})", histtype="stepfilled")
    ax.set_xlabel("Per-image Mean Hue (0–1)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Hue Distribution (per-image mean, sat>0.1)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # H pixel distribution
    ax = axes[1]
    for cls in CLASSES:
        arr = np.array(stats_25[cls].h_pixels)
        ax.hist(arr, bins=np.linspace(0, 1, 80), color=CLASS_COLORS[cls], alpha=0.45,
                label=f"{cls} (n={len(arr):,})", histtype="stepfilled", density=True)
    ax.set_xlabel("Pixel Hue (0–1, sat>0.1)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Hue Pixel Distribution by Class (crops_25pct)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "hsv_by_class.png")


# ── analysis 2 ────────────────────────────────────────────────────────────────

def read_class_means_only(pad_key: str, cls: str) -> tuple[float, float, float]:
    """Return (R_mean, G_mean, B_mean) for given padding × class."""
    d = os.path.join(DATASET_CLS, pad_key, cls)
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")]
    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_read_one, fp) for fp in files]
        for fut in as_completed(futures):
            ps = fut.result()
            r_acc += ps["r_mean"]
            g_acc += ps["g_mean"]
            b_acc += ps["b_mean"]
    n = len(files)
    return r_acc / n, g_acc / n, b_acc / n


def analysis_2(stats_25: dict[str, ClassStats]) -> dict[str, float]:
    """Compare channel means across 0pct → 10pct → 25pct for cut and danger."""
    print("\n" + "=" * 70)
    print("  [분석 2] 패딩별 색상 이동 (채널 mean, 0-1 scale)")
    print("=" * 70)

    pad_means: dict[str, dict[str, tuple[float, float, float]]] = {p: {} for p in PAD_KEYS}

    # crops_25pct는 이미 읽었으므로 재사용
    for cls in ("cut", "danger"):
        s = stats_25[cls]
        pad_means["crops_25pct"][cls] = (
            float(np.mean(s.r_means)),
            float(np.mean(s.g_means)),
            float(np.mean(s.b_means)),
        )

    print("  0pct, 10pct 읽는 중…")
    for pad_key in ("crops_0pct", "crops_10pct"):
        for cls in ("cut", "danger"):
            pad_means[pad_key][cls] = read_class_means_only(pad_key, cls)

    # console table
    print(f"\n  {'':>10s} | {'채널':>4s} | {'0pct':>8s} | {'10pct':>8s} | {'25pct':>8s} | {'delta(0→25)':>11s}")
    print("  " + "-" * 62)
    deltas: dict[str, float] = {}
    for cls in ("cut", "danger"):
        for ch_idx, ch in enumerate(("R", "G", "B")):
            v0 = pad_means["crops_0pct"][cls][ch_idx]
            v10 = pad_means["crops_10pct"][cls][ch_idx]
            v25 = pad_means["crops_25pct"][cls][ch_idx]
            delta = abs(v25 - v0)
            print(f"  {cls:>10s} | {ch:>4s} | {v0:8.4f} | {v10:8.4f} | {v25:8.4f} | {delta:>11.4f}")
        # scalar delta: mean of 3 channel deltas
        d = float(np.mean([
            abs(pad_means["crops_25pct"][cls][i] - pad_means["crops_0pct"][cls][i])
            for i in range(3)
        ]))
        deltas[cls] = d
        flag = "⚠ YES" if d > PADDING_SHIFT_THR else "ok"
        print(f"  {cls:>10s} | {'mean Δ':>4s} | {'':>8s} | {'':>8s} | {'':>8s} | {d:>11.4f}  {flag}")
        print("  " + "·" * 60)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.suptitle("Channel Mean Shift by Padding Level", fontsize=11, fontweight="bold")
    x = np.arange(3)
    x_labels = ["0pct", "10pct", "25pct"]
    ch_colors = {"R": "#E53935", "G": "#43A047", "B": "#1E88E5"}
    for ax, cls in zip(axes, ("cut", "danger")):
        for ch_idx, ch in enumerate(("R", "G", "B")):
            vals = [pad_means[p][cls][ch_idx] for p in PAD_KEYS]
            ax.plot(x, vals, marker="o", color=ch_colors[ch], label=ch, linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{cls}", fontsize=10)
        ax.set_ylabel("Channel Mean (0–1)", fontsize=9)
        ax.set_xlabel("Padding Level", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "color_shift_by_padding.png")

    return deltas


# ── analysis 3 ────────────────────────────────────────────────────────────────

def analysis_3(
    stats_25: dict[str, ClassStats],
    small_ids: frozenset[int],
    class_dirs: dict[str, str],
) -> None:
    """Compare channel means: small (<224) vs normal (>=224) per class."""
    print("\n" + "=" * 70)
    print("  [분석 3] 소형 vs 일반 색상 차이 (crops_25pct)")
    print("=" * 70)
    print(f"  {'클래스':>10s} | {'그룹':>8s} | {'R_μ':>7s} | {'G_μ':>7s} | {'B_μ':>7s} | {'n':>6s}")
    print("  " + "-" * 58)

    for cls in CLASSES:
        cls_dir = class_dirs[cls]
        files = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]
        small_files = [f for f in files if (aid := ann_id_from_filename(f)) is not None and aid in small_ids]
        normal_files = [f for f in files if (aid := ann_id_from_filename(f)) is not None and aid not in small_ids]

        def batch_means(fnames: list[str]) -> tuple[float, float, float]:
            r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
            fps = [os.path.join(cls_dir, fn) for fn in fnames]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for ps in ex.map(_read_one, fps):
                    r_acc += ps["r_mean"]
                    g_acc += ps["g_mean"]
                    b_acc += ps["b_mean"]
            n = len(fnames)
            return r_acc / n, g_acc / n, b_acc / n

        if small_files:
            sr, sg, sb = batch_means(small_files)
            print(f"  {cls:>10s} | {'small':>8s} | {sr:7.4f} | {sg:7.4f} | {sb:7.4f} | {len(small_files):>6,d}")
        if normal_files:
            nr, ng, nb = batch_means(normal_files)
            print(f"  {cls:>10s} | {'normal':>8s} | {nr:7.4f} | {ng:7.4f} | {nb:7.4f} | {len(normal_files):>6,d}")
        if small_files and normal_files:
            delta = float(np.mean([abs(sr - nr), abs(sg - ng), abs(sb - nb)]))
            flag = "⚠ 차이 유의" if delta > 0.03 else "유사"
            print(f"  {cls:>10s} | {'|delta|':>8s} | {abs(sr-nr):7.4f} | {abs(sg-ng):7.4f} | {abs(sb-nb):7.4f}  {flag}")
        print("  " + "·" * 55)


# ── analysis 4 ────────────────────────────────────────────────────────────────

def compute_custom_norm(stats_25: dict[str, ClassStats]) -> tuple[np.ndarray, np.ndarray]:
    all_r = sum([s.r_means for s in stats_25.values()], [])
    all_g = sum([s.g_means for s in stats_25.values()], [])
    all_b = sum([s.b_means for s in stats_25.values()], [])
    all_r_sq = sum([s.r_sqs for s in stats_25.values()], [])
    all_g_sq = sum([s.g_sqs for s in stats_25.values()], [])
    all_b_sq = sum([s.b_sqs for s in stats_25.values()], [])
    mean = np.array([np.mean(all_r), np.mean(all_g), np.mean(all_b)])
    std = np.array([
        pooled_std(all_r, all_r_sq),
        pooled_std(all_g, all_g_sq),
        pooled_std(all_b, all_b_sq),
    ])
    return mean, std


# ── json + save ────────────────────────────────────────────────────────────────

def build_json_payload(
    by_cls: dict[str, dict[str, float]],
    dists: dict[str, float],
    deltas: dict[str, float],
    custom_mean: np.ndarray,
    custom_std: np.ndarray,
    imagenet_diff: np.ndarray,
) -> dict:
    color_bias_risk = any(v > COLOR_BIAS_THR for v in dists.values())
    background_influence = any(v > PADDING_SHIFT_THR for v in deltas.values())
    return {
        "color_analysis": {
            "by_class": {cls: by_cls[cls] for cls in CLASSES},
            "class_color_distance": {
                "danger_vs_cut": dists.get("danger_vs_cut", 0.0),
                "danger_vs_excluded": dists.get("danger_vs_excluded", 0.0),
                "cut_vs_excluded": dists.get("cut_vs_excluded", 0.0),
            },
            "color_bias_risk": color_bias_risk,
            "padding_color_shift": {
                "danger_delta": deltas.get("danger", 0.0),
                "cut_delta": deltas.get("cut", 0.0),
            },
            "background_influence": background_influence,
            "custom_normalization": {
                "mean": custom_mean.tolist(),
                "std": custom_std.tolist(),
            },
            "imagenet_diff": imagenet_diff.tolist(),
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


def _save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("소형 annotation ID 로드 중…")
    small_ids = load_small_ann_ids()
    print(f"  소형 annotation 수: {len(small_ids):,}")

    print("\ncops_25pct 읽는 중…")
    stats_25: dict[str, ClassStats] = {}
    class_dirs: dict[str, str] = {}
    for cls in CLASSES:
        d = os.path.join(DATASET_CLS, "crops_25pct", cls)
        class_dirs[cls] = d
        print(f"  {cls}…", end=" ", flush=True)
        stats_25[cls] = read_class_dir(d, cls)
        print(f"{len(stats_25[cls].r_means):,}장 완료")

    # ── 분석 1
    by_cls_summary = {cls: class_color_summary(stats_25[cls]) for cls in CLASSES}
    print_color_table(by_cls_summary)

    print("\n" + "=" * 70)
    print("  [분석 1-1 continued] 클래스 간 색상 거리 + ImageNet 비교")
    print("=" * 70)
    dists = print_color_distances(by_cls_summary)

    global_mean, global_std = compute_custom_norm(stats_25)
    imagenet_diff = check_imagenet_diff(global_mean, global_std)

    print("\n  [분석 1-3] 밝기 히스토그램 생성 중…")
    plot_brightness(stats_25)
    print("  저장: figures/brightness_by_class.png")

    print("\n  [분석 1-4] HSV H채널 히스토그램 생성 중…")
    plot_hsv(stats_25)
    print("  저장: figures/hsv_by_class.png")

    # ── 분석 2
    deltas = analysis_2(stats_25)
    print("  저장: figures/color_shift_by_padding.png")

    # ── 분석 3
    analysis_3(stats_25, small_ids, class_dirs)

    # ── 분석 4
    print("\n" + "=" * 70)
    print("  [분석 4] 커스텀 정규화값 (crops_25pct 전체)")
    print("=" * 70)
    print(f"  mean = [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]")
    print(f"  std  = [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]")

    # ── JSON
    payload = build_json_payload(
        by_cls_summary, dists, deltas, global_mean, global_std, imagenet_diff
    )
    save_eda_json(payload)

    print("\n" + "=" * 70)
    print("  저장 완료")
    print("=" * 70)
    print("  figures/brightness_by_class.png")
    print("  figures/hsv_by_class.png")
    print("  figures/color_shift_by_padding.png")
    print("  reports/eda/eda_results.json  (color_analysis 추가)")


if __name__ == "__main__":
    main()
