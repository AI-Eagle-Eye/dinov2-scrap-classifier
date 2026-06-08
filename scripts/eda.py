"""EDA script for the hazard-detection dataset.

Execution order: validate → basic → pixel → quality → features → label_noise (--full) → report

Usage:
    python scripts/eda.py --mode raw
    python scripts/eda.py --mode crop
    python scripts/eda.py --mode crop --data-dir dataset/classification/crops_25pct
    python scripts/eda.py --mode raw --bbox-format xyxy --check-duplicates
    python scripts/eda.py --mode crop --max-samples 200
    python scripts/eda.py --mode crop --full
    python scripts/eda.py --mode crop --force-reextract
"""
from __future__ import annotations

import argparse
import base64
import csv
import gc
import hashlib
import io
import json
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from scipy import ndimage as sp_ndimage
from scipy.stats import mannwhitneyu
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import shannon_entropy
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
import torch

# ── Optional libraries ────────────────────────────────────────────────────────

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from imagededup.methods import PHash as _PHash
    IMAGEDEDUP_AVAILABLE = True
except ImportError:
    IMAGEDEDUP_AVAILABLE = False

try:
    import cleanlab  # noqa: F401
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False

try:
    import imagehash as _imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES: tuple[str, ...] = ("위험", "안전", "제외")
CLASS_COLORS: dict[str, str] = {"위험": "#d62728", "안전": "#2ca02c", "제외": "#7f7f7f"}
CLASS_MARKERS: dict[str, str] = {"위험": "o", "안전": "s", "제외": "^"}

# 산점도 전용 시각 파라미터 — 제외를 최상위에 그려 소수 클래스 묻힘 방지
# cut(많음) → danger(중간) → excluded(적음, 최상위) 순서
SCATTER_DRAW_ORDER: tuple[str, ...] = ("안전", "위험", "제외")
SCATTER_PARAMS: dict[str, dict] = {
    "안전": {"color": "#2196F3", "marker": "s", "s": 10,  "alpha": 0.30,
             "edgecolors": "#1565C0", "linewidths": 0.3},
    "위험": {"color": "#F44336", "marker": "o", "s": 20,  "alpha": 0.50,
             "edgecolors": "#B71C1C", "linewidths": 0.4},
    "제외": {"color": "#9E9E9E", "marker": "^", "s": 30,  "alpha": 0.80,
             "edgecolors": "#424242", "linewidths": 0.5},
}

CLASS_DISPLAY: dict[str, str] = {"위험": "danger", "안전": "cut", "제외": "excluded"}

_LABEL_ALIASES: dict[str, str] = {
    "위험": "위험", "danger": "위험", "0": "위험",
    "안전": "안전", "safe": "안전", "cut": "안전", "1": "안전",
    "제외": "제외", "exclude": "제외", "excluded": "제외", "2": "제외",
}

IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)
CUSTOM_NORM_THRESHOLD: float = 0.05

# 데이터셋 실측 정규화값 (EDA 픽셀분석 산출)
DATASET_MEAN: tuple[float, ...] = (0.484, 0.483, 0.496)
DATASET_STD: tuple[float, ...] = (0.188, 0.188, 0.194)
FEATURES_SIZE: int = 336  # DINOv2 feature 추출용 LetterBox 해상도

RESOLUTION_4K_W: int = 3840
RESOLUTION_FHD_W: int = 1920

BLUR_LOW_QUALITY_THRESHOLD: float = 100.0
HEAVY_OCCLUSION_THRESHOLD: float = 0.3
MMD_DOMAIN_SHIFT_THRESHOLD: float = 0.1

BBOX_ABS_SMALL_PX: int = 100
BBOX_ABS_LARGE_PX: int = 300
BBOX_REL_SMALL: float = 0.05
BBOX_REL_LARGE: float = 0.20

IMG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
THUMB_SIZE: int = 224
BATCH_SIZE: int = 32  # 메모리 부족 시 줄일 수 있도록 상수로 분리
FEATURES_BATCH_SIZE: int = 32
PROTOTYPE_COUNT: int = 5
_PHASH_THRESHOLD: int = 12

_DBSCAN_EPSILON: float = 0.15
_DBSCAN_MIN_SAMPLES: int = 2
_SPLIT_RATIO: tuple[float, float, float] = (0.70, 0.15, 0.15)

_DOMINANT_COLOR_N_SAMPLES: int = 200
_DOMINANT_COLOR_K: int = 5
_DOMINANT_COLOR_RESIZE: int = 64  # resize before KMeans for speed

_GABOR_N_SAMPLES: int = 300
_GABOR_ORIENTATIONS: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4)
_GABOR_FREQUENCIES: tuple[float, ...] = (0.1, 0.3)
_GABOR_ORIENT_LABELS: tuple[str, ...] = ("0°", "45°", "90°", "135°")

BBox: TypeAlias = tuple[int, int, int, int]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class Sample:
    file_id: str
    label: str
    path: Path
    camera: str
    camera_source: str
    width: int
    height: int
    bbox_w: float
    bbox_h: float
    bbox_area: float
    bbox_area_ratio: float
    bbox_size_abs: str
    bbox_size_rel: str
    bbox_raw: BBox | None
    blur: float = 0.0
    edge_density: float = 0.0
    texture_entropy: float = 0.0
    occlusion_ratio: float = 0.0
    is_duplicate: bool = False
    is_low_quality: bool = False
    is_label_suspect: bool = False
    split: str = ""
    slice_camera: str = ""
    slice_blur: str = ""
    slice_bbox: str = ""
    slice_occlusion: str = ""


# ── Utility helpers ───────────────────────────────────────────────────────────

def _setup_korean_font() -> None:
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    from matplotlib import ft2font as _ft2

    # 폰트 캐시 강제 재로드 — 시스템 폰트 갱신 누락 방지
    try:
        mpl.font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    available = {f.name for f in fm.fontManager.ttflist}
    candidates = (
        "NanumGothic", "NanumBarunGothic",
        "Malgun Gothic", "AppleGothic", "DejaVu Sans",
    )
    _TEST_CHARS = "가나다라마바사"

    for name in candidates:
        if name not in available:
            continue
        try:
            path = fm.findfont(fm.FontProperties(family=name))
            ft = _ft2.FT2Font(path)
            # 실제 글리프 존재 여부로 OOV 확인
            if all(ft.get_char_index(ord(c)) != 0 for c in _TEST_CHARS):
                plt.rcParams["font.family"] = name
                break
        except Exception:
            continue

    plt.rcParams["axes.unicode_minus"] = False


def _normalise_label(raw: str) -> str | None:
    return _LABEL_ALIASES.get(str(raw).strip()) or _LABEL_ALIASES.get(str(raw).strip().lower())


def _infer_camera(file_name: str, json_camera: str | None, width: int) -> tuple[str, str]:
    """Returns (camera, camera_source)."""
    if json_camera:
        return str(json_camera), "annotation"
    stem = Path(file_name).stem.lower().replace("-", "_")
    for part in stem.split("_"):
        if part.startswith("cam") and part[3:].isdigit():
            return f"cam_{part[3:]}", "filename"
    if "4k" in stem:
        return "4K", "filename"
    if "fhd" in stem or "1080" in stem:
        return "FHD", "filename"
    camera = "4K" if width >= RESOLUTION_4K_W else ("FHD" if width >= RESOLUTION_FHD_W else "unknown")
    return camera, "resolution_heuristic"


def _bbox_sizes(bbox_w: float, bbox_h: float, image_area: float) -> tuple[str, str, float]:
    bbox_area = bbox_w * bbox_h
    ratio = bbox_area / max(image_area, 1.0)
    min_dim = min(bbox_w, bbox_h)
    abs_sz = "small" if min_dim < BBOX_ABS_SMALL_PX else ("large" if min_dim > BBOX_ABS_LARGE_PX else "medium")
    rel_sz = "small" if ratio < BBOX_REL_SMALL else ("large" if ratio > BBOX_REL_LARGE else "medium")
    return abs_sz, rel_sz, ratio


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _letterbox_resize(arr: np.ndarray, target: int) -> np.ndarray:
    """가로세로 비율을 유지하며 target×target 캔버스에 중앙 배치."""
    h, w = arr.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    b64 = _fig_to_b64(fig)
    path.write_bytes(base64.b64decode(b64))
    return b64


def _load_crop_arr(sample: Sample) -> np.ndarray | None:
    """Load and crop image; return HxWx3 uint8 RGB array or None."""
    try:
        with Image.open(sample.path) as im:
            im = im.convert("RGB")
            if sample.bbox_raw is not None:
                im = im.crop(sample.bbox_raw)
            return np.array(im)
    except Exception:
        return None


def _to_gray_f32(arr: np.ndarray) -> np.ndarray:
    """RGB uint8 → grayscale float32 [0, 1]."""
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return (gray / 255.0).astype(np.float32)


def _contrast_color(rgb: np.ndarray) -> str:
    """Return 'black' or 'white' for legible text on an RGB background swatch."""
    lum = 0.299 * float(rgb[0]) + 0.587 * float(rgb[1]) + 0.114 * float(rgb[2])
    return "black" if lum > 128.0 else "white"


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    X_sq = (X ** 2).sum(axis=1)
    Y_sq = (Y ** 2).sum(axis=1)
    dist_XX = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
    dist_YY = Y_sq[:, None] + Y_sq[None, :] - 2 * Y @ Y.T
    dist_XY = X_sq[:, None] + Y_sq[None, :] - 2 * X @ Y.T
    K_XX = np.exp(-gamma * np.clip(dist_XX, 0, None))
    K_YY = np.exp(-gamma * np.clip(dist_YY, 0, None))
    K_XY = np.exp(-gamma * np.clip(dist_XY, 0, None))
    return float(K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())


def _mannwhitney_pairs(values_by_class: dict[str, list[float]]) -> dict[str, dict]:
    pairs = [("위험", "안전"), ("위험", "제외"), ("안전", "제외")]
    results: dict[str, dict] = {}
    key_map = {"위험_안전": "danger_vs_safe", "위험_제외": "danger_vs_exclude", "안전_제외": "safe_vs_exclude"}
    for a, b in pairs:
        va, vb = values_by_class.get(a, []), values_by_class.get(b, [])
        key = key_map[f"{a}_{b}"]
        if len(va) < 2 or len(vb) < 2:
            results[key] = {"pvalue": 1.0, "significant": False}
            continue
        _, pvalue = mannwhitneyu(va, vb, alternative="two-sided")
        results[key] = {"pvalue": float(pvalue), "significant": bool(pvalue < 0.05)}
    return results


# ── Memory-efficient batch utilities ─────────────────────────────────────────

def iter_images_batched(paths: list[Path], batch_size: int = BATCH_SIZE):
    """배치 단위로 이미지를 읽고 yield.
    배치 처리 후 메모리 즉시 해제.
    이미지를 전부 메모리에 올리지 않기 위함.
    """
    import cv2
    for i in range(0, len(paths), batch_size):
        batch: list[np.ndarray] = []
        for p in paths[i:i + batch_size]:
            img = cv2.imread(str(p))
            if img is not None:
                batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        yield batch
        del batch
        gc.collect()


def log_memory() -> None:
    """현재 프로세스 메모리 사용량 로그 출력."""
    import logging
    mem = psutil.Process().memory_info().rss / 1024 ** 3
    logging.getLogger(__name__).info(f"[memory] {mem:.2f} GB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hazard detection dataset EDA")
    p.add_argument("--mode", choices=["raw", "crop"], required=True)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Data root (default: data/raw for raw, dataset/classification/crops_25pct for crop)")
    p.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    p.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy")
    p.add_argument("--check-duplicates", action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="빠른 모드: N개 랜덤 샘플만 분석")
    p.add_argument("--full", action="store_true", help="레이블 노이즈 분석 포함 (cleanlab)")
    p.add_argument("--force-reextract", action="store_true", help="feature 캐시 무시 후 재추출")
    return p.parse_args()


# ── Validation (Analysis 0) ───────────────────────────────────────────────────

def _validate_raw(data_dir: Path, bbox_format: str, report_path: Path) -> list[dict]:
    ann_path = data_dir / "annotations.json"
    images_dir = data_dir / "images"
    rows: list[dict] = []
    valid_records: list[dict] = []

    if not ann_path.exists():
        sys.exit("[오류] 데이터 디렉토리에서 annotations.json을 찾을 수 없습니다")

    with ann_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    records: list[dict] = raw if isinstance(raw, list) else _flatten_coco(raw)
    print(f"[validate] {len(records)} annotation records loaded")

    for rec in records:
        file_name = rec.get("file_name") or rec.get("filename") or rec.get("image_file") or ""
        file_id = Path(file_name).stem or f"rec_{len(rows)}"
        errors: list[str] = []

        # Image exists?
        img_path = images_dir / file_name if file_name else None
        if img_path is None or not img_path.exists():
            hits = list(data_dir.rglob(file_name)) if file_name else []
            img_path = hits[0] if hits else None
        if img_path is None:
            rows.append({"file_id": file_id, "valid": False,
                         "error_type": "missing_image", "error_detail": "image file not found"})
            continue

        # Label valid?
        label_raw = rec.get("label") or rec.get("category") or rec.get("class") or ""
        label = _normalise_label(str(label_raw))
        if label is None:
            errors.append(f"invalid_label:{label_raw!r}")

        # Image integrity
        orig_w = orig_h = 0
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
        except Exception as e:
            errors.append(f"corrupted_image:{e}")

        # BBox validation
        bbox_raw = rec.get("bbox") or rec.get("bounding_box")
        if bbox_raw and len(bbox_raw) == 4:
            if bbox_format == "xywh":
                x1, y1 = int(bbox_raw[0]), int(bbox_raw[1])
                x2, y2 = x1 + int(bbox_raw[2]), y1 + int(bbox_raw[3])
            else:
                x1, y1, x2, y2 = (int(v) for v in bbox_raw)
            if x2 <= x1 or y2 <= y1:
                errors.append("invalid_bbox:width_or_height_not_positive")
            elif orig_w and orig_h:
                if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h:
                    errors.append("invalid_bbox:out_of_bounds")

        if errors:
            rows.append({"file_id": file_id, "valid": False,
                         "error_type": errors[0].split(":")[0],
                         "error_detail": "; ".join(errors)})
        else:
            rows.append({"file_id": file_id, "valid": True,
                         "error_type": "", "error_detail": ""})
            valid_records.append({**rec, "_img_path": img_path, "_label": label,
                                   "_orig_w": orig_w, "_orig_h": orig_h})

    _write_validation_csv(rows, report_path)
    n_err = sum(1 for r in rows if not r["valid"])
    if n_err:
        print(f"[validate] ⚠  {n_err}/{len(rows)} 오류 — validation_report.csv 확인")
    else:
        print(f"[validate] ✅ 전체 {len(rows)}개 이상 없음")
    return valid_records


def _flatten_coco(data: dict) -> list[dict]:
    if "annotations" not in data or "images" not in data:
        return []
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_size = {img["id"]: (img.get("width", 0), img.get("height", 0)) for img in data["images"]}
    flat: list[dict] = []
    for ann in data["annotations"]:
        img_id = ann.get("image_id") or ann.get("img_id")
        entry = dict(ann)
        entry.setdefault("file_name", id_to_file.get(img_id, ""))
        w, h = id_to_size.get(img_id, (0, 0))
        entry.setdefault("width", w); entry.setdefault("height", h)
        flat.append(entry)
    return flat


def _write_validation_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "valid", "error_type", "error_detail"])
        writer.writeheader()
        writer.writerows(rows)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _pad_bbox(x1: int, y1: int, x2: int, y2: int, iw: int, ih: int,
              pad: float = 0.15) -> BBox:
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw * pad), int(bh * pad)
    return (max(0, x1 - pw), max(0, y1 - ph), min(iw, x2 + pw), min(ih, y2 + ph))


def load_raw(valid_records: list[dict], bbox_format: str) -> list[Sample]:
    samples: list[Sample] = []
    for rec in valid_records:
        img_path: Path = rec["_img_path"]
        label: str = rec["_label"]
        orig_w: int = rec["_orig_w"]
        orig_h: int = rec["_orig_h"]
        file_id = Path(rec.get("file_name", "")).stem or f"img_{len(samples)}"

        camera, camera_source = _infer_camera(
            rec.get("file_name", ""),
            rec.get("camera_id") or rec.get("camera"),
            orig_w,
        )

        bbox_raw_list = rec.get("bbox") or rec.get("bounding_box")
        padded: BBox | None = None
        bbox_w, bbox_h = float(orig_w), float(orig_h)
        if bbox_raw_list and len(bbox_raw_list) == 4:
            if bbox_format == "xywh":
                x1, y1 = int(bbox_raw_list[0]), int(bbox_raw_list[1])
                x2, y2 = x1 + int(bbox_raw_list[2]), y1 + int(bbox_raw_list[3])
            else:
                x1, y1, x2, y2 = (int(v) for v in bbox_raw_list)
            padded = _pad_bbox(x1, y1, x2, y2, orig_w, orig_h)
            bbox_w = float(padded[2] - padded[0])
            bbox_h = float(padded[3] - padded[1])

        abs_sz, rel_sz, ratio = _bbox_sizes(bbox_w, bbox_h, orig_w * orig_h)
        samples.append(Sample(
            file_id=file_id, label=label, path=img_path,
            camera=camera, camera_source=camera_source,
            width=orig_w, height=orig_h,
            bbox_w=bbox_w, bbox_h=bbox_h,
            bbox_area=bbox_w * bbox_h, bbox_area_ratio=ratio,
            bbox_size_abs=abs_sz, bbox_size_rel=rel_sz,
            bbox_raw=padded,
        ))
    return samples


def load_crop(data_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = _normalise_label(subdir.name)
        if label is None:
            continue
        for img_path in sorted(subdir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            # crop 이미지 해상도 = bbox 크기, 원본 카메라 해상도와 무관
            abs_sz, rel_sz, ratio = _bbox_sizes(float(w), float(h), w * h)
            samples.append(Sample(
                file_id=f"{label}_{img_path.stem}", label=label, path=img_path,
                camera="unknown", camera_source="not_available",
                width=w, height=h,
                bbox_w=float(w), bbox_h=float(h),
                bbox_area=float(w * h), bbox_area_ratio=ratio,
                bbox_size_abs=abs_sz, bbox_size_rel=rel_sz,
                bbox_raw=None,
            ))
    return samples


# ── Single-pass image processing ─────────────────────────────────────────────

def _compute_blur(gray: np.ndarray) -> float:
    return float(sp_ndimage.laplace(gray).var())


def _compute_edge_density(gray: np.ndarray) -> float:
    edges = canny(gray, sigma=1.0)
    return float(edges.mean())


def _compute_entropy(gray: np.ndarray) -> float:
    # shannon_entropy measures surface complexity, not fractal dimension
    return float(shannon_entropy(gray))


def _compute_occlusion(gray: np.ndarray) -> float:
    try:
        thresh = threshold_otsu(gray)
        foreground = gray > thresh
        return float(foreground.mean())
    except Exception:
        return 0.5


def _apply_blur_threshold(samples: list[Sample]) -> float:
    """하위 5%ile을 저품질 기준으로 설정하고 is_low_quality를 채운다."""
    blur_vals = [s.blur for s in samples if s.blur > 0]
    if not blur_vals:
        return BLUR_LOW_QUALITY_THRESHOLD
    threshold = float(np.percentile(blur_vals, 5))
    for s in samples:
        s.is_low_quality = s.blur < threshold
    lq = sum(1 for s in samples if s.is_low_quality)
    print(f"[blur] p5 threshold={threshold:.6f}  저품질: {lq}/{len(samples)}")
    return threshold


def process_images(samples: list[Sample]) -> None:
    """Single pass: fill per-sample metrics using batched loading."""
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="[image_proc] 이미지 처리"):
        batch = samples[i:i + BATCH_SIZE]
        arrs = [_load_crop_arr(s) for s in batch]
        for s, arr in zip(batch, arrs):
            if arr is None:
                continue
            gray = _to_gray_f32(arr)
            s.blur = _compute_blur(gray)
            s.edge_density = _compute_edge_density(gray)
            s.texture_entropy = _compute_entropy(gray)
            s.occlusion_ratio = _compute_occlusion(gray)
        del arrs
        gc.collect()


# ── Analysis 1: Basic statistics ──────────────────────────────────────────────

def _detect_duplicates_md5(samples: list[Sample]) -> dict[str, list[str]]:
    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for s in tqdm(samples, desc="[dup] MD5 해시"):
        try:
            h = hashlib.md5(s.path.read_bytes()).hexdigest()
            hash_to_ids[h].append(s.file_id)
        except Exception:
            pass
    return {k: v for k, v in hash_to_ids.items() if len(v) > 1}


def _detect_duplicates_simhash(samples: list[Sample]) -> dict[str, list[str]]:
    if not IMAGEDEDUP_AVAILABLE:
        return {}
    phasher = _PHash()
    encodings: dict[str, str] = {}
    for s in tqdm(samples, desc="[dup] PHash 인코딩"):
        try:
            enc = phasher.encode_image(image_file=str(s.path))
            if enc:
                encodings[s.file_id] = enc
        except Exception:
            pass
    dups = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=10)
    return {k: v for k, v in dups.items() if v}


def plot_class_distribution(samples: list[Sample], path: Path) -> str:
    counts = Counter(s.label for s in samples)
    total = max(sum(counts.values()), 1)
    labels = [c for c in CLASS_NAMES if c in counts]
    vals = [counts[c] for c in labels]
    colors = [CLASS_COLORS[c] for c in labels]
    display_labels = [CLASS_DISPLAY[c] for c in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(display_labels, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{v}\n({100*v/total:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("클래스별 이미지 수 및 비율", fontsize=13)
    ax.set_ylabel("이미지 수")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_camera_distribution(samples: list[Sample], path: Path) -> str:
    cameras = sorted({s.camera for s in samples})
    classes = [c for c in CLASS_NAMES if any(s.label == c for s in samples)]
    x = np.arange(len(cameras))
    bottoms = np.zeros(len(cameras))
    fig, ax = plt.subplots(figsize=(max(6, len(cameras) * 1.2), 4))
    for cls in classes:
        vals = np.array([sum(1 for s in samples if s.camera == cam and s.label == cls)
                         for cam in cameras], dtype=float)
        ax.bar(x, vals, 0.6, bottom=bottoms, color=CLASS_COLORS[cls], label=CLASS_DISPLAY[cls],
               edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(cameras, rotation=30, ha="right")
    ax.set_title("카메라별 클래스 분포", fontsize=13)
    ax.set_ylabel("이미지 수")
    ax.legend(title="클래스", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_bbox_absolute(samples: list[Sample], path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, dim in zip(axes,
                              [[s.bbox_w for s in samples], [s.bbox_h for s in samples]],
                              ["가로 크기 (px)", "세로 크기 (px)"]):
        ax.hist(vals, bins=30, color="#4878d0", edgecolor="white", linewidth=0.5)
        ax.axvline(BBOX_ABS_SMALL_PX, color="#d62728", ls="--", lw=1.2, label=f"소형 <{BBOX_ABS_SMALL_PX}px")
        ax.axvline(BBOX_ABS_LARGE_PX, color="#2ca02c", ls="--", lw=1.2, label=f"대형 >{BBOX_ABS_LARGE_PX}px")
        ax.set_title(f"BBox {dim}"); ax.set_xlabel("px"); ax.set_ylabel("건수")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    n_small = sum(1 for s in samples if s.bbox_size_abs == "small")
    fig.suptitle(f"BBox 절대 크기 분포 — small: {n_small}/{len(samples)}", fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_bbox_relative(samples: list[Sample], path: Path) -> str:
    ratios = [s.bbox_area_ratio for s in samples if s.bbox_area_ratio >= 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=30, color="#6acc65", edgecolor="white", linewidth=0.5)
    ax.axvline(BBOX_REL_SMALL, color="#d62728", ls="--", lw=1.2, label=f"소형 <{BBOX_REL_SMALL:.0%}")
    ax.axvline(BBOX_REL_LARGE, color="#2ca02c", ls="--", lw=1.2, label=f"대형 >{BBOX_REL_LARGE:.0%}")
    ax.set_title("BBox 상대 면적 비율 분포")
    ax.set_xlabel("bbox 면적 / 이미지 면적"); ax.set_ylabel("건수")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_aspect_ratio_by_class(samples: list[Sample], path: Path) -> str:
    """Aspect ratio (width/height) by class — horizontal boxplot + jitter overlay."""
    ratios_by_class: dict[str, list[float]] = {cls: [] for cls in CLASS_NAMES}
    for s in samples:
        if s.height > 0:
            ratios_by_class[s.label].append(s.width / s.height)

    classes = [c for c in CLASS_NAMES if ratios_by_class[c]]
    if not classes:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", transform=ax.transAxes)
        fig.tight_layout()
        return _save_fig(fig, path)

    data = [ratios_by_class[c] for c in classes]
    labels = [CLASS_DISPLAY[c] for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, vert=False, labels=labels, patch_artist=True,
                    widths=0.45, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter,
                   color=color, alpha=0.35, s=8, linewidths=0)

    ax.axvline(1.0, color="#888888", ls="--", lw=1.2, label="ratio=1.0 (정사각형)")
    ax.set_xlabel("aspect ratio (width / height)")
    ax.set_title("클래스별 Aspect Ratio 분포")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for cls in classes:
        print(f"  [{CLASS_DISPLAY[cls]}] median aspect ratio: {np.median(ratios_by_class[cls]):.3f}")

    return _save_fig(fig, path)


def analyse_basic(samples: list[Sample], figures_dir: Path,
                  check_duplicates: bool, mode: str = "raw") -> tuple[dict, dict[str, str]]:
    b64s: dict[str, str] = {}
    b64s["01_class_distribution"] = plot_class_distribution(
        samples, figures_dir / "01_class_distribution.png")
    b64s["03_bbox_absolute_size"] = plot_bbox_absolute(
        samples, figures_dir / "03_bbox_absolute_size.png")

    # crop 모드: 원본 해상도 미보유 → 카메라 분석 불가
    camera_available = mode != "crop"
    if camera_available:
        b64s["02_camera_distribution"] = plot_camera_distribution(
            samples, figures_dir / "02_camera_distribution.png")

    # crop 모드: bbox = 이미지 전체라 상대 면적 항상 1.0 → 의미없음
    if mode != "crop":
        b64s["04_bbox_relative_size"] = plot_bbox_relative(
            samples, figures_dir / "04_bbox_relative_size.png")

    b64s["05_aspect_ratio_by_class"] = plot_aspect_ratio_by_class(
        samples, figures_dir / "05_aspect_ratio_by_class.png")

    counts = Counter(s.label for s in samples)
    total = max(len(samples), 1)

    # Duplicate detection
    md5_groups = _detect_duplicates_md5(samples) if check_duplicates else {}
    sim_groups = _detect_duplicates_simhash(samples) if (check_duplicates and IMAGEDEDUP_AVAILABLE) else {}
    dup_ids = {fid for grp in md5_groups.values() for fid in grp}
    for s in samples:
        if s.file_id in dup_ids:
            s.is_duplicate = True

    max_cls = max(counts.values(), default=1)
    min_cls = max(min(counts.values(), default=1), 1)

    stats: dict = {
        "total": len(samples),
        "class_dist": {c: counts.get(c, 0) / total for c in CLASS_NAMES},
        "duplicate_md5_groups": len(md5_groups),
        "duplicate_simhash_groups": len(sim_groups),
        "weighted_sampler_needed": max_cls / min_cls > 2.0,
    }
    if camera_available:
        camera_counts = Counter(s.camera for s in samples)
        stats["camera_dist"] = dict(camera_counts)
        stats["camera_source"] = (
            Counter(s.camera_source for s in samples).most_common(1)[0][0] if samples else ""
        )
    else:
        stats["camera_info"] = "not_available"

    return stats, b64s


# ── Analysis 2: Pixel / colour analysis ──────────────────────────────────────

def analyse_pixel(samples: list[Sample],
                  figures_dir: Path) -> tuple[dict, dict[str, str]]:
    b64s: dict[str, str] = {}

    # Welford online algorithm: 이미지 단위로 채널 mean/std 누적 (픽셀 리스트 불필요)
    # ch_std는 이미지별 평균값의 std — 정규화 필요 여부 판단에 충분
    n_imgs = 0
    welford_mean = np.zeros(3, dtype=np.float64)
    welford_M2 = np.zeros(3, dtype=np.float64)

    # 배치별 np.histogram 누적 — 전체 픽셀 리스트 불필요
    bright_edges = np.linspace(0.0, 1.0, 31)
    hue_edges = np.linspace(0.0, 256.0, 37)
    bright_hist: dict[str, np.ndarray] = {c: np.zeros(30, dtype=np.float64) for c in CLASS_NAMES}
    hue_hist: dict[str, np.ndarray] = {c: np.zeros(36, dtype=np.float64) for c in CLASS_NAMES}

    # 평균 이미지: 누적 합산 방식
    sum_imgs: dict[str, np.ndarray] = {}
    img_counts: dict[str, int] = defaultdict(int)
    cam_brightness: dict[str, list[float]] = defaultdict(list)

    log_memory()
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="[pixel] 배치 처리"):
        for s in samples[i:i + BATCH_SIZE]:
            arr = _load_crop_arr(s)
            if arr is None:
                continue
            resized = np.array(Image.fromarray(arr).resize((THUMB_SIZE, THUMB_SIZE)))
            arr_f32 = resized.astype(np.float32) / 255.0

            # Welford: n+=1, delta=x-mean, mean+=delta/n, delta2=x-mean, M2+=delta*delta2
            img_ch_mean = arr_f32.mean(axis=(0, 1)).astype(np.float64)
            n_imgs += 1
            delta = img_ch_mean - welford_mean
            welford_mean += delta / n_imgs
            delta2 = img_ch_mean - welford_mean
            welford_M2 += delta * delta2

            # 밝기 히스토그램 누적
            brightness = float(arr_f32.mean())
            bh, _ = np.histogram([brightness], bins=bright_edges)
            bright_hist[s.label] += bh

            # H채널 히스토그램 누적
            pil_hsv = Image.fromarray(resized).convert("HSV")
            h_ch = np.array(pil_hsv)[:, :, 0]
            hh, _ = np.histogram(h_ch.ravel(), bins=hue_edges)
            hue_hist[s.label] += hh

            cam_brightness[s.camera].append(brightness)

            # 평균 이미지 누적 — DINOv2 실제 입력 해상도(336×336)로 별도 리사이즈
            mean_resized = np.array(Image.fromarray(arr).resize((FEATURES_SIZE, FEATURES_SIZE)))
            if s.label not in sum_imgs:
                sum_imgs[s.label] = np.zeros((FEATURES_SIZE, FEATURES_SIZE, 3), dtype=np.float64)
            sum_imgs[s.label] += mean_resized.astype(np.float64)
            img_counts[s.label] += 1

            del arr, resized, arr_f32, pil_hsv, h_ch, mean_resized
        gc.collect()

    if n_imgs == 0:
        log_memory()
        return {}, b64s

    ch_mean = welford_mean.astype(np.float32)
    ch_std = np.sqrt(welford_M2 / max(n_imgs - 1, 1)).astype(np.float32)
    diff_mean = (ch_mean - np.array(IMAGENET_MEAN)).tolist()
    custom_norm = bool(np.abs(diff_mean).max() > CUSTOM_NORM_THRESHOLD)

    # Figure 5: channel stats
    fig, ax = plt.subplots(figsize=(7, 3))
    channels = ["R", "G", "B"]
    x = np.arange(3)
    ax.bar(x - 0.2, ch_mean, 0.35, label="데이터 평균", color=["#d62728", "#2ca02c", "#4878d0"])
    ax.bar(x + 0.2, list(IMAGENET_MEAN), 0.35, label="ImageNet 평균",
           color=["#d62728", "#2ca02c", "#4878d0"], alpha=0.4, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(channels)
    ax.set_title("채널별 평균 (데이터 vs ImageNet)"); ax.legend(); ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b64s["06_channel_stats"] = _save_fig(fig, figures_dir / "06_channel_stats.png")

    # Figure 6: 밝기 히스토그램 (누적 히스토그램으로 렌더링)
    bright_centers = (bright_edges[:-1] + bright_edges[1:]) / 2
    bright_w = bright_edges[1] - bright_edges[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in CLASS_NAMES:
        if bright_hist[cls].sum() > 0:
            ax.bar(bright_centers, bright_hist[cls], width=bright_w,
                   alpha=0.5, color=CLASS_COLORS[cls], label=CLASS_DISPLAY[cls])
    ax.set_title("클래스별 밝기 분포"); ax.set_xlabel("평균 밝기"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b64s["07_brightness_by_class"] = _save_fig(fig, figures_dir / "07_brightness_by_class.png")

    # Figure 7: HSV H채널 분포 — 클래스별 픽셀 비율로 정규화 (클래스 간 비교 직관성)
    hue_centers = (hue_edges[:-1] + hue_edges[1:]) / 2
    hue_w = hue_edges[1] - hue_edges[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in CLASS_NAMES:
        total_px = hue_hist[cls].sum()
        if total_px > 0:
            normed = hue_hist[cls] / total_px
            ax.bar(hue_centers, normed, width=hue_w,
                   alpha=0.5, color=CLASS_COLORS[cls], label=CLASS_DISPLAY[cls])
    ax.set_title("클래스별 HSV H채널 분포"); ax.set_xlabel("색조 (Hue)")
    ax.set_ylabel("비율 (클래스 내 픽셀 비율)"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b64s["08_hsv_distribution"] = _save_fig(fig, figures_dir / "08_hsv_distribution.png")

    # Figure 8: 카메라별 밝기 비교 (카메라 정보 있을 때만)
    real_cams = {c for c in cam_brightness if c != "unknown"}
    if real_cams:
        fig, ax = plt.subplots(figsize=(max(5, len(cam_brightness)), 4))
        ax.boxplot([cam_brightness[c] for c in sorted(cam_brightness)],
                   tick_labels=sorted(cam_brightness), patch_artist=True)
        ax.set_title("카메라별 밝기 분포 비교"); ax.set_ylabel("평균 밝기")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        b64s["09_camera_brightness_compare"] = _save_fig(
            fig, figures_dir / "09_camera_brightness_compare.png"
        )

    # 평균 이미지: sum_img / count
    mean_imgs: dict[str, np.ndarray] = {
        cls: (sum_imgs[cls] / img_counts[cls]).astype(np.uint8)
        for cls in sum_imgs
        if img_counts[cls] > 0
    }
    del sum_imgs
    gc.collect()

    # Figure 9: 클래스별 평균 이미지
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 4))
    if len(CLASS_NAMES) == 1:
        axes = [axes]
    for ax, cls in zip(axes, CLASS_NAMES):
        if cls in mean_imgs:
            ax.imshow(mean_imgs[cls])
        ax.set_title(f"{CLASS_DISPLAY[cls]} 평균"); ax.axis("off")
    fig.suptitle(f"클래스별 평균 이미지 ({FEATURES_SIZE}×{FEATURES_SIZE})"); fig.tight_layout()
    b64s["10_mean_image_per_class"] = _save_fig(fig, figures_dir / "10_mean_image_per_class.png")

    # Figure 10: 차이 이미지 — 3쌍 diff (2×2) + RGB 동시 비교
    if mean_imgs:
        _diff_pairs: list[tuple[str, str]] = [
            ("안전", "위험"),   # cut - danger
            ("안전", "제외"),   # cut - excluded
            ("위험", "제외"),   # danger - excluded
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for _ax, (a, b) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], _diff_pairs):
            if a in mean_imgs and b in mean_imgs:
                _diff_gray = (mean_imgs[a].astype(float) - mean_imgs[b].astype(float)).mean(axis=2)
                _vabs = max(abs(_diff_gray.max()), abs(_diff_gray.min()), 1.0)
                _ax.imshow(_diff_gray, cmap="RdBu", vmin=-_vabs, vmax=_vabs)
                _ax.set_title(f"{CLASS_DISPLAY.get(a, a)} - {CLASS_DISPLAY.get(b, b)}")
            else:
                _ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                         transform=_ax.transAxes)
            _ax.axis("off")
        # 패널 4: RGB 동시 비교 (R=danger, G=cut, B=excluded)
        _rgb = np.zeros((FEATURES_SIZE, FEATURES_SIZE, 3), dtype=np.float32)
        for _cls, _ch in [("위험", 0), ("안전", 1), ("제외", 2)]:
            if _cls in mean_imgs:
                _gray = mean_imgs[_cls].astype(np.float32).mean(axis=2)
                _rgb[:, :, _ch] = (_gray - _gray.min()) / max(_gray.max() - _gray.min(), 1.0)
        axes[1, 1].imshow(np.clip(_rgb, 0.0, 1.0))
        axes[1, 1].set_title("RGB 동시 비교 (R=danger / G=cut / B=excluded)")
        axes[1, 1].axis("off")
        fig.suptitle("차이 이미지 (RdBu, 중앙=0)")
        fig.tight_layout()
        b64s["11_diff_image"] = _save_fig(fig, figures_dir / "11_diff_image.png")

    log_memory()
    pixel_stats = {
        "channel_mean": ch_mean.tolist(),
        "channel_std": ch_std.tolist(),
        "imagenet_diff_mean": diff_mean,
        "custom_normalization_needed": custom_norm,
    }
    return pixel_stats, b64s


# ── Analysis 3: Edge density & texture entropy ────────────────────────────────

def _class_boxplot(values_by_class: dict[str, list[float]], title: str,
                   ylabel: str, path: Path) -> str:
    classes = [c for c in CLASS_NAMES if values_by_class.get(c)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([values_by_class[c] for c in classes],
                    tick_labels=[CLASS_DISPLAY[c] for c in classes],
                    patch_artist=True, notch=False)
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS[cls])
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, path)


def analyse_quality_features(samples: list[Sample],
                              figures_dir: Path) -> tuple[dict, dict[str, str]]:
    b64s: dict[str, str] = {}

    edge_by_class: dict[str, list[float]] = defaultdict(list)
    entropy_by_class: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        if s.edge_density > 0 or s.texture_entropy > 0:
            edge_by_class[s.label].append(s.edge_density)
            entropy_by_class[s.label].append(s.texture_entropy)

    b64s["12_edge_density_boxplot"] = _class_boxplot(
        edge_by_class, "클래스별 엣지 밀도 (Canny, σ=1.0)", "엣지 밀도",
        figures_dir / "12_edge_density_boxplot.png")
    b64s["13_texture_entropy_boxplot"] = _class_boxplot(
        entropy_by_class, "클래스별 텍스처 엔트로피 (Shannon)", "텍스처 엔트로피",
        figures_dir / "13_texture_entropy_boxplot.png")

    edge_mw = _mannwhitney_pairs(edge_by_class)
    entropy_mw = _mannwhitney_pairs(entropy_by_class)

    stats = {
        "edge_stats": {
            "by_class": {c: {"median": float(np.median(v)), "mean": float(np.mean(v))}
                         for c, v in edge_by_class.items()},
            "mannwhitney": edge_mw,
        },
        "entropy_stats": {
            "by_class": {c: {"median": float(np.median(v))}
                         for c, v in entropy_by_class.items()},
            "mannwhitney": entropy_mw,
        },
    }
    return stats, b64s


# ── Analysis 4: Blur & occlusion ──────────────────────────────────────────────

def analyse_blur_occlusion(samples: list[Sample], figures_dir: Path,
                           blur_threshold: float) -> tuple[dict, dict[str, str]]:
    b64s: dict[str, str] = {}
    blur_vals = [s.blur for s in samples]
    low_quality = [s for s in samples if s.is_low_quality]

    # Figure 13: blur histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(blur_vals, bins=40, color="#4878d0", edgecolor="white", linewidth=0.5)
    ax.axvline(blur_threshold, color="#d62728", ls="--", lw=1.5,
               label=f"p5 임계값={blur_threshold:.5f}")
    ax.set_title(f"흐림도 점수 분포 (Laplacian 분산) — 저품질: {len(low_quality)}개")
    ax.set_xlabel("Laplacian 분산"); ax.set_ylabel("건수"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b64s["14_blur_histogram"] = _save_fig(fig, figures_dir / "14_blur_histogram.png")

    # Figure 14: blur vs edge scatter — 안전→제외→위험 순서로 위험 최상위 표시
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in SCATTER_DRAW_ORDER:
        cls_s = [s for s in samples if s.label == cls]
        if not cls_s:
            continue
        p = SCATTER_PARAMS[cls]
        ax.scatter([s.blur for s in cls_s], [s.edge_density for s in cls_s],
                   color=p["color"], marker=p["marker"], s=p["s"], alpha=p["alpha"],
                   edgecolors=p["edgecolors"], linewidths=p["linewidths"],
                   label=CLASS_DISPLAY[cls], zorder=3)
    ax.axvline(blur_threshold, color="black", ls=":", lw=1.0,
               label=f"p5 임계값={blur_threshold:.5f}")
    ax.set_title("흐림도 vs 엣지 밀도 (클래스별)"); ax.set_xlabel("흐림도 (Laplacian 분산)")
    ax.set_ylabel("엣지 밀도"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    b64s["15_blur_vs_edge_scatter"] = _save_fig(fig, figures_dir / "15_blur_vs_edge_scatter.png")

    # Figure 15: occlusion distribution
    occ_by_class: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        occ_by_class[s.label].append(s.occlusion_ratio)
    b64s["16_occlusion_distribution"] = _class_boxplot(
        occ_by_class, "클래스별 전경 비율 (Otsu occlusion 추정)", "전경 비율",
        figures_dir / "16_occlusion_distribution.png")

    heavy_occ = [s for s in samples if s.occlusion_ratio < HEAVY_OCCLUSION_THRESHOLD]
    stats = {
        "blur_stats": {
            "low_quality_count": len(low_quality),
            "low_quality_ratio": len(low_quality) / max(len(samples), 1),
            "low_quality_file_ids": [s.file_id for s in low_quality],
        },
        "occlusion_stats": {
            "by_class": {c: {"median": float(np.median(v))} for c, v in occ_by_class.items()},
            "heavy_occlusion_ratio": len(heavy_occ) / max(len(samples), 1),
        },
    }
    return stats, b64s


# ── Analysis 5: DINOv2 features ───────────────────────────────────────────────

def _extract_features(samples: list[Sample], device: torch.device,
                      cache_path: Path, force_reextract: bool,
                      skipped: list[dict]) -> np.ndarray | None:
    if cache_path.exists() and not force_reextract:
        print(f"[features] 캐시 로드: {cache_path.name}")
        return np.load(cache_path)
    try:
        with warnings.catch_warnings():
            # xFormers 미설치 환경에서 발생하는 불필요한 경고 억제
            warnings.filterwarnings("ignore", message="xFormers is not available")
            dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        dino.eval().to(device)
        for p in dino.parameters():
            p.requires_grad = False
    except Exception as e:
        skipped.append({"analysis": "DINOv2 features", "reason": str(e),
                        "install": "torch.hub (네트워크/캐시 확인)"})
        return None

    # 데이터셋 실측 통계 기반 정규화 (ImageNet 아님)
    norm_mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
    norm_std = torch.tensor(DATASET_STD).view(3, 1, 1)
    all_feats: list[np.ndarray] = []

    for i in tqdm(range(0, len(samples), FEATURES_BATCH_SIZE), desc="[features] CLS 추출"):
        batch_imgs = []
        for s in samples[i:i + FEATURES_BATCH_SIZE]:
            arr = _load_crop_arr(s)
            if arr is None:
                arr = np.zeros((FEATURES_SIZE, FEATURES_SIZE, 3), dtype=np.uint8)
            arr = _letterbox_resize(arr, FEATURES_SIZE)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            t = (t - norm_mean) / norm_std
            batch_imgs.append(t)
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feat = dino.forward_features(batch)["x_norm_clstoken"]
        all_feats.append(feat.cpu().numpy())
        del batch, feat, batch_imgs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    features = np.concatenate(all_feats, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    print(f"[features] 저장: {cache_path.name} (shape={features.shape})")
    return features


def analyse_features(samples: list[Sample], features: np.ndarray,
                     figures_dir: Path, interactive_dir: Path,
                     skipped: list[dict]) -> tuple[dict, dict[str, str]]:
    b64s: dict[str, str] = {}
    label_ids = [CLASS_NAMES.index(s.label) if s.label in CLASS_NAMES else -1 for s in samples]
    label_arr = np.array(label_ids)
    valid_mask = label_arr >= 0
    feats = features[valid_mask]
    labels_v = label_arr[valid_mask]
    samples_v = [s for s, m in zip(samples, valid_mask) if m]

    if len(feats) < 3:
        return {}, b64s

    # Silhouette
    sil_all = float(silhouette_score(feats, labels_v)) if len(set(labels_v)) > 1 else 0.0
    sil_samples = silhouette_samples(feats, labels_v) if len(set(labels_v)) > 1 else np.zeros(len(feats))
    sil_by_class = {CLASS_NAMES[i]: float(sil_samples[labels_v == i].mean())
                    for i in range(len(CLASS_NAMES)) if (labels_v == i).any()}

    # UMAP embedding
    embedding: np.ndarray | None = None
    if UMAP_AVAILABLE:
        try:
            reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
                                    low_memory=True)
            embedding = reducer.fit_transform(feats)
        except Exception as e:
            skipped.append({"analysis": "UMAP", "reason": str(e), "install": "pip install umap-learn"})
    else:
        skipped.append({"analysis": "UMAP", "reason": "umap-learn 미설치",
                        "install": "pip install umap-learn"})

    if embedding is not None:
        # Static matplotlib UMAP — 안전→제외→위험 순서, KDE 등고선 추가
        fig, ax = plt.subplots(figsize=(8, 6))

        # KDE 등고선 먼저 (배경): 클러스터 경계 시각화
        try:
            import seaborn as sns
            for cls in SCATTER_DRAW_ORDER:
                mask = np.array([s.label == cls for s in samples_v])
                if mask.sum() < 10:
                    continue
                sns.kdeplot(
                    x=embedding[mask, 0], y=embedding[mask, 1], ax=ax,
                    color=SCATTER_PARAMS[cls]["color"],
                    fill=True, alpha=0.08, levels=4, bw_adjust=0.7,
                )
        except ImportError:
            pass

        # Scatter: 위험이 마지막(최상위)에 그려져 겹침 최소화
        for cls in SCATTER_DRAW_ORDER:
            mask = np.array([s.label == cls for s in samples_v])
            if not mask.any():
                continue
            p = SCATTER_PARAMS[cls]
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       color=p["color"], marker=p["marker"],
                       s=p["s"], alpha=p["alpha"],
                       edgecolors=p["edgecolors"], linewidths=p["linewidths"],
                       label=CLASS_DISPLAY[cls], zorder=3)

        ax.set_title(f"UMAP 클러스터 (silhouette={sil_all:.3f})")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        b64s["17_umap_combined"] = _save_fig(fig, figures_dir / "17_umap_combined.png")

        # Interactive — 데이터 분리형 HTML (Plotly.js CDN 렌더링)
        try:
            import json as _json

            # ── 2D 데이터 추출 ────────────────────────────────────
            _labels_all = [s.file_id for s in samples_v]
            _classes_all = [CLASS_DISPLAY[s.label] for s in samples_v]
            _x2d = embedding[:, 0].tolist()
            _y2d = embedding[:, 1].tolist()

            # ── 3D UMAP 계산 ──────────────────────────────────────
            _x3d: list[float] = []
            _y3d: list[float] = []
            _z3d: list[float] = []
            if UMAP_AVAILABLE:
                try:
                    _red3d = umap_lib.UMAP(
                        n_components=3, random_state=42,
                        n_neighbors=15, min_dist=0.1, low_memory=True,
                    )
                    _emb3d = _red3d.fit_transform(feats)
                    _x3d = _emb3d[:, 0].tolist()
                    _y3d = _emb3d[:, 1].tolist()
                    _z3d = _emb3d[:, 2].tolist()
                except Exception as _e3:
                    skipped.append({
                        "analysis": "UMAP 3D",
                        "reason": str(_e3),
                        "install": "",
                    })

            # ── DATA JSON 직렬화 ──────────────────────────────────
            _n_samples = len(samples_v)
            _data_json = _json.dumps(
                {
                    "labels":  _labels_all,
                    "classes": _classes_all,
                    "x2d":     _x2d,
                    "y2d":     _y2d,
                    "x3d":     _x3d,
                    "y3d":     _y3d,
                    "z3d":     _z3d,
                },
                ensure_ascii=False,
            )

            # ── HTML 템플릿 (삼중따옴표, 구역 분리) ──────────────
            _html_tpl = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>UMAP Interactive</title>
  <style>

    /* === 1. CSS 영역 === */

    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100vh;
      background: #fff;
    }

    .container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    .info-bar {
      height: 50px;
      display: flex;
      align-items: center;
      padding: 0 16px;
      font-family: sans-serif;
      font-size: 14px;
      color: #333;
      background: #f8f9fa;
      border-bottom: 1px solid #dee2e6;
      flex-shrink: 0;
    }

    .tab-bar {
      height: 40px;
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 8px;
      background: #fff;
      border-bottom: 1px solid #dee2e6;
      flex-shrink: 0;
    }

    .tab-btn {
      padding: 4px 20px;
      cursor: pointer;
      border: 1px solid #ccc;
      border-radius: 4px;
      background: #f5f5f5;
      font-size: 13px;
    }

    .tab-btn.active {
      background: #4878d0;
      color: #fff;
      border-color: #4878d0;
    }

    .plot-area {
      flex: 1;
      width: 100%;
    }

    .plot-area.hidden {
      display: none;
    }

  </style>
</head>
<body>

  <!-- === 2. 레이아웃 영역 === -->

  <div class="container">
    <div class="info-bar">
      UMAP 클러스터
      &nbsp;|&nbsp; silhouette = __SIL__
      &nbsp;|&nbsp; 샘플 수 = __N_SAMPLES__
    </div>
    <div class="tab-bar">
      <button id="btn2d" class="tab-btn active"
              onclick="switchTab('2d')">
        2D
      </button>
      <button id="btn3d" class="tab-btn"
              onclick="switchTab('3d')">
        3D
      </button>
    </div>
    <div class="plot-area" id="plot2d"></div>
    <div class="plot-area hidden" id="plot3d"></div>
  </div>

  <script
    src="https://cdn.plot.ly/plotly-latest.min.js">
  </script>
  <script>

    /* === 3-A. 데이터 영역 === */

    const DATA = __DATA_JSON__;

    /* === 3-B. 로직 영역 === */

    const CLASS_CFG = {
      danger: {
        color: '#e74c3c', sym2d: 'circle', sym3d: 'circle',
      },
      cut: {
        color: '#2ecc71', sym2d: 'square', sym3d: 'square',
      },
      excluded: {
        color: '#95a5a6', sym2d: 'triangle-up', sym3d: 'diamond',
      },
    };

    const LAYOUT_BASE = {
      margin:        { t: 30, b: 40, l: 50, r: 20 },
      legend:        { title: { text: '클래스' } },
      paper_bgcolor: '#fff',
      plot_bgcolor:  '#fff',
    };

    const CFG = { responsive: true };

    function indices(cls) {
      return DATA.classes.reduce(function(acc, c, i) {
        if (c === cls) acc.push(i);
        return acc;
      }, []);
    }

    function buildTraces2d() {
      return Object.keys(CLASS_CFG).map(function(cls) {
        var cfg = CLASS_CFG[cls];
        var idx = indices(cls);
        return {
          type: 'scatter',
          mode: 'markers',
          name: cls,
          x:    idx.map(function(i) { return DATA.x2d[i]; }),
          y:    idx.map(function(i) { return DATA.y2d[i]; }),
          text: idx.map(function(i) { return DATA.labels[i]; }),
          hovertemplate: '%{text}<extra></extra>',
          marker: {
            color: cfg.color, symbol: cfg.sym2d,
            size: 5, opacity: 0.7,
          },
        };
      });
    }

    function buildTraces3d() {
      if (DATA.x3d.length === 0) return [];
      return Object.keys(CLASS_CFG).map(function(cls) {
        var cfg = CLASS_CFG[cls];
        var idx = indices(cls);
        return {
          type: 'scatter3d',
          mode: 'markers',
          name: cls,
          x:    idx.map(function(i) { return DATA.x3d[i]; }),
          y:    idx.map(function(i) { return DATA.y3d[i]; }),
          z:    idx.map(function(i) { return DATA.z3d[i]; }),
          text: idx.map(function(i) { return DATA.labels[i]; }),
          hovertemplate: '%{text}<extra></extra>',
          marker: {
            color: cfg.color, symbol: cfg.sym3d,
            size: 5, opacity: 0.7,
          },
        };
      });
    }

    Plotly.newPlot(
      'plot2d',
      buildTraces2d(),
      Object.assign({}, LAYOUT_BASE, {
        xaxis: { title: 'UMAP-1' },
        yaxis: { title: 'UMAP-2' },
      }),
      CFG
    );

    var traces3d = buildTraces3d();
    if (traces3d.length > 0) {
      Plotly.newPlot(
        'plot3d',
        traces3d,
        Object.assign({}, LAYOUT_BASE, {
          scene: {
            xaxis: { title: 'UMAP-1' },
            yaxis: { title: 'UMAP-2' },
            zaxis: { title: 'UMAP-3' },
          },
        }),
        CFG
      );
    } else {
      var b3 = document.getElementById('btn3d');
      b3.disabled = true;
      b3.title = 'UMAP 3D 미지원';
    }

    function switchTab(tab) {
      var p2d = document.getElementById('plot2d');
      var p3d = document.getElementById('plot3d');
      var b2d = document.getElementById('btn2d');
      var b3d = document.getElementById('btn3d');
      if (tab === '2d') {
        p2d.classList.remove('hidden');
        p3d.classList.add('hidden');
        b2d.classList.add('active');
        b3d.classList.remove('active');
        Plotly.Plots.resize(p2d);
      } else {
        p2d.classList.add('hidden');
        p3d.classList.remove('hidden');
        b2d.classList.remove('active');
        b3d.classList.add('active');
        Plotly.Plots.resize(p3d);
      }
    }

  </script>
</body>
</html>"""

            _html = (
                _html_tpl
                .replace("__SIL__", f"{sil_all:.3f}")
                .replace("__N_SAMPLES__", str(_n_samples))
                .replace("__DATA_JSON__", _data_json)
            )

            interactive_dir.mkdir(parents=True, exist_ok=True)
            (interactive_dir / "umap_interactive.html").write_text(
                _html, encoding="utf-8",
            )
            print("[features] 인터랙티브 UMAP (2D+3D) 저장 완료")
        except Exception as e:
            skipped.append({
                "analysis": "인터랙티브 UMAP",
                "reason": str(e),
                "install": "pip install umap-learn",
            })

    # Prototypes: nearest to class centroid
    proto_paths: dict[str, list[str]] = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels_v == i
        if not mask.any():
            continue
        cls_feats = feats[mask]
        centroid = cls_feats.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(cls_feats - centroid, axis=1)
        top_idx = np.argsort(dists)[:PROTOTYPE_COUNT]
        cls_samples = [s for s, m in zip(samples_v, mask) if m]
        proto_paths[cls] = [cls_samples[j].file_id for j in top_idx]

    n_proto_classes = len(proto_paths)
    if n_proto_classes:
        fig = plt.figure(figsize=(PROTOTYPE_COUNT * 2.5, n_proto_classes * 2.5))
        gs = gridspec.GridSpec(n_proto_classes, PROTOTYPE_COUNT, figure=fig)
        for row, (cls, file_ids) in enumerate(proto_paths.items()):
            id_to_sample = {s.file_id: s for s in samples_v}
            for col, fid in enumerate(file_ids):
                ax = fig.add_subplot(gs[row, col])
                s = id_to_sample.get(fid)
                if s:
                    arr = _load_crop_arr(s)
                    if arr is not None:
                        ax.imshow(Image.fromarray(arr).resize((128, 128)))
                ax.set_title(f"{CLASS_DISPLAY[cls]}[{col}]", fontsize=7); ax.axis("off")
        fig.suptitle("클래스별 대표 샘플 이미지 (centroid 최근접)")
        fig.tight_layout()
        b64s["18_prototype_images"] = _save_fig(fig, figures_dir / "18_prototype_images.png")

    # 18: DBSCAN 기반 중복 / 유사 탐지
    import itertools as _itertools
    from sklearn.cluster import DBSCAN as _DBSCAN
    from sklearn.preprocessing import normalize as _sk_normalize

    # ── Step 1: 전처리 (전체 이미지, 루프 밖에서 한 번만) ────────────────────
    if not IMAGEHASH_AVAILABLE:
        print(
            "[18] ERROR: imagehash 미설치 — pHash 실행 불가.\n"
            "       pip install imagehash 후 재실행하세요."
        )
        sys.exit(1)

    md5_of: dict[str, str] = {}
    for s in samples_v:
        try:
            md5_of[s.file_id] = hashlib.md5(s.path.read_bytes()).hexdigest()
        except OSError:
            md5_of[s.file_id] = ""

    phash_of: dict[str, object] = {}
    for s in samples_v:
        try:
            phash_of[s.file_id] = _imagehash.phash(Image.open(s.path))
        except Exception:
            pass

    # L2 정규화: cosine_dist ≤ ε ↔ euclidean_dist ≤ √(2ε) (L2 정규화 벡터 기준)
    _feats_norm = _sk_normalize(feats, norm="l2")
    _eps_euc = float(np.sqrt(2.0 * _DBSCAN_EPSILON))
    _fid_list = [s.file_id for s in samples_v]

    # ── Step 2: DBSCAN 클러스터링 ────────────────────────────────────────────
    _db = _DBSCAN(
        eps=_eps_euc,
        min_samples=_DBSCAN_MIN_SAMPLES,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    _clabels = _db.fit_predict(_feats_norm)  # shape (n_valid,), -1 = noise

    # ── Step 3: 클러스터 내 쌍 분류 ─────────────────────────────────────────
    _clu_to_idx: dict[int, list[int]] = defaultdict(list)
    for _i, _cid in enumerate(_clabels.tolist()):
        if _cid >= 0:
            _clu_to_idx[_cid].append(_i)

    confusion_pairs: list[dict] = []
    _seen: set[tuple[str, str]] = set()

    for _cid, _idxs in _clu_to_idx.items():
        for _ii, _jj in _itertools.combinations(_idxs, 2):
            _fi = _fid_list[_ii]
            _fj = _fid_list[_jj]
            _key = (min(_fi, _fj), max(_fi, _fj))
            if _key in _seen:
                continue
            _seen.add(_key)

            _m5i = md5_of.get(_fi, "")
            _m5j = md5_of.get(_fj, "")
            if _m5i and _m5j and _m5i == _m5j:
                _mtype, _score = "exact", 0.0
            else:
                _phi = phash_of.get(_fi)
                _phj = phash_of.get(_fj)
                if _phi is not None and _phj is not None:
                    _ham = int(_phi - _phj)
                    if _ham <= _PHASH_THRESHOLD:
                        _mtype = "resolution_diff"
                        _score = float(_ham)
                    else:
                        _cos = float(
                            1.0 - float(np.dot(_feats_norm[_ii], _feats_norm[_jj]))
                        )
                        _mtype, _score = "similar", round(max(0.0, _cos), 6)
                else:
                    _cos = float(
                        1.0 - float(np.dot(_feats_norm[_ii], _feats_norm[_jj]))
                    )
                    _mtype, _score = "similar", round(max(0.0, _cos), 6)

            confusion_pairs.append({
                "img_a": _fi, "img_b": _fj,
                "match_type": _mtype, "score": _score,
                "cluster_id": _cid,
            })

    # ── Step 4: 클러스터 단위 split 할당 ────────────────────────────────────
    _clu_to_samps: dict[int, list[Sample]] = defaultdict(list)
    for _i, s in enumerate(samples_v):
        _clu_to_samps[int(_clabels[_i])].append(s)

    _pos_cids = [c for c in _clu_to_samps if c >= 0]
    _rng_sp = random.Random(42)
    _rng_sp.shuffle(_pos_cids)

    # 클러스터 greedy 할당: 비율 부족한 split에 우선 배정
    _n_total_clu = sum(len(_clu_to_samps[c]) for c in _pos_cids)
    _sp_n: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    _sp_tgt = {
        "train": _SPLIT_RATIO[0],
        "val":   _SPLIT_RATIO[1],
        "test":  _SPLIT_RATIO[2],
    }
    _cid_sp: dict[int, str] = {}
    for _cid in _pos_cids:
        _best = min(
            ("train", "val", "test"),
            key=lambda _sp: _sp_n[_sp] / max(_n_total_clu, 1) - _sp_tgt[_sp],
        )
        _cid_sp[_cid] = _best
        _sp_n[_best] += len(_clu_to_samps[_cid])

    # noise (cluster_id=-1): 클래스별 stratified split
    _noise_by_cls: dict[str, list[Sample]] = defaultdict(list)
    for _s in _clu_to_samps.get(-1, []):
        _noise_by_cls[_s.label].append(_s)

    _fid_sp: dict[str, str] = {}
    for _cls_samps in _noise_by_cls.values():
        _rng_sp.shuffle(_cls_samps)
        _n = len(_cls_samps)
        _n_tr = round(_n * _SPLIT_RATIO[0])
        _n_va = round(_n * _SPLIT_RATIO[1])
        for _k, _s in enumerate(_cls_samps):
            if _k < _n_tr:
                _fid_sp[_s.file_id] = "train"
            elif _k < _n_tr + _n_va:
                _fid_sp[_s.file_id] = "val"
            else:
                _fid_sp[_s.file_id] = "test"

    for _cid, _samps in _clu_to_samps.items():
        if _cid >= 0:
            for _s in _samps:
                _fid_sp[_s.file_id] = _cid_sp[_cid]

    _final_sp: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for _v in _fid_sp.values():
        _final_sp[_v] += 1

    # cluster_groups.csv 저장
    _out_dir = figures_dir.parent
    _cg_path = _out_dir / "cluster_groups.csv"
    _cg_path.parent.mkdir(parents=True, exist_ok=True)
    with _cg_path.open("w", newline="", encoding="utf-8") as _cf:
        _cgw = csv.DictWriter(
            _cf,
            fieldnames=["file_path", "cluster_id", "class_label", "split"],
        )
        _cgw.writeheader()
        for _i, _s in enumerate(samples_v):
            _cgw.writerow({
                "file_path": str(_s.path),
                "cluster_id": int(_clabels[_i]),
                "class_label": _s.label,
                "split": _fid_sp.get(_s.file_id, "train"),
            })

    # ── 콘솔 출력 ────────────────────────────────────────────────────────────
    _n_clu_cnt  = len(_pos_cids)
    _n_clu_imgs = sum(len(_clu_to_samps[c]) for c in _pos_cids)
    _n_noise_imgs = len(_clu_to_samps.get(-1, []))
    _n_exact = sum(1 for p in confusion_pairs if p["match_type"] == "exact")
    _n_res   = sum(1 for p in confusion_pairs if p["match_type"] == "resolution_diff")
    _n_sim   = sum(1 for p in confusion_pairs if p["match_type"] == "similar")
    print(f"[18] DBSCAN 클러스터 수: {_n_clu_cnt}개")
    print(f"[18] exact: {_n_exact}쌍 / resolution_diff: {_n_res}쌍 / similar: {_n_sim}쌍")
    print(f"[18] 클러스터 포함 이미지: {_n_clu_imgs}장 / 단독 이미지: {_n_noise_imgs}장")
    print(
        f"[18] split 할당 완료 → "
        f"train: {_final_sp['train']} / val: {_final_sp['val']} / test: {_final_sp['test']}"
    )

    if confusion_pairs:
        sim_pairs = [p for p in confusion_pairs if p["match_type"] == "similar"]
        viz_pairs = (sim_pairs or confusion_pairs)[:5]
        n_viz = len(viz_pairs)
        fig, axes = plt.subplots(n_viz, 2, figsize=(5, n_viz * 2.5))
        if n_viz == 1:
            axes = [axes]
        id_to_sample_map = {s.file_id: s for s in samples}
        for viz_row, viz_pair in enumerate(viz_pairs):
            for col, fid in enumerate([viz_pair["img_a"], viz_pair["img_b"]]):
                ax = axes[viz_row][col]
                s = id_to_sample_map.get(fid)
                if s:
                    arr = _load_crop_arr(s)
                    if arr is not None:
                        ax.imshow(Image.fromarray(arr).resize((112, 112)))
                cls_label = CLASS_DISPLAY.get(s.label, s.label) if s else "?"
                ax.set_title(f"[{cls_label}]\n{fid}", fontsize=6)
                ax.axis("off")
        fig.suptitle(f"중복/유사 샘플 쌍 (상위 {n_viz}쌍)")
        fig.tight_layout()
        b64s["19_confusion_prone_pairs"] = _save_fig(fig, figures_dir / "19_confusion_prone_pairs.png")

    # MMD: 4K vs FHD
    cam_4k = np.array([s.camera == "4K" for s in samples_v])
    cam_fhd = np.array([s.camera == "FHD" for s in samples_v])
    mmd_val = 0.0
    domain_shift = False
    if cam_4k.sum() >= 2 and cam_fhd.sum() >= 2:
        mmd_val = _mmd_rbf(feats[cam_4k], feats[cam_fhd])
        domain_shift = mmd_val > MMD_DOMAIN_SHIFT_THRESHOLD

    stats = {
        "umap_stats": {
            "silhouette_score": sil_all,
            "silhouette_by_class": sil_by_class,
            "confusion_prone_pairs": confusion_pairs,
        },
        "mmd_stats": {
            "4k_vs_fhd": mmd_val,
            "domain_shift_detected": domain_shift,
        },
    }
    return stats, b64s


# ── Analysis 6: Label noise (--full + cleanlab) ───────────────────────────────

_CLEANLAB_MAX_SAMPLES: int = 5000
# silhouette 이 이 값 미만이면 분류기가 무작위 수준 → cleanlab 결과 신뢰 불가
SILHOUETTE_LOW_CONFIDENCE: float = 0.1


def analyse_label_noise(samples: list[Sample], features: np.ndarray,
                        figures_dir: Path, skipped: list[dict],
                        sil_score: float = 0.0) -> dict | None:
    if not CLEANLAB_AVAILABLE:
        skipped.append({"analysis": "레이블 노이즈 (cleanlab)", "reason": "cleanlab 미설치",
                        "install": "pip install cleanlab"})
        return None
    try:
        from cleanlab.filter import find_label_issues
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        label_ids = le.fit_transform([s.label for s in samples])

        # 클래스 비율 유지 stratified 샘플링 (최대 5,000장)
        n_total = len(features)
        rng = np.random.default_rng(42)
        if n_total > _CLEANLAB_MAX_SAMPLES:
            parts: list[np.ndarray] = []
            for cls_id in np.unique(label_ids):
                cls_idx = np.where(label_ids == cls_id)[0]
                n_cls = max(1, round(_CLEANLAB_MAX_SAMPLES * len(cls_idx) / n_total))
                parts.append(rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False))
            sampled_indices = np.sort(np.concatenate(parts))
            print(f"[label_noise] stratified 샘플링: {n_total} → {len(sampled_indices)}")
        else:
            sampled_indices = np.arange(n_total)

        X = features[sampled_indices]
        y = label_ids[sampled_indices]

        # SGDClassifier — 배치 학습으로 메모리 절약, cv=3
        clf = SGDClassifier(
            loss="log_loss", random_state=42, max_iter=1000,
            class_weight="balanced", n_jobs=1,
        )
        pred_probs = cross_val_predict(clf, X, y, cv=3, method="predict_proba")
        issues = find_label_issues(y, pred_probs, return_indices_ranked_by="self_confidence")

        # pred_probs 삭제 전에 argmax 추출 (del 이후 접근 불가)
        pred_label_ids = np.argmax(pred_probs, axis=1)
        del X, pred_probs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 샘플링 인덱스 → 원본 samples 인덱스 역매핑
        orig_issue_indices = [int(sampled_indices[i]) for i in issues]
        for orig_idx in orig_issue_indices:
            samples[orig_idx].is_label_suspect = True

        # 상위 20개: given/predicted label 쌍 기록
        suspect_ids = [samples[orig_idx].file_id for orig_idx in orig_issue_indices[:20]]
        suspect_samples: list[dict] = []
        for sampled_i, orig_idx in zip(issues[:20], orig_issue_indices[:20]):
            pred_label = le.inverse_transform([int(pred_label_ids[sampled_i])])[0]
            s = samples[orig_idx]
            suspect_samples.append({
                "file_id": s.file_id,
                "given_label": CLASS_DISPLAY[s.label],
                "predicted_label": CLASS_DISPLAY[pred_label],
                "mismatch": pred_label != s.label,
            })

        # Figure 19: thumbnails — "pred: X / given: Y" 형태로 표시
        low_conf = sil_score < SILHOUETTE_LOW_CONFIDENCE
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        for ax, (sampled_i, orig_idx) in zip(axes.ravel(),
                                              zip(issues[:20], orig_issue_indices[:20])):
            s = samples[orig_idx]
            pred_label = le.inverse_transform([int(pred_label_ids[sampled_i])])[0]
            arr = _load_crop_arr(s)
            if arr is not None:
                ax.imshow(Image.fromarray(arr).resize((112, 112)))
            flag = "!=" if pred_label != s.label else "=="
            ax.set_title(
                f"pred:{CLASS_DISPLAY[pred_label]} {flag}\ngiven:{CLASS_DISPLAY[s.label]}\n{s.file_id[:10]}",
                fontsize=6,
            )
            ax.axis("off")
        for ax in axes.ravel()[len(orig_issue_indices[:20]):]:
            ax.axis("off")
        warn_suffix = (
            f"\n⚠ 신뢰도 낮음 (silhouette={sil_score:.4f} < {SILHOUETTE_LOW_CONFIDENCE}) — 참고용"
            if low_conf else ""
        )
        fig.suptitle(f"레이블 노이즈 의심 샘플 상위 20개 (n={len(sampled_indices)}){warn_suffix}")
        fig.tight_layout()
        _save_fig(fig, figures_dir / "20_label_noise_samples.png")
        print(f"[label_noise] 의심 샘플 {len(issues)}개 탐지, 상위 20개 시각화"
              + (" ⚠ silhouette 낮음 — 신뢰도 낮음" if low_conf else ""))
        return {
            "suspect_count": len(issues),
            "suspect_file_ids": suspect_ids,
            "suspect_samples": suspect_samples,
            "sampled_n": int(len(sampled_indices)),
            "low_confidence": low_conf,
            "low_confidence_reason": (
                f"silhouette={sil_score:.4f} < {SILHOUETTE_LOW_CONFIDENCE}" if low_conf else None
            ),
        }
    except Exception as e:
        skipped.append({"analysis": "레이블 노이즈 (cleanlab)", "reason": str(e),
                        "install": "pip install cleanlab scikit-learn"})
        return None


# ── Dominant color ────────────────────────────────────────────────────────────

def analyse_dominant_color(
    samples: list[Sample],
    figures_dir: Path,
    skipped: list[dict],
) -> tuple[dict, dict[str, str]]:
    """Dominant color swatches per class via K-means(k=5) on up to 200 random crops."""
    from sklearn.cluster import KMeans as _KMeans

    b64s: dict[str, str] = {}
    rng = random.Random(42)

    palette: dict[str, list[tuple[np.ndarray, float]]] = {}
    for cls in CLASS_NAMES:
        cls_samples = [s for s in samples if s.label == cls]
        chosen = rng.sample(cls_samples, min(_DOMINANT_COLOR_N_SAMPLES, len(cls_samples)))

        pixels: list[np.ndarray] = []
        for s in chosen:
            arr = _load_crop_arr(s)
            if arr is None:
                continue
            pil = Image.fromarray(arr).resize(
                (_DOMINANT_COLOR_RESIZE, _DOMINANT_COLOR_RESIZE), Image.BILINEAR,
            )
            pixels.append(np.array(pil).reshape(-1, 3).astype(np.float32))

        if not pixels:
            palette[cls] = []
            continue

        all_px = np.vstack(pixels)
        if len(all_px) > 50_000:
            idx = rng.sample(range(len(all_px)), 50_000)
            all_px = all_px[idx]

        km = _KMeans(n_clusters=_DOMINANT_COLOR_K, random_state=42, n_init=3)
        km.fit(all_px)
        centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)
        counts = np.bincount(km.labels_, minlength=_DOMINANT_COLOR_K)
        pcts = counts / counts.sum() * 100.0
        order = np.argsort(-pcts)
        palette[cls] = [(centers[i], float(pcts[i])) for i in order]

    n_rows = len(CLASS_NAMES)
    fig, axes = plt.subplots(
        n_rows, _DOMINANT_COLOR_K,
        figsize=(_DOMINANT_COLOR_K * 2.0, n_rows * 2.2),
    )
    for row_i, cls in enumerate(CLASS_NAMES):
        swatches = palette.get(cls, [])
        for col_i in range(_DOMINANT_COLOR_K):
            ax = axes[row_i, col_i]
            if col_i < len(swatches):
                rgb, pct = swatches[col_i]
                fc = tuple(int(v) / 255.0 for v in rgb)
                ax.set_facecolor(fc)
                txt_color = _contrast_color(rgb)
                ax.text(
                    0.5, 0.62,
                    f"RGB\n({rgb[0]},{rgb[1]},{rgb[2]})",
                    ha="center", va="center", fontsize=7,
                    color=txt_color, transform=ax.transAxes,
                )
                ax.text(
                    0.5, 0.22,
                    f"{pct:.1f}%",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=txt_color, transform=ax.transAxes,
                )
            else:
                ax.set_facecolor("#dddddd")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(
                    cls, rotation=0, ha="right", va="center",
                    labelpad=40, fontsize=9,
                )
            if row_i == 0:
                ax.set_title(f"Color {col_i + 1}", fontsize=9)

    fig.suptitle(
        f"클래스별 대표 색상 (K-means k={_DOMINANT_COLOR_K}, 샘플 {_DOMINANT_COLOR_N_SAMPLES}장)",
        fontsize=12,
    )
    fig.tight_layout()
    b64s["21_dominant_color"] = _save_fig(fig, figures_dir / "21_dominant_color.png")
    print(f"[dominant_color] 21_dominant_color.png 저장 완료")
    return {}, b64s


# ── Gabor texture ─────────────────────────────────────────────────────────────

def analyse_gabor_texture(
    samples: list[Sample],
    figures_dir: Path,
    skipped: list[dict],
) -> tuple[dict, dict[str, str]]:
    """Gabor texture analysis — 4 orientations × 2 frequencies, up to 300 crops/class."""
    try:
        from skimage.filters import gabor as _gabor_filter
    except ImportError:
        skipped.append({
            "analysis": "Gabor 텍스처",
            "reason": "skimage 미설치",
            "install": "pip install scikit-image",
        })
        return {}, {}

    b64s: dict[str, str] = {}
    rng = random.Random(42)

    n_orient = len(_GABOR_ORIENTATIONS)
    n_freq = len(_GABOR_FREQUENCIES)
    n_channels = n_orient * n_freq  # 8

    # channel layout: [fi * n_orient + oi] i.e. [f0_o0 … f0_o3, f1_o0 … f1_o3]
    energy_by_class: dict[str, np.ndarray] = {}
    for cls in CLASS_NAMES:
        cls_samples = [s for s in samples if s.label == cls]
        chosen = rng.sample(cls_samples, min(_GABOR_N_SAMPLES, len(cls_samples)))

        feats: list[np.ndarray] = []
        for s in chosen:
            arr = _load_crop_arr(s)
            if arr is None:
                continue
            gray = _to_gray_f32(arr)
            gray_pil = Image.fromarray((gray * 255).astype(np.uint8)).resize(
                (128, 128), Image.BILINEAR,
            )
            gray128 = np.array(gray_pil).astype(np.float32) / 255.0

            vec: list[float] = []
            for freq in _GABOR_FREQUENCIES:
                for theta in _GABOR_ORIENTATIONS:
                    filt_r, filt_i = _gabor_filter(gray128, frequency=freq, theta=theta)
                    mag = np.sqrt(filt_r ** 2 + filt_i ** 2)
                    vec.append(float(mag.mean()))
            feats.append(np.array(vec, dtype=np.float32))

        energy_by_class[cls] = (
            np.vstack(feats) if feats else np.zeros((0, n_channels), dtype=np.float32)
        )
        print(f"[gabor_texture] {cls}: {len(feats)}장 처리 완료")

    # ── Panel 1: orientation boxplot ──────────────────────────────────────────
    # For each orientation, average energy across both frequencies (per sample)
    orient_data: dict[str, list[list[float]]] = {}
    for cls in CLASS_NAMES:
        e = energy_by_class[cls]
        if len(e) == 0:
            orient_data[cls] = [[] for _ in range(n_orient)]
            continue
        # indices for orientation oi: [0*n_orient+oi, 1*n_orient+oi, ...]
        orient_data[cls] = [
            e[:, [fi * n_orient + oi for fi in range(n_freq)]].mean(axis=1).tolist()
            for oi in range(n_orient)
        ]

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")

    n_cls = len(CLASS_NAMES)
    group_w = 0.7
    bar_w = group_w / n_cls
    offsets = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2, n_cls)

    for ci, cls in enumerate(CLASS_NAMES):
        for oi in range(n_orient):
            vals = orient_data[cls][oi]
            if not vals:
                continue
            ax1.boxplot(
                vals,
                positions=[oi + offsets[ci]],
                widths=bar_w * 0.85,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 1.5},
                boxprops={"facecolor": CLASS_COLORS[cls], "alpha": 0.7},
                whiskerprops={"color": CLASS_COLORS[cls]},
                capprops={"color": CLASS_COLORS[cls]},
                flierprops={"marker": ".", "markersize": 2, "color": CLASS_COLORS[cls]},
                showfliers=False,
            )

    ax1.set_xticks(range(n_orient))
    ax1.set_xticklabels(_GABOR_ORIENT_LABELS, fontsize=9)
    ax1.set_xlabel("Gabor 방향", fontsize=9)
    ax1.set_ylabel("평균 에너지", fontsize=9)
    ax1.set_title("방향별 Gabor 에너지", fontsize=10)
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c], alpha=0.7, label=c)
        for c in CLASS_NAMES
    ]
    ax1.legend(handles=legend_patches, fontsize=8)

    # ── Panel 2: frequency scatter ─────────────────────────────────────────────
    # x = low-freq mean energy (freq[0]), y = high-freq mean energy (freq[1])
    for cls in CLASS_NAMES:
        e = energy_by_class[cls]
        if len(e) == 0:
            continue
        low_e = e[:, :n_orient].mean(axis=1)   # fi=0 channels
        high_e = e[:, n_orient:].mean(axis=1)  # fi=1 channels
        ax2.scatter(
            low_e, high_e,
            color=CLASS_COLORS[cls], label=cls,
            marker=CLASS_MARKERS[cls], s=14, alpha=0.5,
        )
    ax2.set_xlabel(f"저주파 에너지 (freq={_GABOR_FREQUENCIES[0]})", fontsize=9)
    ax2.set_ylabel(f"고주파 에너지 (freq={_GABOR_FREQUENCIES[1]})", fontsize=9)
    ax2.set_title("주파수별 에너지 분포", fontsize=10)
    ax2.legend(fontsize=8)

    # ── Panel 3: radar chart ───────────────────────────────────────────────────
    # 8 spokes: [0°(저), 45°(저), 90°(저), 135°(저), 0°(고), 45°(고), 90°(고), 135°(고)]
    spoke_labels = (
        [f"{lbl}(저)" for lbl in _GABOR_ORIENT_LABELS]
        + [f"{lbl}(고)" for lbl in _GABOR_ORIENT_LABELS]
    )
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)

    all_mats = [e for e in energy_by_class.values() if len(e) > 0]
    if all_mats:
        all_vals = np.vstack(all_mats)
        ch_min = all_vals.min(axis=0)
        ch_max = all_vals.max(axis=0)
        ch_range = np.where(ch_max > ch_min, ch_max - ch_min, 1.0)
    else:
        ch_min = np.zeros(n_channels)
        ch_range = np.ones(n_channels)

    for cls in CLASS_NAMES:
        e = energy_by_class[cls]
        if len(e) == 0:
            continue
        mean_feat = e.mean(axis=0)
        normed = np.append((mean_feat - ch_min) / ch_range, 0)  # close polygon
        theta_closed = np.append(theta, theta[0])
        normed[-1] = normed[0]
        ax3.plot(theta_closed, normed, color=CLASS_COLORS[cls], label=cls, linewidth=1.5)
        ax3.fill(theta_closed, normed, color=CLASS_COLORS[cls], alpha=0.12)

    ax3.set_thetagrids(np.degrees(theta), spoke_labels, fontsize=7)
    ax3.set_title("Gabor 에너지 레이더 (정규화)", fontsize=10, pad=18)
    ax3.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.suptitle(
        f"Gabor 텍스처 분석 (4방향 × 2주파수, 샘플 {_GABOR_N_SAMPLES}장/클래스)",
        fontsize=12,
    )
    fig.tight_layout()
    b64s["22_gabor_texture"] = _save_fig(fig, figures_dir / "22_gabor_texture.png")
    print("[gabor_texture] 22_gabor_texture.png 저장 완료")
    return {}, b64s


# ── Slice assignment ──────────────────────────────────────────────────────────

def assign_slices(samples: list[Sample], blur_threshold: float) -> None:
    for s in samples:
        s.slice_camera = s.camera
        s.slice_blur = "high" if s.blur >= blur_threshold else "low"
        s.slice_bbox = s.bbox_size_rel
        s.slice_occlusion = "heavy" if s.occlusion_ratio < HEAVY_OCCLUSION_THRESHOLD else "normal"


# ── Cache helpers (manifest + pixel) ─────────────────────────────────────────

_MANIFEST_CACHE_COLS: list[str] = [
    "path_key", "blur", "edge_density", "texture_entropy", "occlusion_ratio",
]


def _make_path_key(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _load_manifest_cache(samples: list[Sample], cache_path: Path, data_dir: Path) -> int:
    """Fill per-image metrics from CSV cache; returns number of samples matched."""
    base = data_dir.parent
    lookup: dict[str, dict] = {}
    with cache_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["path_key"]] = row
    filled = 0
    for s in samples:
        key = _make_path_key(s.path, base)
        if (cached := lookup.get(key)) is not None:
            s.blur = float(cached.get("blur") or 0.0)
            s.edge_density = float(cached.get("edge_density") or 0.0)
            s.texture_entropy = float(cached.get("texture_entropy") or 0.0)
            s.occlusion_ratio = float(cached.get("occlusion_ratio") or 0.0)
            filled += 1
    return filled


def _save_manifest_cache(samples: list[Sample], cache_path: Path, data_dir: Path) -> None:
    """Save per-image metrics to CSV cache keyed by relative path."""
    base = data_dir.parent
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_CACHE_COLS)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "path_key": _make_path_key(s.path, base),
                "blur": s.blur,
                "edge_density": s.edge_density,
                "texture_entropy": s.texture_entropy,
                "occlusion_ratio": s.occlusion_ratio,
            })
    print(f"[Cache] manifest.csv 저장 ({len(samples)}행) → {cache_path}")


def _load_pixel_cache(cache_path: Path) -> tuple[dict, dict[str, str]]:
    """Load pixel stats and figure b64 strings from npz; no pickle required."""
    d = np.load(cache_path, allow_pickle=False)
    pixel_stats: dict = {
        "channel_mean": d["channel_mean"].tolist(),
        "channel_std": d["channel_std"].tolist(),
        "imagenet_diff_mean": d["imagenet_diff_mean"].tolist(),
        "custom_normalization_needed": bool(d["custom_norm"][0]),
    }
    b64s: dict[str, str] = {
        key[4:]: d[key].tobytes().decode("ascii")
        for key in d.files
        if key.startswith("fig_")
    }
    return pixel_stats, b64s


def _save_pixel_cache(cache_path: Path, pixel_stats: dict, b64s: dict[str, str]) -> None:
    """Save pixel stats and figure b64 strings to npz (uint8 byte arrays, no pickle)."""
    arrays: dict[str, np.ndarray] = {
        "channel_mean": np.array(pixel_stats.get("channel_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
        "channel_std": np.array(pixel_stats.get("channel_std", [0.0, 0.0, 0.0]), dtype=np.float32),
        "imagenet_diff_mean": np.array(
            pixel_stats.get("imagenet_diff_mean", [0.0, 0.0, 0.0]), dtype=np.float64
        ),
        "custom_norm": np.array(
            [int(pixel_stats.get("custom_normalization_needed", False))], dtype=np.int32
        ),
    }
    for fig_key, b64_str in b64s.items():
        arrays[f"fig_{fig_key}"] = np.frombuffer(b64_str.encode("ascii"), dtype=np.uint8).copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    print(f"[Cache] pixel_cache.npz 저장 → {cache_path}")


# ── Report writers ────────────────────────────────────────────────────────────

_MANIFEST_FIELDS = [
    "file_id", "label", "camera", "camera_source",
    "width", "height", "bbox_w", "bbox_h", "bbox_area", "bbox_area_ratio",
    "bbox_size_abs", "bbox_size_rel",
    "blur", "edge_density", "texture_entropy", "occlusion_ratio",
    "is_duplicate", "is_low_quality", "is_label_suspect",
    "split", "slice_camera", "slice_blur", "slice_bbox", "slice_occlusion",
]


def write_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for s in samples:
            d = asdict(s)
            d.pop("path", None); d.pop("bbox_raw", None)
            writer.writerow({k: d.get(k, "") for k in _MANIFEST_FIELDS})


_CONFUSION_CSV_FIELDS: list[str] = ["img_a", "img_b", "match_type", "score", "cluster_id"]


def write_confusion_pairs_csv(pairs: list[dict], samples: list[Sample], path: Path) -> None:
    """Write cross-class duplicate/similar pairs to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CONFUSION_CSV_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "img_a": pair["img_a"],
                "img_b": pair["img_b"],
                "match_type": pair["match_type"],
                "score": round(pair["score"], 6),
                "cluster_id": pair.get("cluster_id", -1),
            })
    print(f"[confusion_pairs] {len(pairs)}쌍 저장 → {path.name}")


def write_slice_index(samples: list[Sample], path: Path) -> None:
    index = {s.file_id: {
        "slice_camera": s.slice_camera,
        "slice_blur": s.slice_blur,
        "slice_bbox": s.slice_bbox,
        "slice_occlusion": s.slice_occlusion,
    } for s in samples}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def write_eda_results(results: dict, skipped: list[dict], path: Path) -> None:
    # 기존 결과 로드 후 머지 — 이번 실행에서 None인 섹션은 기존 값 보존
    # (예: --full 없이 재실행 시 이전 cleanlab 결과 유지)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    merged = {**existing, **results}
    for key in ("label_noise",):
        if results.get(key) is None and existing.get(key) is not None:
            merged[key] = existing[key]
    merged["skipped_analyses"] = skipped

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_rows(results: dict, samples: list[Sample]) -> list[dict]:
    rows: list[dict] = []

    def row(section: str, metric: str, value: Any, unit: str, flag: str) -> dict:
        return {"section": section, "metric": metric, "value": value, "unit": unit, "flag": flag}

    b = results.get("basic_stats", {})
    total = b.get("total", 0)
    rows.append(row("기본현황", "전체 샘플 수", total, "장", "-"))
    for cls in CLASS_NAMES:
        ratio = b.get("class_dist", {}).get(cls, 0.0)
        rows.append(row("기본현황", f"{CLASS_DISPLAY[cls]} 비율", round(ratio, 4), "-", "-"))
    rows.append(row("기본현황", "중복 그룹(MD5)", b.get("duplicate_md5_groups", 0), "개",
                    "WARNING" if b.get("duplicate_md5_groups", 0) > 0 else "-"))

    p = results.get("pixel_stats", {})
    norm_needed = p.get("custom_normalization_needed", False)
    rows.append(row("픽셀특성", "커스텀 정규화 필요", str(norm_needed).lower(), "-",
                    "WARNING" if norm_needed else "-"))

    e = results.get("edge_stats", {})
    mw = e.get("mannwhitney", {})
    rows.append(row("절단면특징", "edge 위험vs안전 유의",
                    str(mw.get("danger_vs_safe", {}).get("significant", False)).lower(), "-", "-"))
    rows.append(row("절단면특징", "entropy 위험vs안전 유의",
                    str(results.get("entropy_stats", {}).get("mannwhitney", {})
                        .get("danger_vs_safe", {}).get("significant", False)).lower(), "-", "-"))

    bl = results.get("blur_stats", {})
    lq = bl.get("low_quality_count", 0)
    rows.append(row("샘플품질", "저품질 샘플 수", lq, "장", "WARNING" if lq > 0 else "-"))
    rows.append(row("샘플품질", "저품질 비율", round(bl.get("low_quality_ratio", 0), 4), "-", "-"))

    u = results.get("umap_stats", {})
    sil = u.get("silhouette_score")
    if sil is not None:
        rows.append(row("feature분석", "silhouette 점수", round(sil, 4), "-", "-"))
    mmd = results.get("mmd_stats", {})
    mmd_val = mmd.get("4k_vs_fhd", 0.0)
    rows.append(row("feature분석", "MMD 4K vs FHD", round(mmd_val, 4), "-",
                    "WARNING" if mmd.get("domain_shift_detected") else "-"))

    ln = results.get("label_noise")
    if ln:
        rows.append(row("레이블품질", "의심 샘플 수", ln.get("suspect_count", 0), "장",
                        "WARNING" if ln.get("suspect_count", 0) > 0 else "-"))

    dec = results.get("decisions", {})
    rows.append(row("결정사항", "WeightedSampler 필요", str(dec.get("weighted_sampler_needed", False)).lower(),
                    "-", "ACTION" if dec.get("weighted_sampler_needed") else "-"))
    rows.append(row("결정사항", "커스텀 정규화 적용", str(dec.get("custom_normalization_needed", False)).lower(),
                    "-", "ACTION" if dec.get("custom_normalization_needed") else "-"))
    if dec.get("camera_domain_shift"):
        rows.append(row("결정사항", "카메라 도메인 시프트", "true", "-", "WARNING"))
    return rows


def write_eda_summary(results: dict, samples: list[Sample],
                      csv_path: Path, md_path: Path) -> None:
    rows = _summary_rows(results, samples)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value", "unit", "flag"])
        writer.writeheader(); writer.writerows(rows)
    # Markdown table
    lines = ["# EDA 요약\n",
             "| section | metric | value | unit | flag |",
             "|---------|--------|-------|------|------|"]
    for r in rows:
        lines.append(f"| {r['section']} | {r['metric']} | {r['value']} | {r['unit']} | {r['flag']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(b64_figs: dict[str, str], results: dict,
                      samples: list[Sample], path: Path) -> None:
    total = len(samples)
    counts = Counter(s.label for s in samples)
    lq = results.get("blur_stats", {}).get("low_quality_count", 0)
    dup = results.get("basic_stats", {}).get("duplicate_md5_groups", 0)
    sil = results.get("umap_stats", {}).get("silhouette_score")

    chart_html = ""
    titles = {
        "01_class_distribution": "클래스별 이미지 수 및 비율",
        "02_camera_distribution": "카메라별 클래스 분포",
        "03_bbox_absolute_size": "BBox 절대 크기 분포",
        "04_bbox_relative_size": "BBox 상대 면적 비율",
        "05_aspect_ratio_by_class": "클래스별 Aspect Ratio 분포",
        "06_channel_stats": "채널별 통계",
        "07_brightness_by_class": "클래스별 밝기 분포",
        "08_hsv_distribution": "HSV H채널 분포",
        "09_camera_brightness_compare": "카메라별 밝기 비교",
        "10_mean_image_per_class": "클래스별 평균 이미지",
        "11_diff_image": "차이 이미지",
        "12_edge_density_boxplot": "엣지 밀도 분포",
        "13_texture_entropy_boxplot": "텍스처 엔트로피 분포",
        "14_blur_histogram": "흐림도 점수 분포",
        "15_blur_vs_edge_scatter": "흐림도 vs 엣지 밀도",
        "16_occlusion_distribution": "전경 비율 분포 (Occlusion)",
        "17_umap_combined": "UMAP 클러스터",
        "18_prototype_images": "대표 샘플 이미지",
        "19_confusion_prone_pairs": "혼동 가능성 높은 샘플 쌍",
        "20_label_noise_samples": "레이블 노이즈 의심 샘플",
    }
    for key, b64 in b64_figs.items():
        title = titles.get(key, key)
        chart_html += (f'<h2>{title}</h2>'
                       f'<img src="data:image/png;base64,{b64}" '
                       f'style="max-width:100%;border-radius:6px;margin-bottom:24px;"/>\n')

    stat_cards = "".join(
        f'<div class="card"><div class="num">{v}</div><div class="lbl">{lbl}</div></div>'
        for v, lbl in [
            (total, "전체 샘플"),
            ("/".join(f"{CLASS_DISPLAY[c]}:{counts.get(c,0)}" for c in CLASS_NAMES), "클래스 분포"),
            (lq, "저품질 샘플"),
            (dup, "중복 그룹"),
            (f"{sil:.3f}" if sil is not None else "N/A", "Silhouette"),
        ]
    )

    decisions = results.get("decisions", {})
    dec_html = "<ul>"
    for k, v in decisions.items():
        if k.endswith("file_ids"):
            dec_html += f"<li><b>{k}</b>: {len(v)}개</li>"
        else:
            dec_html += f"<li><b>{k}</b>: {v}</li>"
    dec_html += "</ul>"

    skipped = results.get("skipped_analyses", [])
    skip_html = ""
    if skipped:
        skip_html = "<h2>Skip된 분석</h2><ul>" + "".join(
            f"<li><b>{s['analysis']}</b>: {s['reason']} — <code>{s['install']}</code></li>"
            for s in skipped
        ) + "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<title>Hazard Detection EDA Report</title>
<style>
body{{font-family:'Malgun Gothic','NanumGothic',sans-serif;max-width:1100px;margin:auto;padding:32px;color:#222;background:#fafafa}}
h1{{color:#1a1a2e}}h2{{color:#16213e;border-bottom:2px solid #e0e0e0;padding-bottom:6px;margin-top:40px}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}}
.card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.num{{font-size:22px;font-weight:700;color:#16213e}}.lbl{{font-size:11px;color:#666;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}}
th,td{{border:1px solid #ddd;padding:7px 10px;text-align:left}}th{{background:#f0f0f0}}
</style>
</head>
<body>
<h1>Hazard Detection — EDA Report</h1>
<p>총 샘플: <strong>{total}</strong></p>
<div class="cards">{stat_cards}</div>
{chart_html}
<h2>결정사항 (decisions)</h2>{dec_html}
{skip_html}
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[report] 저장: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _setup_korean_font()

    data_dir = args.data_dir or (
        Path("data/raw") if args.mode == "raw"
        else Path("dataset/classification/crops_25pct")
    )
    if not data_dir.exists():
        sys.exit("[오류] 데이터 디렉토리를 찾을 수 없습니다 (--data-dir로 지정)")

    out = args.output_dir
    figures_dir = out / "figures"
    interactive_dir = out / "interactive"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = data_dir.parent
    manifest_cache_path = cache_dir / "manifest.csv"

    skipped: list[dict] = []
    all_b64s: dict[str, str] = {}

    # ── Validate ──────────────────────────────────────────────────────────
    print("\n[0/6] Annotation Schema 검증")
    if args.mode == "raw":
        valid_records = _validate_raw(data_dir, args.bbox_format, out / "validation_report.csv")
        samples = load_raw(valid_records, args.bbox_format)
    else:
        samples = load_crop(data_dir)
        print(f"[validate] crop 모드: {len(samples)}개 샘플 로드")

    if not samples:
        sys.exit("[ERROR] 유효한 샘플이 없습니다.")

    if args.max_samples and len(samples) > args.max_samples:
        random.seed(42)
        samples = random.sample(samples, args.max_samples)
        print(f"[fast mode] {args.max_samples}개로 샘플링")

    print(f"[load] {len(samples)}개 샘플 준비 완료")

    # ── Image processing (single pass) ───────────────────────────────────
    print("\n[1/6] 이미지 처리 (단일 패스)")
    log_memory()
    if manifest_cache_path.exists() and not args.force_reextract:
        n_loaded = _load_manifest_cache(samples, manifest_cache_path, data_dir)
        print(f"[Cache HIT] manifest.csv 로드 ({n_loaded}행)")
    else:
        print("[Cache MISS] 이미지 처리 시작...")
        process_images(samples)
        _save_manifest_cache(samples, manifest_cache_path, data_dir)
    blur_threshold = _apply_blur_threshold(samples)
    log_memory()

    # ── Basic stats ───────────────────────────────────────────────────────
    print("\n[2/6] 기본 통계")
    log_memory()
    basic_stats, b64s = analyse_basic(samples, figures_dir, args.check_duplicates, mode=args.mode)
    all_b64s.update(b64s)
    log_memory()

    # ── Pixel analysis ────────────────────────────────────────────────────
    # 캐시 대상 아님 — 시각화(06-11)는 항상 재실행, per-image 수치는 manifest.csv 에서 로드
    print("\n[3/6] 픽셀/색상 분석")
    pixel_stats, b64s = analyse_pixel(samples, figures_dir)
    all_b64s.update(b64s)

    # ── Quality features (edge/entropy) ───────────────────────────────────
    print("\n[4/6] 절단면 특징 분석")
    log_memory()
    quality_stats, b64s = analyse_quality_features(samples, figures_dir)
    all_b64s.update(b64s)

    # ── Blur / occlusion ──────────────────────────────────────────────────
    blur_stats, b64s = analyse_blur_occlusion(samples, figures_dir, blur_threshold)
    all_b64s.update(b64s)
    log_memory()

    # ── DINOv2 features ───────────────────────────────────────────────────
    print("\n[5/6] DINOv2 Feature 분석")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_npy = out / "features.npy"
    features = _extract_features(samples, device, features_npy, args.force_reextract, skipped)
    feat_stats: dict = {}
    if features is not None and len(features) == len(samples):
        feat_stats, b64s = analyse_features(samples, features, figures_dir, interactive_dir, skipped)
        all_b64s.update(b64s)
        confusion_pairs = feat_stats.get("umap_stats", {}).get("confusion_prone_pairs", [])
        if confusion_pairs:
            write_confusion_pairs_csv(confusion_pairs, samples, out / "confusion_prone_pairs.csv")
    else:
        skipped.append({"analysis": "Feature 분석", "reason": "feature 추출 실패 또는 건너뜀",
                        "install": "torch.hub (네트워크/캐시 확인)"})

    # ── Label noise ───────────────────────────────────────────────────────
    label_noise_result = None
    if args.full:
        print("\n[6/6] 레이블 노이즈 (cleanlab)")
        if features is not None:
            sil_score = feat_stats.get("umap_stats", {}).get("silhouette_score", 0.0)
            label_noise_result = analyse_label_noise(
                samples, features, figures_dir, skipped, sil_score
            )
        else:
            skipped.append({"analysis": "레이블 노이즈", "reason": "feature 없음", "install": ""})

    # ── Dominant color ────────────────────────────────────────────────────
    print("\n[색상] 대표 색상 분석 (K-means)")
    _, dom_b64s = analyse_dominant_color(samples, figures_dir, skipped)
    all_b64s.update(dom_b64s)

    # ── Gabor texture ─────────────────────────────────────────────────────
    print("\n[텍스처] Gabor 텍스처 분석")
    _, gab_b64s = analyse_gabor_texture(samples, figures_dir, skipped)
    all_b64s.update(gab_b64s)

    # ── Slice assignment ──────────────────────────────────────────────────
    assign_slices(samples, blur_threshold)

    # ── Decisions ─────────────────────────────────────────────────────────
    decisions = {
        "custom_normalization_needed": pixel_stats.get("custom_normalization_needed", False),
        "weighted_sampler_needed": basic_stats.get("weighted_sampler_needed", False),
        "camera_domain_shift": feat_stats.get("mmd_stats", {}).get("domain_shift_detected", False),
        "remove_low_quality_file_ids": blur_stats.get("blur_stats", {}).get("low_quality_file_ids", []),
        "recheck_label_file_ids": label_noise_result.get("suspect_file_ids", []) if label_noise_result else [],
    }

    # ── Aggregate results ─────────────────────────────────────────────────
    eda_results: dict = {
        "basic_stats": basic_stats,
        "pixel_stats": pixel_stats,
        **quality_stats,
        **blur_stats,
        **feat_stats,
        "label_noise": label_noise_result,
        "split_quality": None,
        "slice_stats": {
            "combinations": dict(Counter(
                f"{s.slice_camera}_{s.slice_blur}_{s.slice_bbox}_{s.slice_occlusion}"
                for s in samples
            ))
        },
        "decisions": decisions,
    }

    # ── Write all outputs ─────────────────────────────────────────────────
    print("\n[report] 출력 파일 생성 중")
    manifest_path = out / "manifest.csv"
    if not manifest_path.exists() or args.force_reextract:
        write_manifest(samples, manifest_path)
    else:
        print(f"[manifest] 기존 파일 재사용 (강제 재작성: --force-reextract)")
    write_slice_index(samples, out / "slice_index.json")
    write_eda_results(eda_results, skipped, out / "eda_results.json")
    write_eda_summary(eda_results, samples, out / "eda_summary.csv", out / "eda_summary.md")
    write_html_report(all_b64s, eda_results, samples, out / "eda_report.html")

    # ── Console summary ───────────────────────────────────────────────────
    completed = ["기본통계", "픽셀분석", "절단면특징", "흐림도/전경비율", "manifest.csv"]
    if features is not None:
        completed.append("DINOv2 features")
    if label_noise_result:
        completed.append("레이블노이즈")

    print("\n" + "─" * 60)
    print(f"  ✅ 완료: {', '.join(completed)}")
    if skipped:
        skip_names = [s["analysis"] for s in skipped]
        print(f"  ⚠️  건너뜀: {', '.join(skip_names)}")
        installs = list({s["install"] for s in skipped if s["install"]})
        if installs:
            print(f"  → pip install {' '.join(installs)} 후 재실행하면 추가됩니다.")
    print(f"  📁 출력: reports/eda/ ({out.resolve().name})")
    print("─" * 60)


if __name__ == "__main__":
    main()
