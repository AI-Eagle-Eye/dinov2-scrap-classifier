"""Step 2-E: DINOv2-S/14 CLS token UMAP analysis.

Usage:
    python scripts/eda_umap.py
    python scripts/eda_umap.py --force-reextract
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Paths (never printed) ─────────────────────────────────────────────────────
_CROPS_DIR = _BASE / "dataset" / "classification" / "crops_25pct"
_REPORTS = _BASE / "reports" / "eda"
_FIGURES = _REPORTS / "figures"
_INTERACTIVE = _REPORTS / "interactive"
_FEATURES_PATH = _REPORTS / "features_cls.npy"
_META_PATH = _REPORTS / "features_meta.json"
_EDA_JSON = _REPORTS / "eda_results.json"

# ── Constants ─────────────────────────────────────────────────────────────────
_CLASSES: list[str] = ["cut", "danger", "excluded"]
_LABEL: dict[str, int] = {"cut": 0, "danger": 1, "excluded": 2}
_SMALL_W: int = 224
_SMALL_H: int = 224
_BATCH_SIZE: int = 32

_CLASS_COLORS: dict[str, str] = {
    "cut": "#2196F3",
    "danger": "#F44336",
    "excluded": "#9E9E9E",
}
_SIZE_BANDS: list[tuple[str, str]] = [
    ("<112", "#F44336"),
    ("112-224", "#FF9800"),
    ("224-336", "#4CAF50"),
    (">336", "#2196F3"),
]


# ── 1. Feature extraction ─────────────────────────────────────────────────────

def _extract_features(force: bool = False) -> tuple[np.ndarray, dict]:
    """Extract DINOv2-S CLS tokens. Cache to disk; reuse on subsequent runs."""
    if not force and _FEATURES_PATH.exists() and _META_PATH.exists():
        feats = np.load(str(_FEATURES_PATH))
        meta = json.loads(_META_PATH.read_text())
        print(f"[cache] Loaded {feats.shape[0]} features from cache.")
        return feats, meta

    samples: list[tuple[Path, int, str]] = []
    for cls in _CLASSES:
        for img_path in sorted((_CROPS_DIR / cls).glob("*.jpg")):
            samples.append((img_path, _LABEL[cls], img_path.stem))
    print(f"[info] {len(samples)} samples found.")

    # Read original crop sizes before transform (object size, not resized)
    widths: list[int] = []
    heights: list[int] = []
    for img_path, _, _ in tqdm(samples, desc="Reading sizes", ncols=80):
        with Image.open(img_path) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)
    is_small: list[bool] = [w < _SMALL_W and h < _SMALL_H for w, h in zip(widths, heights)]

    from src.data.transforms import get_val_transforms
    from src.models.backbone import DINOv2Backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    backbone = DINOv2Backbone("dinov2_vits14").to(device).eval()
    transform = get_val_transforms(336)

    all_feats: list[np.ndarray] = []
    t0 = time.time()
    for i in tqdm(range(0, len(samples), _BATCH_SIZE), desc="Extracting CLS tokens", ncols=80):
        batch = samples[i : i + _BATCH_SIZE]
        imgs = [transform(Image.open(p).convert("RGB")) for p, _, _ in batch]
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            cls_tok, _ = backbone(tensor)
        all_feats.append(cls_tok.cpu().numpy())
    print(f"[info] Extraction done in {time.time() - t0:.1f}s")

    feats = np.concatenate(all_feats, axis=0)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    np.save(str(_FEATURES_PATH), feats)
    meta = {
        "labels": [s[1] for s in samples],
        "widths": widths,
        "heights": heights,
        "is_small": is_small,
        "file_ids": [s[2] for s in samples],
    }
    _META_PATH.write_text(json.dumps(meta))
    print(f"[saved] features shape={feats.shape}, meta={len(meta['labels'])} entries")
    return feats, meta


# ── 2. UMAP ───────────────────────────────────────────────────────────────────

def _run_umap(feats: np.ndarray) -> np.ndarray:
    import umap as umap_lib
    print(f"[info] Fitting UMAP on {feats.shape[0]} samples (may take ~2-5 min)...")
    t0 = time.time()
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(feats)
    print(f"[info] UMAP done in {time.time() - t0:.1f}s")
    return emb


def _plot_umap_by_class(
    emb: np.ndarray,
    labels: list[int],
    is_small: list[bool],
) -> None:
    """2-1: UMAP colored by class; small objects use smaller markers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    la = np.array(labels)
    sa = np.array(is_small)
    label_names = {0: "cut", 1: "danger", 2: "excluded"}

    for lbl, name in label_names.items():
        color = _CLASS_COLORS[name]
        normal = (la == lbl) & ~sa
        small = (la == lbl) & sa
        if normal.any():
            ax.scatter(emb[normal, 0], emb[normal, 1],
                       c=color, s=6, alpha=0.5, label=f"{name} (normal)", rasterized=True)
        if small.any():
            ax.scatter(emb[small, 0], emb[small, 1],
                       c=color, s=2, alpha=0.35, marker=".", label=f"{name} (small<224)",
                       rasterized=True)

    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_title("UMAP — DINOv2-S CLS token, by class", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("off")
    _FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(_FIGURES / "umap_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/umap_by_class.png")


def _plot_umap_interactive(
    emb: np.ndarray,
    labels: list[int],
    widths: list[int],
    heights: list[int],
    is_small: list[bool],
    file_ids: list[str],
) -> None:
    import plotly.graph_objects as go

    la = np.array(labels)
    label_names = {0: "cut", 1: "danger", 2: "excluded"}
    fig = go.Figure()

    for lbl, name in label_names.items():
        mask = la == lbl
        idx = np.where(mask)[0]
        fig.add_trace(go.Scattergl(
            x=emb[idx, 0].tolist(),
            y=emb[idx, 1].tolist(),
            mode="markers",
            name=name,
            marker=dict(color=_CLASS_COLORS[name], size=4, opacity=0.6),
            customdata=[[file_ids[i], widths[i], heights[i], is_small[i]] for i in idx],
            hovertemplate=(
                "id: %{customdata[0]}<br>"
                "size: %{customdata[1]}x%{customdata[2]}<br>"
                "small: %{customdata[3]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="UMAP — DINOv2-S CLS token (interactive)",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        width=950, height=700,
    )
    _INTERACTIVE.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(_INTERACTIVE / "umap_interactive.html"))
    print("[saved] interactive/umap_interactive.html")


def _plot_umap_by_size(
    emb: np.ndarray,
    widths: list[int],
    heights: list[int],
) -> None:
    """2-2: UMAP colored by object size band (min side)."""
    min_side = np.minimum(np.array(widths), np.array(heights))
    masks = {
        "<112":   min_side < 112,
        "112-224": (min_side >= 112) & (min_side < 224),
        "224-336": (min_side >= 224) & (min_side < 336),
        ">336":   min_side >= 336,
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in _SIZE_BANDS:
        mask = masks[label]
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, s=4, alpha=0.5, label=label, rasterized=True)

    ax.legend(markerscale=3, fontsize=9, title="min(W,H)", loc="best")
    ax.set_title("UMAP — DINOv2-S CLS token, by object size", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.savefig(str(_FIGURES / "umap_by_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/umap_by_size.png")


# ── 3. Silhouette ─────────────────────────────────────────────────────────────

def _compute_silhouette(
    emb: np.ndarray,
    labels: list[int],
    is_small: list[bool],
) -> dict:
    """Silhouette scores on UMAP 2D embedding (cheaper than full 384-dim)."""
    from sklearn.metrics import silhouette_samples, silhouette_score

    la = np.array(labels)
    sa = np.array(is_small)

    sil_overall = float(silhouette_score(emb, la))
    sample_scores = silhouette_samples(emb, la)

    sil_by_class = {
        name: float(sample_scores[la == lbl].mean())
        for lbl, name in {0: "cut", 1: "danger", 2: "excluded"}.items()
    }
    sil_small = float(sample_scores[sa].mean()) if sa.any() else 0.0
    sil_normal = float(sample_scores[~sa].mean()) if (~sa).any() else 0.0

    return {
        "silhouette_overall": sil_overall,
        "silhouette_by_class": sil_by_class,
        "silhouette_small": sil_small,
        "silhouette_normal": sil_normal,
    }


# ── 4. Prototypes ─────────────────────────────────────────────────────────────

def _compute_prototypes(
    feats: np.ndarray,
    labels: list[int],
    file_ids: list[str],
    widths: list[int],
    heights: list[int],
    topk: int = 5,
) -> None:
    la = np.array(labels)
    label_names = {0: "cut", 1: "danger", 2: "excluded"}
    n_classes = len(label_names)

    fig, axes = plt.subplots(n_classes, topk, figsize=(topk * 2.5, n_classes * 2.5))

    for row, (lbl, cls_name) in enumerate(label_names.items()):
        idx = np.where(la == lbl)[0]
        centroid = feats[idx].mean(axis=0)
        dists = np.linalg.norm(feats[idx] - centroid, axis=1)
        top_local = np.argsort(dists)[:topk]

        print(f"[prototype] {cls_name}:")
        for rank, local_i in enumerate(top_local):
            global_i = idx[local_i]
            fid = file_ids[global_i]
            print(f"  rank {rank+1}: id={fid}, size={widths[global_i]}x{heights[global_i]}, "
                  f"dist={dists[local_i]:.4f}")
            ax = axes[row, rank]
            img_path = _CROPS_DIR / cls_name / f"{fid}.jpg"
            if img_path.exists():
                ax.imshow(Image.open(img_path))
            ax.set_title(f"{widths[global_i]}x{heights[global_i]}", fontsize=7)
            ax.axis("off")
        axes[row, 0].set_ylabel(cls_name, fontsize=9)

    fig.suptitle("Per-class prototypes (nearest to centroid)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(_FIGURES / "prototype_by_class.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/prototype_by_class.png")


# ── 4. Confusion-prone pairs ──────────────────────────────────────────────────

def _compute_confusion_prone(
    feats: np.ndarray,
    labels: list[int],
    file_ids: list[str],
    topk: int = 10,
    k_neighbors: int = 3,
) -> list[dict]:
    """Find topk (danger, cut) pairs with smallest cosine distance.

    Chunked computation over danger batches to stay within RAM budget.
    """
    la = np.array(labels)
    danger_idx = np.where(la == 1)[0]
    cut_idx = np.where(la == 0)[0]

    d_norm = feats[danger_idx].copy()
    d_norm /= np.linalg.norm(d_norm, axis=1, keepdims=True) + 1e-12
    c_norm = feats[cut_idx].copy()
    c_norm /= np.linalg.norm(c_norm, axis=1, keepdims=True) + 1e-12

    print(f"[info] Computing danger({len(danger_idx)}) × cut({len(cut_idx)}) cosine distances...")
    CHUNK = 256
    k = min(k_neighbors, len(cut_idx))
    # Accumulate: (cosine_dist, danger_local_idx, cut_local_idx)
    candidates: list[tuple[float, int, int]] = []

    for di_start in tqdm(range(0, len(danger_idx), CHUNK), desc="Pairwise dist", ncols=80):
        di_end = min(di_start + CHUNK, len(danger_idx))
        sim = d_norm[di_start:di_end] @ c_norm.T  # [chunk, n_cut]
        dist = 1.0 - sim
        # For each danger in chunk, keep k nearest cuts
        for offset in range(di_end - di_start):
            row = dist[offset]
            best_ci = np.argpartition(row, k)[:k]
            for ci in best_ci:
                candidates.append((float(row[ci]), di_start + offset, int(ci)))

    candidates.sort(key=lambda x: x[0])

    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for dist_val, di_local, ci_local in candidates:
        di_global = int(danger_idx[di_local])
        ci_global = int(cut_idx[ci_local])
        key = (di_global, ci_global)
        if key in seen:
            continue
        seen.add(key)
        fid_d = file_ids[di_global]
        fid_c = file_ids[ci_global]
        pairs.append({"file_ids": [fid_d, fid_c], "cosine_distance": dist_val})
        print(f"  danger={fid_d}, cut={fid_c}, cosine_dist={dist_val:.4f}")
        if len(pairs) >= topk:
            break

    return pairs


# ── 5. Small cluster analysis ─────────────────────────────────────────────────

def _analyze_small_cluster(labels: list[int], is_small: list[bool]) -> None:
    la = np.array(labels)
    sa = np.array(is_small)
    n_total = len(la)
    n_small = int(sa.sum())
    label_names = {0: "cut", 1: "danger", 2: "excluded"}

    print(f"\n[small cluster] total={n_total}, small={n_small} ({n_small/n_total:.1%})")
    print("  class distribution — all vs small vs normal:")
    for lbl, name in label_names.items():
        all_r = float((la == lbl).sum() / n_total)
        small_r = float(((la == lbl) & sa).sum() / max(n_small, 1))
        normal_r = float(((la == lbl) & ~sa).sum() / max(n_total - n_small, 1))
        print(f"  {name}: all={all_r:.1%}, small={small_r:.1%}, normal={normal_r:.1%}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cluster_separation(sil: float) -> str:
    if sil > 0.3:
        return "good"
    if sil >= 0.1:
        return "moderate"
    return "poor"


def _vpt_paa_necessity(sep: str) -> str:
    return {"good": "low", "moderate": "medium", "poor": "high"}[sep]


def _update_eda_json(umap_result: dict) -> None:
    data: dict = {}
    if _EDA_JSON.exists():
        data = json.loads(_EDA_JSON.read_text())
    data["umap_analysis"] = umap_result
    _EDA_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("[saved] eda_results.json (umap_analysis)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2-E: UMAP feature analysis")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Ignore cache and re-extract DINOv2 features")
    args = parser.parse_args()

    # 1. Feature extraction
    feats, meta = _extract_features(force=args.force_reextract)
    labels: list[int] = meta["labels"]
    widths: list[int] = meta["widths"]
    heights: list[int] = meta["heights"]
    is_small: list[bool] = meta["is_small"]
    file_ids: list[str] = meta["file_ids"]

    # 2. UMAP
    emb = _run_umap(feats)

    _plot_umap_by_class(emb, labels, is_small)
    _plot_umap_interactive(emb, labels, widths, heights, is_small, file_ids)
    _plot_umap_by_size(emb, widths, heights)

    # 3. Silhouette
    print("[info] Computing silhouette scores on UMAP embedding...")
    sil = _compute_silhouette(emb, labels, is_small)
    sep = _cluster_separation(sil["silhouette_overall"])
    nec = _vpt_paa_necessity(sep)

    print(f"\n[silhouette results]")
    print(f"  overall:  {sil['silhouette_overall']:.4f}  → separation={sep}")
    for cls_name, v in sil["silhouette_by_class"].items():
        print(f"  {cls_name}: {v:.4f}")
    print(f"  small:    {sil['silhouette_small']:.4f}")
    print(f"  normal:   {sil['silhouette_normal']:.4f}")
    print(f"  vpt_paa_necessity: {nec}")

    # 4. Prototypes
    print("\n[info] Computing prototypes...")
    _compute_prototypes(feats, labels, file_ids, widths, heights)

    # 4. Confusion-prone pairs
    print("\n[info] Computing confusion-prone pairs (danger vs cut)...")
    confusion_pairs = _compute_confusion_prone(feats, labels, file_ids)

    # 5. Small cluster analysis
    _analyze_small_cluster(labels, is_small)

    # 6. Update eda_results.json
    umap_result = {
        **sil,
        "confusion_prone_pairs": confusion_pairs,
        "cluster_separation": sep,
        "vpt_paa_necessity": nec,
    }
    _update_eda_json(umap_result)
    print("\n[done] Step 2-E complete.")


if __name__ == "__main__":
    main()
