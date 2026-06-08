#!/usr/bin/env python3
"""Standalone test for eda.py block 18 — DBSCAN-based dedup/similarity detection.

Reads features.npy + manifest.csv from --out-dir and runs only the block 18
pipeline, printing the 4-line [18] console output.

Usage:
    python scripts/test_dedup.py
    python scripts/test_dedup.py --out-dir reports/eda/crops_25pct
    python scripts/test_dedup.py --out-dir reports/eda/crops_25pct \\
        --data-dir dataset/classification/crops_25pct
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# ── Constants (mirrored from eda.py) ─────────────────────────────────────────
CLASS_NAMES: tuple[str, ...] = ("위험", "안전", "제외")
IMG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_PHASH_THRESHOLD: int = 12
_DBSCAN_EPSILON: float = 0.15
_DBSCAN_MIN_SAMPLES: int = 2
_SPLIT_RATIO: tuple[float, float, float] = (0.70, 0.15, 0.15)

# ── Optional imagehash ────────────────────────────────────────────────────────
try:
    import imagehash as _imagehash
    from PIL import Image
    _IMAGEHASH_OK = True
except ImportError:
    _IMAGEHASH_OK = False


@dataclass(slots=True)
class _Sample:
    file_id: str
    label: str
    path: Path


def _scan_images(data_dir: Path) -> dict[str, Path]:
    """Return {stem: path} for all images under data_dir."""
    result: dict[str, Path] = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTENSIONS:
            result[p.stem] = p
    return result


def _load_samples(
    manifest_path: Path,
    stem_to_path: dict[str, Path],
) -> tuple[list[_Sample], list[int]]:
    """Parse manifest.csv and return (samples_v, valid_row_indices).

    valid_row_indices[i] is the 0-based row number in manifest.csv (= row in features.npy)
    that corresponds to samples_v[i].
    """
    samples_v: list[_Sample] = []
    valid_rows: list[int] = []

    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            label = row.get("label", "")
            if label not in CLASS_NAMES:
                continue
            file_id = row["file_id"]
            # file_id = "{label}_{stem}" — label has no underscore so split(1) is safe
            stem = file_id.split("_", 1)[1] if "_" in file_id else file_id
            path = stem_to_path.get(stem, Path(file_id))
            samples_v.append(_Sample(file_id=file_id, label=label, path=path))
            valid_rows.append(row_idx)

    return samples_v, valid_rows


def run(out_dir: Path, data_dir: Path | None) -> None:
    features_path = out_dir / "features.npy"
    manifest_path = out_dir / "manifest.csv"

    for p in (features_path, manifest_path):
        if not p.exists():
            sys.exit(f"[ERROR] 파일 없음: {p}")

    # ── Load ──────────────────────────────────────────────────────────────────
    features_all: np.ndarray = np.load(features_path)
    print(f"[load] features.npy  shape={features_all.shape}")

    stem_to_path: dict[str, Path] = {}
    if data_dir is not None:
        if not data_dir.exists():
            sys.exit(f"[ERROR] data-dir 없음: {data_dir}")
        stem_to_path = _scan_images(data_dir)
        print(f"[load] data-dir 이미지: {len(stem_to_path)}장")
    else:
        print("[load] --data-dir 없음 → MD5/pHash 스킵 (cosine-only 모드)")

    samples_v, valid_rows = _load_samples(manifest_path, stem_to_path)
    print(f"[load] manifest.csv  유효 샘플: {len(samples_v)}행")

    if not samples_v:
        sys.exit("[ERROR] 유효 샘플이 없습니다.")

    if max(valid_rows) >= len(features_all):
        sys.exit(
            f"[ERROR] manifest 행({max(valid_rows)}) ≥ features 행({len(features_all)}): "
            "파일 불일치"
        )

    feats = features_all[valid_rows]  # shape (n_valid, D)

    # ── Step 1: MD5 / pHash 일괄 계산 ────────────────────────────────────────
    if data_dir is not None and not _IMAGEHASH_OK:
        print("[WARNING] imagehash 미설치 → pHash 스킵 (pip install imagehash)")

    md5_of: dict[str, str] = {}
    phash_of: dict[str, object] = {}

    if data_dir is not None:
        for s in samples_v:
            if not s.path.exists():
                continue
            try:
                md5_of[s.file_id] = hashlib.md5(s.path.read_bytes()).hexdigest()
            except OSError:
                pass
            if _IMAGEHASH_OK:
                try:
                    phash_of[s.file_id] = _imagehash.phash(Image.open(s.path))
                except Exception:
                    pass

    # ── Step 2: DBSCAN (L2-normalized, euclidean ≡ cosine eps=0.15) ──────────
    feats_norm: np.ndarray = normalize(feats, norm="l2")
    eps_euc = float(np.sqrt(2.0 * _DBSCAN_EPSILON))
    fid_list = [s.file_id for s in samples_v]

    db = DBSCAN(
        eps=eps_euc,
        min_samples=_DBSCAN_MIN_SAMPLES,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    clabels: np.ndarray = db.fit_predict(feats_norm)

    # ── Step 3: 클러스터 내 쌍 분류 ──────────────────────────────────────────
    clu_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(clabels.tolist()):
        if cid >= 0:
            clu_to_idx[cid].append(i)

    confusion_pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for cid, idxs in clu_to_idx.items():
        for ii, jj in itertools.combinations(idxs, 2):
            fi, fj = fid_list[ii], fid_list[jj]
            key = (min(fi, fj), max(fi, fj))
            if key in seen:
                continue
            seen.add(key)

            m5i = md5_of.get(fi, "")
            m5j = md5_of.get(fj, "")
            if m5i and m5j and m5i == m5j:
                mtype, score = "exact", 0.0
            else:
                phi = phash_of.get(fi)
                phj = phash_of.get(fj)
                if phi is not None and phj is not None:
                    ham = int(phi - phj)
                    if ham <= _PHASH_THRESHOLD:
                        mtype, score = "resolution_diff", float(ham)
                    else:
                        cos = float(1.0 - float(np.dot(feats_norm[ii], feats_norm[jj])))
                        mtype, score = "similar", round(max(0.0, cos), 6)
                else:
                    cos = float(1.0 - float(np.dot(feats_norm[ii], feats_norm[jj])))
                    mtype, score = "similar", round(max(0.0, cos), 6)

            confusion_pairs.append({
                "img_a": fi, "img_b": fj,
                "match_type": mtype, "score": score,
                "cluster_id": cid,
            })

    # ── Step 4: 클러스터 단위 split 할당 ─────────────────────────────────────
    clu_to_samps: dict[int, list[_Sample]] = defaultdict(list)
    for i, s in enumerate(samples_v):
        clu_to_samps[int(clabels[i])].append(s)

    pos_cids = [c for c in clu_to_samps if c >= 0]
    rng = random.Random(42)
    rng.shuffle(pos_cids)

    n_total_clu = sum(len(clu_to_samps[c]) for c in pos_cids)
    sp_n: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    sp_tgt: dict[str, float] = {
        "train": _SPLIT_RATIO[0],
        "val":   _SPLIT_RATIO[1],
        "test":  _SPLIT_RATIO[2],
    }
    cid_sp: dict[int, str] = {}
    for cid in pos_cids:
        best = min(
            ("train", "val", "test"),
            key=lambda sp: sp_n[sp] / max(n_total_clu, 1) - sp_tgt[sp],
        )
        cid_sp[cid] = best
        sp_n[best] += len(clu_to_samps[cid])

    noise_by_cls: dict[str, list[_Sample]] = defaultdict(list)
    for s in clu_to_samps.get(-1, []):
        noise_by_cls[s.label].append(s)

    fid_sp: dict[str, str] = {}
    for cls_samps in noise_by_cls.values():
        rng.shuffle(cls_samps)
        n = len(cls_samps)
        n_tr = round(n * _SPLIT_RATIO[0])
        n_va = round(n * _SPLIT_RATIO[1])
        for k, s in enumerate(cls_samps):
            if k < n_tr:
                fid_sp[s.file_id] = "train"
            elif k < n_tr + n_va:
                fid_sp[s.file_id] = "val"
            else:
                fid_sp[s.file_id] = "test"

    for cid, samps in clu_to_samps.items():
        if cid >= 0:
            for s in samps:
                fid_sp[s.file_id] = cid_sp[cid]

    final_sp: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for v in fid_sp.values():
        final_sp[v] += 1

    # ── 콘솔 출력 (eda.py block 18과 동일) ────────────────────────────────────
    n_clu       = len(pos_cids)
    n_clu_imgs  = sum(len(clu_to_samps[c]) for c in pos_cids)
    n_noise_imgs = len(clu_to_samps.get(-1, []))
    n_exact = sum(1 for p in confusion_pairs if p["match_type"] == "exact")
    n_res   = sum(1 for p in confusion_pairs if p["match_type"] == "resolution_diff")
    n_sim   = sum(1 for p in confusion_pairs if p["match_type"] == "similar")
    print(f"[18] DBSCAN 클러스터 수: {n_clu}개")
    print(f"[18] exact: {n_exact}쌍 / resolution_diff: {n_res}쌍 / similar: {n_sim}쌍")
    print(f"[18] 클러스터 포함 이미지: {n_clu_imgs}장 / 단독 이미지: {n_noise_imgs}장")
    print(
        f"[18] split 할당 완료 → "
        f"train: {final_sp['train']} / val: {final_sp['val']} / test: {final_sp['test']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 18 standalone dedup test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default="reports/eda/crops_25pct",
        help="features.npy와 manifest.csv가 있는 출력 디렉토리 (기본: reports/eda/crops_25pct)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="이미지 원본 디렉토리 (MD5/pHash 계산용; 없으면 cosine-only 모드)",
    )
    args = parser.parse_args()
    run(Path(args.out_dir), Path(args.data_dir) if args.data_dir else None)


if __name__ == "__main__":
    main()
