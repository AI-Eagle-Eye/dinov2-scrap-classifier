"""FastAPI backend for the hazard-classification delivery demo.

Serves the single-page `demo.html`, exposes 12 endpoints for inference,
attention, t-SNE and cost profiling, and lets the operator hot-swap weight
checkpoints (.pt/.pth/.ckpt) at runtime. Default mode picks CUDA when present
and falls back to CPU. The dataset is read only; sample crops are copied into
demo/testset_samples/ on first launch.

Paths are relative to this file (demo/ under the model repo root) and can be
overridden via env vars so the demo/ folder is portable:
    DEMO_REPO_ROOT     repo root that holds src/ and dataset/  (default: parents[2])
    DEMO_TESTSET       testset dir with cut/danger/excluded/    (default: <repo>/dataset/testset/crops_25pct)
    DEMO_DEFAULT_CKPT  checkpoint to auto-load on startup        (default: newest experiments/*/checkpoints/best_f1*.ckpt)
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from sklearn.manifold import TSNE

from inference import ModelAdapter, select_device

# ----------------------------------------------------------------------- paths
BACKEND_DIR = Path(__file__).resolve().parent
DEMO_DIR = BACKEND_DIR.parent
REPO_ROOT = Path(os.environ.get("DEMO_REPO_ROOT") or DEMO_DIR.parent)
DATASET_TESTSET = Path(
    os.environ.get("DEMO_TESTSET") or REPO_ROOT / "dataset" / "testset" / "crops_25pct"
)


def _resolve_default_ckpt() -> Path | None:
    """Pick the startup checkpoint: env override, else newest best_f1 .ckpt, else any .ckpt/.pth."""
    env = os.environ.get("DEMO_DEFAULT_CKPT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    patterns = ("experiments/*/checkpoints/best_f1*.ckpt",
                "experiments/*/checkpoints/best_model.pth",
                "experiments/*/checkpoints/*.ckpt")
    for pat in patterns:
        found = sorted(REPO_ROOT.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return found[0]
    return None


DEFAULT_CKPT = _resolve_default_ckpt()

SAMPLES_DIR = DEMO_DIR / "testset_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"
WEIGHTS_DIR = BACKEND_DIR / "weights"
CACHE_DIR = BACKEND_DIR / "cache"
DEMO_HTML = DEMO_DIR / "demo.html"
PLOTLY_JS = DEMO_DIR / "plotly.min.js"

SAMPLES_PER_CLASS = 10
CLASS_NAMES: tuple[str, ...] = ("cut", "danger", "excluded")

# --------------------------------------------------------------- global state
_adapter: ModelAdapter | None = None
_gallery: list[dict[str, Any]] = []
_tsne_lock = threading.Lock()
_tsne_state: dict[str, Any] = {"hash": None, "status": "idle", "done": 0, "total": 0, "points": []}


# --------------------------------------------------------------- sample setup
def _copy_testset_samples() -> list[dict[str, Any]]:
    """Copy N crops per class from the read-only dataset into demo/testset_samples/."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())

    manifest: list[dict[str, Any]] = []
    for label in CLASS_NAMES:
        src_dir = DATASET_TESTSET / label
        if not src_dir.is_dir():
            continue
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        dst_dir = SAMPLES_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in files[:SAMPLES_PER_CLASS]:
            shutil.copy2(src, dst_dir / src.name)  # dataset stays untouched (read-only)
            manifest.append({
                "id": f"{label}__{src.stem}",
                "file": f"{label}/{src.name}",
                "gt_label": label,
            })
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _image_path(image_id: str) -> Path:
    for item in _gallery:
        if item["id"] == image_id:
            return SAMPLES_DIR / item["file"]
    raise HTTPException(status_code=404, detail=f"unknown image id: {image_id}")


def _open(image_id: str):
    from PIL import Image  # local import keeps startup light

    return Image.open(_image_path(image_id)).convert("RGB")


# --------------------------------------------------------------- model + tsne
def _require_model() -> ModelAdapter:
    if _adapter is None:
        raise HTTPException(status_code=409, detail="모델이 로드되지 않았습니다. 가중치를 업로드하세요.")
    return _adapter


def _load_model(ckpt_path: Path) -> None:
    global _adapter
    _adapter = ModelAdapter(ckpt_path, device="auto")
    _start_tsne(_adapter)


def _start_tsne(adapter: ModelAdapter) -> None:
    """Load cached t-SNE for this model hash, or compute it in the background."""
    cache_file = CACHE_DIR / f"{adapter.hash}.json"
    if cache_file.exists():
        with _tsne_lock:
            points = json.loads(cache_file.read_text())
            _tsne_state.update(hash=adapter.hash, status="ready",
                               done=len(points), total=len(points), points=points)
        return
    with _tsne_lock:
        _tsne_state.update(hash=adapter.hash, status="computing",
                           done=0, total=len(_gallery), points=[])
    threading.Thread(target=_compute_tsne, args=(adapter, cache_file), daemon=True).start()


def _compute_tsne(adapter: ModelAdapter, cache_file: Path) -> None:
    try:
        embeddings: list[np.ndarray] = []
        for i, item in enumerate(_gallery):
            embeddings.append(adapter.embedding(_open(item["id"])))
            with _tsne_lock:
                if _tsne_state["hash"] != adapter.hash:
                    return  # a newer model superseded this run
                _tsne_state["done"] = i + 1
        x = np.stack(embeddings)
        perplexity = max(2.0, min(30.0, (len(x) - 1) / 3.0))
        coords = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                      init="pca").fit_transform(x)
        points = [
            {"id": _gallery[i]["id"], "x": float(coords[i, 0]), "y": float(coords[i, 1]),
             "gt_label": _gallery[i]["gt_label"]}
            for i in range(len(_gallery))
        ]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(points, ensure_ascii=False))
        with _tsne_lock:
            if _tsne_state["hash"] == adapter.hash:
                _tsne_state.update(status="ready", points=points, done=len(points))
    except Exception as exc:  # surface compute failure to the SSE client
        with _tsne_lock:
            if _tsne_state["hash"] == adapter.hash:
                _tsne_state.update(status="error", points=[], error=str(exc))


# ----------------------------------------------------------------------- app
app = FastAPI(title="Steel-Scrap Hazard Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global _gallery
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _gallery = _copy_testset_samples()
    if DEFAULT_CKPT is not None and DEFAULT_CKPT.exists():
        _load_model(DEFAULT_CKPT)


# ---------------------------------------------------------------- schemas
class InferRequest(BaseModel):
    id: str


class BatchRequest(BaseModel):
    ids: list[str]


class AttentionRequest(BaseModel):
    id: str


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _adapter is not None,
        "device": str(select_device("auto")),
    }


@app.get("/gallery")
def gallery() -> list[dict[str, Any]]:
    return [{"id": g["id"], "url": f"/image/{g['id']}", "gt_label": g["gt_label"]} for g in _gallery]


@app.get("/image/{image_id}")
def image(image_id: str) -> FileResponse:
    return FileResponse(_image_path(image_id))


@app.get("/model")
def model() -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(_require_model().model_info())


@app.post("/model/upload")
async def model_upload(file: UploadFile) -> dict[str, Any]:
    from dataclasses import asdict

    suffix = Path(file.filename or "model.pth").suffix or ".pth"
    if suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        raise HTTPException(status_code=400, detail="지원 형식: .pt / .pth / .ckpt")
    dst = WEIGHTS_DIR / (file.filename or "uploaded.pth")
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        _load_model(dst)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"체크포인트 로드 실패: {exc}") from exc
    return asdict(_require_model().model_info())


@app.post("/infer")
def infer(req: InferRequest, tta: bool = False) -> dict[str, Any]:
    adapter = _require_model()
    probs, latency_ms = adapter.infer(_open(req.id), tta=tta)
    return {"id": req.id, "probs": probs, "latency_ms": latency_ms}


@app.post("/infer/batch")
def infer_batch(req: BatchRequest, tta: bool = False) -> list[dict[str, Any]]:
    adapter = _require_model()
    images = [_open(i) for i in req.ids]
    probs, latency_ms = adapter.infer_batch(images, tta=tta)
    return [{"id": req.ids[i], "probs": probs[i], "latency_ms": latency_ms} for i in range(len(req.ids))]


@app.get("/infer/batch/stream")
def infer_batch_stream(ids: str, tta: bool = False) -> StreamingResponse:
    adapter = _require_model()
    id_list = [s for s in ids.split(",") if s]

    def gen() -> Iterator[str]:
        results = []
        for i, image_id in enumerate(id_list):
            probs, latency_ms = adapter.infer(_open(image_id), tta=tta)
            results.append({"id": image_id, "probs": probs, "latency_ms": latency_ms})
            yield _sse("progress", {"done": i + 1, "total": len(id_list)})
        yield _sse("done", {"results": results})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/attention")
def attention(req: AttentionRequest) -> dict[str, Any]:
    adapter = _require_model()
    grid = adapter.attention(_open(req.id))
    return {"id": req.id, "grid": grid.tolist(), "gh": grid.shape[0], "gw": grid.shape[1]}


@app.get("/tsne")
def tsne() -> dict[str, Any]:
    with _tsne_lock:
        return {"status": _tsne_state["status"], "points": _tsne_state["points"]}


@app.get("/tsne/stream")
def tsne_stream() -> StreamingResponse:
    def gen() -> Iterator[str]:
        while True:
            with _tsne_lock:
                status = _tsne_state["status"]
                done, total = _tsne_state["done"], _tsne_state["total"]
                points = _tsne_state["points"]
            if status in {"ready", "error"}:
                yield _sse("done", {"status": status, "points": points})
                return
            yield _sse("progress", {"status": status, "done": done, "total": total})
            time.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/cost")
def cost() -> dict[str, Any]:
    return _require_model().cost_profile()


# ---------------------------------------------------------------- static
@app.get("/")
def index() -> Response:
    if DEMO_HTML.exists():
        return FileResponse(DEMO_HTML)
    return Response("demo.html이 아직 없습니다. demo/ 폴더에 추가하세요.", media_type="text/plain")


@app.get("/plotly.min.js")
def plotly() -> Response:
    if PLOTLY_JS.exists():
        return FileResponse(PLOTLY_JS, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="plotly.min.js 없음 (demo/ 에 로컬 파일 배치)")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
