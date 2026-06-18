"""Unified model adapter for the delivery demo backend (hazard-detection-agent).

Wraps `HazardModel` checkpoints behind one interface so heads switch transparently:

    - AttentionHead (head_type == "attention") -> "내 모델": attention map is read
      straight off the head's MultiheadAttention over patch tokens.
    - other heads (mlp / linear / class_aware) -> cosine-to-CLS patch saliency
      (grad-free, works on the frozen DINOv2 backbone).

The model repo (src/) and dataset roots are resolved relative to this file or via
the DEMO_REPO_ROOT env var, so the demo/ folder can be relocated under any repo
that ships hazard-detection-agent's src layout. Checkpoint config is used when
present; a bare state_dict is supported by inferring the architecture from keys.
All attention paths return a raw (gh, gw) float map normalised 0~1.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Resolve the model repo (the dir that holds src/). demo/ sits directly under it,
# so parents[2] of demo/backend/inference.py is the repo root by default.
REPO_ROOT = Path(os.environ.get("DEMO_REPO_ROOT") or Path(__file__).resolve().parents[2])
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.transforms import get_val_transforms  # noqa: E402  (deferred until path set)
from models import HazardModel, ModelConfig, VPTConfig  # noqa: E402

CLASS_ORDER: tuple[str, ...] = ("cut", "danger", "excluded")
_COST_BATCH_SIZES: tuple[int, ...] = (1, 4, 8, 16)
_LATENCY_WARMUP = 3
_LATENCY_RUNS = 50
_DEFAULT_IMAGE_SIZE = 336
# embed_dim (patch_embed.proj out channels) -> DINOv2 backbone name.
_EMBED_DIM_TO_BACKBONE: dict[int, str] = {384: "dinov2_vits14", 768: "dinov2_vitb14"}

ClassProb = dict[str, float]


@dataclass(slots=True)
class ModelInfo:
    name: str           # checkpoint filename
    model_type: str     # head_type
    head_kind: str      # "attention" | "vanilla"
    params: int
    file_size_bytes: int
    vram_mb: float | None
    img_size: int
    num_classes: int
    classes: list[str]
    device: str
    hash: str


def select_device(arg: str = "auto") -> torch.device:
    """Resolve 'auto' to cuda when present, else cpu."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def checkpoint_hash(path: Path) -> str:
    """Cheap stable key: name + size + first 1MB. Avoids hashing a 1GB file fully."""
    h = hashlib.sha1()
    h.update(path.name.encode())
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def _load_raw_checkpoint(path: Path) -> Any:
    """Load a checkpoint robustly: torch.load default -> weights_only=False -> raw pickle.

    Handles .ckpt files that fail with "PytorchStreamReader failed reading zip
    archive" (e.g. legacy non-zip pickles) by falling back through each loader.
    """
    path = Path(path)
    errors: list[str] = []
    for kwargs in ({}, {"weights_only": False}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception as exc:  # broad on purpose: try the next loader strategy
            errors.append(f"torch.load({kwargs or 'default'}): {exc}")
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        errors.append(f"pickle.load: {exc}")
        raise RuntimeError("체크포인트 로드 실패:\n  " + "\n  ".join(errors)) from exc


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Find the weights inside a checkpoint: model_state_dict / state_dict / model."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                value = ckpt[key]
                if isinstance(value, torch.nn.Module):
                    return value.state_dict()
                if isinstance(value, dict):
                    return value
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt  # the file itself is a bare state_dict
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    raise KeyError("state_dict를 찾을 수 없습니다 (탐색 키: model_state_dict / state_dict / model).")


def _model_config_from_dict(mcfg: dict[str, Any]) -> ModelConfig:
    vpt = mcfg.get("vpt", {}) or {}
    return ModelConfig(
        backbone_name=mcfg.get("backbone_name", "dinov2_vitb14"),
        head_type=mcfg.get("head_type", "attention"),
        vpt=VPTConfig(
            enabled=bool(vpt.get("enabled", False)),
            num_tokens=int(vpt.get("num_tokens", 10)),
            insert_from_layer=int(vpt.get("insert_from_layer", 0)),
        ),
        dropout=float(mcfg.get("dropout", 0.3)),
        num_classes=int(mcfg.get("num_classes", len(CLASS_ORDER))),
        use_grad_checkpoint=False,                       # inference: no activation checkpointing
        backbone_frozen=True,                            # weights come from the checkpoint
        unfreeze_last_n=0,                               # architecture-irrelevant for loading
        head_use_cls=bool(mcfg.get("head_use_cls", False)),
    )


def _infer_model_config(state: dict[str, torch.Tensor]) -> ModelConfig:
    """Reconstruct a ModelConfig from a bare state_dict (no embedded config)."""
    proj = state.get("backbone._dino.patch_embed.proj.weight")
    embed_dim = int(proj.shape[0]) if proj is not None else 768
    backbone_name = _EMBED_DIM_TO_BACKBONE.get(embed_dim, "dinov2_vitb14")

    if "head.attn.in_proj_weight" in state and "head.classifier.weight" in state:
        head_type, ncls = "attention", int(state["head.classifier.weight"].shape[0])
    elif "head.class_tokens" in state:
        head_type, ncls = "class_aware", int(state["head.classifier.weight"].shape[0])
    elif "head.fc.weight" in state:
        head_type, ncls = "linear", int(state["head.fc.weight"].shape[0])
    elif any(k.startswith("head.net.") for k in state):
        last = max(int(k.split(".")[2]) for k in state if k.startswith("head.net.") and k.endswith(".weight"))
        head_type, ncls = "mlp", int(state[f"head.net.{last}.weight"].shape[0])
    else:
        raise KeyError("state_dict에서 head 종류를 추론할 수 없습니다.")

    return _model_config_from_dict(
        {"backbone_name": backbone_name, "head_type": head_type, "num_classes": ncls}
    )


class ModelAdapter:
    """One HazardModel checkpoint loaded for inference, attention, embedding, profiling."""

    def __init__(self, ckpt_path: str | Path, device: torch.device | str = "auto"):
        self.ckpt_path = Path(ckpt_path)
        self.device = select_device(device) if isinstance(device, str) else device
        self.hash = checkpoint_hash(self.ckpt_path)

        ckpt = _load_raw_checkpoint(self.ckpt_path)
        state = _extract_state_dict(ckpt)
        cfg = ckpt.get("config") if isinstance(ckpt, dict) else None

        if isinstance(cfg, dict) and "model" in cfg:
            model_config = _model_config_from_dict(cfg["model"])
            dcfg = cfg.get("data", {}) or {}
            self.img_size = int(dcfg.get("image_size", _DEFAULT_IMAGE_SIZE))
            self._padding_color = str(dcfg.get("padding_color", "black"))
        else:
            model_config = _infer_model_config(state)
            self.img_size = _DEFAULT_IMAGE_SIZE
            self._padding_color = "black"

        self.head_type = model_config.head_type
        self.head_kind = "attention" if self.head_type == "attention" else "vanilla"
        self.num_classes = model_config.num_classes
        self.patch_size = int(getattr(self, "_patch_size", 14))
        self._idx_to_name = list(CLASS_ORDER[: self.num_classes])

        model = HazardModel(model_config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Surface only head/structural mismatches; backbone hub keys always match.
            bad = [k for k in list(missing) + list(unexpected) if not k.startswith("backbone._dino")]
            if bad:
                raise RuntimeError(f"state_dict 불일치: {bad[:6]}")
        self.model = model.to(self.device).eval()
        self.patch_size = int(self.model.backbone.patch_size)

        self._eval_tfm = get_val_transforms(self.img_size, padding_color=self._padding_color)

    # ---------------------------------------------------------------- helpers
    def _to_batch(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self._eval_tfm(img.convert("RGB")) for img in images]
        return torch.stack(tensors).to(self.device)

    def _probs_to_dict(self, prob_row: np.ndarray) -> ClassProb:
        return {self._idx_to_name[i]: float(prob_row[i]) for i in range(self.num_classes)}

    # ---------------------------------------------------------------- inference
    @torch.inference_mode()
    def infer(self, image: Image.Image, tta: bool = False) -> tuple[ClassProb, float]:
        probs, latency_ms = self.infer_batch([image], tta=tta)
        return probs[0], latency_ms

    @torch.inference_mode()
    def infer_batch(self, images: list[Image.Image], tta: bool = False) -> tuple[list[ClassProb], float]:
        x = self._to_batch(images)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        logits = self._forward_tta(x) if tta else self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return [self._probs_to_dict(probs[i]) for i in range(len(images))], latency_ms

    def _forward_tta(self, x: torch.Tensor) -> torch.Tensor:
        """TTA: mean of logits over the image and its horizontal flip (softmax applied by caller)."""
        return (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2.0

    @torch.inference_mode()
    def embedding(self, image: Image.Image) -> np.ndarray:
        """Backbone CLS feature, consistent across heads for t-SNE."""
        cls, _ = self.model._extract_features(self._to_batch([image]))
        return cls.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        cls, _ = self.model._extract_features(self._to_batch(images))
        return cls.cpu().numpy()

    # ---------------------------------------------------------------- attention
    @torch.inference_mode()
    def attention(self, image: Image.Image) -> np.ndarray:
        """Return a (gh, gw) attention map in 0~1, architecture-agnostic."""
        cls, patch = self.model._extract_features(self._to_batch([image]))  # [1,D], [1,N,D]
        if self.head_type == "attention":
            importance = self._head_attention(cls, patch)
        else:
            importance = self._cls_cosine_saliency(cls, patch)
        n = importance.shape[0]
        side = int(round(math.sqrt(n)))
        grid = importance[: side * side].reshape(side, side).float().cpu().numpy()
        return _normalise(grid)

    def _head_attention(self, cls: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        head = self.model.head
        tokens = torch.cat([cls.unsqueeze(1), patch], dim=1) if head.use_cls else patch
        _, weights = head.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        importance = weights.mean(dim=1)[0]            # mean over queries -> [L]
        return importance[1:] if head.use_cls else importance

    @staticmethod
    def _cls_cosine_saliency(cls: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        pn = F.normalize(patch, dim=-1)
        cn = F.normalize(cls, dim=-1).unsqueeze(1)
        return (pn * cn).sum(-1)[0].clamp(min=0)       # [N]

    # ---------------------------------------------------------------- profiling
    def model_info(self) -> ModelInfo:
        params = sum(p.numel() for p in self.model.parameters())
        return ModelInfo(
            name=self.ckpt_path.name,
            model_type=self.head_type,
            head_kind=self.head_kind,
            params=int(params),
            file_size_bytes=self.ckpt_path.stat().st_size,
            vram_mb=self._peak_vram_mb(batch_size=1),
            img_size=self.img_size,
            num_classes=self.num_classes,
            classes=list(self._idx_to_name),
            device=str(self.device),
            hash=self.hash,
        )

    @torch.inference_mode()
    def _peak_vram_mb(self, batch_size: int) -> float | None:
        if self.device.type != "cuda":
            return None
        dummy = torch.randn(batch_size, 3, self.img_size, self.img_size, device=self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model(dummy)
        torch.cuda.synchronize(self.device)
        return torch.cuda.max_memory_allocated(self.device) / 1024**2

    @torch.inference_mode()
    def cost_profile(self) -> dict[str, Any]:
        """VRAM per batch size + latency p50/p95/p99 (warmup 3, cuda-synced)."""
        vram_by_batch = []
        for bs in _COST_BATCH_SIZES:
            try:
                vram_by_batch.append({"batch": bs, "vram_mb": self._peak_vram_mb(bs)})
            except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
                vram_by_batch.append({"batch": bs, "vram_mb": None})

        dummy = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)

        def _sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        for _ in range(_LATENCY_WARMUP):
            self.model(dummy)
        _sync()
        times: list[float] = []
        for _ in range(_LATENCY_RUNS):
            t0 = time.perf_counter()
            self.model(dummy)
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        p50, p95, p99 = (float(v) for v in np.percentile(times, [50, 95, 99]))

        info = self.model_info()
        return {
            "params": info.params,
            "file_size_bytes": info.file_size_bytes,
            "vram_by_batch": vram_by_batch,
            "latency": {"p50": p50, "p95": p95, "p99": p99},
        }


def _normalise(grid: np.ndarray) -> np.ndarray:
    grid = grid.astype(np.float32)
    lo, hi = float(grid.min()), float(grid.max())
    if hi - lo < 1e-8:
        return np.zeros_like(grid)
    return (grid - lo) / (hi - lo)
