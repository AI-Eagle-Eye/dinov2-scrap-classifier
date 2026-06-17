from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import pandas as pd

_N_WARMUP = 5
_N_RUNS = 100


def measure_vram(
    model: nn.Module,
    image_size: int = 224,
    batch_size: int = 1,
    device: torch.device | str = "cuda",
    fp16: bool = True,
) -> dict[str, float]:
    """Peak VRAM 사용량 측정 (CUDA 전용).

    fp16=True (기본): 추론 환경(기업 PC RTX 4090) 기준 FP16 모델로 측정.
    fp16=False: FP32 기준 측정 — 비교 및 디버깅용.
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda":
        return {"error": "VRAM 측정은 CUDA 환경에서만 가능합니다."}

    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model(dummy)
    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return {
        "peak_vram_mb": peak_bytes / 1024**2,
        "peak_vram_gb": peak_bytes / 1024**3,
        "batch_size": float(batch_size),
        "image_size": float(image_size),
        "dtype": "float16" if fp16 else "float32",
    }


def measure_latency(
    model: nn.Module,
    image_size: int = 224,
    batch_size: int = 1,
    device: torch.device | str = "cuda",
    fp16: bool = True,
    n_warmup: int = _N_WARMUP,
    n_runs: int = _N_RUNS,
) -> dict[str, float]:
    """추론 latency(ms) 측정: cold start + warmup 후 mean/std/min/max/p50/p95/p99 + FPS.

    fp16=True (기본): FP16 모델 기준 측정.
    cold_start_ms: warmup 이전 첫 추론 1회 (CUDA context init 등 포함).
    fps: 처리량 images/sec = batch_size / mean_ms * 1000.
    """
    device = torch.device(device) if isinstance(device, str) else device
    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        # cold start: warmup 전 첫 추론 1회 별도 측정
        t0 = time.perf_counter()
        _ = model(dummy)
        _sync()
        cold_start_ms = (time.perf_counter() - t0) * 1000

        for _ in range(n_warmup):
            _ = model(dummy)
        _sync()

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(times)
    p50, p95, p99 = (float(v) for v in np.percentile(times, [50, 95, 99]))
    return {
        "mean_ms": mean_ms,
        "std_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "fps": batch_size / mean_ms * 1000.0,
        "cold_start_ms": cold_start_ms,
        "batch_size": float(batch_size),
        "dtype": "float16" if fp16 else "float32",
    }


def measure_model_info(model: nn.Module, ckpt_path: str | Path | None) -> dict[str, float]:
    """파라미터 수(total/trainable) + 체크포인트 파일 크기(MB)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size_mb = float("nan")
    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt_size_mb = Path(ckpt_path).stat().st_size / 1024**2
    return {
        "total_params": float(total),
        "trainable_params": float(trainable),
        "ckpt_size_mb": round(ckpt_size_mb, 2),
    }


def _append_model_info(out_csv: Path, info: dict[str, Any]) -> None:
    """model_profile.csv 끝에 빈 줄 + (info,value) 섹션으로 모델 정보 추가."""
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(["info", "value"])
        for key, value in info.items():
            w.writerow([key, value])


def run_batch_sweep(
    model: nn.Module,
    image_size: int,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    device: torch.device | str = "cuda",
    fp16: bool = True,
    out_csv: str | Path | None = None,
    model_info: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """배치별 peak_vram_mb / mean_ms / p95_ms / fps 측정 → DataFrame 반환(+CSV 저장)."""
    import pandas as pd  # noqa: PLC0415 — profiling 시에만 필요

    rows: list[dict[str, float]] = []
    for bs in batch_sizes:
        vram = measure_vram(model, image_size, bs, device, fp16)
        lat = measure_latency(model, image_size, bs, device, fp16)
        rows.append({
            "batch_size": bs,
            "peak_vram_mb": round(vram.get("peak_vram_mb", float("nan")), 2),
            "mean_ms": round(lat["mean_ms"], 3),
            "p95_ms": round(lat["p95_ms"], 3),
            "fps": round(lat["fps"], 2),
        })
    df = pd.DataFrame(rows)

    print(f"\n[profile] batch sweep (image_size={image_size}, dtype={'fp16' if fp16 else 'fp32'})")
    print(df.to_string(index=False))

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if model_info is not None:
            _append_model_info(out_csv, model_info)
    return df
