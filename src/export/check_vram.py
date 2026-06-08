from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn

_N_WARMUP = 5
_N_RUNS = 20


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
    """평균 추론 latency(ms) 측정.

    fp16=True (기본): FP16 모델 기준 측정.
    """
    device = torch.device(device) if isinstance(device, str) else device
    dtype = torch.float16 if fp16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "batch_size": float(batch_size),
        "dtype": "float16" if fp16 else "float32",
    }
