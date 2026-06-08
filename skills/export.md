# skills/export.md
# ONNX Export 및 추론 파이프라인

---

## 추론 환경

```text
GPU: RTX 4090
VRAM: 전체 파이프라인 5GB 이하
Runtime: ONNX Runtime
입력: bbox crop 이미지
```

---

## Export Precision 정책

```text
기본: FP16 export (fp16=True)
  - export_onnx(model, path, fp16=True): model.half() + float16 dummy 입력
  - RTX 4090 ONNX Runtime GPU 가속 최대화
  - 추론 VRAM: DINOv2-S ~550MB, DINOv2-B ~900MB (5GB 제약 여유)

FP32 export (fp16=False):
  - 디버깅 또는 CPU 배포 시
  - 추론 VRAM: DINOv2-S ~1.0GB, DINOv2-B ~1.8GB

양자화(INT8/FP8) 검토 보류:
  - RTX 4090은 INT8 지원하나, FP8은 Ada Lovelace 이후 전용 연산기 필요
  - 검증 복잡도 대비 이득이 제한적 — 현재 실험 범위 제외
  - 검토 시점: FP16 VRAM이 5GB를 초과하는 경우에만 재논의

verify_onnx():
  - 모델 dtype 자동 감지 → test_input 변환 후 비교
  - FP16 export: atol=1e-3 권장 (FP16 수치 범위 제한)
  - FP32 export: atol=1e-4 (기본값 유지)
```

## Export 정책

```python
EXPORT_TARGETS = ["best_safe_precision", "best_f1"]
OPSET_VERSION = 17
INPUT_SHAPE = (1, 3, 224, 224)
```

dynamic batch는 가능하면 지원하되, 납품 기본은 batch=1입니다.

---

## Export 검증

```python
def verify_onnx_export(pt_model, onnx_path, test_input):
    """
    PyTorch 출력과 ONNX 출력 일치 검증.
    """
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(test_input).detach().cpu().numpy()

    ort_out = run_onnx(onnx_path, test_input)
    max_diff = np.abs(pt_out - ort_out).max()
    assert np.allclose(pt_out, ort_out, atol=1e-4), (
        f"ONNX output mismatch: max_diff={max_diff:.6f}"
    )
```

---

## export_info.json

```json
{
  "model_name": "dinov2s_vpt_paa_texture_attn",
  "checkpoint": "best_safe_precision_ep035_0.9521.ckpt",
  "export_date": "2026-05-13",
  "opset": 17,
  "input_shape": [1, 3, 224, 224],
  "output_shape": [1, 3],
  "classes": ["위험", "안전", "제외"],
  "temperature": 1.35,
  "thresholds": {
    "safe_threshold": 0.70,
    "danger_threshold": 0.60,
    "margin": 0.20
  },
  "metrics": {
    "safe_precision": 0.951,
    "danger_as_safe": 0.008,
    "danger_miss": 0.042,
    "f1": 0.858
  },
  "vram_mb": 550,
  "inference_ms": 12.3
}
```

---

## 추론 후처리

ONNX 모델이 logits를 출력하면 후처리에서 temperature scaling과 threshold decision을 적용합니다.

```python
def infer(
    onnx_path: Path,
    image: np.ndarray,
    temperature: float,
    safe_thr: float = 0.70,
    danger_thr: float = 0.60,
    margin: float = 0.20,
) -> dict:
    preprocessed = preprocess(image)
    logits = run_onnx(onnx_path, preprocessed)
    logits = logits / temperature
    probs = softmax(logits)

    p_danger = float(probs[0])
    p_safe = float(probs[1])
    second = sorted(probs)[-2]
    safe_margin = p_safe - second

    if p_safe >= safe_thr and safe_margin >= margin:
        decision = "안전"
    elif p_danger >= danger_thr:
        decision = "위험"
    else:
        decision = "제외/재검토"

    return {
        "decision": decision,
        "probabilities": {
            "위험": float(probs[0]),
            "안전": float(probs[1]),
            "제외": float(probs[2])
        },
        "thresholds_used": {
            "safe_threshold": safe_thr,
            "danger_threshold": danger_thr,
            "margin": margin
        },
        "temperature": temperature
    }
```

---

## VRAM/속도 검증

```bash
python check_vram.py --onnx exports/final_best_safe_precision_fp16.onnx
python benchmark.py --onnx exports/final_best_safe_precision_fp16.onnx --repeat 200
```

목표:

```text
Peak VRAM < 5GB
Latency < 30ms/image, 가능하면
PyTorch/ONNX 출력 일치
```

---

## 배포 체크리스트

```text
- [ ] 기업 PC CUDA 버전 확인
- [ ] ONNX Runtime GPU 설치 확인
- [ ] verify_onnx_export 통과
- [ ] VRAM 5GB 이하 확인
- [ ] latency 측정
- [ ] export_info.json 포함
- [ ] threshold_results.json 포함
- [ ] threshold 조정 문서 포함
```

