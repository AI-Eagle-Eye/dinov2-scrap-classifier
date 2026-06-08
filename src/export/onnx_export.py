from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    image_size: int = 224,
    opset_version: int = 17,
    fp16: bool = True,
) -> None:
    """모델을 ONNX로 export.

    fp16=True (기본): 모델과 입력을 float16으로 변환 후 export.
    원본 모델은 변경하지 않는다 (deepcopy 후 변환).
    batch dimension은 dynamic axes로 설정.
    """
    export_model = copy.deepcopy(model).eval()
    if fp16:
        export_model = export_model.half()
        dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float16)
    else:
        dummy = torch.randn(1, 3, image_size, image_size)

    # dynamo=False: legacy TorchScript tracer 사용.
    # 새 dynamo exporter는 backbone의 inference_mode 텐서와 충돌하므로 제외.
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
    )


def verify_onnx(
    onnx_path: str | Path,
    model: nn.Module,
    test_input: torch.Tensor,
    atol: float = 1e-3,
) -> bool:
    """PyTorch 출력과 ONNX Runtime 출력이 일치하는지 검증.

    ONNX 모델의 입력 dtype을 자동 감지해 PyTorch 모델도 동일 dtype으로 실행한다.
    FP16 export 기본값 atol=1e-3; FP32의 경우 1e-4로 좁혀서 호출 가능.
    원본 모델은 변경하지 않는다 (deepcopy 후 dtype 변환).
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime 설치 필요: pip install onnxruntime") from e

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # ONNX 모델의 기대 입력 dtype을 session 메타에서 감지
    input_type_str = sess.get_inputs()[0].type
    torch_dtype = torch.float16 if "float16" in input_type_str else torch.float32
    np_dtype = np.float16 if "float16" in input_type_str else np.float32

    # PyTorch도 ONNX와 동일 dtype으로 실행 (deepcopy로 원본 보호)
    cmp_model = copy.deepcopy(model).eval().to(dtype=torch_dtype)
    cmp_input = test_input.to(torch_dtype)
    with torch.inference_mode():
        pt_out = cmp_model(cmp_input).float().numpy()

    onnx_out = sess.run(None, {"image": cmp_input.numpy().astype(np_dtype)})[0].astype(np.float32)

    return bool(np.allclose(pt_out, onnx_out, atol=atol))
