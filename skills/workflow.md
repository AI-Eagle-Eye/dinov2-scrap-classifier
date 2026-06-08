# skills/workflow.md
# 반복 등장 작업 패턴 모음

이 파일은 구현 과정에서 2회 이상 등장한 패턴을 정리한다.
새 작업을 시작하기 전에 해당 섹션을 참고한다.

---

## 1. 테스트 실패 대응 패턴

실패를 마주쳤을 때 원인을 두 가지로 먼저 분류한다.

```text
A. fp32 수치 특성 → atol 조정 또는 허용 설명 추가
B. 실제 코드 버그  → 코드 수정 후 재검증
```

### A. 수치 오차 (fp32 non-determinism)

```python
# 증상: batch=2 처리 결과가 batch=1과 미세하게 다름
# 원인: 12블록 Transformer에서 BLAS 병렬 처리 시 fp32 누적 오차
# 진단: atol=1e-5 실패, atol=1e-4 통과

# 수정 전
assert torch.allclose(patch_batch[:1], patch_single)          # atol=1e-5 (기본값)
# 수정 후 — 주석으로 이유 명시
assert torch.allclose(patch_batch[:1], patch_single, atol=1e-4)
# 12블록 transformer, 98,304개(256×384) 비교 → fp32 특성, 코드 버그 아님
```

판단 기준: CLS (384개 비교)는 통과, patch (256×384 = 98,304개 비교)는 실패 → 수치 오차.

### B. Gradient 경로 단절

```text
증상: param.grad is None (학습이 실제로 일어나지 않는 버그)
원인: torch.topk()는 Long 인덱스를 반환 → gather가 인덱스 경유 gradient를 전달 못 함
```

```python
# 진단 — 아래 경로를 추적
query → scores → indices(Long) → gather(selected)
#                 ↑ 여기서 gradient 단절

# 수정 패턴: softmax weights를 선택된 토큰에 곱해 soft path 복원
weights = torch.softmax(scores, dim=-1)          # [B, N]
_, indices = weights.topk(k, dim=-1)             # Long, gradient 단절
selected = torch.gather(patch_tokens, 1, ...)    # patch_tokens 경유 gradient는 유지
selected_weights = torch.gather(weights, 1, indices).unsqueeze(-1)
selected = selected * selected_weights
# 복원된 경로: selected → selected_weights → weights → scores → query ✓
```

### 테스트 실패 체크리스트

```text
1. 에러 메시지에서 실패 값을 읽는다 (None인지, 수치 차이인지)
2. requires_grad 경로를 코드에서 직접 추적한다
3. Long tensor / inference_mode tensor가 경로에 있는지 확인
4. 최소 변경으로 수정하고, 기존 통과 테스트가 유지되는지 확인한다
```

---

## 2. Mock Backbone으로 E2E 테스트하는 구조

실제 DINOv2를 로드하지 않고 동일한 인터페이스의 mock을 만들어 전체 파이프라인을 검증한다.
네트워크/캐시 없이도 CI가 돌아가고, 테스트 속도가 수십 배 빠르다.

### Mock backbone 패턴

```python
class _MockBackbone(nn.Module):
    """DINOv2Backbone 인터페이스를 만족하는 최소 mock."""
    embed_dim = 384    # dinov2_vits14 기준
    patch_size = 14
    num_heads = 6
    num_blocks = 12

    def __init__(self) -> None:
        super().__init__()
        # Conv2d로 패치 임베딩 구조를 모방
        self._proj = nn.Conv2d(3, 384, kernel_size=14, stride=14, bias=False)
        for p in self._proj.parameters():
            p.requires_grad = False   # backbone은 항상 frozen

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self._proj(x)              # [B, 384, 16, 16]
        patches = out.flatten(2).transpose(1, 2)   # [B, 256, 384]
        return patches.mean(1), patches            # (cls, patches)

    def train(self, mode: bool = True) -> _MockBackbone:
        super().train(mode)
        return self
```

### HazardModel을 mock backbone으로 조립하는 패턴

```python
# __new__로 생성 → __init__ 없이 직접 속성 주입
model = HazardModel.__new__(HazardModel)
torch.nn.Module.__init__(model)
model.config = EXP_B_S
model.backbone = mock_backbone   # 실제 DINOv2 대신
model.vpt = None
model.head = MLPHead(384, 256, 3)   # classifier_type에 맞는 head 선택
model = model.eval()
# paa/fusion_norm 속성은 제거됨 — classifier_type=paa_mlp면 PAAMLPHead 내부에서 PAA 처리
```

### E2E 픽스처 구조

```python
# module scope로 한 번만 실행, 결과를 모든 테스트가 공유
@pytest.fixture(scope="module")
def e2e(mock_model, dummy_loader):
    """전체 파이프라인을 한 번 실행하고 결과 dict를 반환."""
    logits_list = []
    labels_list = []
    # LBFGS backward를 위해 inference_mode가 아닌 no_grad 사용
    with torch.no_grad():
        for images, labels in dummy_loader:
            logits_list.append(mock_model(images))
            labels_list.append(labels)
    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)
    return {
        "logits": all_logits,
        "labels": all_labels,
        # ... metrics, sweep_results, calibrated_probs
    }
```

### `inference_mode` vs `no_grad` 구분 원칙

```text
torch.inference_mode() → inference tensor 생성
  - forward-only, 가장 빠름
  - LBFGS 같은 backward가 필요한 곳에서 사용 불가

torch.no_grad() → 일반 tensor, requires_grad=False
  - calibration logit 수집에 사용 (LBFGS가 나중에 backward 필요)
  - E2E 테스트의 logit 수집에 사용

규칙:
  Trainer._validate()    → @torch.inference_mode()  (평가 전용)
  calibration logit 수집 → torch.no_grad()          (LBFGS 대비)
```

---

## 3. 새 실험 추가 순서

```text
config → model build → train → evaluate → threshold sweep → (calibration) → ONNX export
```

### Step 1. Config 파일 생성

```bash
cp configs/exp_b_s.yaml configs/exp_X_Y.yaml
```

수정 항목:
```yaml
experiment:
  name: exp_x_y
model:
  use_vpt: true              # Exp C/D
  classifier_type: paa_mlp  # Exp D: paa_mlp | Exp B/C: mlp | linear | attn_pooling
```

### Step 2. ModelConfig 상수 추가

`src/models/hazard_model.py` 하단:
```python
EXP_X_Y = ModelConfig(backbone_name="dinov2_vits14", use_vpt=True, classifier_type="paa_mlp")
```

### Step 3. 학습 실행

```bash
python train.py --config configs/exp_x_y.yaml \
                --train-dir data/processed/train \
                --val-dir data/processed/val
# 출력: experiments/exp_x_y/checkpoints/, experiments/exp_x_y/logs/train_log.csv
```

### Step 4. 평가 (evaluator)

```python
from src.evaluation.evaluator import compute_metrics, compute_confusion_matrix

# 핵심 지표 순서: danger_as_safe_rate → safe_precision → f1_macro
metrics = compute_metrics(y_true, y_pred)
# danger_as_safe_rate: 이 값이 낮을수록 우선
```

### Step 5. Threshold Sweep

```python
from src.evaluation.threshold import ThresholdSweeper

sweeper = ThresholdSweeper(safe_start=0.30, safe_end=0.90, safe_step=0.05)
results = sweeper.sweep(val_probs, val_labels)
best = sweeper.select_best(results, constraint={"danger_as_safe_rate": 0.05})
# val에서 선택한 threshold를 test에 고정 적용
```

### Step 6. (Exp E만) Temperature Calibration

```python
from src.evaluation.calibration import TemperatureScaler

scaler = TemperatureScaler()
# no_grad로 수집 (LBFGS backward 대비)
with torch.no_grad():
    val_logits = collect_logits(model, val_loader)
scaler.fit(val_logits, val_labels)    # val set만 사용
probs = scaler.calibrate(test_logits)
```

### Step 7. ONNX Export

```bash
python -c "
from src.export.onnx_export import export_onnx
export_onnx(model, 'experiments/exp_x_y/model.onnx', input_shape=(1,3,224,224))
"
```

---

## 4. ONNX Export 검증 패턴

Export 직후 반드시 PyTorch 출력과 비교 검증한다.

```python
import onnxruntime as ort
import torch
import numpy as np

def verify_onnx_export(model: nn.Module, onnx_path: str) -> float:
    """PyTorch 출력과 ONNX 출력의 최대 절댓값 차이를 반환."""
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]

    diff = np.abs(pt_out - ort_out).max()
    assert diff < 1e-4, f"ONNX 출력 불일치: max_diff={diff:.2e}"
    return float(diff)
```

테스트에서의 패턴:
```python
def test_onnx_output_matches_pytorch(self, ...):
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime 미설치")
    # ... verify_onnx_export 호출
    assert max_diff < 1e-4
```

검증 기준:
```text
atol = 1e-4   : fp32 export 허용 오차 (BLAS 누적 오차 포함)
dynamic axes  : batch 차원은 항상 dynamic으로 export
입력 이름     : "input" 고정 (OnnxExporter에서 설정)
출력 이름     : "logits" 고정
```

VRAM / latency 측정은 export 직후 수행:
```bash
python -c "from src.export.check_vram import measure_vram; measure_vram('experiments/exp_x_y/model.onnx')"
# 목표: 전체 파이프라인 5GB 이하 (기업 PC RTX 4090 기준)
```

---

## 5. EDA 실행 순서 및 결과 해석 기준

### 실행 순서

```bash
# 1. 원본 이미지 + annotations.json 형태
python scripts/eda.py \
  --mode raw \
  --data-dir data/raw \
  --bbox-format xyxy \
  --check-duplicates \
  --output-dir reports/

# 2. 이미 크롭된 폴더 구조
python scripts/eda.py \
  --mode crop \
  --data-dir data/processed/train \
  --output-dir reports/

# 결과: reports/eda_report.html (브라우저로 열기)
```

### 해석 기준 — 순서대로 확인

```text
1. 클래스 불균형
   기준: 위험:안전:제외 비율이 1:2 초과 시 WeightedRandomSampler 또는 class weight CE 고려
   핵심: 위험 클래스가 가장 적을 가능성이 높음 → oversample 또는 class_weight 우선

2. 소형 Crop 경보 (<64px)
   기준: 전체의 5% 초과 시 bbox_padding 조정 검토 (0.15 → 0.20)
   이유: 너무 타이트한 crop은 절단면 끝단과 주변 맥락을 잃음

3. 카메라별 분포
   기준: 특정 카메라가 한 클래스를 80% 이상 점유 → camera leakage 위험
   대응: camera-stratified split 적용 (train/val/test에 카메라 분산)

4. 중복 이미지
   기준: 중복 그룹이 1개라도 있으면 train/test split 전에 제거
   이유: 동일 프레임이 train과 test에 나뉘면 지표가 부풀려짐

5. 해상도 분포
   기준: 4K/FHD 비율 확인 → DINOv2 224×224 resize 전에 품질 손실 추정
   4K 원본이 있으면 bbox crop 시 해상도 여유가 충분함 → padding 범위 확장 가능
```

### EDA → 학습 전 의사결정 흐름

```text
EDA 실행
  ↓
클래스 불균형 심함?
  → Yes: configs/exp_x_y.yaml에 class_weight 또는 sampler 설정 추가
  → No:  기본 CE (label_smoothing=0.1)

소형 crop 5% 초과?
  → Yes: bbox_padding 0.15 → 0.20으로 조정 후 재확인

카메라 leakage 의심?
  → Yes: camera-stratified split 스크립트 작성 후 재분리

중복 존재?
  → Yes: 중복 제거 후 EDA 재실행

→ 모두 통과: train.py 실행
```

---

## 6. Precision 관련 패턴

### 학습(BF16) vs Export(FP16) 분리 이유

```text
학습: BF16 mixed precision
  - 목적: VRAM 절약 + 속도 향상
  - backbone은 frozen FP32 유지 (DINOv2 내부 inference_mode와 충돌 방지)
  - head/VPT/PAA만 autocast 범위 내에서 BF16 연산
  - BF16 ≠ FP16: 지수 비트가 FP32와 동일(8비트) → overflow 없음
  - GradScaler 불필요

ONNX Export: FP16
  - 목적: 추론 VRAM 절반 + RTX 4090 Tensor Core 활용
  - model.half()로 전체 변환 → ONNX에 FP16 가중치로 저장
  - 입력도 float16 필요 (ONNX Runtime GPU provider 자동 처리)
  - BF16으로 export하지 않는 이유: ONNX Runtime BF16 지원이 제한적
```

### BF16 autocast 확인 패턴

```python
# 학습 중 실제 BF16 연산이 일어나는지 확인
import torch

class _PrecisionProbe(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"input dtype inside model: {x.dtype}")  # bfloat16이어야 함
        return x

# Trainer._use_amp 확인
trainer = Trainer(..., device="cuda")
assert trainer._use_amp is True, "CUDA에서 AMP가 비활성화됨"
```

### ONNX FP16 검증 체크리스트

```text
1. export_onnx(model, path, fp16=True) 호출 후:
   - 파일이 생성되었는지 확인
   - verify_onnx()에서 atol=1e-3 사용 (FP16 수치 범위 제한)

2. VRAM 측정:
   - measure_vram(model, fp16=True) → "dtype": "float16" 확인
   - 목표: DINOv2-S ~550MB, DINOv2-B ~900MB

3. 양자화(INT8/FP8) 시도 전 GPU 지원 여부 확인:
   python -c "import torch; print(torch.cuda.get_device_capability())"
   # (8, 9) 이상 = Ada Lovelace → FP8 가능
   # (8, 6) = Ampere → INT8 가능, FP8 불가
   # RTX 4090 = (8, 9): FP8 가능하나 ONNX Runtime 지원 확인 필요
   → 현재 실험 범위 제외, FP16으로 5GB 제약 충족 시 불필요
```

---

## 7. 테스트 후 산출물 정리 패턴

### 삭제 대상 (smoke / 더미 테스트 산출물)

```text
tests/fixtures/dummy_crop/       더미 이미지 — 픽스처가 생성, 커밋 불필요
tests/fixtures/tmp/              픽스처 임시 디렉토리

experiments/*_smoke/             smoke 학습 결과 (예: exp_b_s_smoke/)
reports/eda_smoke/               smoke EDA 결과
reports/*_smoke/                 기타 smoke 리포트
```

이 디렉토리들은 `.gitignore`에 등록되어 있으며, 실수로 커밋되지 않는다.
삭제해도 재현 가능하므로 주기적으로 정리한다.

### 절대 삭제하지 않는 것 (실제 실험 산출물)

```text
experiments/exp_b_s/             실제 데이터로 학습한 결과
experiments/exp_d_s/             Primary MVP 실험 결과
reports/eda/                     실제 데이터 EDA 결과
reports/threshold_report/        threshold sweep 분석 결과
exports/*.onnx                   납품용 ONNX (기업 PC 배포용)
```

이 결과물은 재현에 시간이 걸리므로 삭제 전 반드시 확인한다.
바이너리(*.ckpt, *.onnx)는 .gitignore에 막혀 커밋되지 않지만, **로컬에서 보존**해야 한다.

### 구분 기준

```text
smoke 여부 판별 방법 (우선순위 순):

1. 디렉토리명 suffix
   smoke 실험:  experiments/exp_b_s_smoke/   (suffix: _smoke)
   실제 실험:   experiments/exp_b_s/         (suffix 없음)
   smoke EDA:   reports/eda_smoke/           (suffix: _smoke)
   실제 EDA:    reports/eda/

2. config 플래그 (YAML)
   is_smoke: true   → smoke 실험 — 출력 디렉토리에 _smoke suffix 강제
   is_smoke: false  → 실제 실험 (기본값, 명시 불필요)

3. 실험 명명 규칙
   실제 실험: exp_b_s, exp_b_b, exp_c_s, exp_c_b, exp_d_s, exp_d_b
   smoke:    위 이름에 _smoke 추가, 또는 _dummy 추가
             (예: exp_b_s_smoke, exp_b_s_dummy)
```

smoke 실험 실행 예시:

```bash
# smoke — _smoke suffix로 출력 격리
python train.py --config configs/exp_b_s.yaml \
                --data-dir tests/fixtures/dummy_crop \
                --output-dir experiments/exp_b_s_smoke \
                --epochs 3

# 실제 실험 — suffix 없음
python train.py --config configs/exp_b_s.yaml \
                --train-dir data/processed/train \
                --val-dir data/processed/val
```

### pytest fixture: yield + finally 패턴

```python
import shutil

@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory):
    """module scope: 모든 테스트 완료 후 tmp 디렉토리를 명시적으로 삭제."""
    tmp = tmp_path_factory.mktemp("e2e_pipeline")
    try:
        # ... 파이프라인 setup ...
        yield {
            "ckpt_dir": tmp / "checkpoints",
            "log_dir":  tmp / "logs",
            "onnx_path": tmp / "model.onnx",
        }
    finally:
        # pytest 자동 정리(마지막 3회 보존)에 더해 명시적으로 삭제
        shutil.rmtree(tmp, ignore_errors=True)
```

- `ignore_errors=True` — 이미 삭제됐거나 권한 오류 시 무시
- `scope="module"` → 모듈 내 모든 테스트 완료 후 teardown
- `scope="function"` → 각 테스트 종료 직후 teardown

### 더미 이미지 fixture 패턴 (smoke train.py용)

```python
from PIL import Image

@pytest.fixture(scope="session")
def dummy_crop_dir(tmp_path_factory: pytest.TempPathFactory):
    """위험/안전/제외 각 N장의 랜덤 이미지를 tmp에 생성하고 경로를 반환."""
    root = tmp_path_factory.mktemp("dummy_crop")
    for cls in ("위험", "안전", "제외"):
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(10):
            img = Image.fromarray(
                torch.randint(0, 256, (224, 224, 3)).numpy().astype("uint8")
            )
            img.save(cls_dir / f"{cls}_{i:02d}.jpg")
    yield root
    # tmp_path_factory가 세션 종료 후 삭제 — 별도 rmtree 불필요
```

### .gitignore 관리 원칙

```text
smoke/더미 산출물      experiments/*_smoke/  reports/*_smoke/  reports/eda_smoke/
테스트 픽스처          tests/fixtures/dummy_crop/  tests/fixtures/tmp/
납품·중간 산출물 폴더  exports/
바이너리 모델 파일     *.ckpt  *.pth  *.pt  *.onnx  *.npy  (커밋 금지, 로컬 보존)
기업 이미지 데이터     data/  +  *.jpg  *.png  *.jpeg  *.bmp 등

절대 커밋 금지:
  data/raw/       ← 기업 원본
  data/processed/ ← 전처리 결과
  *.ckpt          ← 모델 가중치 (실험 결과라도 예외 없음)

커밋 가능 (실제 실험):
  experiments/exp_b_s/logs/train_log.csv   ← 에포크별 지표 기록
  experiments/exp_d_s/logs/train_log.csv
  reports/eda/eda_report.html              ← 실제 데이터 EDA HTML
  reports/threshold_report/               ← threshold sweep 결과
```

---

## 공통 주의 사항

```text
Metric 키 일관성
  evaluator 반환 키: f1_macro, f2_macro, loss, danger_as_safe_rate
  checkpoint.py 기대 키: val_loss, f1, f2, safe_precision
  → trainer.py fit()에서 리매핑 후 ckpt_manager.save() 호출

Backbone frozen 확인
  HazardModel 생성 직후: assert all(not p.requires_grad for p in model.backbone.parameters())
  train() 호출 후에도 backbone._dino.training == False 유지해야 함

실험 결과 경로
  experiments/{exp_name}/checkpoints/  → 5종 ckpt
  experiments/{exp_name}/logs/train_log.csv → 에포크별 지표
  reports/eda_report.html → EDA 결과

val/test 분리 원칙
  threshold는 val에서 선택 → test에 고정 적용
  calibration은 val logit만 사용
  test 지표는 최종 1회만 계산 (oracle 사용 금지)
```
