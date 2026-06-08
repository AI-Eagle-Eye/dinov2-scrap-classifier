# skills/evaluation.md
# 평가 지표 및 리포트 정책

---

## 핵심 지표 정의

Confusion matrix 행은 실제 클래스, 열은 예측/decision 클래스입니다.

```text
class index:
  0 = 위험
  1 = 안전
  2 = 제외/재검토
```

### Danger-as-Safe Rate

실제 위험 샘플 중 **안전으로 판정된** 비율입니다. 이 프로젝트에서 가장 중요한 실패입니다.

```python
danger_as_safe_rate = cm[0, 1] / cm[0, :].sum()
```

### Danger Miss Rate

실제 위험 샘플 중 위험으로 잡지 못한 비율입니다. `위험→안전`과 `위험→제외`를 모두 포함합니다.

```python
danger_miss_rate = (cm[0, 1] + cm[0, 2]) / cm[0, :].sum()
```

`Danger-as-Safe Rate`와 구분해서 보고합니다. `위험→제외`는 재검토로 이어질 수 있지만, `위험→안전`은 통과 오류라 비용이 훨씬 큽니다.

### Safe Precision

안전으로 판정된 샘플 중 실제 안전 샘플 비율입니다.

```python
safe_precision = cm[1, 1] / cm[:, 1].sum()
```

### Safe Coverage

실제 안전 샘플 중 자동으로 안전 통과된 비율입니다.

```python
safe_coverage = cm[1, 1] / cm[1, :].sum()
```

---

## 필수 지표

```text
Accuracy
Precision / Recall / F1, macro + per-class
F2 score
Confusion matrix
PR curve, safe-vs-rest
PR curve, danger-vs-rest
Danger-as-Safe Rate
Danger Miss Rate
Safe Precision
Threshold sweep
```

가능하면:

```text
ECE
Reliability diagram
Safe Coverage
카메라별 성능
```

---

## Threshold Sweep

MVP에서는 `safe_threshold`만 sweep합니다.

```python
SWEEP_TARGET = "safe_threshold"
DANGER_THRESHOLD = 0.60
MARGIN = 0.20
SWEEP_START = 0.30
SWEEP_END = 0.90
SWEEP_STEP = 0.05
```

Optional:

```text
safe_threshold × margin 2D sweep
```

`threshold_results.json`:

```json
{
  "config": {
    "danger_threshold": 0.60,
    "margin": 0.20
  },
  "sweep": {
    "0.30": {
      "safe_precision": 0.78,
      "safe_coverage": 0.91,
      "danger_as_safe": 0.03,
      "danger_miss": 0.12,
      "f1": 0.87
    }
  },
  "recommended": {
    "max_f1": {"safe_thr": 0.42, "f1": 0.891},
    "max_f2": {"safe_thr": 0.35, "f2": 0.912},
    "safe_precision_95_max_coverage": {
      "safe_thr": 0.61,
      "safe_precision": 0.951,
      "safe_coverage": 0.784
    },
    "min_danger_as_safe": {
      "safe_thr": 0.75,
      "danger_as_safe": 0.005
    }
  }
}
```

---

## Calibration

Temperature scaling은 val set으로만 수행합니다.

```python
def fit_temperature(logits_val, labels_val) -> float:
    """
    모델 파라미터는 고정하고 temperature T만 학습한다.
    train set 또는 test set으로 T를 고르면 안 된다.
    """
    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_val / T, labels_val)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach())
```

평가:

```text
calibration 전후 ECE
calibration 전후 reliability diagram
temperature 값 export_info.json 기록
```

---

## 체크포인트별 사용처

```text
best_val_loss:
  ablation 비교 기준

best_f1:
  균형 성능 비교

best_f2:
  위험 recall 중심 비교

best_safe_precision:
  calibration/threshold 적용 후 납품 후보

last:
  재시작/디버깅
```

---

## Ablation 테이블

```markdown
| 모델 | Accuracy | F1 | Danger-as-Safe ↓ | Danger Miss ↓ | Safe Precision ↑ | Params | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-S: DINOv2-S + Attn | | | | | | | |
| B-B: DINOv2-B + Attn | | | | | | | |
| C-S: DINOv2-S + VPT + Attn | | | | | | | |
| C-B: DINOv2-B + VPT + Attn | | | | | | | |
| D-S: DINOv2-S + VPT + PAA + Texture + Attn | | | | | | | |
| D-B: DINOv2-B + VPT + PAA + Texture + Attn | | | | | | | |
| E: Final + Calibration + Threshold | | | | | | | |
```

---

## report.html

포함 내용:

```text
1. 실험 요약
2. 학습 곡선
3. confusion matrix
4. safe-vs-rest PR curve
5. danger-vs-rest PR curve
6. safe_threshold slider
7. ablation 비교 테이블
8. attention map
9. Danger-as-Safe 오분류 갤러리
10. reliability diagram
```

기술:

```text
plotly.js 오프라인 번들 사용
외부 CDN 금지
HTML 단일 파일 지향
```

---

## CLI

```bash
python evaluate.py --exp experiments/exp_d_s_... \
  --checkpoint best_f1 \
  --safe_thr 0.70 \
  --danger_thr 0.60 \
  --margin 0.20

python evaluate.py --exp experiments/exp_d_s_... --sweep

python evaluate.py --exp experiments/exp_d_s_... \
  --checkpoint best_f1 \
  --camera_split

python report.py --compare exp_b_s exp_b_b exp_c_s exp_c_b exp_d_s \
  --output reports/ablation_final.html
```

