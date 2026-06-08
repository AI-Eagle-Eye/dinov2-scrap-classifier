# skills/dataset.md
# 데이터셋 처리 규칙

---

## 핵심 원칙

```text
학습 데이터: 기업 제공 데이터만 사용
검증 데이터: 기업 제공 데이터만 사용
최종 평가: 기업 제공 데이터만 사용

공개 데이터셋:
  실험 파이프라인에서 완전 제외
  VPT 사전 적응 금지
  self-supervised tuning 금지
  cross-dataset 학습 금지
  문헌/데이터 조사 항목으로만 기록
```

---

## 데이터 보안

```text
data/ 디렉토리는 .gitignore 필수
데이터 경로를 로그/출력/커밋 메시지에 포함 금지
데이터 관련 코드에 외부 통신 금지
원본 데이터 수정 금지
로그에는 데이터 건수와 통계만 기록
```

---

## 데이터 소스 (확정)

```text
라벨 CSV   : dataset/label_with_split_260527.csv
이미지 루트 : dataset/classification/crops_25pct/{danger,cut,excluded}/

전체        : 17,901장
  cut(0)      9,713장
  danger(1)   5,393장
  excluded(2) 2,797장

제외/포함 규칙:
  short_side < 32  → 143장 제거
  unk 표시 샘플    → excluded로 포함 (HazardDataset unk_label="excluded")

클래스 매핑: cut=0 / danger=1 / excluded=2  (src/data/dataset.py LABEL_MAP)
```

> ⚠️ dataset/ 폴더(이미지·CSV)는 절대 수정 금지 — 원본 보존, .gitignore 필수.

---

## 디렉토리 구조

```text
dataset/                          # .gitignore 필수 (원본 수정 금지)
├── label_with_split_260527.csv   # split 컬럼 포함 라벨 CSV
└── classification/
    └── crops_25pct/              # 학습/EDA 기본 경로
        ├── danger/               # 위험 클래스 (label=1)
        ├── cut/                  # 안전 클래스, 절단 확인 (label=0)
        └── excluded/             # 제외 클래스 (label=2)
```

### crops_25pct 선택 이유
```text
- bbox에 25% padding을 준 crop. 0pct/10pct/25pct 비교(EDA) 결과
  25%가 절단면 끝단과 주변 맥락을 가장 안정적으로 보존.
- 10~25% 권장 범위(AGENT.md) 내 상단값으로, 타이트 crop의 맥락 손실을 방지.
```

---

## bbox Crop 처리

```python
def crop_with_padding(image, bbox, padding: float = 0.15):
    """
    bbox에 padding을 추가해 crop한다.
    너무 타이트한 crop은 절단면 끝단과 주변 맥락을 잃을 수 있다.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(image.width, x2 + pad_w)
    y2 = min(image.height, y2 + pad_h)
    return image.crop((x1, y1, x2, y2))
```

확정값:

```text
bbox_padding = 0.25  (crops_25pct)
  - AGENT.md 권장 범위(0.10~0.20)보다 상단이나, EDA 패딩 비교(0/10/25pct)에서
    25%가 절단면 끝단·주변 맥락 보존에 가장 안정적이어서 채택.
  - crop 이미지는 dataset/classification/crops_25pct/에 사전 생성되어 있음
    (학습 시 재크롭하지 않음).
```

---

## Split 정책 (확정)

```python
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
```

```text
방식: 클러스터 기반 split (DINOv2 feature DBSCAN 클러스터 단위 배정)
  - 같은/거의 동일한 프레임이 train/test에 나뉘지 않도록 클러스터를 통째로 배정 → 데이터 유출 방지.
  - noise(단독) 샘플은 클래스별 stratified split.
  - split 결과는 dataset/label_with_split_260527.csv의 split 컬럼에 고정 저장.
  - HazardDataset은 이 split 컬럼만 읽어 train/val/test를 구성 (재계산 없음).
```

---

## 전처리 (확정 — src/data/transforms.py)

```text
입력 크기   : 336×336 LetterBox (가로세로 비율 유지 + 중앙 패딩)
  - 단순 Resize가 아닌 LetterBox로 객체 왜곡 방지.
  - 패딩 색은 CUSTOM_MEAN×255 = (123,123,126)로 분포 shift 최소화.

정규화 (데이터 실측값, ImageNet 아님):
  mean = [0.484, 0.483, 0.496]
  std  = [0.188, 0.188, 0.194]
  - EDA 픽셀 분석에서 스크랩야드 이미지의 B채널 shift가 ImageNet과 차이가 커
    커스텀 정규화를 채택.
```

학습:

```python
train_transforms = transforms.Compose([
    LetterBox(336),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.484, 0.483, 0.496],
                         std=[0.188, 0.188, 0.194]),
])
```

검증/테스트:

```python
val_transforms = transforms.Compose([
    LetterBox(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.484, 0.483, 0.496],
                         std=[0.188, 0.188, 0.194]),
])
```

---

## 클래스 불균형 처리 (확정)

```text
채택: Focal Loss(γ=2.0) + inverse-frequency class weight
  - compute_class_weights(dataset)로 inverse-freq 가중치 산출 후 Focal Loss에 적용.

제외: WeightedRandomSampler
  - Focal Loss와 동시 적용 시 소수 클래스가 이중 보정되어 과보정·불안정.
  - 실험 결과 단독 Focal Loss 대비 이득 없음. 자세한 근거는 [skills/experiments.md].
```

```python
# src/data/dataset.py — inverse-frequency class weight
class_weights = compute_class_weights(train_ds)  # FocalLoss(weight=...)에 전달
```

---

## EDA 체크리스트

```text
- [ ] 클래스별 이미지 수와 비율
- [ ] 카메라별 이미지 수, 4K/FHD 비율
- [ ] bbox 크기 분포
- [ ] crop 후 객체가 너무 작은 샘플
- [ ] occlusion 샘플
- [ ] 위험/안전 경계 사례
- [ ] 제외 클래스 사례 유형
- [ ] 중복 이미지
- [ ] 레이블 오류 의심 샘플
```

---

## 공개 데이터셋 조사 결과

| 데이터셋 | 라이선스 | 비고 |
|---|---|---|
| CylinDeRS | CC BY 4.0 | 가스통 bbox, 위험/안전 라벨 없음 |
| Gas Cylinder Roboflow | CC BY 4.0 | 가스통 bbox, 품질 검수 필요 |
| DOES | CC BY 4.0 | 고철 등급 분류, 위험/안전 라벨 없음 |
| AIHub 산업 폐기물 | 개별 확인 필요 | 위험/안전 라벨 없음 |
| NEU Surface Defect | 비상업 조건 | 상업 사용 부적합 |

사용하지 않는 이유:

```text
핵심 라벨이 없음
VPT/SSL에 쓰면 실험 범위가 커짐
기업 데이터만 쓰는 편이 결과 해석이 명확함
40일 일정에서는 평가와 ablation 집중이 더 중요함
```

