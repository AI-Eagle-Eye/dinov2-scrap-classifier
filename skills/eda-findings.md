# EDA 결과 요약 — crops_25pct 기준

> 기준 파일: `reports/eda/crops_25pct/eda_results.json`, `eda_summary.md`
> 수치 기반 기록. 추측 기반 항목 없음.

---

## 1. 데이터셋 현황 요약

### 클래스별 샘플 수 및 불균형

| 클래스 | 샘플 수 | 비율 |
|---|---|---|
| 안전 (cut) | ~9,713 | 54.3% |
| 위험 (danger) | ~5,392 | 30.1% |
| 제외 (excluded) | ~2,797 | 15.6% |
| 합계 | 17,903 | — |

- 최대 불균형 비율: 안전:제외 = **3.47:1**, 안전:위험 = 1.80:1
- EDA 결론: `weighted_sampler_needed = true`

### 저품질 및 데이터 오염

| 항목 | 수치 |
|---|---|
| 저품질 샘플 (blur < 100) | 897장 (5.01%) |
| 중증 전경비율 샘플 (heavy occlusion) | 1,702장 (9.51%) |
| MD5 완전 중복 (위험↔안전 동일 이미지) | 1쌍 |
| 클래스 간 유사 쌍 (cosine 유사도 기준) | 369쌍 (위험↔안전 중심) |
| 레이블 노이즈 의심 (cleanlab) | 1,946장 (10.9%) — 신뢰도 낮음 |
| 레이블 최종 재검토 목록 | 20개 |

---

## 2. src/ 코드 수정 방향

### A. `train.py` — WeightedSampler 미적용 (필수)

- `make_weighted_sampler`가 `dataset.py`에 구현돼 있으나 `train.py` DataLoader에 전달되지 않음
- 현재 `shuffle=True`만 사용 중
- 수정 방향: sampler 사용 시 `shuffle=False`로 전환하고 sampler 인자 전달
- 근거: `weighted_sampler_needed=True`, 클래스 비율 3.47:1

### B. `configs/*.yaml` — `use_weighted_sampler` 키 누락

- 6개 config 모두 `data:` 블록에 해당 키 없음
- 수정 방향: `data.use_weighted_sampler: true` 추가
- 근거: EDA 결정 사항 일치

### C. `data/dataset.py` — 저품질 샘플 제외 기능 없음

- `HazardDataset` 로딩 시 저품질 file_id를 제외하는 파라미터 없음
- 수정 방향: `exclude_stems: frozenset[str] | None = None` 파라미터 추가 후 로딩 시 스킵
- 근거: EDA `remove_low_quality_file_ids` 897개

### D. `train.py` — 데이터 오염 사전 차단 없음

- EDA에서 위험↔안전 완전 동일 이미지 1쌍 확인 (`img1203`, 인접 annotation ID)
- 동일 입력에 서로 다른 레이블 존재 → 학습 시 loss 발진 원인
- 수정 방향: 해당 쌍 중 하나를 수동으로 레이블 확정 후 `exclude_stems`로 제거
- 근거: MD5 exact match, `match_type="exact"`, score=0.0

### E. `transforms.py` — `CUSTOM_STD` 값 출처 검증 필요

- `CUSTOM_STD = (0.188, 0.188, 0.194)` vs EDA `channel_std = [0.089, 0.085, 0.089]`
- 두 값이 약 2배 차이
- 수정 방향: `channel_std`가 이미지 간 평균값의 표준편차(between-image)인지, 픽셀 단위 표준편차(per-pixel)인지 EDA 집계 방식 확인 후 결정
- 잘못된 정규화는 학습 발산 원인이 될 수 있음

---

## 3. 학습 하이퍼파라미터 권장값

EDA 수치 근거가 있는 항목만 기재.

| 항목 | 현재 값 | 권장 방향 | 근거 수치 |
|---|---|---|---|
| `use_weighted_sampler` | 없음 (미사용) | true로 활성화 | 불균형 3.47:1 |
| `label_smoothing` | 0.1 | 유지 | 위험↔안전 유사 쌍 369개, 레이블 혼재 확인 |
| 정규화 mean | [0.484, 0.483, 0.496] | 유지 | EDA channel_mean 일치 |
| 정규화 std | [0.188, 0.188, 0.194] | 검증 후 결정 | EDA channel_std=0.089와 2배 괴리 |
| `early_stopping_patience` | 10 | 유지 또는 증가 검토 | silhouette=0.0025, 초반 수렴 매우 느릴 수 있음 |

EDA 수치로 도출 불가한 항목 (추측 기반 권장 금지):
- `lr`, `weight_decay`, `dropout`, `paa.topk`, `vpt.num_tokens`, `batch_size`

---

## 4. 확인이 필요한 불확실한 항목

### ① DINOv2 silhouette ≈ 0

- 전체 silhouette = **0.0025**, 클래스별: 위험 −0.008 / 안전 +0.010 / 제외 −0.002
- Frozen DINOv2 feature 공간에서 3개 클래스가 완전히 혼재
- "태스크 자체가 어렵다(외관 유사)"인지 "feature 추출이 실패했다"인지 이미지 직접 확인 필요
- VPT+PAA로 얼마나 개선되는지는 실험 전까지 판단 불가

### ② 레이블 노이즈 1,946장 — 신뢰도 낮음

- cleanlab 결과 `low_confidence=True` (silhouette=0.0025 < 기준 0.1)
- 최종 재검토 목록 20개는 수동 확인 가능
- 1,946장 전체는 수치만으로 판단 불가, 맹목적 제거 금지

### ③ `CUSTOM_STD` 0.188 vs EDA `channel_std` 0.089

- 어느 값이 픽셀 단위 표준편차인지 EDA 집계 방식 확인 필요
- 2항목 B, E와 연동

### ④ camera_info = "not_available"

- 카메라 메타데이터 부재로 4K/FHD domain shift 검증 불가
- MMD=0.0은 분리 불가 결과이며 shift 없음을 증명하지 않음

### ⑤ MD5 완전 중복 쌍의 정답 레이블

- `위험_ann3903_img1203` ↔ `안전_ann3904_img1203` 중 정답 레이블 원본 확인 필요
- 해결 전까지 두 샘플 모두 학습에서 제외 권장
