# skills/coding-policy.md
# 코딩 정책

---

## 언어

```text
Python: 학습/평가/추론/리포트
YAML: 설정
Markdown: 문서
```

---

## 모듈화

```text
src/models/backbone.py      DINOv2 로드
src/models/vpt.py           VPT prompt tokens
src/models/paa.py           PAA token selection
src/models/head.py          attention pooling head
src/models/hazard_model.py  전체 조합
src/evaluation/threshold.py threshold decision
src/evaluation/calibration.py temperature scaling
```

파일은 가능하면 300줄 이하, 함수는 50줄 이하를 목표로 합니다.

---

## 설정 관리

하드코딩하지 않고 config에서 관리합니다.

```python
model = HazardModel(cfg.model)
dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size)
```

---

## 타입 힌트

모든 함수에 타입 힌트를 작성합니다.

```python
def compute_safe_precision(confusion_matrix: np.ndarray) -> float:
    ...
```

---

## 주석

주석은 what보다 why를 설명합니다.

```python
# 안전 판정은 고신뢰일 때만 허용한다. 위험→안전 오류 비용이 가장 크다.
if p_safe >= safe_thr and safe_margin >= margin:
    ...
```

---

## 보안

```text
데이터 경로 출력 금지
외부 API 호출 금지
원본 데이터 수정 금지
checkpoint/data 커밋 금지
```

---

## 출력 언어 정책

```text
스크립트 콘솔 출력(print, tqdm 설명 등): 한국어
오류 메시지 (sys.exit, ValueError 등): 한국어
matplotlib 그래프 제목/축 이름/설명 텍스트: 한국어
JSON/CSV 결과 파일의 key: 영어 유지
로그 태그 (예: [features], [validate]): 영어 유지
데이터 절대경로: 출력 금지

클래스 레이블 (danger / cut / excluded):
  figure 범례, 축 tick, 수치 테이블: 영어 원문 유지
  제목, 축 이름, 설명 텍스트: 한국어 유지
  내부 데이터 표현 (s.label 등): 한국어 유지, CLASS_DISPLAY로 변환 후 출력
```

---

## 커밋 메시지

```text
feat: add attention pooling head
feat: add threshold decision layer
exp: run dinov2-s vpt ablation
fix: correct danger-as-safe metric
docs: update public dataset policy
```

