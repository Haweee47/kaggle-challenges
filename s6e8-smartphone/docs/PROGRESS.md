# Playground Series S6E8 — 진행 과정 요약

> Predicting Smartphone Addiction · 스마트폰 중독 여부 예측
> 저장소: [Haweee47/kaggle-challenges](https://github.com/Haweee47/kaggle-challenges)
> 작성일: 2026-08-30

---

## 1. 대회 개요

| 항목 | 내용 |
|---|---|
| 대회 | [Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) |
| 문제 유형 | 이진 분류 — **확률**을 제출 |
| 평가 지표 | ROC-AUC *(제출 형식으로 추정, Evaluation 탭 확인 필요)* |
| train | 691,369 행 × 13 피처 |
| test | 296,302 행 |
| 타깃 | `addicted_label` — 양성 70.9% : 음성 29.1% |

### 데이터 구성

| 종류 | 변수 |
|---|---|
| 수치형 9개 | `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time` |
| 범주형 3개 | `gender`, `stress_level`, `academic_work_impact` |

---

## 2. 단계별 진행 현황

| 단계 | 노트북 | 상태 | 핵심 산출물 |
|---|---|---|---|
| 01 | `01_eda.ipynb` | ✅ 완료 | 결측 구조 파악, 변수 중요도 사전 진단 |
| 02 | `02_preprocessing.ipynb` | ✅ 완료 | 전처리 방침 확정, 5-Fold 검증 틀 |
| 03 | `03_baseline_lgbm.ipynb` | ✅ 완료 | **베이스라인 CV AUC 0.963574** |
| 04 | `04_feature_engineering.ipynb` | 🔄 진행중 | 파생변수 선별 + Optuna 튜닝 |
| 05 | `05_ensemble_submit.ipynb` | ⏳ 대기 | LGB/XGB/CatBoost 앙상블 |

---

## 3. 01단계 — EDA에서 알아낸 것

### 3-1. 스크린타임 3형제가 압도적

| 변수 | 타깃 상관 | 효과크기 | 비중독 평균 → 중독 평균 |
|---|---|---|---|
| `daily_screen_time_hours` | **0.611** | **1.35** | 5.04h → 8.71h |
| `weekend_screen_time` | **0.590** | **1.30** | 6.85h → 10.56h |
| `social_media_hours` | **0.532** | **1.17** | 1.38h → 2.92h |
| `work_study_hours` | 0.251 | 0.55 | 1.87h → 2.57h |
| `gaming_hours` | 0.205 | 0.45 | 1.16h → 1.58h |
| `app_opens_per_day` | 0.064 | 0.14 | 97.9 → 104.6 |
| `sleep_hours` | 0.043 | 0.09 | 6.72h → 6.84h |
| `notifications_per_day` | −0.012 | −0.03 | 147.1 → 145.4 |
| `age` | 0.004 | 0.01 | 26.58 → 26.63 |

> **인사이트**: 알림 개수와 앱 실행 횟수는 중독과 **무관**하다.
> "얼마나 자주 켜느냐"가 아니라 **"얼마나 오래 쓰느냐"** 가 중독을 가른다.

### 3-2. 결측치가 최대 난관

- 전 열 **4.18% ~ 19.38%** 결측
- 결측이 하나도 없는 완전한 행은 **38.9%뿐** → `dropna()` 쓰면 61% 손실
- train/test 결측률 차이는 작아 동일 전처리 적용 가능

### 3-3. 범주형은 신호가 거의 없음

| 변수 | 중독률 편차 |
|---|---|
| `gender` | 2.2%p |
| `stress_level` | 0.6%p |
| `academic_work_impact` | 0.3%p |

### 3-4. 다중공선성 주의

`daily_screen_time_hours` ↔ `weekend_screen_time` 상관 **0.80**
→ 두 값의 **비율/차이**를 파생변수로 만들 여지가 있음 (04단계에서 검증)

---

## 4. 02단계 — 전처리 방침

| 항목 | 결정 | 근거 |
|---|---|---|
| 결측치 | 원본 유지를 1순위로, 대치안과 CV 비교 | 중앙값 대치 시 인위적 봉우리 발생 |
| 인코딩 | `category` dtype (LightGBM 네이티브) | 열 증가 없음 + 순서 오해 없음 |
| 스케일링 | **하지 않음** | 트리 모델은 값의 *순서*만 사용 |
| 검증 | `StratifiedKFold(5, shuffle=True, random_state=42)` | 불균형 데이터 + 재현성 |

### 구조적으로 방지한 함정 2가지

**1. 데이터 누수 (Data Leakage)**
대치값을 전체 데이터에서 계산하면 검증 폴드의 정보가 학습에 샌다.
`impute_median()`이 계산한 중앙값을 **반환**하도록 설계해 학습 폴드에서만
계산하고 검증/테스트에는 *적용만* 하도록 강제했다.

**2. train/test 카테고리 순서 불일치**
각각 `astype('category')`를 하면 `Male`이 train에서 1번, test에서 0번이 될 수 있다.
`align_categories()`가 카테고리 목록을 합집합으로 통일한다.

---

## 5. 03단계 — 베이스라인

### 결과: **CV AUC 0.963574**

| 항목 | 값 |
|---|---|
| 폴드 평균 ± 표준편차 | 0.963576 ± 0.000562 |
| 사용 피처 | 12개 (원본) |
| 평균 트리 개수 | 1,761 |
| 학습 시간 | 194.6초 (5-Fold) |

폴드 표준편차가 0.00056으로 작아 **검증 틀이 안정적**이다.

### 결측 처리 4전략 실측 비교 (고속 설정)

| 전략 | CV AUC | 베이스라인 대비 |
|---|---|---|
| **D. 원본 + 행별 결측개수** | **0.963299** | **+0.000549** |
| C. 원본 + 결측표시변수 12개 | 0.962897 | +0.000148 |
| A. 원본 유지 | 0.962750 | 기준 |
| B. 중앙값/최빈값 대치 | 0.962478 | −0.000272 |

### ⚠️ EDA 가설이 빗나간 지점

01단계에서 *"결측 여부와 타깃의 관계가 최대 0.42%p이므로 결측 표시 변수는
쓸모없을 것"* 이라 예측했다. **실측 결과는 반대였다.**

**왜 빗나갔나**

1. **상호작용(interaction)** — EDA에서 본 것은 "한 변수의 결측 여부"와 타깃의
   1:1 관계다. 하지만 트리 모델은 *"스크린타임이 8시간이면서 결측이 3개인 사람"*
   같은 **조합**을 자동으로 찾아낸다. 단변량 분석으로는 절대 보이지 않는다.

2. **요약이 개별보다 나았다** — 개별 표시 변수 12개(C)보다 이를 하나로 합친
   `n_missing`(D)이 더 좋았다. 열을 12개 늘리면 정보는 희석되고 노이즈만 늘어난다.

**교훈**
- EDA는 **가설을 만드는 단계**지 결론을 내는 단계가 아니다. 최종 판단은 항상 CV다.
- 피처는 많을수록 좋은 게 아니다. 같은 정보라면 **압축된 형태**가 유리하다.

> 단, 개선폭(+0.0005)은 폴드 표준편차(0.00056)와 비슷한 수준이다.
> "확실한 개선"보다는 **"약한 신호"** 로 보는 것이 정직하다.
> 다만 모든 실험이 **동일한 폴드 분할**을 쓴 짝지은 비교라 우연보다는 신뢰할 만하다.

---

## 6. 04단계 — 피처 엔지니어링 (실험 결과)

### 6-1. 그룹별 단독 효과

원본 12피처에 그룹을 **하나씩만** 추가해 순수 효과를 측정 (애블레이션 실험).

| 그룹 | 추가 변수 | CV AUC | 베이스라인 대비 |
|---|---|---|---|
| **구성비율** | `social_ratio`, `gaming_ratio`, `study_ratio` | **0.963443** | **+0.000694** |
| 여가지표 | `leisure_hours`, `leisure_vs_study` | 0.963176 | +0.000426 |
| 사용강도 | `mins_per_open`, `opens_per_notif` | 0.962893 | +0.000143 |
| 주말패턴 | `weekend_ratio`, `weekend_diff` | 0.962882 | +0.000132 |
| 시간예산 | `awake_hours`, `screen_per_awake`, `sleep_screen_sum` | 0.962780 | +0.000030 |
| — | *(전체 12개 모두 추가)* | 0.963253 | +0.000503 |

> **전체 12개를 다 넣은 것이 좋은 그룹 하나만 넣은 것보다 나빴다.**

### 6-2. 조합 실험 — 피처를 늘릴수록 나빠진다

| 조합 | 피처 수 | CV AUC | 베이스라인 대비 |
|---|---|---|---|
| **K1. 원본 + `n_missing` + 구성비율** | **16** | **0.963615** | **+0.000865** |
| K2. K1 + 여가지표 | 18 | 0.963611 | +0.000861 |
| K5. K2 + 결측표시변수 12개 | 30 | 0.963382 | +0.000632 |
| K3. K2 + 주말패턴 | 20 | 0.963244 | +0.000495 |
| K4. K3 + 사용강도 | 22 | 0.963070 | +0.000321 |
| K0. 베이스라인 | 12 | 0.962750 | 기준 |

**16피처가 최적점.** 이후로는 피처를 더할수록 단조 감소한다.

### 6-3. 왜 쓸모없는 피처가 점수를 깎는가

LightGBM은 트리를 만들 때마다 모든 후보 변수를 검토해 최적 분기를 찾는다.
기여도 낮은 변수가 섞이면:

1. 그 변수로 분기하는 트리가 **우연히** 만들어진다 (노이즈 학습)
2. `feature_fraction=0.8` 때문에 **정작 좋은 변수가 후보에서 빠지는** 일이 잦아진다

즉 **쓸모없는 피처는 중립이 아니라 마이너스**다.

### 6-4. 확정된 최종 피처셋 (16개)

```python
원본 12개
  + n_missing        # 행별 결측 개수
  + social_ratio     # social_media_hours / daily_screen_time_hours
  + gaming_ratio     # gaming_hours / daily_screen_time_hours
  + study_ratio      # work_study_hours / daily_screen_time_hours
```

`src/preprocess.py`의 `build_final()`로 고정해 05단계에서도 동일하게 재현한다.

---

## 7. 트러블슈팅 기록

| # | 문제 | 원인 | 해결 |
|---|---|---|---|
| 1 | `kaggle.json` 다운로드 안 됨 | 캐글이 OAuth 방식으로 전환 | `kaggle auth login` 브라우저 인증 |
| 2 | PowerShell `UnauthorizedAccess` | 스크립트 실행 정책 차단 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 3 | `.venv` 활성화 실패 | 홈 디렉토리에서 실행 | 프로젝트 폴더로 `cd` 후 실행 |
| 4 | matplotlib 한글 깨짐 | 기본 폰트가 한글 미지원 | `plt.rcParams['font.family'] = 'Malgun Gothic'` |
| 5 | 노트북 `ModuleNotFoundError` | 커널이 시스템 파이썬을 가리킴 | `ipykernel install --name kaggle` 등록 |
| 6 | 노트북에 그래프 미저장 | `MPLBACKEND=Agg` 강제 | 환경변수 제거 → inline 백엔드 사용 |
| 7 | CatBoost `cat_features must be...` | pandas 3.0에서 `astype(str)`이 NA 보존 | `np.where(s.isna(), 'NA', s.astype(str))` |

---

## 8. 프로젝트 구조

```
s6e8-smartphone/
├── data/                       # 원본 CSV (git 제외)
├── notebooks/
│   ├── 01_eda.ipynb            # 데이터 탐색
│   ├── 02_preprocessing.ipynb  # 전처리 + 검증 틀
│   ├── 03_baseline_lgbm.ipynb  # 베이스라인
│   ├── 04_feature_engineering.ipynb
│   └── 05_ensemble_submit.ipynb
├── src/
│   ├── preprocess.py           # 로드/인코딩/파생변수/최종 피처셋
│   └── cv.py                   # 5-Fold 고정, run_cv, ExperimentLog
├── submissions/                # 제출 파일
├── preds/                      # OOF/테스트 예측 (git 제외)
├── docs/
│   ├── PROGRESS.md             # 이 문서
│   └── progress.html           # HTML 버전
└── README.md
```

---

## 9. 성능 추적

| 버전 | 구성 | 피처 | CV AUC | LB |
|---|---|---|---|---|
| v1 | LightGBM 베이스라인 | 12 | 0.963574 | — |
| v2 | + 피처 엔지니어링 + Optuna | 16 | *(04단계 진행중)* | — |
| v3 | + LGB/XGB/CatBoost 앙상블 | 16 | *(05단계 대기)* | — |

---

## 10. 남은 작업

- [ ] 04단계 Optuna 튜닝 완료 및 CV 재측정
- [ ] 05단계 3-모델 앙상블 + 블렌딩 가중치 탐색
- [ ] 캐글 제출 후 LB 점수 기록 → CV/LB 상관 확인
- [ ] 대회 Evaluation 탭에서 평가 지표 최종 확인
