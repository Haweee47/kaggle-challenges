# Playground Series S6E8 — Predicting Smartphone Addiction

스마트폰 사용 습관 데이터로 **중독 여부를 예측**하는 이진 분류 대회.

- **대회 링크**: https://www.kaggle.com/competitions/playground-series-s6e8
- **문제 유형**: 이진 분류 (확률 제출)
- **평가 지표**: ROC-AUC *(제출 형식 기준 추정 — Evaluation 탭 확인 필요)*
- **데이터**: train 691,369행 × 13피처 / test 296,302행
- **타깃**: `addicted_label` — 양성 70.9%

📄 **[전체 진행 과정 보기 → docs/PROGRESS.md](docs/PROGRESS.md)** ([HTML 버전](docs/progress.html))

---

## 최종 결과

| 항목 | 값 |
|---|---|
| 베이스라인 CV AUC | 0.963574 |
| **최종 CV AUC** | **0.964890** |
| 누적 개선 | +0.001316 |
| 최종 구성 | LightGBM + XGBoost + CatBoost 단순 평균 |
| 최종 피처 | 16개 |

---

## 실험 로그

| # | 단계 | 구성 | 피처 | CV AUC | 단계별 개선 | LB |
|---|---|---|---|---|---|---|
| v1 | 03 | LightGBM 베이스라인 | 12 | 0.963574 | — | — |
| — | 04 | + 피처 엔지니어링 | 16 | 0.963951 | +0.000378 | — |
| v2 | 04 | + Optuna 튜닝 | 16 | 0.964516 | +0.000565 | — |
| v3 | 05 | + 3-모델 앙상블 | 16 | **0.964890** | +0.000374 | — |

### 개별 모델 (05단계)

| 모델 | CV AUC | 학습 시간 |
|---|---|---|
| XGBoost | 0.964521 | 487초 |
| LightGBM | 0.964516 | 225초 |
| CatBoost | 0.963579 | 3,978초 |

---

## 폴더 구조

```
s6e8-smartphone/
├── data/                            # 원본 CSV (git 제외)
├── notebooks/                       # 01~05 단계별 노트북 (실행 결과 포함)
├── src/
│   ├── preprocess.py                # 로드/인코딩/파생변수/build_final()
│   └── cv.py                        # 5-Fold 고정, run_cv, ExperimentLog
├── submissions/                     # 제출 파일 (v1~v3)
├── preds/                           # OOF/테스트 예측 (git 제외)
├── docs/                            # 진행 요약 (MD/HTML)
└── README.md
```

---

## 재현 방법

```powershell
# 1. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 2. 데이터 다운로드 (kaggle auth login 선행 필요)
kaggle competitions download -c playground-series-s6e8 -p s6e8-smartphone/data
Expand-Archive s6e8-smartphone/data/playground-series-s6e8.zip -DestinationPath s6e8-smartphone/data

# 3. 노트북을 01 → 05 순서로 실행 (커널: Python (kaggle))
```

> 모든 실험은 `SEED = 42`와 동일한 `StratifiedKFold(5, shuffle=True)` 분할을 사용하므로
> 실험 간 점수를 직접 비교할 수 있다.

**⏱ 실행 시간 참고**: 04단계 약 50분(Optuna 포함), 05단계 약 80분(CatBoost가 대부분).

---

## 배운 것 3가지

**1. EDA는 가설을 만드는 단계지 결론을 내는 단계가 아니다**
EDA에서 "결측 표시 변수는 쓸모없다"고 판단했지만, CV로 재보니 반대였다.
단변량 분석은 트리 모델이 찾아내는 **변수 간 상호작용**을 보지 못한다.

**2. 피처는 많을수록 좋은 게 아니다**
파생변수 12개를 전부 넣은 것보다 효과가 확인된 4개만 넣은 것이 더 높았다.
기여하지 않는 피처는 중립이 아니라 **점수를 깎는다**.

**3. 통념보다 자기 데이터의 숫자를 믿어라**
"피처가 튜닝보다 중요하다", "상관 0.99 이상이면 앙상블 이득이 없다" — 두 통념 모두
이 대회에서는 맞지 않았다. 자세한 이유는 [PROGRESS.md](docs/PROGRESS.md)에 정리했다.

---

## 제출

```powershell
kaggle competitions submit -c playground-series-s6e8 `
  -f s6e8-smartphone/submissions/submission_v3_ensemble.csv `
  -m "LGB+XGB+CatBoost simple average (CV 0.964890)"
```
