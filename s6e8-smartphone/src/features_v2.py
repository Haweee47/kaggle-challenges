# -*- coding: utf-8 -*-
"""
피처 엔지니어링 v2 — 합성 데이터 구조를 활용한 확장 피처셋

06단계에서 CV 0.963546 → 0.966568 (+0.003022)을 만든 피처 파이프라인.

## 기여도 (탐색용 고속 설정 5-Fold 기준)
| 블록 | 순수 기여 |
|---|---|
| 빈도 인코딩          | +0.00197 |
| 결측 복원값           | +0.00047 |
| 시간 예산 제약        | +0.00028 |
| 소수점 지문           | +0.00017 |
| 생성 규칙 구간        | +0.00013 |

## 아이디어 출처 (Kaggle 공개 노트북 / Discussion)
- 시간 예산 제약: tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041
- 빈도 인코딩·소수점 지문: najiama/pure-lgbm-model-lb-0-97007-cv-0-96897
- 원본 생성 규칙: 대회 Discussion 제보 (원본 7,500행에서 확인)
모두 우리 데이터로 재검증한 뒤 채택했다.

## 검증된 구조적 사실
- `daily_screen_time >= social + gaming + work` : train/test 60만 행 **위반 0건**
- 주말에는 같은 제약 없음 (위반 5.3%)
- 소수부가 0.00~0.99의 100개 값뿐 → 생성기가 2자리 반올림
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from preprocess import NUM_COLS, CAT_COLS, TARGET, to_category, align_categories

SEED = 42
COMP = ['social_media_hours', 'gaming_hours', 'work_study_hours']
D = 'daily_screen_time_hours'
S = 'social_media_hours'
W = 'weekend_screen_time'

# 결측 복원 모델 설정 (변수 하나당 LightGBM 회귀 1개)
IMP_PARAMS = dict(n_estimators=300, learning_rate=0.08, num_leaves=63,
                  min_child_samples=40, feature_fraction=0.9,
                  verbose=-1, n_jobs=-1, random_state=SEED)


def fit_impute(full_cat, verbose=True):
    """
    결측값을 다른 변수로 예측해 복원한다.

    타깃을 쓰지 않으므로 train+test를 합쳐 학습해도 누수가 아니다.
    (이런 걸 transductive 전처리라고 한다)

    복원이 잘 되는 변수는 시간 예산 제약으로 묶인 것들이다:
      daily R2=0.82 / weekend 0.65 / social 0.57  vs  age·sleep 0.03~0.05
    """
    out = {}
    for c in NUM_COLS:
        na = full_cat[c].isnull()
        if not na.any():
            continue
        others = [x for x in NUM_COLS + CAT_COLS if x != c]
        reg = lgb.LGBMRegressor(**IMP_PARAMS)
        reg.fit(full_cat.loc[~na, others], full_cat.loc[~na, c])
        col = full_cat[c].copy()
        col[na] = reg.predict(full_cat.loc[na, others])
        out[c] = col.values
        if verbose:
            print(f'  복원 {c:26s} 결측 {na.mean():5.1%}', flush=True)
    return out


def build_v2(train_df, test_df, verbose=True):
    """
    최종 피처셋을 만든다.

    반환: (X_train, X_test, 열 목록)
    """
    n_tr = len(train_df)
    full = pd.concat([train_df[NUM_COLS + CAT_COLS], test_df[NUM_COLS + CAT_COLS]],
                     ignore_index=True)
    full_cat = to_category(full)

    X = full_cat.copy()
    d, s, w = full[D], full[S], full[W]

    # ── 1. 기본 비율 (04단계에서 검증된 3개) ──
    eps = 1e-6
    X['social_ratio'] = s / (d + eps)
    X['gaming_ratio'] = full['gaming_hours'] / (d + eps)
    X['study_ratio'] = full['work_study_hours'] / (d + eps)

    # ── 2. 결측 개수 ──
    X['n_missing'] = full[NUM_COLS + CAT_COLS].isnull().sum(axis=1).astype(np.int8).values

    # ── 3. 시간 예산 제약 ──
    # daily >= social+gaming+work 가 100% 성립하므로 '잔여 화면시간'이 실체를 갖는다.
    # 트리는 4항 선형결합을 축 정렬 분할로 표현하지 못하므로 직접 계산해준다.
    # NaN을 0으로 취급하는 편이 전파시키는 것보다 CV가 높았다(+0.00022).
    sgw = full[COMP].sum(axis=1)
    X['other_screen'] = d - sgw
    X['other_frac'] = (d - sgw) / d.clip(lower=0.1)
    X['sgw_frac'] = sgw / d.clip(lower=0.1)
    X['wk_minus_sgw'] = w - sgw
    X['wk_other'] = w - (d - sgw)

    # ── 4. 빈도 인코딩 (★ 최대 기여 +0.00197) ──
    # 합성 데이터의 생성기는 특정 값을 반복 샘플링한다.
    # "이 값이 전체에서 몇 번 나오는가"가 생성 밀도의 대리 지표가 된다.
    for c in NUM_COLS:
        st = full[c].astype(str)
        X[f'{c}_freq'] = st.map(st.value_counts()).fillna(0).astype(np.int32).values

    # ── 5. 소수점 지문 ──
    # 생성기가 2자리로 반올림한 흔적. 소수부 분포 자체가 정보를 갖는다.
    for c in NUM_COLS:
        X[f'{c}_dec'] = (full[c] % 1).round(2).values

    # ── 6. 결측 복원값 (원본과 '병기') ──
    # 원본을 대체하면 오히려 손해(-0.00078). "값이 없다"는 정보도 유지해야 한다.
    if verbose:
        print('결측 복원 모델 학습 중...', flush=True)
    imp = fit_impute(full_cat, verbose=verbose)
    for c, v in imp.items():
        X[f'{c}_imp'] = v

    # ── 7. 원본 생성 규칙 구간 ──
    # 원본 7,500행: daily>8 or social>4 → 1 / daily<=6 and social<=4 → 0
    # 대회 합성 데이터에서는 각각 97.6% / 32.5%로 뭉개져 있으므로
    # 하드코딩하지 않고 '힌트'로만 제공한다.
    # 트리는 OR 조건을 한 번에 못 자르므로 구간 번호를 주면 이득이 있다.
    zone = pd.Series(np.nan, index=full.index)
    ok = d.notna() & s.notna()
    hi = (d > 8) | (s > 4)
    lo = (d <= 6) & (s <= 4)
    zone[ok & hi] = 1
    zone[ok & lo] = 0
    zone[ok & ~hi & ~lo] = 2
    X['gen_zone'] = zone.values
    X['dist_d8'] = (d - 8).values
    X['dist_d6'] = (d - 6).values
    X['dist_s4'] = (s - 4).values
    X['rule_max'] = np.maximum((d - 8).fillna(-99), (s - 4).fillna(-99)).values

    Xtr = X.iloc[:n_tr].reset_index(drop=True)
    Xte = X.iloc[n_tr:].reset_index(drop=True)
    Xtr, Xte = align_categories(Xtr, Xte)

    assert list(Xtr.columns) == list(Xte.columns)
    if verbose:
        print(f'피처 {Xtr.shape[1]}개 생성 완료', flush=True)
    return Xtr, Xte, list(Xtr.columns)
