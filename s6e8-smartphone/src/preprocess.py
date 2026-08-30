# -*- coding: utf-8 -*-
"""
전처리 재사용 모듈 — Playground Series S6E8

노트북에서 이렇게 불러 쓴다:

    import sys; sys.path.append('../src')
    from preprocess import load_data, build_features, NUM_COLS, CAT_COLS, TARGET

같은 전처리를 02~05단계 노트북에서 반복해서 쓰기 때문에 함수로 묶어둔다.
노트북마다 코드를 복붙하면 한 곳만 고쳐도 다른 노트북과 어긋나서
"CV 점수가 왜 다르지?" 하는 사고가 난다.
"""
import numpy as np
import pandas as pd

SEED = 42
TARGET = 'addicted_label'

# 원본 데이터의 열 구성 (01_eda에서 확인한 그대로)
NUM_COLS = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time',
]
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']

# 01_eda에서 타깃과의 상관이 특히 높았던 '스크린타임 3형제'
SCREEN_COLS = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours']


def load_data(data_dir='../data/'):
    """train / test / sample_submission 을 한 번에 읽어온다."""
    train = pd.read_csv(data_dir + 'train.csv')
    test = pd.read_csv(data_dir + 'test.csv')
    sub = pd.read_csv(data_dir + 'sample_submission.csv')
    return train, test, sub


def to_category(df, cat_cols=CAT_COLS):
    """
    문자열 범주형을 pandas의 category dtype으로 바꾼다.

    왜 필요한가: LightGBM은 category dtype 열을 '범주형'으로 인식해서
    원-핫 인코딩 없이 바로 학습할 수 있다. 문자열(str) 그대로면 에러가 난다.
    또 category는 내부적으로 정수로 저장돼 메모리도 아낀다.
    """
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype('category')
    return out


def align_categories(train_df, test_df, cat_cols=CAT_COLS):
    """
    train과 test의 category 코드 순서를 맞춘다.

    왜 필요한가: 각각 따로 astype('category')를 하면 카테고리 정렬 순서가
    달라질 수 있고, 그러면 train에서 0번이던 값이 test에서 1번이 되어
    모델이 엉뚱하게 예측한다. 학습/추론 불일치의 대표적인 원인이다.
    """
    tr, te = train_df.copy(), test_df.copy()
    for c in cat_cols:
        if c in tr.columns and c in te.columns:
            cats = sorted(set(tr[c].dropna().unique()) | set(te[c].dropna().unique()))
            dtype = pd.CategoricalDtype(categories=cats)
            tr[c] = tr[c].astype(dtype)
            te[c] = te[c].astype(dtype)
    return tr, te


def add_missing_indicators(df, source_df, cols=None):
    """
    '값이 비어 있다'는 사실을 0/1 열로 추가한다 (missing indicator).

    01_eda 결론: 이 데이터에서는 결측이 거의 완전 무작위(MCAR)라
    효과가 크지 않을 것으로 예상된다. 그래도 검증은 해봐야 한다.

    source_df: 결측 여부를 판단할 원본 (대치 전 데이터)
    """
    out = df.copy()
    cols = cols or (NUM_COLS + CAT_COLS)
    for c in cols:
        out[f'{c}_isna'] = source_df[c].isnull().astype(np.int8)
    return out


def add_missing_count(df, source_df, cols=None):
    """한 행에 결측이 몇 개인지를 하나의 열로 요약한다."""
    out = df.copy()
    cols = cols or (NUM_COLS + CAT_COLS)
    out['n_missing'] = source_df[cols].isnull().sum(axis=1).astype(np.int8)
    return out


def impute_median(df, medians=None, modes=None):
    """
    수치형은 중앙값, 범주형은 최빈값으로 결측을 채운다.

    ⚠️ 데이터 누수(data leakage) 주의
    중앙값은 반드시 '학습 데이터에서만' 계산해서 검증/테스트에 적용해야 한다.
    전체 데이터로 중앙값을 구하면 검증 세트의 정보가 학습에 새어 들어가
    CV 점수가 실제보다 좋게 나온다(=리더보드에서 배신당한다).

    medians/modes 를 넘기지 않으면 df 자체에서 계산한다(학습용).
    넘기면 그 값을 그대로 적용한다(검증/테스트용).
    """
    out = df.copy()
    if medians is None:
        medians = {c: out[c].median() for c in NUM_COLS if c in out.columns}
    if modes is None:
        modes = {c: out[c].mode()[0] for c in CAT_COLS if c in out.columns}

    for c, v in medians.items():
        if c in out.columns:
            out[c] = out[c].fillna(v)
    for c, v in modes.items():
        if c in out.columns:
            # category dtype은 없는 값을 못 채우므로 카테고리에 먼저 추가
            if isinstance(out[c].dtype, pd.CategoricalDtype) and v not in out[c].cat.categories:
                out[c] = out[c].cat.add_categories([v])
            out[c] = out[c].fillna(v)
    return out, medians, modes


def build_features(df, fe=True):
    """
    파생변수 생성 (04단계에서 사용).

    fe=False 면 원본 열만 반환 → 베이스라인과 공정하게 비교할 수 있다.

    설계 근거 (01_eda):
      - daily_screen_time 과 weekend_screen_time 의 상관이 0.80으로 매우 높다.
        → 두 값의 '비율'과 '차이'는 원본에 없는 새 정보다.
          (평일 대비 주말에 얼마나 더 쓰는가 = 사용 패턴의 성격)
      - 스크린타임 안에서 SNS/게임/공부가 차지하는 '비중'은
        절대 시간과 다른 정보를 담는다.
    """
    X = df.copy()
    if not fe:
        return X

    eps = 1e-6  # 0으로 나누는 것을 막는 아주 작은 수

    d = X['daily_screen_time_hours']
    w = X['weekend_screen_time']
    s = X['social_media_hours']
    g = X['gaming_hours']
    k = X['work_study_hours']
    sl = X['sleep_hours']

    # 1) 주말 대 평일 사용 패턴
    X['weekend_ratio'] = w / (d + eps)          # 주말에 몇 배 더 쓰는가
    X['weekend_diff'] = w - d                   # 절대 증가량

    # 2) 스크린타임 구성 비율 — 같은 5시간도 '무엇에 썼는가'가 다르다
    X['social_ratio'] = s / (d + eps)
    X['gaming_ratio'] = g / (d + eps)
    X['study_ratio'] = k / (d + eps)

    # 3) 여가성 사용 vs 생산적 사용
    X['leisure_hours'] = s + g                  # SNS + 게임
    X['leisure_vs_study'] = (s + g) / (k + eps)

    # 4) 하루 시간 예산 — 잠 + 스크린이 24시간을 얼마나 잡아먹는가
    X['awake_hours'] = 24 - sl
    X['screen_per_awake'] = d / (24 - sl + eps)  # 깨어있는 시간 중 화면 비중
    X['sleep_screen_sum'] = sl + d

    # 5) 사용 강도 — 한 번 열 때마다 얼마나 오래 보는가
    X['mins_per_open'] = d * 60 / (X['app_opens_per_day'] + eps)
    X['opens_per_notif'] = X['app_opens_per_day'] / (X['notifications_per_day'] + eps)

    return X


def feature_list(fe=True, with_indicators=False):
    """모델에 넣을 열 이름 목록을 반환한다."""
    cols = NUM_COLS + CAT_COLS
    if fe:
        cols = cols + [
            'weekend_ratio', 'weekend_diff',
            'social_ratio', 'gaming_ratio', 'study_ratio',
            'leisure_hours', 'leisure_vs_study',
            'awake_hours', 'screen_per_awake', 'sleep_screen_sum',
            'mins_per_open', 'opens_per_notif',
        ]
    if with_indicators:
        cols = cols + [f'{c}_isna' for c in NUM_COLS + CAT_COLS]
    return cols
