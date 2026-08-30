# -*- coding: utf-8 -*-
"""
교차검증(CV) 재사용 모듈 — Playground Series S6E8

캐글에서 가장 중요한 건 '믿을 수 있는 검증 틀'이다.
검증 방식이 실험마다 다르면 점수를 비교할 수 없다.
그래서 폴드 분할과 평가 로직을 이 파일 하나로 고정한다.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SPLITS = 5


def get_folds(X, y, n_splits=N_SPLITS, seed=SEED):
    """
    Stratified K-Fold 분할기를 만든다.

    - K-Fold: 데이터를 K조각으로 나눠 K번 학습/검증을 반복하고 평균을 낸다.
      한 번만 나눠서 평가하면 '운 좋게 쉬운 검증셋'을 뽑았을 수 있기 때문이다.
    - Stratified(층화): 각 조각의 타깃 비율(양성 70.9%)을 원본과 같게 유지한다.
      불균형 데이터에서 이걸 안 하면 어떤 폴드는 양성이 60%, 다른 폴드는 80%가 되어
      점수가 들쭉날쭉해진다.
    - shuffle=True + random_state 고정: 매번 같은 방식으로 섞어 재현성을 확보한다.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def run_cv(model_fn, X, y, X_test=None, folds=None, name='model',
           n_splits=N_SPLITS, seed=SEED, verbose=True, fit_params=None):
    """
    교차검증을 돌리고 OOF 예측 / 테스트 예측 / AUC를 반환한다.

    model_fn : 인자 없이 호출하면 '학습 안 된 새 모델'을 돌려주는 함수.
               폴드마다 완전히 새 모델을 써야 하므로 인스턴스가 아니라 함수를 받는다.
    X_test   : 넘기면 폴드별 예측을 평균낸다 (= 5개 모델의 앙상블 효과).

    OOF(Out-Of-Fold) 예측이란?
      각 행을 '그 행이 학습에 쓰이지 않은 폴드의 모델'로 예측한 값.
      전체 학습 데이터에 대해 편향 없는 예측을 얻을 수 있어서,
      모델 비교와 앙상블 가중치 탐색의 기준이 된다.
    """
    folds = folds if folds is not None else get_folds(X, y, n_splits, seed)
    fit_params = fit_params or {}

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    fold_scores, best_iters = [], []
    t0 = time.time()

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = model_fn()
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **fit_params)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, oof[va_idx])
        fold_scores.append(fold_auc)

        bi = getattr(model, 'best_iteration_', None) or getattr(model, 'best_iteration', None)
        if bi:
            best_iters.append(bi)

        if X_test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / len(folds)

        if verbose:
            print(f'  fold {i}/{len(folds)}  AUC={fold_auc:.6f}', flush=True)

    cv_auc = roc_auc_score(y, oof)
    elapsed = time.time() - t0

    if verbose:
        print(f'[{name}] CV AUC = {cv_auc:.6f} '
              f'(폴드 평균 {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}) '
              f'| {elapsed:.1f}s', flush=True)

    return {
        'name': name,
        'cv_auc': cv_auc,
        'fold_scores': fold_scores,
        'fold_mean': float(np.mean(fold_scores)),
        'fold_std': float(np.std(fold_scores)),
        'oof': oof,
        'test_pred': test_pred,
        'best_iters': best_iters,
        'elapsed': elapsed,
    }


class ExperimentLog:
    """
    실험 결과를 표로 누적한다.

    캐글에서 실험을 20번쯤 하다 보면 '어떤 조합이 몇 점이었는지'를 반드시 잊는다.
    기록이 없으면 같은 실험을 두 번 하게 된다.
    """

    def __init__(self, baseline=None):
        self.rows = []
        self.baseline = baseline

    def add(self, name, cv_auc, note='', **kwargs):
        if self.baseline is None:
            self.baseline = cv_auc
        self.rows.append({
            '실험': name,
            'CV AUC': round(cv_auc, 6),
            '베이스라인 대비': round(cv_auc - self.baseline, 6),
            '비고': note,
            **kwargs,
        })
        return self

    def to_frame(self):
        return pd.DataFrame(self.rows).sort_values('CV AUC', ascending=False)

    def show(self):
        df = self.to_frame()
        print(df.to_string(index=False))
        return df
