# -*- coding: utf-8 -*-
"""
최종 프로덕션 학습 — features_v2 + 다중 시드 + LGBM/XGB 앙상블

실행:
    python src/train_final.py            # 기본: 10-Fold, LGBM 2시드 + XGB 1시드
    python src/train_final.py --folds 5  # 빠르게

설계 원칙
- **폴드 분할 시드는 42로 고정**하고 모델 시드만 바꾼다.
  분할을 바꾸면 OOF가 어긋나 정직한 블렌딩이 불가능해진다.
- 각 모델의 OOF/테스트 예측을 개별 저장 → 나중에 재학습 없이 조합 실험 가능.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
ROOT = os.path.dirname(HERE)

from preprocess import load_data, TARGET
from features_v2 import build_v2

SEED = 42
PREDS = os.path.join(ROOT, 'preds')
SUBS = os.path.join(ROOT, 'submissions')

# 06단계 실험으로 확정 (53피처 5-Fold 기준)
#   A: Optuna 결과            CV 0.967573 / 270초
#   B: 저학습률 + 강정규화     CV 0.967689 / 581초
# B가 근소하게 높지만 2.2배 느리다. 둘을 **함께** 쓰면 점수와 다양성을 모두 얻는다.
# 시드만 바꾸는 것보다 파라미터 성격이 다른 모델이 덜 상관되어 앙상블 이득이 크다.
LGB_A = dict(objective='binary', metric='auc', verbose=-1, n_jobs=-1,
             learning_rate=0.05, num_leaves=84, min_child_samples=42,
             feature_fraction=0.5325, bagging_fraction=0.9744, bagging_freq=1,
             lambda_l1=4.9056, lambda_l2=0.1886, n_estimators=6000)

LGB_B = dict(objective='binary', metric='auc', verbose=-1, n_jobs=-1,
             learning_rate=0.02, num_leaves=127, min_child_samples=200,
             feature_fraction=0.34, bagging_fraction=0.75, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0, max_bin=1023, n_estimators=8000)

LGB_CONFIGS = [('lgbB_s42', LGB_B, 42), ('lgbA_s2026', LGB_A, 2026)]

XGB_BEST = dict(objective='binary:logistic', eval_metric='auc',
                tree_method='hist', enable_categorical=True,
                learning_rate=0.04, max_depth=8, min_child_weight=30,
                subsample=0.85, colsample_bytree=0.5,
                reg_alpha=1.0, reg_lambda=3.0,
                n_estimators=5000, early_stopping_rounds=150,
                n_jobs=-1, verbosity=0)


def cv_lgb(X, y, X_test, folds, params, seed, tag):
    oof = np.zeros(len(X)); tp = np.zeros(len(X_test)); t0 = time.time()
    for i, (tr, va) in enumerate(folds, 1):
        m = lgb.LGBMClassifier(**{**params, 'random_state': seed})
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / len(folds)
        print(f'  [{tag}] fold{i}/{len(folds)} {roc_auc_score(y[va], oof[va]):.6f} '
              f'(iter {m.best_iteration_})', flush=True)
    auc = roc_auc_score(y, oof)
    print(f'>> {tag} CV AUC = {auc:.6f}  ({time.time()-t0:.0f}s)\n', flush=True)
    return oof, tp, auc


def cv_xgb(X, y, X_test, folds, seed, tag):
    oof = np.zeros(len(X)); tp = np.zeros(len(X_test)); t0 = time.time()
    for i, (tr, va) in enumerate(folds, 1):
        m = xgb.XGBClassifier(**{**XGB_BEST, 'random_state': seed})
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / len(folds)
        print(f'  [{tag}] fold{i}/{len(folds)} {roc_auc_score(y[va], oof[va]):.6f}', flush=True)
    auc = roc_auc_score(y, oof)
    print(f'>> {tag} CV AUC = {auc:.6f}  ({time.time()-t0:.0f}s)\n', flush=True)
    return oof, tp, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--xgb-seeds', type=int, nargs='*', default=[42])
    ap.add_argument('--tag', default='v5')
    args = ap.parse_args()

    os.makedirs(PREDS, exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)

    train, test, sub = load_data(os.path.join(ROOT, 'data') + os.sep)
    y = train[TARGET].values

    t0 = time.time()
    X, X_test, cols = build_v2(train, test)
    print(f'피처 {X.shape[1]}개 ({time.time()-t0:.0f}초)\n', flush=True)

    # 폴드 분할 시드는 절대 바꾸지 않는다 (정직한 블렌딩의 전제)
    folds = list(StratifiedKFold(args.folds, shuffle=True,
                                 random_state=SEED).split(X, y))
    print(f'{args.folds}-Fold / LGBM {[c[0] for c in LGB_CONFIGS]} '
          f'/ XGB 시드 {args.xgb_seeds}\n', flush=True)

    oofs, tests, scores = {}, {}, {}

    for tag, params, seed in LGB_CONFIGS:
        o, t, a = cv_lgb(X, y, X_test, folds, params, seed, tag)
        oofs[tag], tests[tag], scores[tag] = o, t, a
        np.save(os.path.join(PREDS, f'{args.tag}_oof_{tag}.npy'), o)
        np.save(os.path.join(PREDS, f'{args.tag}_test_{tag}.npy'), t)

    for xs in args.xgb_seeds:
        tag = f'xgb{xs}'
        o, t, a = cv_xgb(X, y, X_test, folds, xs, tag)
        oofs[tag], tests[tag], scores[tag] = o, t, a
        np.save(os.path.join(PREDS, f'{args.tag}_oof_{tag}.npy'), o)
        np.save(os.path.join(PREDS, f'{args.tag}_test_{tag}.npy'), t)

    # ── 블렌딩 ──
    print('=== 개별 모델 ===')
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f'  {k:10s} {v:.6f}')

    names = list(oofs)
    simple_oof = np.mean([oofs[k] for k in names], axis=0)
    simple_test = np.mean([tests[k] for k in names], axis=0)
    auc_simple = roc_auc_score(y, simple_oof)

    from scipy.stats import rankdata
    rn = lambda a: rankdata(a) / len(a)
    rank_oof = np.mean([rn(oofs[k]) for k in names], axis=0)
    rank_test = np.mean([rn(tests[k]) for k in names], axis=0)
    auc_rank = roc_auc_score(y, rank_oof)

    print(f'\n  단순평균  {auc_simple:.6f}')
    print(f'  순위평균  {auc_rank:.6f}')

    best_single = max(scores.values())
    if auc_rank >= auc_simple:
        chosen, final_test, final_auc = '순위평균', rank_test, auc_rank
    else:
        chosen, final_test, final_auc = '단순평균', simple_test, auc_simple
    if final_auc < best_single:
        bk = max(scores, key=scores.get)
        chosen, final_test, final_auc = f'단일({bk})', tests[bk], scores[bk]
    print(f'  -> 채택: {chosen} (CV {final_auc:.6f})')

    out = sub.copy()
    out[TARGET] = final_test
    path = os.path.join(SUBS, f'submission_{args.tag}_final.csv')
    out.to_csv(path, index=False)

    ok = (len(out) == len(sub) and (out['id'].values == sub['id'].values).all()
          and out[TARGET].notna().all() and out[TARGET].between(0, 1).all())
    print(f'\n제출 파일 점검: {"OK" if ok else "NG"} | 행 {len(out):,} | '
          f'평균 {out[TARGET].mean():.6f} (train {y.mean():.6f})')
    print(f'저장: submissions/submission_{args.tag}_final.csv')

    with open(os.path.join(PREDS, f'{args.tag}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'individual': {k: float(v) for k, v in scores.items()},
                   'simple': float(auc_simple), 'rank': float(auc_rank),
                   'chosen': chosen, 'cv': float(final_auc),
                   'folds': args.folds, 'n_features': int(X.shape[1])},
                  f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
