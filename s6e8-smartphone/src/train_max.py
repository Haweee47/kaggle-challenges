# -*- coding: utf-8 -*-
"""
최대 규모 앙상블 학습 — 정직한 CV 최대화 전략

## 왜 이 전략인가
공개 노트북 "Why every S6E8 notebook above 0.97110 overfits"(65표)의 분석:
  - Public LB는 test의 **약 20%(≈59,260행)** 로만 채점된다.
  - Season 6의 7개 에피소드 중 **3개는 Public Top10 중 Private Top10 생존자가 0명**이었다.
  - S6E7에서는 Public 1위가 Private 440위로 추락했다.
=> 리더보드 점수를 보고 미세 조정을 고르는 것은 '검증'이 아니라 '선택 편향'이다.

우리는 반대로 간다:
  - Public LB를 보고 튜닝하지 않는다.
  - **CV를 정직하게 최대화**하고, 최종 제출도 CV 기준으로 고른다.
  - 우리 CV-LB 순위상관은 5개 점에서 1.000이므로 이 방침이 정당하다.

## 다양성 설계
같은 모델의 시드만 바꾸는 것보다 **알고리즘/성장방식이 다른 모델**이 덜 상관된다.
  A: LGBM 저학습률 + 강정규화      D: LGBM ExtraTrees 모드 (분할점을 무작위로)
  B: LGBM 고학습률 + 강한 L1       E: XGBoost (depth-wise 성장)
  C: LGBM GOSS (기울기 기반 샘플링) F: CatBoost (대칭 트리 + 순서형 부스팅)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
ROOT = os.path.dirname(HERE)
from preprocess import load_data, TARGET, CAT_COLS
from features_v2 import build_v2

SEED = 42
PREDS = os.path.join(ROOT, 'preds')
SUBS = os.path.join(ROOT, 'submissions')

L = dict(objective='binary', metric='auc', verbose=-1, n_jobs=-1)

LGB_B = dict(L, learning_rate=0.02, num_leaves=127, min_child_samples=200,
             feature_fraction=0.34, bagging_fraction=0.75, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0, max_bin=1023, n_estimators=8000)
LGB_A = dict(L, learning_rate=0.05, num_leaves=84, min_child_samples=42,
             feature_fraction=0.5325, bagging_fraction=0.9744, bagging_freq=1,
             lambda_l1=4.9056, lambda_l2=0.1886, n_estimators=6000)
LGB_GOSS = dict(L, boosting_type='goss', learning_rate=0.03, num_leaves=96,
                min_child_samples=120, feature_fraction=0.45,
                lambda_l1=1.0, lambda_l2=1.0, max_bin=511, n_estimators=7000)
LGB_XT = dict(L, extra_trees=True, learning_rate=0.03, num_leaves=160,
              min_child_samples=80, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=0.5, lambda_l2=2.0, max_bin=511, n_estimators=8000)

XGB_P = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist',
             enable_categorical=True, learning_rate=0.04, max_depth=8,
             min_child_weight=30, subsample=0.85, colsample_bytree=0.5,
             reg_alpha=1.0, reg_lambda=3.0, n_estimators=5000,
             early_stopping_rounds=150, n_jobs=-1, verbosity=0)


def cv_lgb(X, y, Xt, folds, params, seed, tag):
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt)); t0 = time.time()
    for i, (tr, va) in enumerate(folds, 1):
        m = lgb.LGBMClassifier(**{**params, 'random_state': seed})
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc',
              callbacks=[lgb.early_stopping(120, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / len(folds)
        print(f'  [{tag}] {i}/{len(folds)} {roc_auc_score(y[va], oof[va]):.6f}', flush=True)
    a = roc_auc_score(y, oof)
    print(f'>> {tag} CV={a:.6f} ({time.time()-t0:.0f}s)\n', flush=True)
    return oof, tp, a


def cv_xgb(X, y, Xt, folds, seed, tag):
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt)); t0 = time.time()
    for i, (tr, va) in enumerate(folds, 1):
        m = xgb.XGBClassifier(**{**XGB_P, 'random_state': seed})
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(Xt)[:, 1] / len(folds)
        print(f'  [{tag}] {i}/{len(folds)} {roc_auc_score(y[va], oof[va]):.6f}', flush=True)
    a = roc_auc_score(y, oof)
    print(f'>> {tag} CV={a:.6f} ({time.time()-t0:.0f}s)\n', flush=True)
    return oof, tp, a


def cv_cat(X, y, Xt, folds, seed, tag):
    from catboost import CatBoostClassifier
    def conv(df):
        o = df.copy()
        for c in CAT_COLS:
            o[c] = pd.Series(np.where(o[c].isna(), 'NA', o[c].astype(str)),
                             index=o.index).astype(str)
        return o
    Xc, Xtc = conv(X), conv(Xt)
    ci = [Xc.columns.get_loc(c) for c in CAT_COLS]
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt)); t0 = time.time()
    for i, (tr, va) in enumerate(folds, 1):
        m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC',
                               iterations=3000, learning_rate=0.06, depth=8,
                               l2_leaf_reg=5.0, random_seed=seed, cat_features=ci,
                               early_stopping_rounds=100, verbose=0,
                               thread_count=-1, allow_writing_files=False)
        m.fit(Xc.iloc[tr], y[tr], eval_set=(Xc.iloc[va], y[va]), verbose=0)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        tp += m.predict_proba(Xtc)[:, 1] / len(folds)
        print(f'  [{tag}] {i}/{len(folds)} {roc_auc_score(y[va], oof[va]):.6f}', flush=True)
    a = roc_auc_score(y, oof)
    print(f'>> {tag} CV={a:.6f} ({time.time()-t0:.0f}s)\n', flush=True)
    return oof, tp, a


def hill_climb(oofs, y, names, iters=200, step=0.02):
    """
    탐욕적 가중치 탐색. 매 반복마다 '어느 모델에 가중치를 조금 더 줄까'만 고른다.
    격자 완전탐색보다 모델 수가 많아도 쓸 수 있고, 단순 평균에서 출발하므로
    OOF 과적합이 비교적 덜하다.
    """
    w = np.ones(len(names)) / len(names)
    cur = np.sum([w[i] * oofs[n] for i, n in enumerate(names)], axis=0)
    best = roc_auc_score(y, cur)
    for _ in range(iters):
        gains = []
        for i in range(len(names)):
            w2 = w * (1 - step); w2[i] += step
            p = np.sum([w2[j] * oofs[n] for j, n in enumerate(names)], axis=0)
            gains.append((roc_auc_score(y, p), i, w2))
        g, i, w2 = max(gains)
        if g <= best + 1e-9:
            break
        best, w = g, w2
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--tag', default='v6')
    ap.add_argument('--cat', action='store_true', help='CatBoost 포함 (느림)')
    args = ap.parse_args()

    os.makedirs(PREDS, exist_ok=True); os.makedirs(SUBS, exist_ok=True)
    train, test, sub = load_data(os.path.join(ROOT, 'data') + os.sep)
    y = train[TARGET].values

    t0 = time.time()
    X, Xt, cols = build_v2(train, test, verbose=False)
    print(f'피처 {X.shape[1]}개 ({time.time()-t0:.0f}초)\n', flush=True)

    folds = list(StratifiedKFold(args.folds, shuffle=True,
                                 random_state=SEED).split(X, y))

    # v6 실험에서 확인한 '승리 조합' 5개만 남긴다.
    # ExtraTrees(0.962)와 GOSS(0.9675)는 힐클라임 가중치가 0.003에 그쳐 제외.
    jobs = [
        ('lgbB_42',   'lgb', LGB_B,    42),
        ('xgb_2026',  'xgb', None,     2026),
        ('xgb_42',    'xgb', None,     42),
        ('lgbB_777',  'lgb', LGB_B,    777),
        ('lgbA_2026', 'lgb', LGB_A,    2026),
    ]
    if args.cat:
        jobs.append(('cat_42', 'cat', None, 42))

    oofs, tests, scores = {}, {}, {}
    for tag, kind, params, seed in jobs:
        try:
            if kind == 'lgb':
                o, t, a = cv_lgb(X, y, Xt, folds, params, seed, tag)
            elif kind == 'xgb':
                o, t, a = cv_xgb(X, y, Xt, folds, seed, tag)
            else:
                o, t, a = cv_cat(X, y, Xt, folds, seed, tag)
        except Exception as e:
            print(f'!! {tag} 실패: {type(e).__name__}: {e}\n', flush=True)
            continue
        oofs[tag], tests[tag], scores[tag] = o, t, a
        np.save(os.path.join(PREDS, f'{args.tag}_oof_{tag}.npy'), o)
        np.save(os.path.join(PREDS, f'{args.tag}_test_{tag}.npy'), t)
        # 중간 저장 — 중단되어도 여기까지는 건진다
        with open(os.path.join(PREDS, f'{args.tag}_progress.json'), 'w') as f:
            json.dump({k: float(v) for k, v in scores.items()}, f, indent=2)

    names = list(oofs)
    print('=== 개별 모델 ===')
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f'  {k:12s} {v:.6f}')

    rn = lambda a: rankdata(a) / len(a)
    simple_o = np.mean([oofs[k] for k in names], axis=0)
    simple_t = np.mean([tests[k] for k in names], axis=0)
    rank_o = np.mean([rn(oofs[k]) for k in names], axis=0)
    rank_t = np.mean([rn(tests[k]) for k in names], axis=0)
    a_simple = roc_auc_score(y, simple_o)
    a_rank = roc_auc_score(y, rank_o)

    w, a_hill = hill_climb(oofs, y, names)
    hill_t = np.sum([w[i] * tests[n] for i, n in enumerate(names)], axis=0)

    print(f'\n  단순평균  {a_simple:.6f}')
    print(f'  순위평균  {a_rank:.6f}')
    print(f'  힐클라임  {a_hill:.6f}   가중치 ' +
          ', '.join(f'{n}={w[i]:.3f}' for i, n in enumerate(names)))

    # 선택: 힐클라임이 OOF에 과적합될 수 있으므로 이득이 뚜렷할 때만 채택
    cands = [('단순평균', a_simple, simple_t), ('순위평균', a_rank, rank_t)]
    base_best = max(cands, key=lambda c: c[1])
    if a_hill > base_best[1] + 0.00005:
        chosen, cv, final = '힐클라임', a_hill, hill_t
    else:
        chosen, cv, final = base_best[0], base_best[1], base_best[2]
        print(f'  -> 힐클라임 이득({a_hill-base_best[1]:+.6f})이 작아 {chosen} 채택')

    print(f'\n채택: {chosen} (CV {cv:.6f})')

    out = sub.copy(); out[TARGET] = final
    path = os.path.join(SUBS, f'submission_{args.tag}_max.csv')
    out.to_csv(path, index=False)
    ok = (len(out) == len(sub) and (out['id'].values == sub['id'].values).all()
          and out[TARGET].notna().all() and out[TARGET].between(0, 1).all())
    print(f'제출 점검 {"OK" if ok else "NG"} | 행 {len(out):,} | 평균 {out[TARGET].mean():.6f}')
    print(f'저장: submissions/submission_{args.tag}_max.csv')

    with open(os.path.join(PREDS, f'{args.tag}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'individual': {k: float(v) for k, v in scores.items()},
                   'simple': float(a_simple), 'rank': float(a_rank),
                   'hill': float(a_hill),
                   'hill_weights': {n: float(w[i]) for i, n in enumerate(names)},
                   'chosen': chosen, 'cv': float(cv),
                   'folds': args.folds, 'n_features': int(X.shape[1])},
                  f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
