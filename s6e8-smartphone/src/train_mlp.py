# -*- coding: utf-8 -*-
"""
신경망(MLP) 학습 — 앙상블 다양성 확보용

## 왜 이게 남은 최대 승부수인가
현재 앙상블 5개가 **전부 GBDT**이고 OOF 순위상관이 평균 0.995다.
ExtraTrees, GOSS 같은 변형을 써도 0.990 아래로 안 내려갔다.
앙상블 이득은 '서로 다르게 틀릴 때' 나오므로, 이 상태로는 모델을 더 넣어도 소용없다.

신경망은 **결정 경계의 성질 자체가 다르다.**
  - 트리: 축에 평행한 계단 (daily > 7.5 같은 분할의 조합)
  - MLP : 매끄러운 비선형 곡면 (모든 변수의 가중합을 여러 층 통과)
같은 데이터로 학습해도 **다른 종류의 실수**를 하므로, 성능이 조금 낮아도
블렌딩에서 기여할 수 있다.

## 신경망에 필요한 전처리 (트리에는 필요 없던 것들)
1. **결측 대치** — 신경망은 NaN으로 곱셈을 할 수 없다. 트리처럼 '분기 방향 학습'이 불가능.
2. **스케일 정규화** — 가중치 학습이 변수 스케일에 직접 좌우된다.
   여기서는 QuantileTransformer를 쓴다. 빈도 인코딩 피처가 극단적으로 치우쳐 있어
   (대부분 작고 일부만 9,000 이상) 단순 표준화로는 꼬리에 눌린다.
   분위수 변환은 순위를 정규분포로 펴주므로 이런 분포에 강하다.
3. **범주형 인코딩** — one-hot.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
ROOT = os.path.dirname(HERE)
from preprocess import load_data, TARGET, CAT_COLS
from features_v2 import build_v2

SEED = 42
PREDS = os.path.join(ROOT, 'preds')


def to_numeric(X, Xt):
    """범주형 one-hot + 결측 대치 + 분위수 정규화."""
    n = len(X)
    full = pd.concat([X, Xt], ignore_index=True)
    cats = [c for c in CAT_COLS if c in full.columns]
    full = pd.get_dummies(full, columns=cats, dummy_na=True, dtype=np.float32)
    full = full.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    imp = SimpleImputer(strategy='median')
    arr = imp.fit_transform(full)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                             subsample=200_000, random_state=SEED)
    arr = qt.fit_transform(arr).astype(np.float32)
    return arr[:n], arr[n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--tag', default='mlp')
    ap.add_argument('--hidden', type=int, nargs='*', default=[256, 128])
    ap.add_argument('--max-iter', type=int, default=60)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    # 학습 폴드에서 일부만 뽑아 학습한다. 검증 폴드는 전부 예측하므로
    # OOF 정렬은 그대로 유지되고 GBDT와 정직하게 블렌딩할 수 있다.
    # sklearn MLP는 CPU에서 매우 느려(69만행 1폴드에 54분+) 시간 안에 끝내려면 필수.
    ap.add_argument('--subsample', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(PREDS, exist_ok=True)
    train, test, sub = load_data(os.path.join(ROOT, 'data') + os.sep)
    y = train[TARGET].values

    t0 = time.time()
    X, Xt, _ = build_v2(train, test, verbose=False)
    print(f'피처 {X.shape[1]}개 ({time.time()-t0:.0f}초)', flush=True)

    t0 = time.time()
    A, At = to_numeric(X, Xt)
    print(f'수치 변환 완료 {A.shape} ({time.time()-t0:.0f}초)\n', flush=True)

    # 폴드는 GBDT와 반드시 동일해야 블렌딩이 정직하다
    folds = list(StratifiedKFold(args.folds, shuffle=True,
                                 random_state=SEED).split(A, y))

    oof = np.zeros(len(A)); tp = np.zeros(len(At))
    t_all = time.time()
    rng = np.random.RandomState(SEED)
    for i, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        if args.subsample < 1.0:
            tr = rng.choice(tr, int(len(tr) * args.subsample), replace=False)
        m = MLPClassifier(hidden_layer_sizes=tuple(args.hidden),
                          activation='relu', solver='adam',
                          alpha=args.alpha, batch_size=args.batch,
                          learning_rate_init=1e-3, max_iter=args.max_iter,
                          early_stopping=True, n_iter_no_change=6,
                          validation_fraction=0.08, random_state=args.seed,
                          verbose=False)
        m.fit(A[tr], y[tr])
        oof[va] = m.predict_proba(A[va])[:, 1]
        tp += m.predict_proba(At)[:, 1] / len(folds)
        print(f'  [{args.tag}] {i}/{len(folds)} AUC={roc_auc_score(y[va], oof[va]):.6f} '
              f'(epochs {m.n_iter_}, {time.time()-t0:.0f}s)', flush=True)

    auc = roc_auc_score(y, oof)
    print(f'\n>> {args.tag} CV AUC = {auc:.6f}  ({time.time()-t_all:.0f}s)', flush=True)

    np.save(os.path.join(PREDS, f'v8_oof_{args.tag}.npy'), oof)
    np.save(os.path.join(PREDS, f'v8_test_{args.tag}.npy'), tp)

    # GBDT와 얼마나 다른가 — 이게 이 모델의 존재 이유다
    from scipy.stats import rankdata
    for f in ['v8_oof_xgb_42.npy', 'v8_oof_lgbB_42.npy']:
        p = os.path.join(PREDS, f)
        if os.path.exists(p):
            other = np.load(p)
            r = np.corrcoef(rankdata(oof), rankdata(other))[0, 1]
            print(f'   순위상관 vs {f[8:-4]:12s} {r:.5f}   (GBDT끼리는 0.995)')

    json.dump({'cv': float(auc), 'hidden': args.hidden, 'seed': args.seed},
              open(os.path.join(PREDS, f'{args.tag}_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
