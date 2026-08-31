# -*- coding: utf-8 -*-
"""
가중 평균 제출 파일 생성 (v4)

05단계에서 가중 평균은 OOF 과적합 위험 때문에 **채택하지 않았다.**
(가중 평균 CV 0.964951 vs 단순 평균 0.964890 — 차이 0.000061)

이 스크립트는 그 판단이 실제로 옳았는지 **리더보드로 검증하기 위해** 만든다.
- 단순 평균 v3의 LB가 0.96613 이었다.
- 가중 평균 v4의 LB가 v3보다 낮으면 → "OOF 가중치 최적화는 위험하다"가 실증된다.
- 높으면 → 이 데이터에서는 OOF 과적합이 크지 않았다는 뜻이다.

저장된 예측(preds/*.npy)을 재사용하므로 재학습이 필요 없다.

실행:
    python src/make_weighted_submission.py      # 대회 폴더에서
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # s6e8-smartphone/
PREDS = os.path.join(ROOT, 'preds')
SUBS = os.path.join(ROOT, 'submissions')
DATA = os.path.join(ROOT, 'data')

TARGET = 'addicted_label'
MODELS = ['lightgbm', 'xgboost', 'catboost']

# 05단계 완전탐색(0.05 단위)으로 찾은 최적 가중치
WEIGHTS = {'lightgbm': 0.40, 'xgboost': 0.45, 'catboost': 0.15}


def main():
    # ── 1. 저장된 예측 불러오기 ──
    missing = [m for m in MODELS
               if not os.path.exists(os.path.join(PREDS, f'oof_{m}.npy'))]
    if missing:
        sys.exit(f'❌ 예측 파일이 없습니다: {missing}\n'
                 f'   05_ensemble_submit.ipynb 를 먼저 실행하세요.')

    oof = {m: np.load(os.path.join(PREDS, f'oof_{m}.npy')) for m in MODELS}
    test = {m: np.load(os.path.join(PREDS, f'test_{m}.npy')) for m in MODELS}

    y = pd.read_csv(os.path.join(DATA, 'train.csv'))[TARGET].values
    sub = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))

    print('=== 개별 모델 OOF AUC (검증) ===')
    for m in MODELS:
        print(f'  {m:10s} {roc_auc_score(y, oof[m]):.6f}')

    # ── 2. 블렌딩 ──
    w = np.array([WEIGHTS[m] for m in MODELS])
    assert abs(w.sum() - 1.0) < 1e-9, '가중치 합이 1이 아닙니다'

    oof_simple = np.mean([oof[m] for m in MODELS], axis=0)
    oof_weighted = sum(WEIGHTS[m] * oof[m] for m in MODELS)
    test_weighted = sum(WEIGHTS[m] * test[m] for m in MODELS)

    auc_simple = roc_auc_score(y, oof_simple)
    auc_weighted = roc_auc_score(y, oof_weighted)

    print()
    print('=== 블렌딩 비교 (OOF 기준) ===')
    print(f'  단순 평균  {auc_simple:.6f}   ← v3 제출분 (LB 0.96613)')
    print(f'  가중 평균  {auc_weighted:.6f}   ← v4 (이번에 생성)')
    print(f'  차이       {auc_weighted - auc_simple:+.6f}')
    print()
    print('  가중치:', ', '.join(f'{m}={WEIGHTS[m]:.2f}' for m in MODELS))

    # ── 3. 제출 파일 작성 ──
    out = sub.copy()
    out[TARGET] = test_weighted
    path = os.path.join(SUBS, 'submission_v4_weighted.csv')
    out.to_csv(path, index=False)

    # ── 4. 제출 전 점검 ──
    print()
    print('=== 제출 파일 점검 ===')
    checks = {
        '행 개수 일치': len(out) == len(sub),
        '열 이름 일치': list(out.columns) == list(sub.columns),
        'id 순서 일치': (out['id'].values == sub['id'].values).all(),
        '결측 없음': out[TARGET].notna().all(),
        '값이 0~1 범위': out[TARGET].between(0, 1).all(),
    }
    for k, v in checks.items():
        print(f'  {"OK " if v else "NG "}{k}')
    if not all(checks.values()):
        sys.exit('❌ 점검 실패 — 제출하지 마세요')

    print()
    print(f'  행 {len(out):,} / 평균 {out[TARGET].mean():.6f} '
          f'(train 양성비율 {y.mean():.6f})')
    print(f'  범위 [{out[TARGET].min():.6f}, {out[TARGET].max():.6f}]')

    # v3와 얼마나 다른지 — 순위 상관이 1에 가까우면 LB도 거의 같게 나온다
    v3_path = os.path.join(SUBS, 'submission_v3_ensemble.csv')
    if os.path.exists(v3_path):
        v3 = pd.read_csv(v3_path)[TARGET].values
        from scipy.stats import spearmanr
        rho = spearmanr(v3, test_weighted).statistic
        print()
        print(f'=== v3(단순) vs v4(가중) 차이 ===')
        print(f'  순위 상관(Spearman) : {rho:.6f}')
        print(f'  예측값 최대 차이     : {np.abs(v3 - test_weighted).max():.6f}')
        print(f'  예측값 평균 차이     : {np.abs(v3 - test_weighted).mean():.6f}')
        if rho > 0.9999:
            print('  → 순위가 거의 동일하다. LB 차이도 미미할 가능성이 높다.')

    print()
    print(f'✅ 생성 완료: submissions/submission_v4_weighted.csv')
    print()
    print('제출 명령:')
    print('  kaggle competitions submit -c playground-series-s6e8 \\')
    print('    -f s6e8-smartphone/submissions/submission_v4_weighted.csv \\')
    print(f'    -m "Weighted blend .40/.45/.15 | 16 feats | CV {auc_weighted:.6f}"')

    # 요약 저장
    with open(os.path.join(PREDS, 'v4_weighted_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'weights': WEIGHTS,
                   'oof_auc_weighted': float(auc_weighted),
                   'oof_auc_simple': float(auc_simple)}, f,
                  ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
