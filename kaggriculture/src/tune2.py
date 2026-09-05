# -*- coding: utf-8 -*-
"""
2차 파라미터 탐색 — 강한 상대 기준으로 자를 바로잡는다.

## 왜 다시 하는가
1차 탐색은 튜토리얼 봇(v0_melon) 하나만 상대로 했다.
그 결과 평균 29,000 / 승률 100%가 나왔지만 **실제 레이팅은 466**이었다
(1위는 3,000대). 상대가 약하면 "약한 상대를 이기는 법"만 배운다.

메타 분석 노트북이 이 함정을 정확히 경고했었다:
  "starter 상대로 10만~17만을 찍는 봇도 사다리 중위권에 머물 수 있다.
   사다리의 상대는 다른 강한 농장들이기 때문이다."

S6E8로 치면 검증 세트가 너무 쉬워서 CV가 실제 성능을 반영하지 못한 상황이다.
**자(ruler)를 바꿔야 한다.**

## 이번 설계
- 상대를 **여러 명**으로 (약한 봇 + 현재 v2 자신 + 가능하면 공개 상위권)
- 승률뿐 아니라 **절대 잔고**도 함께 본다
  (승률만 보면 "특정 상대 특화"가 또 생긴다)
- 목적함수 = 상대별 승률의 최솟값 위주 → 어떤 상대에게도 지지 않는 쪽으로

실행:
    python src/tune2.py --trials 60 --games 6
"""
import argparse
import json
import os
import statistics
import sys
import time

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'agents'))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from arena import match                      # noqa: E402
import v1_demand                             # noqa: E402

BASE = dict(v1_demand.P)

# 1차 탐색이 찾은 설정 = 현재 제출본(v2). 이걸 스파링 상대로 쓴다.
def _load_v2():
    p = os.path.join(ROOT, 'sims', 'tune_v1.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))['best']['params']
    return {}


V2 = _load_v2()

SPACE = {
    'max_hands':        ('int', 4, 12),
    'target_cows':      ('int', 2, 12),
    'target_sheep':     ('int', 0, 8),
    'target_straw':     ('int', 0, 20),
    'wheat_tiles':      ('int', 2, 20),
    'load_per_unit':    ('float', 1.0, 3.5),
    'sell_frac':        ('float', 0.5, 3.0),
    'wheat_first_days': ('int', 0, 5),
    'animal_start_day': ('int', 0, 5),
    'animal_per_day':   ('int', 1, 5),
    'straw_start_day':  ('int', 0, 14),
    'work_capital':     ('int', 50, 1200),
    'buy_ne_day':       ('int', 1, 16),
    'land_cash':        ('int', 300, 3000),
    'feed_carry':       ('int', 3, 12),
    'shed_soft_cap':    ('int', 45, 95),
}


def _with(params):
    v1_demand.P.clear()
    v1_demand.P.update(BASE)
    v1_demand.P.update(params)
    return v1_demand.act


def make_opponent(params):
    """
    파라미터를 고정한 상대 함수를 만든다.

    ⚠️ v1_demand.P는 모듈 전역이라, 상대와 내가 같은 모듈을 쓰면 서로 덮어쓴다.
    그래서 상대는 호출 시점에 자기 파라미터를 넣고 원래대로 되돌린다.
    """
    frozen = dict(BASE)
    frozen.update(params)

    def opp(obs):
        saved = dict(v1_demand.P)
        v1_demand.P.clear()
        v1_demand.P.update(frozen)
        try:
            return v1_demand.act(obs)
        finally:
            v1_demand.P.clear()
            v1_demand.P.update(saved)
    return opp


def evaluate(params, opponents, n_games, seed_base=42):
    """상대별로 대전하고 (최소 승률, 평균 잔고)를 돌려준다."""
    out = []
    for name, opp in opponents:
        me = _with(params)
        r = match(me, opp, n_games=n_games, workers=1,
                  seed_base=seed_base, verbose=False)
        out.append((name, r['win_rate'], r['a_mean'], r['margin_mean']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--out', default=os.path.join(ROOT, 'sims', 'tune_v2.json'))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from v0_melon import act as weak
    opponents = [('v0_melon', weak), ('v2_self', make_opponent(V2))]
    # 외부 상대(공개 상위권 에이전트)가 있으면 자동으로 추가한다
    ext = os.path.join(ROOT, 'agents', 'external')
    if os.path.isdir(ext):
        for f in sorted(os.listdir(ext)):
            if f.endswith('.py'):
                opponents.append((f[:-3], os.path.join(ext, f)))
    print('상대:', [o[0] for o in opponents], flush=True)

    t0 = time.time()
    base = evaluate(V2, opponents, args.games)
    print('[기준] 현재 제출본 v2')
    for nm, wr, mean, mg in base:
        print(f'   vs {nm:12s} 승률 {wr:5.0%}  평균 {mean:9,.0f}  격차 {mg:+9,.0f}')
    base_score = min(w for _, w, _, _ in base) * 1e6 + statistics.mean(
        m for _, _, _, m in base)
    print(f'   목적값 {base_score:,.0f}\n', flush=True)

    best = {'value': base_score, 'params': V2, 'detail': base}

    def objective(trial):
        p = {k: (trial.suggest_int(k, s[1], s[2]) if s[0] == 'int'
                 else trial.suggest_float(k, s[1], s[2]))
             for k, s in SPACE.items()}
        res = evaluate(p, opponents, args.games)
        # 최소 승률을 크게 가중 → 어떤 상대에게도 안 지는 쪽을 선호.
        # 동점이면 평균 격차로 가른다.
        score = min(w for _, w, _, _ in res) * 1e6 + statistics.mean(
            m for _, _, _, m in res)
        if score > best['value']:
            best.update(value=score, params=p, detail=res)
            wr = ' '.join(f'{nm}:{w:.0%}' for nm, w, _, _ in res)
            print(f'  ★ trial {trial.number:3d}  {wr}  목적값 {score:,.0f}', flush=True)
        elif trial.number % 5 == 0:
            wr = ' '.join(f'{nm}:{w:.0%}' for nm, w, _, _ in res)
            print(f'    trial {trial.number:3d}  {wr}', flush=True)
        json.dump({'best': {k: v for k, v in best.items() if k != 'detail'},
                   'detail': [list(d) for d in best['detail']]},
                  open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        return score

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=args.trials)

    print(f'\n=== 완료 ({time.time()-t0:.0f}초) ===')
    for nm, wr, mean, mg in best['detail']:
        print(f'  vs {nm:12s} 승률 {wr:5.0%}  평균 {mean:9,.0f}  격차 {mg:+9,.0f}')
    print('\n변경된 파라미터:')
    for k, v in sorted(best['params'].items()):
        if V2.get(k, BASE.get(k)) != v:
            print(f'  {k:20s} {V2.get(k, BASE.get(k))}  ->  {v}')
    json.dump({'best': {k: v for k, v in best.items() if k != 'detail'},
               'detail': [list(d) for d in best['detail']]},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n저장: {args.out}')


if __name__ == '__main__':
    main()
