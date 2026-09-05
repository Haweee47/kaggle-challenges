# -*- coding: utf-8 -*-
"""
파라미터 자동 탐색 (Optuna) — 손으로 더듬는 대신 기계에 맡긴다.

## 왜 필요한가
v1의 파라미터는 20개다. 손으로 하나씩 바꿔보다가 "소를 더 일찍 사자"는
그럴듯한 가설이 오히려 −386을 냈다. 20차원 공간에서 사람의 직관은
사실상 무작위 걷기다. S6E8에서 Optuna로 +0.00057을 얻은 것과 같은 상황.

## 잡음 관리 — S6E8의 교훈을 그대로 적용
자체 대전은 시드마다 결과가 크게 흔들린다(표준편차 약 1,500).
그래서 **모든 시행이 똑같은 시드 집합**을 쓰게 고정한다.
분할이 바뀌면 실험을 비교할 수 없다는 원칙은 여기서도 같다.

실행:
    python src/tune.py --trials 60 --games 8
"""
import argparse
import json
import os
import sys
import time

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'agents'))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from arena import match                      # noqa: E402
import importlib                             # noqa: E402
# 상대를 환경변수로 교체 가능하게 한다.
# 약한 상대로 튜닝하면 '약한 상대를 이기는 법'만 배운다는 것을 466점으로 배웠다.
_OPP = os.environ.get('TUNE_OPP', '')
if _OPP:
    OPPONENT = _OPP            # .py 경로 문자열도 kaggle_environments가 받아준다
else:
    from v0_melon import act as OPPONENT   # noqa: E402

AGENT_MOD = os.environ.get('TUNE_AGENT', 'v1_demand')
v1_demand = importlib.import_module(AGENT_MOD)   # 이름은 유지 (아래 코드 재사용)

# 탐색 공간 — 영향이 클 것으로 보이는 16개
SPACE = {
    'max_hands':        ('int', 4, 14),
    'hire_slots':       ('int', 4, 10),
    'target_cows':      ('int', 3, 14),
    'target_sheep':     ('int', 0, 10),
    'target_straw':     ('int', 0, 30),
    'wheat_tiles':      ('int', 4, 34),
    # ⚠️ 이전 탐색은 상한 4.2였다. 10명 x 4.2 = 42타일이라
    #    상위권 메타(68타일)가 애초에 후보에 없었다. 상자를 넓힌다.
    'load_per_unit':    ('float', 1.2, 9.0),
    'sell_frac':        ('float', 0.3, 2.5),
    'wheat_first_days': ('int', 0, 6),
    'animal_start_day': ('int', 0, 6),
    'animal_per_day':   ('int', 1, 5),
    'straw_start_day':  ('int', 0, 14),
    'work_capital':     ('int', 50, 900),
    'buy_ne_day':       ('int', 0, 14),
    'buy_sw_day':       ('int', 2, 22),
    'land_cash':        ('int', 100, 2500),
    'feed_carry':       ('int', 3, 12),
    'shed_soft_cap':    ('int', 45, 95),
}

BASE = dict(v1_demand.P)


def evaluate(params, n_games, seed_base=42):
    """고정 시드로 대전해 평균 잔고를 돌려준다."""
    v1_demand.P.clear()
    v1_demand.P.update(BASE)
    v1_demand.P.update(params)
    r = match(v1_demand.act, OPPONENT, n_games=n_games, workers=1,
              seed_base=seed_base, verbose=False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(ROOT, 'sims', 'tune_v1.json'))
    ap.add_argument('--warm', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    base = evaluate({}, args.games)
    print(f'[기준] 현재 설정  평균 {base["a_mean"]:,.0f}  승률 {base["win_rate"]:.0%}  '
          f'({base["elapsed"]}초)', flush=True)

    best = {'value': base['a_mean'], 'params': {}, 'win': base['win_rate']}
    history = []

    def objective(trial):
        p = {}
        for k, spec in SPACE.items():
            if spec[0] == 'int':
                p[k] = trial.suggest_int(k, spec[1], spec[2])
            else:
                p[k] = trial.suggest_float(k, spec[1], spec[2])
        r = evaluate(p, args.games)
        history.append({'trial': trial.number, 'mean': r['a_mean'],
                        'win': r['win_rate'], 'params': p})
        if r['a_mean'] > best['value']:
            best.update(value=r['a_mean'], params=p, win=r['win_rate'])
            print(f'  ★ trial {trial.number:3d}  평균 {r["a_mean"]:8,.0f}  '
                  f'승률 {r["win_rate"]:.0%}   (신기록)', flush=True)
        elif trial.number % 5 == 0:
            print(f'    trial {trial.number:3d}  평균 {r["a_mean"]:8,.0f}  '
                  f'승률 {r["win_rate"]:.0%}   최고 {best["value"]:,.0f}', flush=True)
        # 중간 저장 — 중단되어도 여기까지는 건진다
        json.dump({'best': best, 'base_mean': base['a_mean'],
                   'history': history[-200:]},
                  open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        return r['a_mean']

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    # 1차 탐색이 찾은 최적점을 첫 시행으로 넣는다.
    # 상자를 넓혔으므로 처음부터 다시 헤매지 않고 거기서부터 개선하게 된다.
    prev = os.path.join(ROOT, 'sims', 'tune_v1.json')
    if os.path.exists(prev) and args.warm:
        pv = json.load(open(prev, encoding='utf-8'))['best']['params']
        seedp = {k: pv[k] for k in SPACE if k in pv}
        for k, spec in SPACE.items():
            if k not in seedp:
                seedp[k] = BASE.get(k, spec[1])
            lo, hi = spec[1], spec[2]
            seedp[k] = min(max(seedp[k], lo), hi)
            if spec[0] == 'int':
                seedp[k] = int(round(seedp[k]))
        study.enqueue_trial(seedp)
        print('1차 최적점을 시작점으로 투입', flush=True)
    study.optimize(objective, n_trials=args.trials)

    print(f'\n=== 완료 ({time.time()-t0:.0f}초, {args.trials}회) ===')
    print(f'기준  {base["a_mean"]:,.0f} (승률 {base["win_rate"]:.0%})')
    print(f'최고  {best["value"]:,.0f} (승률 {best["win"]:.0%})  '
          f'{best["value"]-base["a_mean"]:+,.0f}')
    print('\n최적 파라미터 (기본값과 다른 것만):')
    for k, v in sorted(best['params'].items()):
        if BASE.get(k) != v:
            print(f'  {k:20s} {BASE.get(k)}  ->  {v}')
    json.dump({'best': best, 'base_mean': base['a_mean'], 'history': history},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n저장: {args.out}')


if __name__ == '__main__':
    main()
