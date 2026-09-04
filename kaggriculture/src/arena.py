# -*- coding: utf-8 -*-
"""
자체 대전 평가 틀 (Arena) — 이 대회의 '교차검증'에 해당한다.

## 왜 이걸 먼저 만드는가
S6E8에서 배운 것: **믿을 수 있는 검증 틀이 없으면 어떤 실험도 의미가 없다.**
거기서는 StratifiedKFold(5, seed=42)를 고정해 모든 실험이 같은 자를 쓰게 했다.
여기서도 같다. 에이전트 A가 B보다 나은지 판단하려면:

  - **같은 시드 집합**으로 대전해야 한다 (게임마다 상점 해금·잡초가 무작위)
  - **선후공을 바꿔가며** 해야 한다 (플레이어 0/1이 비대칭일 수 있다)
  - **충분한 판수**가 필요하다 (1판은 측정이 아니다)

한 판만 이겼다고 좋아하는 것은, CV 한 폴드만 보고 판단하는 것과 같은 실수다.
"""
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from kaggle_environments import make

SEED_BASE = 42


def play(agent_a, agent_b, seed, episode_steps=720, board_size=10):
    """
    한 판 대전. A가 player 0.

    반환: dict(a_money, b_money, winner, steps)
    """
    cfg = {'episodeSteps': episode_steps, 'boardSize': board_size, 'seed': seed}
    env = make('kaggriculture', configuration=cfg, debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    a = final[0].observation['farms'][0]['money']
    b = final[0].observation['farms'][1]['money']
    return {'seed': seed, 'a_money': a, 'b_money': b,
            'winner': 0 if a > b else (1 if b > a else -1),
            'steps': len(env.steps)}


def _resolve(agent):
    """
    문자열이면 그대로(내장 에이전트 이름 또는 .py 경로), 함수면 그대로 쓴다.

    ⚠️ Windows 주의: 멀티프로세스는 spawn 방식이라 **함수 객체를 넘길 수 없다**.
    병렬로 돌리려면 에이전트를 파일 경로 문자열로 넘겨야 한다.
    (kaggle_environments는 .py 경로를 직접 받아준다)
    """
    return agent


def _job(args):
    a, b, seed, swap, steps, size = args
    a, b = _resolve(a), _resolve(b)
    try:
        r = play(b, a, seed, steps, size) if swap else play(a, b, seed, steps, size)
        if swap:                       # 결과를 A 기준으로 뒤집는다
            r = {'seed': seed, 'a_money': r['b_money'], 'b_money': r['a_money'],
                 'winner': (-1 if r['winner'] == -1 else 1 - r['winner']),
                 'steps': r['steps']}
        r['swap'] = swap
        return r
    except Exception as e:
        return {'seed': seed, 'swap': swap, 'error': f'{type(e).__name__}: {e}'}


def match(agent_a, agent_b, n_games=20, workers=None, episode_steps=720,
          board_size=10, seed_base=SEED_BASE, verbose=True, name_a='A', name_b='B'):
    """
    A vs B 를 n_games 판 붙인다. 절반은 선후공을 바꾼다.

    n_games는 짝수를 권장한다 (선공/후공 균등).
    """
    half = n_games // 2
    jobs = ([(agent_a, agent_b, seed_base + i, False, episode_steps, board_size)
             for i in range(half)] +
            [(agent_a, agent_b, seed_base + i, True, episode_steps, board_size)
             for i in range(half)])

    # 함수 객체는 Windows 멀티프로세스로 못 넘긴다 -> 자동으로 순차 실행
    if workers is None:
        workers = 6 if all(isinstance(x, str) for x in (agent_a, agent_b)) else 1

    t0 = time.time()
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]

    ok = [r for r in results if 'error' not in r]
    errs = [r for r in results if 'error' in r]

    wins = sum(1 for r in ok if r['winner'] == 0)
    draws = sum(1 for r in ok if r['winner'] == -1)
    losses = len(ok) - wins - draws
    a_scores = [r['a_money'] for r in ok]
    b_scores = [r['b_money'] for r in ok]
    margins = [r['a_money'] - r['b_money'] for r in ok]

    out = {
        'name_a': name_a, 'name_b': name_b,
        'games': len(ok), 'errors': len(errs),
        'win_rate': wins / len(ok) if ok else 0.0,
        'wins': wins, 'draws': draws, 'losses': losses,
        'a_mean': statistics.mean(a_scores) if a_scores else 0,
        'a_median': statistics.median(a_scores) if a_scores else 0,
        'a_std': statistics.pstdev(a_scores) if len(a_scores) > 1 else 0,
        'b_mean': statistics.mean(b_scores) if b_scores else 0,
        'margin_mean': statistics.mean(margins) if margins else 0,
        'elapsed': round(time.time() - t0, 1),
        'results': ok, 'errors_detail': errs[:3],
    }

    if verbose:
        print(f'[{name_a}] vs [{name_b}]  {len(ok)}판 ({out["elapsed"]}초)')
        print(f'  승률 {out["win_rate"]:.1%}  ({wins}승 {draws}무 {losses}패)')
        print(f'  내 잔고  평균 {out["a_mean"]:,.0f}  중앙값 {out["a_median"]:,.0f}  '
              f'표준편차 {out["a_std"]:,.0f}')
        print(f'  상대 잔고 평균 {out["b_mean"]:,.0f}   |  평균 격차 {out["margin_mean"]:+,.0f}')
        if errs:
            print(f'  ⚠️ 오류 {len(errs)}건: {errs[0].get("error")}')
    return out


def solo(agent, n_games=10, workers=None, opponent='random', **kw):
    """
    상대를 고정('random' 등)하고 절대 점수만 본다.

    승률보다 '얼마를 버는가'가 중요할 때 쓴다. 개발 초기에는 이쪽이 신호가 강하다.
    (상대가 약하면 승률은 금방 100%가 되어 더 이상 구분이 안 된다)
    """
    return match(agent, opponent, n_games=n_games, workers=workers,
                 name_a='agent', name_b=opponent, **kw)
