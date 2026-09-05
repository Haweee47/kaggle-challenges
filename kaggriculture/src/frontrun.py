# -*- coding: utf-8 -*-
"""
판매 선점 (front-run) — 상대보다 한 턴 먼저 판다.

## 왜 이것인가
v4(관측 기반 '기다렸다 팔기')를 기각하면서 얻은 결론:
  floor_ratio 0.70 -> 0.02 로 낮출수록 점수가 올랐다 (15,650 -> 21,497).
  즉 **기다리는 것은 손해**다. 상대가 계속 물량을 풀어 가격이 회복되지 않고,
  그 사이 창고 100개 상한에 걸려 재고가 폐기된다.

그렇다면 답은 하나다: **상대가 팔기 전에 먼저 판다.**
가격은 매도 *직전* 재고로 호가되므로, 같은 물건이라도 먼저 파는 쪽이 비싸게 받는다.

## 상대의 판매를 어떻게 아는가
우리는 매 턴 `market.inventory`를 본다. 재고 변화는 세 가지로만 설명된다.

    Δ재고 = (내 판매) + (상대 판매) − (마을 소비)

내 판매는 내가 알고, 마을 소비는 상점 수로 추정할 수 있다.
따라서 **나머지가 상대 판매량**이다. 이걸 매 턴 기록하면 상대의 판매 주기가 보인다.

## 한계
- 관측은 턴 종료 후 갱신되므로 한 턴 지연이 있다.
- 마을 소비는 상점 구성이 무작위라 정확히는 모른다(기대값으로 근사).
그래서 정밀 예측이 아니라 **"최근에 팔았으면 곧 또 판다"** 는 단순 신호로 쓴다.
"""

# 품목별 기본가 (판매 우선순위 결정용)
BASE = {'WHEAT': 25, 'CARROT': 35, 'TOMATO': 60, 'STRAWBERRY': 120,
        'MELON': 250, 'EGG': 50, 'MILK': 160, 'WOOL': 200, 'FERTILIZER': 100}
PREMIUM = ('MELON', 'WOOL', 'MILK', 'STRAWBERRY')

# 에이전트는 매 턴 새로 호출되므로 모듈 전역에 기록을 남긴다.
# 키에 player를 넣어 두 자리(선/후공)가 섞이지 않게 한다.
_STATE = {}


def reset(player=0):
    _STATE.pop(player, None)


def observe(obs, my_sales_last_turn=None, player=0):
    """
    이번 턴 관측으로 상대의 판매량을 역산해 기록한다.

    my_sales_last_turn: 지난 턴에 내가 낸 SELL 주문 {품목: 수량}
    반환: {품목: 최근 상대 판매 추정치}
    """
    mkt = (obs.get('market') or {})
    inv = mkt.get('inventory') or {}
    step = obs.get('day', 0) * 24 + obs.get('hour', 0)

    st = _STATE.setdefault(player, {'prev_inv': None, 'prev_step': -1,
                                    'opp': {}, 'hot': {}})
    if st['prev_step'] == step:          # 같은 턴 중복 호출 방지
        return st['hot']

    prev = st['prev_inv']
    my = my_sales_last_turn or {}
    n_shops = len((obs.get('town') or {}).get('unlocked_shops') or [])

    if prev is not None:
        for p, cur in inv.items():
            if p not in BASE:
                continue
            delta = cur - prev.get(p, cur)
            # 마을 소비는 재고를 줄인다. 상점 1개는 4턴마다 1개씩 가져가므로
            # 턴당 기대 소비는 대략 (상점수 x 0.25 x 그 품목을 요구할 확률).
            # 정확할 필요는 없다 — '상대가 팔았는가'의 부호만 보면 된다.
            town = n_shops * 0.12
            opp = delta - my.get(p, 0) + town
            # 지수이동평균으로 최근 경향만 남긴다
            st['opp'][p] = 0.6 * st['opp'].get(p, 0.0) + 0.4 * max(0.0, opp)

    st['prev_inv'] = dict(inv)
    st['prev_step'] = step
    # '뜨거운' 품목 = 상대가 최근 활발히 파는 품목
    st['hot'] = {p: v for p, v in st['opp'].items() if v >= 0.5}
    return st['hot']


def boost(sell_orders, hot, shed, reserve=None, max_extra=25):
    """
    상대가 파는 품목을 **더 많이, 더 먼저** 판다.

    가격은 매도 직전 재고로 호가되므로, 상대가 곧 풀 물량이라면
    지금 파는 편이 항상 낫다. 반대로 상대가 손대지 않는 품목은
    서두를 이유가 없으므로 그대로 둔다.
    """
    reserve = reserve or {}
    if not hot:
        return sell_orders
    have_order = {o[1] for o in sell_orders if o[0] == 'SELL'}
    out = []
    for o in sell_orders:
        if o[0] == 'SELL' and o[1] in hot:
            room = shed.get(o[1], 0) - reserve.get(o[1], 0)
            o = ['SELL', o[1], min(room, o[2] + max_extra)] if room > o[2] else o
        out.append(o)
    # 아직 주문에 없는 뜨거운 품목도 추가한다 (비싼 것부터)
    for p in sorted(hot, key=lambda x: -BASE.get(x, 0)):
        if p in have_order:
            continue
        room = shed.get(p, 0) - reserve.get(p, 0)
        if room > 0:
            out.append(['SELL', p, min(room, max_extra)])
    return out
