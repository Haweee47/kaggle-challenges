# -*- coding: utf-8 -*-
"""
시장 컨트롤러 — 우리가 직접 만드는 부분.

## 왜 여기에 집중하는가
메타 분석 노트북의 결론:
  "c18은 c16 대비 필드 행동은 20턴만 바뀌었고 **시장 행동은 112턴이 바뀌었다.**
   현재의 우위가 축종 구성이 아니라 **재고 판매 타이밍**에서 온다는 증거다."

상위 40개 팀의 필드 행동은 99~100% 동일하다. 즉 **농장 운영은 이미 모두가
최적에 수렴했고, 남은 승부처는 언제 파느냐**다. 그래서 필드는 검증된 스케줄을
쓰고, 시장 결정만 우리가 관측 기반으로 새로 만든다.

## 핵심 아이디어 — 가격은 우리가 만든다
시장 가격은 재고의 함수다.
  price(inv) = base ± amp * f(|inv - I0|)
매 턴 관측에서 `market.inventory`를 볼 수 있으므로 **다음 한 단위를 팔면
얼마를 받는지 정확히 계산할 수 있다.** 그러면 판매를 '규칙'이 아니라
**한계수익 비교**로 결정할 수 있다.

  - 지금 팔면 p(현재 재고)
  - 기다리면 마을이 재고를 빼가 가격이 오른다
  - 단, 창고는 100개 상한이고 시즌은 끝난다

그래서 **"남은 시간 안에 팔 수 있는 총량"을 역산해 하루 배분을 정한다.**
"""
import math

I0 = 10000
FLOOR = 1
HINGE_GAIN = 8.0

MP = {
    'WHEAT':      dict(base=25,  T=400, bf='sqrt',   bt=0.8, af='log',    at=0.2),
    'CARROT':     dict(base=35,  T=450, bf='hinge',  bt=1.0, af='sqrt',   at=0.7),
    'TOMATO':     dict(base=60,  T=200, bf='hinge',  bt=0.4, af='sqrt',   at=0.6),
    'STRAWBERRY': dict(base=120, T=100, bf='sqrt',   bt=0.7, af='linear', at=1.6),
    'MELON':      dict(base=250, T=300, bf='log',    bt=0.2, af='sq',     at=3.6),
    'EGG':        dict(base=50,  T=332, bf='hinge',  bt=0.4, af='log',    at=0.2),
    'MILK':       dict(base=160, T=122, bf='sqrt',   bt=0.6, af='linear', at=1.6),
    'WOOL':       dict(base=200, T=105, bf='log',    bt=0.2, af='sq',     at=3.2),
    'FERTILIZER': dict(base=100, T=200, bf='linear', bt=0.4, af='linear', at=0.4),
}


def _f(name, x, T):
    if name == 'linear':
        return x
    if name == 'sq':
        return x * x
    if name == 'sqrt':
        return math.sqrt(x)
    if name == 'log':
        return math.log(1 + x)
    if name == 'hinge':
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price_at(prod, inv):
    """재고가 inv일 때의 단가. 관측의 market.inventory를 그대로 넣으면 된다."""
    p = MP.get(prod)
    if not p:
        return 0
    x = abs(inv - I0)
    if inv < I0:
        func, target, sign = p['bf'], p['bt'], +1
    else:
        func, target, sign = p['af'], p['at'], -1
    fT = _f(func, p['T'], p['T'])
    amp = target * p['base'] / fT if fT else 0.0
    return max(FLOOR, round(p['base'] + sign * amp * _f(func, x, p['T'])))


def marginal_revenue(prod, inv, qty):
    """
    지금 qty개를 연속으로 팔았을 때의 총수익.

    한 단위 팔 때마다 재고가 늘어 가격이 떨어지는 과정을 그대로 재현한다.
    '몇 개까지 파는 게 이득인가'를 이걸로 판단한다.
    """
    total = 0
    cur = inv
    for _ in range(qty):
        pr = price_at(prod, cur)
        total += pr
        if pr > FLOOR:
            cur += 1
    return total


def best_quantity(prod, inv, have, floor_ratio=0.70, max_q=40):
    """
    한계단가가 base*floor_ratio 밑으로 떨어지기 직전까지만 판다.

    규칙 기반('마을 흡수량의 몇 %')과 달리 **실제 재고를 보고** 판단하므로,
    상대가 이미 시장에 물량을 풀어 가격이 낮으면 자동으로 적게 판다.
    이것이 이 컨트롤러의 존재 이유다.
    """
    if have <= 0:
        return 0
    base = MP[prod]['base']
    limit = base * floor_ratio
    cur = inv
    q = 0
    while q < min(have, max_q):
        pr = price_at(prod, cur)
        if pr < limit:
            break
        q += 1
        if pr > FLOOR:
            cur += 1
    return q


def plan_sales(shed, inventory, day, hour, reserve=None, floor_ratio=0.70,
               end_rush_day=27, max_orders=8):
    """
    이번 턴의 SELL 주문 목록을 만든다.

    reserve: 팔지 않고 남길 수량 {품목: 개수} (예: 사료용 밀)
    end_rush_day: 이 날부터는 바닥가라도 전부 처분한다.
                  미판매 재고는 0원이므로 시즌 끝에 남기면 손해다.
    """
    reserve = reserve or {}
    orders = []
    endgame = day >= end_rush_day
    # 값이 비싼 것부터 판다 (창고 상한이 있으므로 회전이 중요)
    for prod in sorted(shed, key=lambda p: -MP.get(p, {}).get('base', 0)):
        if prod not in MP or len(orders) >= max_orders:
            continue
        have = shed.get(prod, 0) - reserve.get(prod, 0)
        if have <= 0:
            continue
        inv = inventory.get(prod, I0)
        if endgame:
            q = min(have, 40)
        else:
            q = best_quantity(prod, inv, have, floor_ratio)
        if q > 0:
            orders.append(['SELL', prod, q])
    return orders
