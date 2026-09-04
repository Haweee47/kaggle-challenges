# -*- coding: utf-8 -*-
"""
Kaggriculture 경제 분석 — 에이전트를 짜기 전에 '무엇이 돈이 되는지' 먼저 계산한다.

정형 데이터 대회의 EDA에 해당하는 단계다. 이 게임에서 우리가 최적화할 대상은
"시즌 종료 시점의 잔고"이고, 그건 다음 세 가지의 곱으로 결정된다.

    수익 = (타일 수) x (타일당 생산량) x (판매 단가)
                                            └ 팔수록 떨어진다 ← 여기가 핵심

셋 다 제약이 다르다:
  - 타일  : 25칸에서 시작, $1k/$2k/$4k로 100칸까지 확장
  - 생산  : 행동 횟수가 필요 (턴 = 자원)
  - 단가  : 시장에 물량을 풀면 가격이 붕괴 (품목마다 붕괴 속도가 다름)

이 파일은 세 제약을 각각 수치화한다.
"""
import math
from kaggle_environments.envs.kaggriculture import kaggriculture as K

CROPS = K.CROPS
ANIMALS = K.ANIMALS
MP = K.MARKET_PARAMS
SHOPS = K.SHOPS
I0 = K.MARKET_I0
FLOOR = K.PRICE_FLOOR
HINGE_GAIN = K.HINGE_GAIN
LAND_PRICES = K.LAND_PRICES
HAND_MULT = K.FARM_HAND_COST_MULT

TURNS_PER_DAY = 24
DAYS = 30
START_MONEY = 3000
SHED_CAP = 100


# ─────────────────────────────────────────────────────────────
# 1. 가격 곡선 — 팔면 얼마나 떨어지는가
# ─────────────────────────────────────────────────────────────
def _f(name, x, T):
    """가격 곡선의 shape 함수. x = |inv - I0|"""
    if name == 'linear':
        return x
    if name == 'sq':
        return x * x
    if name == 'sqrt':
        return math.sqrt(x)
    if name == 'log':
        return math.log(1 + x)
    if name == 'log10':
        return math.log10(1 + x)
    if name == 'hinge':
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(name)


def price(product, inv):
    """
    시장 재고가 inv일 때의 가격.

    inv < I0 (품귀) → 가격 상승 / inv > I0 (공급과잉) → 가격 하락
    amp는 'T만큼 움직이면 base의 target배만큼 변한다'가 되도록 유도된 값이다.
    """
    p = MP[product]
    base, T = p['base'], p['T']
    x = abs(inv - p['I0'])
    if inv < p['I0']:
        func, target, sign = p['below_func'], p['below_target'], +1
    else:
        func, target, sign = p['above_func'], p['above_target'], -1
    fT = _f(func, T, T)
    amp = target * base / fT if fT else 0.0
    return max(FLOOR, round(base + sign * amp * _f(func, x, T)))


def sell_revenue(product, n, inv0=I0):
    """
    n개를 연속으로 팔았을 때의 총수익과 평균 단가.

    한 단위씩 팔면서 재고가 늘고 가격이 떨어지는 과정을 그대로 재현한다.
    (실제 환경은 두 플레이어가 번갈아 처리하지만, 단독 판매 기준의 상한선을 본다)
    """
    inv, total = inv0, 0
    for _ in range(n):
        pr = price(product, inv)
        total += pr
        if pr > FLOOR:          # 바닥가에서는 재고에 안 쌓인다
            inv += 1
    return total, (total / n if n else 0)


def price_curve_table(sizes=(1, 10, 25, 50, 100, 200, 300, 500)):
    """물량별 평균 단가 표 — '얼마까지 팔아도 되는가'를 보여준다."""
    rows = []
    for prod in K.PRODUCTS:
        r = {'product': prod, 'base': MP[prod]['base'], 'T': MP[prod]['T']}
        for n in sizes:
            _, avg = sell_revenue(prod, n)
            r[f'n={n}'] = round(avg)
        rows.append(r)
    return rows


def units_until_floor(product, thresh_ratio=0.5):
    """가격이 base의 thresh_ratio 밑으로 떨어질 때까지 몇 개나 팔 수 있는가."""
    base = MP[product]['base']
    inv, n = I0, 0
    while price(product, inv) > base * thresh_ratio and n < 5000:
        inv += 1
        n += 1
    return n


# ─────────────────────────────────────────────────────────────
# 2. 생산 효율 — 타일 하나가 하루에 얼마를 버는가
# ─────────────────────────────────────────────────────────────
def crop_profile(crop, fertilize=False):
    """
    작물 한 그루의 생애: 점유 일수, 총 수확량, 필요 행동 수.

    행동 수가 중요한 이유: 턴은 720개뿐이고 일꾼을 고용해도 하루 24턴씩만 늘어난다.
    '수확량/행동' 이 낮은 작물은 일손을 잡아먹는다.
    """
    c = CROPS[crop]
    if c['ongoing']:
        # 정해진 횟수만큼만 생산하고 시든다
        n_yields = c['max_yield']
        interval = max(1, c['interval'])
        last_day = c['first_yield_day'] + interval * (n_yields - 1)
        units = n_yields * (2 if fertilize else 1)
        days = last_day + 1
        # 행동: 심기1 + 매일 물주기 + 수확 n번 (+ 비료)
        actions = 1 + days + n_yields + (n_yields if fertilize else 0)
    else:
        # 일회성: 최대수확일의 절반부터 물 준 날마다 +1 (비료면 +2)
        bonus_start = math.ceil(c['max_yield_day'] / 2)
        bonus_days = max(0, c['max_yield_day'] - bonus_start + 1)
        units = min(c['max_yield'], 1 + bonus_days * (2 if fertilize else 1))
        days = c['max_yield_day'] + 1
        actions = 1 + days + 1 + (3 if fertilize else 0)
    return {'crop': crop, 'fert': fertilize, 'days': days, 'units': units,
            'seed': c['seed'], 'actions': actions,
            'units_per_tile_day': round(units / days, 3),
            'units_per_action': round(units / actions, 3)}


def animal_profile(animal, days_held=None):
    """동물 한 마리의 생애 생산량. 시즌 끝까지 보유한다고 가정."""
    a = ANIMALS[animal]
    days_held = days_held if days_held is not None else DAYS
    prod_days = max(0, days_held - a['first_yield_day'])
    units = prod_days // a['interval'] if a['interval'] else prod_days
    # 행동: 구조물1 + 배치1 + 매일 먹이 + 수확 + (관리)
    actions = 2 + prod_days + units
    return {'animal': animal, 'product': a['product'], 'cost': a['cost'],
            'first_yield_day': a['first_yield_day'], 'interval': a['interval'],
            'units': units, 'actions': actions,
            'units_per_action': round(units / actions, 3) if actions else 0}


# ─────────────────────────────────────────────────────────────
# 3. 노동 — 일꾼은 얼마나 싼가
# ─────────────────────────────────────────────────────────────
def fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost_table(max_hires=15):
    """
    n번째 일꾼까지 고용하는 누적 비용과, 그때 얻는 행동 수.

    이 게임의 핵심 발견 후보: 일꾼 비용이 피보나치라 초반이 거의 공짜다.
    하루 24행동을 단돈 몇 코인에 살 수 있다면, 병목은 노동이 아니라 땅과 가격이다.
    """
    rows, cum = [], 0
    for n in range(max_hires):
        cost = HAND_MULT * fib(n)
        cum += cost
        rows.append({'n번째': n + 1, '비용': cost, '누적': cum,
                     '총 행동': (n + 2) * TURNS_PER_DAY,
                     '행동당 비용': round(cum / ((n + 1) * TURNS_PER_DAY), 3)})
    return rows
