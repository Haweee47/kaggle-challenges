# -*- coding: utf-8 -*-
"""
v4 — 관측 기반 시장 컨트롤러

## 왜 시장인가
메타 분석: "상위 40개 팀의 필드 행동은 99~100% 동일하다. 우위는 판매 타이밍에서 온다."
필드 운영은 이미 모두가 최적에 수렴했으므로, 승부처는 **언제 파느냐**다.

## v3까지의 판매 방식과 무엇이 다른가
v3: `마을 하루 흡수량 x sell_frac` 이라는 **고정 규칙**
    -> 상대가 시장에 물량을 풀어 가격이 무너져도 똑같이 판다.

v4: 관측의 `market.inventory`로 **다음 한 단위의 실제 단가를 계산**하고,
    기준가의 일정 비율 밑으로 떨어지기 직전까지만 판다.
    -> 가격이 낮으면 자동으로 적게 팔고, 회복되면 많이 판다.

가격 함수는 게임 엔진과 261/261 완전 일치를 검증했다 (src/market_brain.py).
즉 **우리는 파는 순간의 가격을 정확히 안다.**

## 추가: 종반 처분
미판매 재고는 0원이다. end_rush_day부터는 바닥가라도 전부 판다.
"""
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS

try:                                   # 로컬 실험용
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src'))
    import market_brain as MB
except Exception:                      # 제출 시에는 아래 인라인 사본을 쓴다
    MB = None

# ── 튜닝 파라미터 (5단계에서 Optuna로 최적화 예정) ──
P = {
    'max_hands': 10,          # 하루 고용 인원 (fib 비용이 싼 구간)
    'hire_slots': 9,          # 하루 첫 턴에 고용에 쓸 시장 주문 슬롯 수
    'target_cows': 8,
    'target_sheep': 6,
    'target_straw': 20,       # 딸기 그루 수
    'wheat_tiles': 20,        # 사료 + 무한 배출구
    'buy_ne_day': 5,          # NE 구매 목표일 (수입이 돌기 시작한 뒤)
    'buy_sw_day': 10,
    'buy_se_day': 18,
    'land_cash': 2600,        # 이 이상 남을 때만 땅을 산다
    'reserve': 30,            # 항상 남겨둘 현금 (고용이 최우선이라 낮게)
    'load_per_unit': 2.6,     # 유닛 1명이 하루에 감당하는 타일 수 (이동 포함)
    'sell_frac': 0.55,        # 마을 하루 흡수량 대비 판매 비율
    'shed_soft_cap': 78,      # 이 이상 쌓이면 강제 매도
    'feed_carry': 6,          # 유닛이 한 번에 집어오는 밀 개수
    # ── 구역 배정 (zoning) ──
    # 측정 결과 행동의 74%가 '이동'이었고 실제 작업은 22%뿐이었다.
    # 모든 유닛이 매 턴 '전역에서 가장 가까운 일'을 새로 고르면
    # 서로 목표를 뺏고 왕복하며 걷기만 한다(thrashing).
    # 유닛마다 담당 구역을 고정하면 걷는 거리가 줄고 배정도 안정된다.
    # 실측 결과 오히려 손해였다(26,562 -> 14,698). 인덱스 나머지로 나눈 '구역'이
    # 실제로는 흩어진 타일 집합이라 이동이 더 늘었다. 기본은 끈다.
    # 제대로 하려면 **연속된 블록**으로 나눠야 한다 (미구현).
    # ── 멜론 (v16rc5 테이프 분석에서 도입) ──
    # 나는 '마을 흡수량 30개뿐'이라며 배제했으나, 상위권은 20그루를 심어
    # 112개를 팔았다. 타일당 수익 1위($109/타일일)라 **초반 자본 형성**에 쓴다.
    # 수요가 적어도 d10/d20처럼 몰아서 팔면 그 시점엔 상점이 늘어 가격이 회복돼 있다.
    # ⚠️ 실측 결과 멜론은 **우리 구조에서는 손해**였다 (0그루 44,282 vs 20그루 34,097).
    #    v16rc5는 d10/d20에 몰아서 파는 정밀한 타이밍이 있어 이득을 보지만,
    #    우리는 마을 흡수량 기준으로 조금씩 파는 구조라 13일간 타일만 묶인다.
    #    같은 재료도 구조가 다르면 결과가 반대다. 기본값 0 유지.
    'target_melon': 0,        # 멜론 타일 수 (0이면 미사용)
    'melon_last_day': 15,     # 이 날 이후엔 심지 않는다 (10~12일 걸려 못 여문다)
    # ── 비료 (v16rc5는 296회 수집, d24에 22개 판매) ──
    # 동물이 매일 1개씩 만든다. 마을 수요는 0이지만 시장 가격 자체가 base $100이고
    # 곡선이 완만해(linear 0.4) 98개까지 80% 가격을 유지한다. 공짜 부수입이다.
    # ⚠️ 실측: 우선순위를 올리거나 판매량을 늘리면 오히려 손해였다
    #    (기준 44,282 vs 우선3/4개 38,486 vs 우선2/8개 37,529).
    #    비료를 줍는 행동이 물주기·먹이주기 턴을 잡아먹는다. 기본값 유지.
    # ── 관측 기반 판매 (v4) ──
    # ⚠️ 실측 기각. floor_ratio를 0.70~0.02까지 훑어도 고정 규칙을 못 이겼다.
    #    vs v16rc5:  0.70->15,650  0.45->20,161  0.25->20,487
    #                0.10->20,953  0.02->21,497  고정규칙->22,204
    #    원인 추정: 가격이 낮을 때 재고를 쥐고 기다리는데, 상대가 계속 물량을
    #    풀어 가격이 안 오른다. 그 사이 창고 100개 상한에 걸려 폐기된다.
    #    "기다리면 오른다"는 전제가 **경쟁 상황에서는 성립하지 않는다.**
    #    -> 진짜 답은 '기다리기'가 아니라 '상대보다 먼저 팔기'(front-run)다.
    'smart_sell': 0,          # 1이면 고정 규칙 대신 실제 가격을 보고 판다
    'floor_ratio': 0.70,      # 기준가의 이 비율 밑으로 떨어지면 그만 판다
    'end_rush_day': 27,       # 이 날부터는 바닥가라도 전부 처분 (재고는 0원)
    'fert_priority': 6,       # COLLECT_FERTILIZER 우선순위 (낮을수록 먼저)
    'fert_sell_cap': 1,       # 한 번에 파는 비료 개수
    # ── 시기별 고용 (v16rc5는 5명 -> 14명으로 조절) ──
    # 우리는 매일 같은 수를 고용한다. 수확이 몰리는 중후반에 증원하면
    # 초반 자본을 아끼면서 후반 처리량을 늘릴 수 있다.
    # 실측 채택: 기준 44,282 -> 45,665 (튜닝시드), 40,175 -> 41,732 (새 시드).
    # 초반 3명/4명은 오히려 크게 손해였다(31,785 / 26,218). 5명이 분기점.
    'hire_ramp': 1,           # 1이면 시기별 조절 사용
    'hire_early': 5,          # 초반(=ramp_day 이전) 고용 수
    'hire_ramp_day': 10,      # 이 날부터 max_hands 만큼 고용
    'zoning': 0,              # 1이면 구역 배정 사용
    'zone_slack': 2,          # 자기 구역에 일이 없을 때 전역에서 찾을 우선순위 여유
    # ── 단계적 개시 (staged opening) ──
    # 0일차에 소를 몰아 사면 자본이 통째로 묶이고 14일간 수입이 0이 된다.
    # 밀은 씨앗 $10에 2일이면 수확되므로 **초반 현금 엔진**으로 쓴다.
    'wheat_first_days': 3,    # 이 날까지는 밀만 심는다 (2일 만에 도는 현금 엔진)
    'animal_start_day': 1,    # 동물 구매 시작일
    'animal_per_day': 3,      # 하루 최대 동물 구매 수 (자본 잠김 방지)
    'straw_start_day': 3,     # 딸기 구매 시작일
    'work_capital': 250,      # 확장 전에 남겨둘 운전자금
    'work_capital_growth': 8,
}

# 마을이 하루에 흡수하는 기대량 (상점 8개 기준, docs/STRATEGY.md 표)
TOWN_DAILY = {'WHEAT': 31, 'CARROT': 19, 'TOMATO': 13, 'STRAWBERRY': 25,
              'MELON': 1, 'EGG': 13, 'MILK': 19, 'WOOL': 13, 'FERTILIZER': 0}
SELLABLE = ['MILK', 'WOOL', 'STRAWBERRY', 'EGG', 'WHEAT', 'FERTILIZER',
            'TOMATO', 'CARROT', 'MELON']
DIRS = {(0, -1): 'NORTH', (0, 1): 'SOUTH', (1, 0): 'EAST', (-1, 0): 'WEST'}


def _shed_tiles(n):
    h = n // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def _quad(x, y, n):
    h = n // 2
    return ('N' if y < h else 'S') + ('W' if x < h else 'E')


def _step(fx, fy, tx, ty):
    """목표를 향해 한 칸. 큰 축부터 줄인다."""
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy) and dx:
        return 'EAST' if dx > 0 else 'WEST'
    if dy:
        return 'SOUTH' if dy > 0 else 'NORTH'
    if dx:
        return 'EAST' if dx > 0 else 'WEST'
    return None


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def configure(**kw):
    """
    파라미터 덮어쓰기 (자동 탐색용).

    손으로 12개 파라미터를 더듬으면 사실상 무작위 걷기가 된다.
    S6E8에서 Optuna를 쓴 것과 같은 이유로, 여기서도 탐색은 기계에 맡긴다.
    """
    unknown = set(kw) - set(P)
    if unknown:
        raise KeyError(f'모르는 파라미터: {unknown}')
    P.update(kw)
    return dict(P)


def _expected_animals_before(day, P):
    """day 이전까지 계획상 사들였어야 할 동물 수 (하루 상한 누적)."""
    d = max(0, day - P['animal_start_day'])
    return d * P['animal_per_day']


def act(obs):
    farms = obs.get('farms') or []
    me = obs.get('player', 0)
    if not farms or me >= len(farms):
        return {'farmer': ['PASS'], 'hands': [], 'market': []}

    farm = farms[me]
    priv = obs.get('private') or {}
    tiles = farm['tiles']
    n = len(tiles)
    day = obs.get('day', 0)
    hour = obs.get('hour', 0)
    money = farm.get('money', 0)
    shed = dict(priv.get('shed') or {})
    seeds = dict(priv.get('seeds') or {})
    invs = priv.get('inventories') or [{}]
    prices = (obs.get('market') or {}).get('prices') or {}
    unlocked = set(farm.get('unlocked_quadrants') or ['NW'])

    units = [tuple(farm['farmer'])] + [tuple(h) for h in (farm.get('hands') or [])]
    n_units = len(units)
    shed_set = set(_shed_tiles(n))

    # ── 농장 현황 집계 ──
    cows = sheep = geese = straw = wheat_t = melon_t = 0
    empty_coop = empty_past = 0
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if not isinstance(t, dict):
                continue
            k = t.get('kind')
            if k == 'PLANT':
                if t['crop'] == 'STRAWBERRY':
                    straw += 1
                elif t['crop'] == 'WHEAT':
                    wheat_t += 1
                elif t['crop'] == 'MELON':
                    melon_t += 1
            elif k == 'COOP':
                if t.get('animal'):
                    geese += 1
                else:
                    empty_coop += 1
            elif k == 'PASTURE':
                a = t.get('animal')
                if a == 'COW':
                    cows += 1
                elif a == 'SHEEP':
                    sheep += 1
                elif not a:
                    empty_past += 1

    shed_total = sum(v for k, v in shed.items() if k in SELLABLE or k in ANIMALS)

    # ── 유지 능력 (capacity) ──
    # 1턴 1행동이므로 하루에 쓸 수 있는 행동은 (유닛 수 x 24)다.
    # 식물은 물주기+수확, 동물은 먹이+수확+관리가 필요하고 이동까지 든다.
    # 감당 못 할 만큼 지으면 잡초가 되고 동물은 굶어 죽는다 — v1 초기판의 실패 원인.
    # 그래서 '목표'를 고정값이 아니라 **현재 인력에 비례**해 정한다.
    planned_units = 1 + min(P['max_hands'], P['hire_slots'])
    capacity = int(planned_units * P['load_per_unit'])
    cap_animals = max(2, min(P['target_cows'] + P['target_sheep'], capacity // 2))
    cap_straw = max(0, min(P['target_straw'], capacity - cap_animals))

    # ═══════════ 시장 주문 ═══════════
    market = []
    cash = money

    def afford(c):
        nonlocal cash
        if cash - c >= P['reserve']:
            cash -= c
            return True
        return False

    # 1) 하루 시작에 일꾼 고용 (가장 싼 투자)
    if hour == 0:
        # 시장 주문은 턴당 10개까지. 고용은 하루 첫 턴에 몰아서 하되,
        # 남은 슬롯을 판매/구매에 쓰려면 전부 고용에 쓸 수는 없다.
        # -> hire_slots 파라미터로 조절 (이전에는 9로 하드코딩되어 있었다)
        target = P['max_hands']
        if P['hire_ramp'] and day < P['hire_ramp_day']:
            target = P['hire_early']
        hires = min(target, P['hire_slots'])
        fibs, a, b = [], 1, 1
        for _ in range(hires):
            fibs.append(a)
            a, b = b, a + b
        for c in fibs:
            if len(market) >= P['hire_slots']:
                break
            if afford(c):
                market.append(['HIRE'])

    # 2) 땅 확장 — 순서 NE, SW, SE
    if hour <= 2:
        plan = [('NE', P['buy_ne_day'], 1000), ('SW', P['buy_sw_day'], 2000),
                ('SE', P['buy_se_day'], 4000)]
        for q, d0, cost in plan:
            if q not in unlocked and day >= d0 and money >= cost + P['land_cash']                     and afford(cost):
                market.append(['BUY_LAND'])
                break

    # 3) 사료 확보가 동물 구매보다 우선.
    #    동물은 이틀 굶으면 도망가고 투자금이 통째로 사라진다.
    #    소 4마리를 사고 사료값이 없어 6일차에 전멸한 것이 v1 초기판의 실패였다.
    n_animals = cows + sheep + geese
    pending_a = shed.get('COW', 0) + shed.get('SHEEP', 0)
    # ⚠️ 매수선과 매도선이 겹치면 같은 밀을 사고팔기를 반복하며 손실이 누적된다.
    #    (매수는 매수후 재고, 매도는 매도전 재고로 호가되므로 왕복하면 손해)
    #    그래서 매수선(3배)과 매도선(7배) 사이에 넓은 완충 구간을 둔다.
    feed_floor = (n_animals + pending_a) * 3
    if len(market) < 9 and feed_floor > shed.get('WHEAT', 0):
        need = feed_floor - shed.get('WHEAT', 0)
        if afford(prices.get('WHEAT', 25) * need):
            market.append(['BUY_PRODUCT', 'WHEAT', need])

    # 4) 동물 구매 — 하루 몇 마리까지만, 그리고 운전자금이 남을 때만.
    #    매 턴 살 수 있으면 사버리면 0일차에 소 7마리를 사고 자본이 전부 묶인다.
    #    (실측: 0일차 지출 $3,146 -> 이후 14일간 수입 0)
    #    턴마다 1건씩 주문하므로 '하루 n마리'는 그날 이미 산 수로 근사한다.
    bought_today = pending_a + cows + sheep - _expected_animals_before(day, P)
    work_cap = P['work_capital'] + P['work_capital_growth'] * day
    if len(market) < 9 and day >= P['animal_start_day'] and bought_today < P['animal_per_day']:
        have_a = cows + sheep + pending_a
        room = cap_animals - have_a
        feed_buffer = prices.get('WHEAT', 25) * 8
        want_cow = min(P['target_cows'] - cows - shed.get('COW', 0), room)
        want_sheep = min(P['target_sheep'] - sheep - shed.get('SHEEP', 0), room)
        need_cash = ANIMALS['COW']['cost'] + feed_buffer + work_cap
        if want_cow > 0 and money >= need_cash and afford(ANIMALS['COW']['cost'] + feed_buffer):
            market.append(['BUY_ANIMAL', 'COW', 1])
        elif want_sheep > 0 and money >= ANIMALS['SHEEP']['cost'] + feed_buffer + work_cap                 and afford(ANIMALS['SHEEP']['cost'] + feed_buffer):
            market.append(['BUY_ANIMAL', 'SHEEP', 1])

    # 멜론 씨앗 — 초반 자본용이라 딸기보다 먼저 산다.
    # 나는 '마을 흡수량 30개뿐'이라며 배제했으나, 상위권(v16rc5)은 20그루를 심어
    # 112개를 팔았다. 타일당 수익 1위($109/타일일)라 자본 형성 속도가 압도적이다.
    if len(market) < 9 and P['target_melon'] > 0 and day <= P['melon_last_day'] \
            and melon_t + seeds.get('MELON', 0) < P['target_melon'] \
            and afford(CROPS['MELON']['seed'] * 2):
        market.append(['BUY_SEED', 'MELON', 2])

    if len(market) < 9 and seeds.get('WHEAT', 0) < 4 and afford(CROPS['WHEAT']['seed'] * 4):
        market.append(['BUY_SEED', 'WHEAT', 4])
    if len(market) < 9 and straw + seeds.get('STRAWBERRY', 0) < P['target_straw'] \
            and afford(CROPS['STRAWBERRY']['seed'] * 2):
        market.append(['BUY_SEED', 'STRAWBERRY', 2])

    # 4) 판매 — 마을 흡수 속도에 맞춰 조금씩. 창고가 차면 강제 매도.
    urgent = shed_total >= P['shed_soft_cap']
    if P['smart_sell'] and MB is not None:
        # 관측 재고로 실제 단가를 계산해 '얼마까지 팔아도 되는지' 정한다.
        inv_now = (obs.get('market') or {}).get('inventory') or {}
        reserve = {'WHEAT': (n_animals + pending_a) * 7}
        fr = 0.35 if urgent else P['floor_ratio']   # 창고가 차면 기준을 낮춘다
        for o in MB.plan_sales(shed, inv_now, day, hour, reserve=reserve,
                               floor_ratio=fr, end_rush_day=P['end_rush_day'],
                               max_orders=10 - len(market)):
            if len(market) >= 10:
                break
            market.append(o)
    else:
        for prod in SELLABLE:
            if len(market) >= 10:
                break
            have = shed.get(prod, 0)
            if have <= 0:
                continue
            if prod == 'WHEAT':
                have -= (n_animals + pending_a) * 7
                if have <= 0:
                    continue
            cap = (P['fert_sell_cap'] if prod == 'FERTILIZER'
                   else max(1, int(TOWN_DAILY.get(prod, 1) * P['sell_frac'])))
            qty = have if urgent else min(have, cap)
            if qty > 0:
                market.append(['SELL', prod, qty])

    # ═══════════ 유닛 행동 ═══════════
    # 할 일 목록: (우선순위, 좌표, 행동, 인자)
    structs_now = cows + sheep + empty_past
    build_budget = max(0, min(cap_animals,
                              cows + sheep + shed.get('COW', 0)
                              + shed.get('SHEEP', 0)) - structs_now)
    straw_budget = max(0, min(seeds.get('STRAWBERRY', 0), cap_straw - straw))
    wheat_budget = max(0, min(seeds.get('WHEAT', 0), P['wheat_tiles'] - wheat_t))
    melon_budget = (max(0, min(seeds.get('MELON', 0), P['target_melon'] - melon_t))
                    if day <= P['melon_last_day'] else 0)
    if day < P['wheat_first_days']:
        # 개시 단계: 목장·딸기보다 밀이 먼저다 (2일 만에 현금이 돈다)
        build_budget = 0
        straw_budget = 0
        wheat_budget = max(wheat_budget, min(seeds.get('WHEAT', 0), 12 - wheat_t))

    tasks = []
    empty_structs = []      # (x, y, kind) 비어있는 코옵/목장
    empty_tiles = []        # 빈 타일 (나중에 창고 거리순으로 정렬)
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t == 'LOCKED':
                continue
            if isinstance(t, dict):
                k = t.get('kind')
                if k == 'PLANT':
                    if not t.get('watered_today'):
                        tasks.append((0, (x, y), 'WATER', None))
                    if t.get('yield_units', 0) > 0:
                        tasks.append((2, (x, y), 'HARVEST', None))
                elif k == 'WEED':
                    tasks.append((7, (x, y), 'DIG', None))
                elif k in ('COOP', 'PASTURE'):
                    a = t.get('animal')
                    if a:
                        if not t.get('fed_today'):
                            tasks.append((1, (x, y), 'FEED', None))   # 밀 필요
                        if t.get('yield_units', 0) > 0:
                            tasks.append((2, (x, y), 'HARVEST', None))
                        if not t.get('cared_today'):
                            tasks.append((3, (x, y), 'CARE', None))  # v16rc5의 2위 행동
                        if t.get('fertilizer_available'):
                            tasks.append((P['fert_priority'], (x, y), 'COLLECT_FERTILIZER', None))
                    else:
                        # 빈 구조물 — 동물을 '들고 있는' 유닛만 배치할 수 있다.
                        # (PLACE는 인벤토리에서 꺼낸다. 창고에 있으면 PICKUP이 먼저다)
                        empty_structs.append((x, y, k))
            elif t is None:
                empty_tiles.append((x, y))

    # ── 밀집 배치 ──
    # 창고에서 가까운 빈 타일부터 채운다. 이동이 행동의 74%를 먹던 원인이
    # '경작 타일이 창고에서 평균 5칸 떨어져 있던 것'이었다.
    if empty_tiles:
        empty_tiles.sort(key=lambda t: min(_dist(t, sp) for sp in shed_set))
        for (x, y) in empty_tiles:
            if build_budget > 0:
                tasks.append((4, (x, y), 'BUILD_PASTURE', None))
                build_budget -= 1
            elif melon_budget > 0:
                tasks.append((5, (x, y), 'PLANT', 'MELON'))
                melon_budget -= 1
            elif straw_budget > 0:
                tasks.append((5, (x, y), 'PLANT', 'STRAWBERRY'))
                straw_budget -= 1
            elif wheat_budget > 0:
                tasks.append((5, (x, y), 'PLANT', 'WHEAT'))
                wheat_budget -= 1
            else:
                break

    tasks.sort(key=lambda z: z[0])

    # 유닛별 배정
    actions = [None] * n_units
    used = set()
    wheat_needed = sum(1 for t in tasks if t[2] == 'FEED')

    for ui, pos in enumerate(units):
        inv = invs[ui] if ui < len(invs) else {}
        carrying = sum(inv.values()) if isinstance(inv, dict) else 0
        have_wheat = (inv or {}).get('WHEAT', 0)
        on_shed = pos in shed_set

        held_animal = next((a for a in ('COW', 'SHEEP', 'GOOSE')
                            if (inv or {}).get(a, 0) > 0), None)

        # (A) 동물을 들고 있으면 빈 구조물로 가서 배치
        if held_animal and empty_structs:
            want_kind = 'COOP' if held_animal == 'GOOSE' else 'PASTURE'
            cand = [e for e in empty_structs if e[2] == want_kind]
            if cand:
                tx, ty, _ = min(cand, key=lambda e: _dist(pos, (e[0], e[1])))
                if (tx, ty) == pos:
                    empty_structs.remove((tx, ty, want_kind))
                    actions[ui] = ['PLACE', held_animal]
                else:
                    mv = _step(pos[0], pos[1], tx, ty)
                    actions[ui] = [mv] if mv else ['PASS']
                continue

        # (B) 창고에 동물이 있고 빈 구조물이 있으면 가지러 간다
        pend = next((a for a in ('COW', 'SHEEP') if shed.get(a, 0) > 0), None)
        if pend and empty_structs and not held_animal:
            if on_shed:
                shed[pend] -= 1
                actions[ui] = ['PICKUP', pend, 1]
                continue
            tgt = min(shed_set, key=lambda t: _dist(pos, t))
            mv = _step(pos[0], pos[1], tgt[0], tgt[1])
            if mv:
                actions[ui] = [mv]
                continue

        # 창고 옆이고 팔 물건을 들고 있으면 내려놓는다 (SELL은 창고만 본다)
        if on_shed and carrying - have_wheat > 0 and not held_animal:
            actions[ui] = ['DROP']
            continue
        # 사료가 필요한데 없으면 창고에서 밀을 집어온다
        if wheat_needed > 0 and have_wheat == 0 and shed.get('WHEAT', 0) > 0:
            if on_shed:
                actions[ui] = ['PICKUP', 'WHEAT', P['feed_carry']]
                wheat_needed -= P['feed_carry']
                continue
            tgt = min(shed_set, key=lambda s: _dist(pos, s))
            mv = _step(pos[0], pos[1], tgt[0], tgt[1])
            if mv:
                actions[ui] = [mv]
                continue

        # ── 연속 처리 ──
        # 지금 서 있는 타일에 할 일이 있으면 걷지 말고 그것부터 한다.
        # 매 턴 '전역 최근접'을 새로 고르면 목표를 서로 뺏으며 왕복만 한다.
        here = None
        for ti, (pr, tp, op, arg) in enumerate(tasks):
            if ti in used or tp != pos:
                continue
            if op == 'FEED' and have_wheat <= 0:
                continue
            if here is None or pr < here[0]:
                here = (pr, ti, op, arg)
        if here is not None:
            _, ti, op, arg = here
            used.add(ti)
            actions[ui] = [op, arg] if arg else [op]
            continue

        # 가장 가까운 미배정 할 일
        # 구역 배정: 유닛 index로 담당 구역을 고정한다. 관측만 보고 매번 같은
        # 구역이 나오므로 상태를 저장하지 않아도 배정이 안정적이다.
        best = None
        for ti, (pr, tp, op, arg) in enumerate(tasks):
            if ti in used:
                continue
            if op == 'FEED' and have_wheat <= 0:
                continue
            d = _dist(pos, tp)
            pen = 0
            if P['zoning'] and n_units > 1:
                owner = (tp[0] + tp[1] * n) % n_units
                if owner != ui:
                    pen = P['zone_slack']       # 남의 구역은 뒤로 미룬다
            key = (pr + pen, d)
            if best is None or key < best[0]:
                best = (key, ti, tp, op, arg)
        if best is None:
            actions[ui] = ['PASS']
            continue
        _, ti, tp, op, arg = best
        used.add(ti)
        if tp == pos:
            actions[ui] = [op, arg] if arg else [op]
        else:
            mv = _step(pos[0], pos[1], tp[0], tp[1])
            actions[ui] = [mv] if mv else ['PASS']

    return {'farmer': actions[0] if actions else ['PASS'],
            'hands': actions[1:],
            'market': market[:10]}
