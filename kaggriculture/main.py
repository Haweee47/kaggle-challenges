# -*- coding: utf-8 -*-
"""
Kaggriculture 제출 에이전트 v3 — 이동 최소화 + 재튜닝

자체 대전 (vs 튜토리얼 melon_maxxer)
  v2            22,560
  v3 구조개선   35,244
  v3 + 재튜닝   48,966  (튜닝 시드) / 40,621 / 36,179 (새 시드)
  전 시드 승률 100%

vs 상위권 공개본(v16rc5): 14,849 -> 19,255 -> 29,223 (아직 0승, 상대 141k)

## 핵심 변경 (docs/TOP_AGENT_ANALYSIS.md)
상위권 테이프를 분석해 최대 격차가 '이동 비율'임을 확인했다.
  v16rc5  이동 42.8% / 작업 52.4%
  v2      이동 75.4% / 작업 21.7%   <- 3걸음 걸어 1번 일함
  v3      이동 26.3% / 작업 48.8%
1) 밀집 배치 — 창고에서 가까운 빈 타일부터 채운다
2) 연속 처리 — 서 있는 자리에 할 일이 있으면 걷지 않는다
3) CARE 우선순위 상향 (상대의 2위 행동인데 우리는 미실행이었다)

## 재튜닝이 바꾼 것 (구조가 바뀌면 최적 파라미터도 바뀐다)
  load_per_unit  1.53 -> 6.40   이동이 싸지니 농장을 4배 크게
  max_hands         9 -> 13     일손을 더 쓴다
  sell_frac      1.36 -> 2.38   더 공격적으로 판다
  hire_slots        9 -> 6      고용에 주문 슬롯을 덜 쓰고 판매에 돌린다
"""
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS

# ── 튜닝 파라미터 (5단계에서 Optuna로 최적화 예정) ──
P = {
    'max_hands': 13,          # 하루 고용 인원 (fib 비용이 싼 구간)
    'hire_slots': 6,          # 하루 첫 턴에 고용에 쓸 시장 주문 슬롯 수
    'target_cows': 8,
    'target_sheep': 2,
    'target_straw': 2,       # 딸기 그루 수
    'wheat_tiles': 5,        # 사료 + 무한 배출구
    'buy_ne_day': 5,          # NE 구매 목표일 (수입이 돌기 시작한 뒤)
    'buy_sw_day': 13,
    'buy_se_day': 18,
    'land_cash': 1675,        # 이 이상 남을 때만 땅을 산다
    'reserve': 30,            # 항상 남겨둘 현금 (고용이 최우선이라 낮게)
    'load_per_unit': 6.3991,     # 유닛 1명이 하루에 감당하는 타일 수 (이동 포함)
    'sell_frac': 2.3788,        # 마을 하루 흡수량 대비 판매 비율
    'shed_soft_cap': 87,      # 이 이상 쌓이면 강제 매도
    'feed_carry': 4,          # 유닛이 한 번에 집어오는 밀 개수
    # ── 구역 배정 (zoning) ──
    # 측정 결과 행동의 74%가 '이동'이었고 실제 작업은 22%뿐이었다.
    # 모든 유닛이 매 턴 '전역에서 가장 가까운 일'을 새로 고르면
    # 서로 목표를 뺏고 왕복하며 걷기만 한다(thrashing).
    # 유닛마다 담당 구역을 고정하면 걷는 거리가 줄고 배정도 안정된다.
    # 실측 결과 오히려 손해였다(26,562 -> 14,698). 인덱스 나머지로 나눈 '구역'이
    # 실제로는 흩어진 타일 집합이라 이동이 더 늘었다. 기본은 끈다.
    # 제대로 하려면 **연속된 블록**으로 나눠야 한다 (미구현).
    'zoning': 0,              # 1이면 구역 배정 사용
    'zone_slack': 2,          # 자기 구역에 일이 없을 때 전역에서 찾을 우선순위 여유
    # ── 단계적 개시 (staged opening) ──
    # 0일차에 소를 몰아 사면 자본이 통째로 묶이고 14일간 수입이 0이 된다.
    # 밀은 씨앗 $10에 2일이면 수확되므로 **초반 현금 엔진**으로 쓴다.
    'wheat_first_days': 3,    # 이 날까지는 밀만 심는다 (2일 만에 도는 현금 엔진)
    'animal_start_day': 0,    # 동물 구매 시작일
    'animal_per_day': 1,      # 하루 최대 동물 구매 수 (자본 잠김 방지)
    'straw_start_day': 12,     # 딸기 구매 시작일
    'work_capital': 477,      # 확장 전에 남겨둘 운전자금
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
    cows = sheep = geese = straw = wheat_t = 0
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
        hires = min(P['max_hands'], P['hire_slots'])
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

    if len(market) < 9 and seeds.get('WHEAT', 0) < 4 and afford(CROPS['WHEAT']['seed'] * 4):
        market.append(['BUY_SEED', 'WHEAT', 4])
    if len(market) < 9 and straw + seeds.get('STRAWBERRY', 0) < P['target_straw'] \
            and afford(CROPS['STRAWBERRY']['seed'] * 2):
        market.append(['BUY_SEED', 'STRAWBERRY', 2])

    # 4) 판매 — 마을 흡수 속도에 맞춰 조금씩. 창고가 차면 강제 매도.
    urgent = shed_total >= P['shed_soft_cap']
    for prod in SELLABLE:
        if len(market) >= 10:
            break
        have = shed.get(prod, 0)
        if have <= 0:
            continue
        if prod == 'WHEAT':
            # 사료 7일치를 넘는 분량만 판다 (매수선 3배와 겹치지 않게)
            have -= (n_animals + pending_a) * 7
            if have <= 0:
                continue
        cap = max(1, int(TOWN_DAILY.get(prod, 1) * P['sell_frac']))
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
                            tasks.append((6, (x, y), 'COLLECT_FERTILIZER', None))
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


def agent(obs):
    return act(obs)
