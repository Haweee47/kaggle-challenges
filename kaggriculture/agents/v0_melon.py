# -*- coding: utf-8 -*-
"""
v0 — 튜토리얼 melon_maxxer (베이스라인)

공식 getting-started 노트북의 에이전트를 그대로 옮긴 것. 개선의 기준점이다.
공식 문서가 스스로 지적한 약점 4가지:
  1. 일꾼을 고용하지 않고 땅도 사지 않는다 (농부 1명 + 1구역)
  2. 멜론만 키운다 (다른 작물 가격이 높아도 팔 게 없다)
  3. 비료를 안 쓴다
  4. 팔 때 재고를 한 번에 쏟아 가격을 스스로 무너뜨린다
"""
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

MELON_SEED_COST = CROPS["MELON"]["seed"]
SELL_THRESHOLD = 200


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _find_target_tile(farm, board_size, have_seed):
    fx, fy = farm["farmer"]
    candidates = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" \
                    and tile["crop"] == "MELON":
                purpose = None
                if tile["yield_units"] > 0 and tile.get("planted_day") is not None:
                    purpose = "harvest"
                if not tile["watered_today"]:
                    purpose = "water" if purpose is None else purpose
                if purpose:
                    candidates.append((x, y, purpose))
            elif tile is None and have_seed:
                candidates.append((x, y, "plant"))
    if not candidates:
        return None
    priority = {"harvest": 0, "water": 1, "plant": 2}
    candidates.sort(key=lambda c: (priority[c[2]], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def act(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]

    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
    melon_price = prices.get("MELON", 0)
    money = farm.get("money", 0)

    market = []
    # 씨앗이 떨어지면 한 개 산다
    if seeds.get("MELON", 0) == 0 and money >= MELON_SEED_COST:
        market.append(["BUY_SEED", "MELON", 1])
    # 값이 좋으면 창고를 전부 판다 (← 이게 가격을 무너뜨린다)
    held = shed.get("MELON", 0)
    if held > 0 and melon_price >= SELL_THRESHOLD:
        market.append(["SELL", "MELON", held])

    have_seed = seeds.get("MELON", 0) > 0
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" \
            and tile["crop"] == "MELON":
        if tile["yield_units"] > 0:
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}

    target = _find_target_tile(farm, board_size, have_seed)
    if target is None:
        return {"farmer": ["PASS"], "hands": [], "market": market}
    tx, ty, purpose = target
    if (tx, ty) == (fx, fy):
        action = {"harvest": "HARVEST", "water": "WATER",
                  "plant": "PLANT"}[purpose]
        farmer = [action, "MELON"] if action == "PLANT" else [action]
        return {"farmer": farmer, "hands": [], "market": market}
    mv = _step_toward(fx, fy, tx, ty)
    return {"farmer": [mv or "PASS"], "hands": [], "market": market}
