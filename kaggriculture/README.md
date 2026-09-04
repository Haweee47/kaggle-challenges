# Kaggriculture — 농장 시뮬레이션 에이전트

- **대회**: https://www.kaggle.com/competitions/kaggriculture
- **유형**: Featured · 시뮬레이션/에이전트 (정형 ML 아님)
- **상금**: $50,000 · **마감**: 2026-09-30 · 7,586팀
- **평가**: 1:1 대전 후 은행 잔고 비교 → **승/패/무 기반 Elo**

📕 **[전략 분석 → docs/STRATEGY.md](docs/STRATEGY.md)**

## 핵심 발견

**"무엇을 키우느냐"가 아니라 "마을이 사주는 만큼만 파느냐"의 게임.**

| 품목 | 타일 효율 | 마을 시즌 흡수량 |
|---|---|---|
| MELON | $109/타일·일 (1위) | **30개 (꼴찌)** ⚠️ |
| STRAWBERRY | $44.7 | 426개 |
| MILK (소) | — | 327개 |
| WOOL (양) | — | 228개 |

멜론은 타일 효율 1위지만 **사가는 상점이 없어** 팔수록 가격이 무너진다.
상위권이 전부 **소+양+딸기**로 수렴한 이유가 이것이다.

## 구조

```
kaggriculture/
├── data/                원본 규칙 (README.md, AGENTS.md)
├── src/
│   ├── economics.py     가격곡선·생산효율·노동비용    ← EDA 역할
│   └── arena.py         자체 대전 평가 틀            ← 교차검증 역할
├── agents/
│   └── v0_melon.py      튜토리얼 베이스라인 (평균 잔고 1,060)
├── docs/STRATEGY.md     전략·메타 분석
└── sims/                실험 로그
```

## 사용법

```powershell
.\.venv\Scripts\Activate.ps1
cd kaggriculture

# 경제 분석
python -c "import sys; sys.path.append('src'); import economics as E; print(E.price_curve_table())"

# 에이전트 대전
python -c "import sys; sys.path.append('src'); sys.path.append('agents'); from arena import match; from v0_melon import act; match(act,'random',n_games=10)"
```

> ⚠️ Windows에서는 함수 객체를 멀티프로세스로 넘길 수 없어 `arena`가 자동으로 순차 실행한다.
> 병렬로 돌리려면 에이전트를 `.py` 파일 경로 문자열로 넘긴다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 규칙·경제 분석 | ✅ |
| 2 | 자체 대전 평가 틀 | ✅ |
| 3 | 메타 분석 (상위권 리플레이) | ✅ |
| 4 | v1 에이전트 (수요 기반 포트폴리오) | ⏳ |
| 5 | 파라미터 튜닝 | ⏳ |
| 6 | 판매 타이밍 (front-running) | ⏳ |
