"""整场状态特征（match_state）+ 叫牌张（called_tile / last_discard）的测试。

这一版把「模型看不见的整场状态」补进局面特征：

* ``View`` 新增 ``furiten / kyoku / honba / kyotaku / dealer_repeat /
  hands_left / ranks / tsumogiri / called_tile``（全部带默认值，
  老构造点不用改）。
* ``encode`` 末尾追加三块：``match_state``(17) / ``called_tile``(34) /
  ``last_discard``(34) —— **已有块的 offset 与宽度一个都没动**。
* 顺带修 ``visible_counts`` 把「自己的副露 / 自己的拔北」算两遍的 bug。

跑法：``.venv/bin/python aimodel/tests/test_state_features.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import RandomAgent  # noqa: E402
from aimodel.encode import (  # noqa: E402
    DIM, encode, feature_blocks, layout_for, visible_counts,
)
from aimodel.feature_dispatch import encode_for, encoder_for, known_dims  # noqa: E402
from aimodel.engine import Game, View  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def close(a, b, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_view(players: int = 3, **overrides) -> View:
    """**只用老字段**构造 View —— 顺便证明新字段的默认值足够老代码跑。

    默认值必须让 ``advise.py:98`` / ``ui/autopilot.py:66`` /
    ``potatomaj/ai.py:516`` 这些老构造点**一行都不用改**。
    """
    rule = S.rule_for(players)
    base = dict(
        player=0,
        hand=S.parse_hand("123456789m1123p"),
        drawn=None,
        melds=[],
        hand_red=[0] * 34,
        discards=[[] for _ in range(players)],
        melds_all=[[] for _ in range(players)],
        kita=[0] * players,
        scores=[rule.start_score] * players,
        dora_indicators=[0],
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=40,
        turn=5,
        riichi=[False] * players,
        ippatsu=[False] * players,
        is_dealer=True,
        last_discard=None,
        rule=rule,
    )
    base.update(overrides)
    return View(**base)


class Fixed:
    """永远返回固定选项的 agent；把 decide 收到的 (decision, view) 都记下来。"""

    def __init__(self, choices: dict | None = None):
        self.choices = dict(choices or {})
        self.calls: list[tuple] = []

    def decide(self, decision, view):
        self.calls.append((decision, view))
        return self.choices.get(decision.kind, 0)


def setup_game(players: int = 3, seat: int = 1, choices: dict | None = None):
    """建一个 Game 并 ``_setup(0)``；只让 ``seat`` 有固定选项，其余永远 0。"""
    agents = [
        Fixed(choices if i == seat else {}) for i in range(players)
    ]
    g = Game(
        agents, seed=7, rule=S.rule_for(players), rounds=players, max_hands=2
    )
    g._setup(0)
    g.turn = 3
    return g, agents


#: 三麻合法的听牌手：234567p 234567s + 5s（单骑 5s，断幺）。
#: 和牌张 = 5s = 22。
TENPAI_3P = "234567p234567s5s"
WAIT_5S = 22


# ---------------------------------------------------------------------------
print("== 1) 布局：新块只追加在末尾，老块 offset / DIM 前缀不动 ==")
for players, old_dim, old_tenpai in ((3, 486, 485), (4, 560, 559)):
    O, dim = layout_for(players)
    blocks = feature_blocks(players)
    check(dim == old_dim + 17 + 34 + 34,
          f"{players} 麻 DIM 应 = 老 {old_dim} + 85，得到 {dim}")
    check(O["tenpai"] == old_tenpai,
          f"{players} 麻老最后一块 tenpai 的 offset 应仍是 {old_tenpai}，"
          f"得到 {O['tenpai']}")
    check(O["match_state"] == old_dim,
          f"{players} 麻 match_state 应紧接老布局（offset {old_dim}），"
          f"得到 {O['match_state']}")
    check(blocks[-3:] == [("match_state", 17), ("called_tile", 34),
                          ("last_discard", 34)],
          f"{players} 麻末尾三块必须是 match_state/called_tile/last_discard："
          f"{blocks[-3:]}")
    v = make_view(players)
    x = encode(v)
    check(x.shape == (dim,), f"{players} 麻 encode 维度应是 {dim}，得到 {x.shape}")
    check(np.isfinite(x).all(), f"{players} 麻 encode 不能有 NaN/inf")
    # 老构造点（不传任何新字段）拿到的默认值
    check(v.furiten is False and v.kyoku == 1 and v.honba == 0
          and v.kyotaku == 0 and v.dealer_repeat is False
          and v.hands_left == 0 and v.ranks == () and v.tsumogiri == ()
          and v.called_tile is None,
          f"{players} 麻 View 新字段的默认值不对")
print(f"  ✅ 3p DIM={layout_for(3)[1]} / 4p DIM={layout_for(4)[1]}，"
      f"老块 offset 不变；老构造点零改动可跑")

O3, D3 = layout_for(3)
O4, D4 = layout_for(4)
MS = O3["match_state"]
MS4 = O4["match_state"]


# ---------------------------------------------------------------------------
print("\n== 2) match_state 逐维（直接改 View 字段，验 encode 读的就是它）==")
v = make_view(3)
v.honba = 2
check(close(encode(v)[MS + 0], 0.5), "honba=2 应编码成 2/4=0.5")
v.honba = 9
check(close(encode(v)[MS + 0], 1.0), "honba 应 clip 到 1")

v = make_view(3, kyoku=3)
x = encode(v)
check(close(x[MS + 1], 2 / 7), "(kyoku-1)/7：kyoku=3 -> 2/7")
check(close(x[MS + 2:MS + 6].sum(), 1.0), "kyoku onehot 必须恰好一位")
check(close(x[MS + 4], 1.0), "kyoku=3 应落在 2..5 的第 3 位（MS+4）")
v.kyoku = 5
x = encode(v)
check(close(x[MS + 2:MS + 6].sum(), 0.0), "kyoku>4（南入）onehot 应全 0")
check(close(x[MS + 1], 4 / 7), "kyoku=5 -> (5-1)/7")

v = make_view(3, kyotaku=3, dealer_repeat=True, hands_left=4)
x = encode(v)
check(close(x[MS + 6], 1.0), "kyotaku=3 -> 3/3=1")
check(close(x[MS + 7], 1.0), "dealer_repeat=True -> 1")
check(close(x[MS + 8], 0.5), "hands_left=4 -> 4/8=0.5")
v.kyotaku = 9
check(close(encode(v)[MS + 6], 1.0), "kyotaku 应 clip 到 1")
v.hands_left = 20
check(close(encode(v)[MS + 8], 1.0), "hands_left 应 clip 到 1")

v = make_view(3, furiten=True)
check(close(encode(v)[MS + 9], 1.0), "furiten=True -> 1")
v.furiten = False
check(close(encode(v)[MS + 9], 0.0), "furiten=False -> 0")

# rank one-hot：rank==1 -> MS4+10 ... rank==4 -> MS4+13（四麻的 match_state 在 560）
v = make_view(4, player=0, ranks=(2, 1, 4, 3))
x = encode(v)
check(close(x[MS4 + 10:MS4 + 14].sum(), 1.0), "rank onehot 必须恰好一位")
check(close(x[MS4 + 11], 1.0), "rank=2 应落在 MS4+11")
v.player = 2
check(close(encode(v)[MS4 + 13], 1.0), "rank=4 应落在 MS4+13")
v.ranks = (0, 0, 0, 0)
check(close(encode(v)[MS4 + 10:MS4 + 14].sum(), 0.0),
      "rank 缺失/0 时 onehot 应全 0（老 View 兼容）")

# 自己牌河摸切占比 / 最近一张
v = make_view(3)
v.tsumogiri = ((True, False, True), (), ())
x = encode(v)
check(close(x[MS + 14], 2 / 3), "自己摸切占比 = 2/3")
check(close(x[MS + 15], 1.0), "自己最近一张是摸切 -> 1")
v.tsumogiri = ((True, False, False), (), ())
check(close(encode(v)[MS + 15], 0.0), "自己最近一张不是摸切 -> 0")
v.tsumogiri = ((), (), ())
x = encode(v)
check(close(x[MS + 14], 0.0) and close(x[MS + 15], 0.0),
      "自己牌河为空时占比 / 最近一张都应是 0")

# 点数差
v = make_view(3, scores=[30000, 50000, 25000])
check(close(encode(v)[MS + 16], -0.4),
      "(30000-50000)/50000 = -0.4")
v.scores = [55000, 60000, 25000]
check(close(encode(v)[MS + 16], -0.1),
      "(55000-60000)/50000 = -0.1")
print("  ✅ match_state 0..16 逐维对得上")

print("\n== 2b) called_tile / last_discard（encode 层）==")
v = make_view(3)
x = encode(v)
check(close(x[O3["called_tile"]:O3["called_tile"] + 34].sum(), 0.0),
      "called_tile=None 应全 0")
v.called_tile = 9
x = encode(v)
check(close(x[O3["called_tile"] + 9], 1.0), "called_tile=1p 应在对应位为 1")
check(close(x[O3["called_tile"]:O3["called_tile"] + 34].sum(), 1.0),
      "called_tile 必须恰好一位")
v = make_view(3)
check(close(encode(v)[O3["last_discard"]:O3["last_discard"] + 34].sum(), 0.0),
      "last_discard=None 应全 0")
v.last_discard = (1, 9)
x = encode(v)
check(close(x[O3["last_discard"] + 9], 1.0),
      "last_discard=(1,1p) 应在 1p 位为 1")
check(close(x[O3["last_discard"]:O3["last_discard"] + 34].sum(), 1.0),
      "last_discard 必须恰好一位")
print("  ✅ called_tile / last_discard 的 one-hot 正确")

# ---------------------------------------------------------------------------
# 与另一组固定局面的交叉核对（协调者 2026-10-21，四麻：碰 5s + 拔 1 北 +
# 牌河有牌）——这里只写解释，不钉死值（逐位金值在 test_legacy_encode.py）：
#   v1（老 encode）sha256 = 73e3dcf2867b2c556c6038cf6beb2b193f73eb7c0d67e75b84a7e16867aaed3b
#   v2（新 encode）sha256 = 1c26846656a491c7a998f1a791407a58967e9c89741fda3bb9b199df1cca6354
#   **前 560 维只有 2 个下标不同**：430（visible 里的 5s）与 438（visible 里的北）。
#   原因：visible_counts 修掉了「自己的副露/拔北算两遍」——5s 从 6 变 3、
#   北从 2 变 1；该局的进张/安全牌恰好不涉及这两种牌，所以 ukeire/safe_exit 没变。
#   其余 558 维逐位相同 —— 这就是「只追加 + 只修这一处」的直接证据。
print("\n== 2c) 维度分发：老/新各自可编码，未知维度必须当场炸 ==")
v4 = make_view(4)
check(known_dims(4) == {560: "v1", 645: "v2"},
      f"四麻 known_dims 应为 {{560:'v1',645:'v2'}}，得到 {known_dims(4)}")
check(encoder_for(560, 4) is not encoder_for(645, 4),
      "560 与 645 必须分发到不同的 encoder 模块")
check(encode_for(560, 4, v4).shape == (560,), "encode_for(560,4) 形状应为 560")
check(encode_for(645, 4, v4).shape == (645,), "encode_for(645,4) 形状应为 645")
try:
    encode_for(999, 4, v4)
    check(False, "encode_for(999,4) 应当抛 ValueError，实际没抛")
except ValueError as exc:
    msg = str(exc)
    check("未知特征维度" in msg and "560" in msg and "645" in msg,
          f"异常消息必须同时包含「未知特征维度」和两个合法维度 560/645：{msg}")
    print(f"  ✅ encode_for(999,4) -> ValueError：{msg.splitlines()[0]}")


# ---------------------------------------------------------------------------
print("\n== 3) 振听三条件（真引擎状态 -> View.furiten -> encode）==")
# ① 自己打过的牌是自己的听牌张
g, _ = setup_game(3, seat=1)
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[1].discards = [WAIT_5S]
g.players[1].pond = [WAIT_5S]
g.players[1].pond_tsumogiri = [False]
v = g.view_for(1)
check(v.furiten is True, "① 自己打过和牌张 -> 振听")
check(close(encode(v)[MS + 9], 1.0), "① 振听必须进 match_state[9]")

# ② 立直后漏和（立直振听，永久）
g, _ = setup_game(3, seat=1)
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[1].riichi = True
g.players[1].missed_ron_riichi = True
g.players[1].missed_ron_turn = -1
v = g.view_for(1)
check(v.furiten is True, "② 立直后漏和 -> 立直振听")
g.players[1].riichi = False
check(g.view_for(1).furiten is False,
      "② 未立直时 missed_ron_riichi 不应单独触发")

# ③ 同巡漏和（临时）；自己打牌后清掉
g, _ = setup_game(3, seat=1)
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[1].missed_ron_turn = g.turn
check(g.view_for(1).furiten is True, "③ 同巡漏和 -> 振听")
g.turn += 1
check(g.view_for(1).furiten is False, "③ 过了这一巡 -> 临时振听消失")

g, _ = setup_game(3, seat=1)
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[1].missed_ron_turn = g.turn
g.players[1].hand[0] = 1          # 手里有 1m，打掉它（不是和牌张）
g._finish_discard(1, 0, g.turn)
check(g.players[1].missed_ron_turn == -1, "自己打牌后 missed_ron_turn 必须清零")
check(g.view_for(1).furiten is False, "打掉非和牌张后不应振听")
print("  ✅ ①自己牌河 ②立直漏和 ③同巡漏和 三条件都正确")

print("\n== 3b) 真引擎：别人打出可和牌而不喊 -> 引擎真的记下漏和 ==")
# 不立直：同巡临时振听；巡目过后消失
g, agents = setup_game(3, seat=1, choices={"ron": 1})
for q in range(3):
    g.players[q].hand = [0] * 34
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[0].pond = [WAIT_5S]
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [WAIT_5S]
g.last_discard = (0, WAIT_5S)
g._after_discard(0, WAIT_5S, g.turn)
pl = g.players[1]
check(pl.passed_ron is True, "拒绝荣和必须置 passed_ron")
check(pl.missed_ron_turn == g.turn, "拒绝荣和必须记下当前巡目")
check(g.view_for(1).furiten is True, "拒绝荣和 -> 本巡振听")
g.turn += 1
check(g.view_for(1).furiten is False, "下一巡临时振听应消失（未立直）")

# 立直：漏和后永久振听
g, agents = setup_game(3, seat=1, choices={"ron": 1})
for q in range(3):
    g.players[q].hand = [0] * 34
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[1].riichi = True
g.players[0].pond = [WAIT_5S]
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [WAIT_5S]
g.last_discard = (0, WAIT_5S)
g._after_discard(0, WAIT_5S, g.turn)
check(g.players[1].missed_ron_riichi is True, "立直后漏和必须记成立直振听")
g.turn += 1
check(g.view_for(1).furiten is True, "立直振听不能因巡目变化消失")
print("  ✅ 引擎跟踪：临时振听 / 立直振听都按规则生效")


# ---------------------------------------------------------------------------
print("\n== 4) called_tile：荣和 / 碰 / 吃 各跑一次真引擎 ==")
# 4a 荣和
g, agents = setup_game(3, seat=1, choices={"ron": 1})
for q in range(3):
    g.players[q].hand = [0] * 34
g.players[1].hand = S.parse_hand(TENPAI_3P)
g.players[0].pond = [WAIT_5S]
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [WAIT_5S]
g.last_discard = (0, WAIT_5S)
g._after_discard(0, WAIT_5S, g.turn)
d, v = agents[1].calls[-1]
check(d.kind == "ron" and d.tile == WAIT_5S,
      f"荣和决策应带 tile={WAIT_5S}，得到 {d.kind}/{d.tile}")
check(v.called_tile == d.tile, "荣和 View.called_tile 必须 == decision.tile")
x = encode(v)
check(close(x[O3["called_tile"] + WAIT_5S], 1.0),
      "荣和时 called_tile 编码位必须是 1")
check(close(x[O3["called_tile"]:O3["called_tile"] + 34].sum(), 1.0),
      "荣和时 called_tile 只能有一位")

# 4b 碰
g, agents = setup_game(3, seat=1, choices={"call": 1})
for q in range(3):
    g.players[q].hand = [0] * 34
g.players[1].hand[WAIT_5S] = 2
g.players[0].pond = [WAIT_5S]
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [WAIT_5S]
caller, _ = g._check_call(WAIT_5S, 0)
check(caller == 1, f"碰的副露者应是 1，得到 {caller}")
d, v = agents[1].calls[-1]
check(d.kind == "call" and d.tile == WAIT_5S, "碰决策应带被碰的那张")
check(v.called_tile == d.tile, "碰 View.called_tile 必须 == decision.tile")
x = encode(v)
check(close(x[O3["called_tile"] + WAIT_5S], 1.0), "碰时 called_tile 编码位必须是 1")
check(g.players[0].pond == [] and g.players[0].pond_tsumogiri == [],
      "被碰走的牌必须从 pond 和 pond_tsumogiri 一起删掉")

# 4c 吃（只有四麻）
O4, D4 = layout_for(4)
g, agents = setup_game(4, seat=1, choices={"call": 0, "chi": 1})
for q in range(4):
    g.players[q].hand = [0] * 34
g.players[1].hand[10] = 1      # 2p
g.players[1].hand[11] = 1      # 3p
g.players[0].pond = [9]        # 1p
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [9]
caller, _ = g._check_call(9, 0)
check(caller == 1, f"吃的副露者应是 1，得到 {caller}")
d, v = agents[1].calls[-1]
check(d.kind == "chi" and d.tile == 9, "吃决策应带被吃的那张（1p）")
check(v.called_tile == d.tile, "吃 View.called_tile 必须 == decision.tile")
x = encode(v)
check(close(x[O4["called_tile"] + 9], 1.0), "吃时 called_tile 编码位必须是 1")
check(g.players[0].pond == [] and g.players[0].pond_tsumogiri == [],
      "被吃走的牌必须从 pond 和 pond_tsumogiri 一起删掉")

# 4d 普通打牌 -> called_tile 全 0
g, agents = setup_game(3, seat=0)
g.players[0].hand = [0] * 34
g.players[0].hand[0] = 1
g._ask_choice(0, "discard", [0])
d, v = agents[0].calls[-1]
check(d.kind == "discard" and d.tile is None, "打牌决策的 tile 应为 None")
check(v.called_tile is None, "打牌 View.called_tile 应为 None")
check(close(encode(v)[O3["called_tile"]:O3["called_tile"] + 34].sum(), 0.0),
      "普通打牌时 called_tile 必须全 0")
print("  ✅ 荣和 / 碰 / 吃 的 called_tile 都正确；普通打牌全 0")


# ---------------------------------------------------------------------------
print("\n== 5) 整场（固定 seed）：kyoku / honba / ranks / tsumogiri 不变量 ==")
# 南入起始必须是 5（三麻也不能把「南1」算成 4 —— 4 是东4 的 one-hot 位）
g_south = Game([RandomAgent(i) for i in range(3)], seed=0,
               round_wind=S.SOUTH, rounds=3)
check(g_south.kyoku == 5,
      f"南入起始 kyoku 应为 5，得到 {g_south.kyoku}")

log: list[tuple] = []


class Spy:
    def __init__(self, seed: int):
        self.inner = RandomAgent(seed)

    def decide(self, decision, view):
        log.append((decision, view))
        return self.inner.decide(decision, view)


g = Game([Spy(1), Spy(2), Spy(3)], seed=20261021, rule=S.rule_for(3), rounds=3)
res = g.play()
check(len(log) > 50, f"整场应有足够多的决策，得到 {len(log)}")
kinds = {d.kind for d, _ in log}
check("discard" in kinds, f"整场应出现打牌决策，得到 {sorted(kinds)}")

kyoku_seq = [v.kyoku for _, v in log]
check(all(a <= b for a, b in zip(kyoku_seq, kyoku_seq[1:])),
      f"kyoku 必须单调不减，得到 {kyoku_seq[:20]}…")
check(all(k >= 1 for k in kyoku_seq), "kyoku 必须从 1 起")
check(all(v.honba >= 0 for _, v in log), "honba 必须 >= 0")
check(all(v.hands_left >= 0 for _, v in log), "hands_left 必须 >= 0")
check(all(v.kyotaku >= 0 for _, v in log), "kyotaku 必须 >= 0")
check(all(isinstance(v.dealer_repeat, bool) for _, v in log),
      "dealer_repeat 必须是 bool")

bad_rank = bad_len = bad_called = bad_last = bad_finite = 0
for d, v in log:
    if sorted(v.ranks) != [1, 2, 3] or sum(v.ranks) != 6:
        bad_rank += 1
    if len(v.tsumogiri) != 3:
        bad_len += 1
    else:
        for q in range(3):
            if len(v.tsumogiri[q]) != len(v.discards[q]):
                bad_len += 1
    # called_tile 只在 ron / call / chi 三类决策里出现，且就是 decision.tile
    if d.kind in ("ron", "call", "chi"):
        if v.called_tile != d.tile:
            bad_called += 1
    elif v.called_tile is not None:
        bad_called += 1
    # last_discard 编码必须和 view.last_discard 一致
    x = encode(v)
    if not np.isfinite(x).all():
        bad_finite += 1
    last = x[O3["last_discard"]:O3["last_discard"] + 34]
    if v.last_discard is None:
        if last.sum() != 0:
            bad_last += 1
    elif last[int(v.last_discard[1])] != 1.0 or last.sum() != 1.0:
        bad_last += 1
check(bad_rank == 0, f"{bad_rank} 个 View 的 ranks 不是 1..3 的排列")
check(bad_len == 0, f"{bad_len} 处 tsumogiri 与 discards 长度不一致")
check(bad_called == 0, f"{bad_called} 个 View 的 called_tile 与决策不符")
check(bad_last == 0, f"{bad_last} 个 View 的 last_discard 编码不一致")
check(bad_finite == 0, f"{bad_finite} 个 View 的 encode 有 NaN/inf")
check(g.kyoku >= 1 and g.honba >= 0 and g.hands_left >= 0,
      "整场结束后 Game 的整场状态字段仍应合法")
print(f"  ✅ {len(log)} 个决策：kyoku 单调、honba>=0、ranks 和为 6、"
      f"tsumogiri 与牌河等长、called_tile/last_discard 一致")


# ---------------------------------------------------------------------------
print("\n== 6) visible_counts：自己的副露 / 拔北只能算一次（修 bug）==")
meld = S.Meld("pon", WAIT_5S, 1)      # 自己碰的 5s
v = make_view(3)
v.hand = S.parse_hand("234567p234s")   # 手里没有 5s
v.melds = [meld]
v.melds_all = [[meld], [], []]
v.dora_indicators = []
v.discards = [[], [], []]
vis = visible_counts(v)
check(vis[WAIT_5S] == 3,
      f"自己碰的 5s：visible 应为 3（副露三张），得到 {vis[WAIT_5S]}")
check(close(encode(v)[O3["visible"] + WAIT_5S], 3 / 4),
      "自己碰的 5s：encode visible 应为 3/4")

v = make_view(3)
v.hand = S.parse_hand("234567p234s")
v.dora_indicators = []
v.kita = [1, 2, 0]
vis = visible_counts(v)
check(vis[S.NORTH] == 3,
      f"拔北：自己 1 + 别家 2 应共 3，得到 {vis[S.NORTH]}")
check(close(encode(v)[O3["visible"] + S.NORTH], 3 / 4),
      "拔北：encode visible 应为 3/4")

# 与「逐项手工数一遍」对拍（含自己副露、别家副露、别家拔北、宝牌指示牌）
v = make_view(3)
v.hand = S.parse_hand("234567p234s")
v.melds = [S.Meld("pon", WAIT_5S, 1)]
v.melds_all = [
    [S.Meld("pon", WAIT_5S, 1)],
    [S.Meld("pon", 0, 0)],
    [S.Meld("pon", 27, 2)],
]
v.discards = [[9, 10], [11], [12, 13, 14]]
v.kita = [1, 0, 2]
v.dora_indicators = [15]
want = [0] * 34
for i in range(34):
    want[i] = v.hand[i]
for m in v.melds:
    for t in m.tiles:
        want[t] += 1
for q in range(3):
    for t in v.discards[q]:
        want[t] += 1
    if q != v.player:
        for m in v.melds_all[q]:
            for t in m.tiles:
                want[t] += 1
for d in v.dora_indicators:
    want[d] += 1
want[S.NORTH] += sum(v.kita)
got = visible_counts(v)
check(got == want, f"visible_counts 与手工逐项计数不一致：\n  got={got}\n  want={want}")
print("  ✅ 自己副露=3、自己的北只算一次；与手工计数完全一致")

# 拿真引擎跑出来的带副露 View 再验一次（防止只用手造 View 自洽）
g, _ = setup_game(3, seat=1, choices={"call": 1})
g.players[1].hand[WAIT_5S] = 2
g.players[0].pond = [WAIT_5S]
g.players[0].pond_tsumogiri = [False]
g.players[0].discards = [WAIT_5S]
g._check_call(WAIT_5S, 0)
v = g.view_for(1)
check(v.melds and v.melds[0].kind == "pon", "真引擎应真的碰了 5s")
check(visible_counts(v)[WAIT_5S] == 3,
      f"真引擎碰 5s 后 visible 应为 3，得到 {visible_counts(v)[WAIT_5S]}")
print("  ✅ 真引擎的碰副露也只算一次")


# ---------------------------------------------------------------------------
if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
