"""局面 -> 特征向量。

特征分四块，全部是「这个玩家本来就能看到的东西」：

1. **自己的牌**：手牌 / 副露 / 牌河 / 赤牌（计数）
2. **他家的公开信息**：两家的牌河、副露、立直/一发标记、点数
3. **场况**：宝牌指示牌与宝牌、可见牌计数、剩余牌数、巡目、自风场风、亲家
4. **听牌信息**：向听数 one-hot、进张枚数（这两项是手写特征，
   不靠网络自己从计数里推 —— 有它俩收敛快很多）

动作掩码也在这里给出：「哪几张能打」「能不能碰」等。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import Sequence

from . import sanma as S
from .engine import ACTION_TABLE, Decision, View, dora_from_index
from .fastshanten import shanten, ukeire_count

N_TYPES = 34

#: 各决策种类的策略头大小
HEAD_SIZE: dict[str, int] = {
    "discard": 34,
    "tsumo": 2,
    "ron": 2,
    "riichi": 2,
    "call": 3,
    "kita": 2,
    "kan": 2,
    "kyuushu": 2,
}

# ---------------------------------------------------------------------------
# 特征布局（顺序固定，改动必须同步改 dim）
# ---------------------------------------------------------------------------

#: (名字, 宽度) —— 三麻/四麻的布局由 :func:`feature_blocks` 生成
def feature_blocks(players: int = 3) -> list[tuple[str, int]]:
    """按人数生成特征布局。

    三麻（``players=3``）生成的顺序和以前**一模一样**，这样老权重照样能读；
    四麻只是多插一组对手信息、并把「每家一块」的那些块加宽。
    """
    n_opp = players - 1
    blocks: list[tuple[str, int]] = [
        ("hand", 34), ("hand_onehot", 34), ("hand_pair", 34), ("hand_triple", 34),
        ("melds", 34), ("own_pond", 34),
    ]
    blocks += [(f"opp{i + 1}_pond", 34) for i in range(n_opp)]
    blocks += [(f"opp{i + 1}_melds", 34) for i in range(n_opp)]
    blocks += [
        ("visible", 34), ("dora_indicator", 34), ("dora_tile", 34), ("yakuhai", 5),
        ("shanten_onehot", 8), ("ukeire", 1), ("hand_dora", 1), ("aka", 1),
        ("wall_left", 1), ("turn", 1), ("hand_size", 1),
        ("scores", players), ("score_lead", 1),
        ("riichi", players), ("ippatsu", players), ("kita", players),
        ("meld_count", players), ("seat_onehot", players), ("round_wind", 2),
        ("is_dealer", 1), ("menzen", 1), ("safe_exit", 1), ("tenpai", 1),
    ]
    return blocks


_LAYOUT_CACHE: dict[int, tuple[dict[str, int], int]] = {}


def layout_for(players: int = 3) -> tuple[dict[str, int], int]:
    """返回 (每块起始下标, 总维度)，按人数缓存。"""
    got = _LAYOUT_CACHE.get(players)
    if got is None:
        offsets: dict[str, int] = {}
        o = 0
        for name, w in feature_blocks(players):
            offsets[name] = o
            o += w
        got = (offsets, o)
        _LAYOUT_CACHE[players] = got
    return got


#: 默认（三麻）布局，兼容直接 import OFFSET / DIM 的老代码
FEATURE_BLOCKS: list[tuple[str, int]] = feature_blocks(3)
OFFSET: dict[str, int] = layout_for(3)[0]
DIM = layout_for(3)[1]


def visible_counts(view: View) -> list[int]:
    """从该玩家视角能看到的每一种牌各有几张（含自己手牌）。"""
    vis = [0] * 34
    for i in range(34):
        vis[i] = view.hand[i]
    for m in view.melds:
        # 注意：吃是顺子，要把三张**不同的**牌分别记上，
        # 不能拿 start 那张加 3（之前的 bug）
        for t in m.tiles:
            vis[t] += 1
    n = len(view.discards)
    for p in range(n):
        for t in view.discards[p]:
            vis[t] += 1
        for m in view.melds_all[p]:
            for t in m.tiles:
                vis[t] += 1
    for d in view.dora_indicators:
        vis[d] += 1
    vis[S.NORTH] += view.kita[view.player]
    for p in range(len(view.kita)):
        vis[S.NORTH] += view.kita[p]
    return vis


# ---------------------------------------------------------------------------
# 打牌动作特征：每一张候选牌各一份
# ---------------------------------------------------------------------------
# 为什么需要它？只给「整手牌」的特征时，网络要从 34 维计数里自己推出
# 「打掉这张之后是几向听、这张牌危不危险」——防守和听牌判断都学得很慢。
# 把这两件事直接算好喂给它，等于把「牌效 + 读牌」的先验给它，
# 剩下的（什么时候弃和、贪不贪）才是真正要学的。

#: (名字, 宽度)
ACTION_BLOCKS: list[tuple[str, int]] = [
    ("shanten_after", 1),   # 打掉它之后几向听 /6
    ("is_win_after", 1),    # 打掉它就听牌了（向听 0）
    ("ukeire_after", 1),    # 打掉它之后的进张枚数 /40
    ("danger", 1),          # 危险度 0~1（对已经立直的人是现物时为 0）
    ("genbutsu", 1),        # 对每一个立直家都是现物
    ("is_dora", 1),
    ("is_aka", 1),
    ("yakuhai_keep", 1),    # 手里有 2 张以上的役牌，留着能做刻子
    ("left", 1),            # 这张牌场上还剩几张
    ("is_drawn", 1),        # 是不是刚摸到的那张（摸切）
    ("isolated", 1),        # 手里没有相邻牌，是孤张
    ("open_risk", 1),       # 非立直家（含副露 / 晚巡）造成的压迫感
    ("suit_pressure", 1),   # 有家在明显做这一门（牌河单色），这张正好是同门
]
ACTION_DIM = sum(w for _, w in ACTION_BLOCKS)

ACTION_OFFSET: dict[str, int] = {}
_o = 0
for _name, _w in ACTION_BLOCKS:
    ACTION_OFFSET[_name] = _o
    _o += _w
del _o, _name, _w


def flush_pressure(view: View, q: int, tile: int) -> float:
    """``q`` 这家有多像在做 ``tile`` 这一门（0~1）。

    读牌的方向千万不能搞反：**牌河是他不要的牌**。
    一个人疯狂打万子和字牌、某一名几乎不碰，才说明他在攒那一门；
    反过来「牌河全是筒子 -> 他在做筒子」是错的（那说明他不要筒子）。
    """
    pond = view.discards[q]
    n = len(pond)
    suit = tile // 9
    press = 0.0
    if n >= 6:
        used = [0, 0, 0, 0]
        for t in pond:
            used[t // 9] += 1
        share = used[suit] / n
        if share <= 0.15 and used[suit] <= 1:
            # 这一门他几乎没打过 -> 可疑
            press = 0.2 + (0.15 - share) / 0.15 * 0.6
    # 副露里就有这一门，那就不用猜了
    if any(m.tile // 9 == suit for m in view.melds_all[q]):
        press = min(1.0, press + 0.4)
    return min(press, 1.0)


def tile_danger(
    view: View,
    tile: int,
    opp_riichi: Sequence[int] | None = None,
    vis: Sequence[int] | None = None,
) -> float:
    """一张牌打出去有多危险（0 = 绝对安全，1 = 很危险）。

    分两块取最大：

    * **立直家**：现物 0；筋牌打折；中张比幺九危险。
    * **没立直的家**：以前这里直接返回 0，等于完全不设防 —— 实战里被
      默听清一色打了 12000 就是这么来的。现在按「副露数 / 巡目」给基线，
      再乘上「他是不是在做这一门」。
    """
    n_players = view.rule.players
    if opp_riichi is None:
        opp_riichi = [q for q in range(n_players) if q != view.player and view.riichi[q]]
    if vis is None:
        vis = visible_counts(view)

    # 这张牌本身有多好打（中张 > 幺九 > 字牌，字牌还要看还剩几张）
    if tile >= 27:
        d = 0.45 if vis[tile] <= 1 else (0.30 if vis[tile] == 2 else 0.08)
    else:
        n = tile % 9
        if n in (0, 8):
            d = 0.22
        elif n in (1, 7):
            d = 0.38
        else:
            d = 0.62

    # --- 立直家 ---
    d_riichi = 0.0
    for q in opp_riichi:
        if tile in view.discards[q]:
            continue  # 现物
        dd = d
        pond = set(view.discards[q])
        for delta in (-3, 3):
            r = tile + delta
            if 0 <= r < 34 and r // 9 == tile // 9 and r in pond:
                dd *= 0.65  # 筋
                break
        d_riichi = max(d_riichi, dd)

    # --- 没立直的家 ---
    d_open = 0.0
    for q in range(n_players):
        if q == view.player or q in opp_riichi:
            continue
        melds = len(view.melds_all[q])
        if melds >= 2:
            base = 0.45
        elif melds == 1:
            base = 0.32
        elif view.turn >= 12:
            base = 0.30
        elif view.turn >= 8:
            base = 0.18
        else:
            base = 0.08
        fp = flush_pressure(view, q, tile)
        if fp >= 0.35:
            base *= 1.0 + fp          # 他在做这一门
        elif tile in view.discards[q]:
            base *= 0.35              # 他打过这张
        d_open = max(d_open, base)

    return min(max(d_riichi, d_open), 1.0)


def action_features(view: View) -> np.ndarray:
    """``(34, ACTION_DIM)``：每一张牌作为「打出去」时的特征。

    只对手里有的牌做计算（其余留 0）——虽然贵一点（每张要算一次进张），
    但这是让网络学会「牌效 + 防守」的关键。
    """
    af = np.zeros((34, ACTION_DIM), dtype=np.float32)
    hand = view.hand
    n_melds = view.meld_count()
    n_players = view.rule.players
    sanma = n_players == 3
    vis = visible_counts(view)
    opp_riichi = [q for q in range(n_players) if q != view.player and view.riichi[q]]
    dora_types = {dora_from_index(d, view.rule) for d in view.dora_indicators}
    drawn = view.drawn
    o = ACTION_OFFSET
    for t in range(34):
        if hand[t] == 0:
            continue
        rest = list(hand)
        rest[t] -= 1
        s = shanten(rest, n_melds)
        af[t, o["shanten_after"]] = min(max(s, 0), 6) / 6.0
        af[t, o["is_win_after"]] = 1.0 if s == 0 else 0.0
        af[t, o["ukeire_after"]] = (
            min(ukeire_count(rest, vis, n_melds, sanma=sanma), 40) / 40.0
        )
        d = tile_danger(view, t, opp_riichi, vis)
        af[t, o["danger"]] = d
        # 单独拆出来：让网络能分清「立直家给的压迫」和「没立直但快听牌了」
        af[t, o["open_risk"]] = tile_danger(
            view, t, opp_riichi=[], vis=vis
        )
        af[t, o["suit_pressure"]] = max(
            (flush_pressure(view, q, t) for q in range(n_players) if q != view.player),
            default=0.0,
        )
        af[t, o["genbutsu"]] = (
            1.0 if opp_riichi and all(t in view.discards[q] for q in opp_riichi) else 0.0
        )
        af[t, o["is_dora"]] = 1.0 if t in dora_types else 0.0
        af[t, o["is_aka"]] = 1.0 if view.hand_red[t] else 0.0
        af[t, o["yakuhai_keep"]] = (
            1.0 if (hand[t] >= 2 and (t in S.DRAGONS or t in (view.round_wind, view.seat_wind))) else 0.0
        )
        af[t, o["left"]] = (4 - vis[t]) / 4.0
        af[t, o["is_drawn"]] = 1.0 if t == drawn else 0.0
        if t >= 27:
            near = hand[t] > 1
        else:
            n = t % 9
            near = (
                (n > 0 and hand[t - 1] > 0)
                or (n < 8 and hand[t + 1] > 0)
                or (n > 1 and hand[t - 2] > 0)
                or (n < 7 and hand[t + 2] > 0)
                or hand[t] > 1
            )
        af[t, o["isolated"]] = 0.0 if near else 1.0
    return af


def encode(view: View) -> np.ndarray:
    """把观测编码成特征向量（维度由人数决定）。"""
    players = view.rule.players
    O, dim = layout_for(players)
    x = np.zeros(dim, dtype=np.float32)
    hand = view.hand
    h = np.asarray(hand, dtype=np.float32)
    n_melds = view.meld_count()
    sanma = players == 3

    x[O["hand"] : O["hand"] + 34] = h / 4.0
    x[O["hand_onehot"] : O["hand_onehot"] + 34] = (h >= 1).astype(np.float32)
    x[O["hand_pair"] : O["hand_pair"] + 34] = (h >= 2).astype(np.float32)
    x[O["hand_triple"] : O["hand_triple"] + 34] = (h >= 3).astype(np.float32)

    own_melds = np.zeros(34, dtype=np.float32)
    for m in view.melds:
        for t in m.tiles:
            own_melds[t] += 1
    x[O["melds"] : O["melds"] + 34] = own_melds / 3.0

    pond = np.zeros((players, 34), dtype=np.float32)
    for p in range(players):
        for t in view.discards[p]:
            pond[p, t] += 1
    x[O["own_pond"] : O["own_pond"] + 34] = pond[view.player] / 4.0

    others = [p for p in range(players) if p != view.player]
    for slot, p in enumerate(others):
        a = O[f"opp{slot + 1}_pond"]
        x[a : a + 34] = pond[p] / 4.0
        m = np.zeros(34, dtype=np.float32)
        for meld in view.melds_all[p]:
            for t in meld.tiles:
                m[t] += 1
        b = O[f"opp{slot + 1}_melds"]
        x[b : b + 34] = m / 3.0

    vis = visible_counts(view)
    x[O["visible"] : O["visible"] + 34] = np.asarray(vis, dtype=np.float32) / 4.0

    for d in view.dora_indicators:
        x[O["dora_indicator"] + d] = 1.0
        x[O["dora_tile"] + dora_from_index(d, view.rule)] = 1.0

    yak = [
        min(hand[S.HAKU], 3), min(hand[S.HATSU], 3), min(hand[S.CHUN], 3),
        min(hand[view.round_wind], 3), min(hand[view.seat_wind], 3),
    ]
    x[O["yakuhai"] : O["yakuhai"] + 5] = np.asarray(yak, dtype=np.float32) / 3.0

    sh = shanten(hand, n_melds)
    base = O["shanten_onehot"]
    x[base if sh < 0 else base + min(sh, 7)] = 1.0
    x[O["ukeire"]] = min(ukeire_count(hand, vis, n_melds, sanma=sanma), 40) / 40.0

    dora_types = [dora_from_index(d, view.rule) for d in view.dora_indicators]
    tiles = [i for i in range(34) for _ in range(hand[i])]
    for m in view.melds:
        tiles.extend(m.tiles)
    tiles.extend([S.NORTH] * view.kita[view.player])
    x[O["hand_dora"]] = min(sum(tiles.count(d) for d in dora_types), 6) / 6.0
    x[O["aka"]] = min(sum(view.hand_red), 2) / 2.0
    x[O["wall_left"]] = view.wall_left / 55.0
    x[O["turn"]] = min(view.turn, 20) / 20.0
    x[O["hand_size"]] = view.hand_size / 14.0
    x[O["scores"] : O["scores"] + players] = (
        np.asarray(view.scores, dtype=np.float32) / 50000.0
    )
    x[O["score_lead"]] = (view.scores[view.player] - max(view.scores)) / 50000.0
    x[O["riichi"] : O["riichi"] + players] = np.asarray(view.riichi, dtype=np.float32)
    x[O["ippatsu"] : O["ippatsu"] + players] = np.asarray(view.ippatsu, dtype=np.float32)
    x[O["kita"] : O["kita"] + players] = np.asarray(view.kita, dtype=np.float32) / 4.0
    x[O["meld_count"] : O["meld_count"] + players] = (
        np.asarray([len(view.melds_all[p]) for p in range(players)], dtype=np.float32) / 4.0
    )
    x[O["seat_onehot"] + (view.seat_wind - S.EAST)] = 1.0
    x[O["round_wind"] + (0 if view.round_wind == S.EAST else 1)] = 1.0
    x[O["is_dealer"]] = 1.0 if view.is_dealer else 0.0
    x[O["menzen"]] = 1.0 if all(m.is_concealed for m in view.melds) else 0.0
    x[O["tenpai"]] = 1.0 if sh == 0 else 0.0
    # 手里最安全的一张有多危险：这张高就说明「想跑也没地方跑」
    if sum(hand):
        x[O["safe_exit"]] = min(
            (tile_danger(view, t, vis=vis) for t in range(34) if hand[t] > 0), default=0.0
        )
    return x


# ---------------------------------------------------------------------------
# 动作掩码
# ---------------------------------------------------------------------------


def action_mask(decision: Decision) -> np.ndarray:
    """该决策下每个策略头位置的合法性。"""
    size = HEAD_SIZE[decision.kind]
    m = np.zeros(size, dtype=np.float32)
    if decision.kind == "discard":
        for t in decision.options:
            m[int(t)] = 1.0
    else:
        table = ACTION_TABLE[decision.kind]
        for opt in decision.options:
            m[table.index(str(opt))] = 1.0
    if m.sum() == 0:  # 兜底，避免 softmax 全零
        m[0] = 1.0
    return m


def action_index(decision: Decision, choice: int) -> int:
    """把「options 里的下标」换算成「策略头里的下标」。"""
    if decision.kind == "discard":
        return int(decision.options[choice])
    table = ACTION_TABLE[decision.kind]
    return table.index(str(decision.options[choice]))


def action_label(decision: Decision, choice: int) -> str:
    if decision.kind == "discard":
        return S.tile_cn(int(decision.options[choice]))
    return str(decision.options[choice])


__all__ = [
    "DIM", "OFFSET", "FEATURE_BLOCKS", "HEAD_SIZE", "encode", "visible_counts",
    "feature_blocks", "layout_for",
    "action_mask", "action_index", "action_label",
    "ACTION_DIM", "ACTION_OFFSET", "ACTION_BLOCKS", "action_features", "tile_danger",
]
