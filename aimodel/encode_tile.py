"""dot V1：**per-tile 特征张量**编码器（路线书 v3.0，F = 20）。

为什么另起一个文件、不动 :mod:`aimodel.encode`
--------------------------------------------------
生产在用的 645 维 ``encode()`` 是**平铺全局向量**：手牌、牌河、宝牌全被
摊成一维，卷积网络看不到「哪张牌在第几巡被打出」。这个模块只做新路线
需要的那件事：

    view -> (tiles[34, 20], globals[21])

* ``tiles[t, c]``：第 ``t`` 种牌的第 ``c`` 个通道（**牌在前、通道在后**）。
  34 个牌位就是 Conv1d 的「长度」维，通道是网络输入通道。
* ``globals``：与具体哪张牌无关的标量（巡目 / 剩牌 / 向听 / 进张 / 点数 / 场况），
  **不塞进 34×F**，由网络在最后 FC 前 concat。

只使用该玩家看得见的信息
--------------------------
``View`` 本身就不含别家手牌；这里再保证所有通道只读下列字段：

* 自己的：``hand`` / ``hand_red`` / ``melds`` / ``drawn`` / ``furiten``
* 公开的：``discards``（牌河）/ ``melds_all``（副露）/ ``kita`` /
  ``dora_indicators`` / ``riichi`` / ``ippatsu`` / ``scores`` / ``wall_left`` /
  ``turn`` / 场风自风 / ``kyoku`` / ``honba`` / ``kyotaku`` / ``is_dealer`` …

``tests/test_encode_tile.py`` 会把「改别家手牌 -> 编码逐位不变」和
「改自己手牌 -> 编码必须变」同时钉死（后者证明检查不是恒真）。

通道表（顺序即通道号，改顺序 = 改语义，金值会红）
--------------------------------------------------
====================  ======================================================
#   名称                含义（全部 0~1）
====================  ======================================================
0   hand               自己手牌张数 /4
1   visible            自己可见张数 /4（含手牌/副露/牌河/宝牌指示/拔北，修过 double-count）
2   remaining          场上剩余 = max(0, 4 - visible) /4
3   dora               是否宝牌（由宝牌指示牌推出）0/1
4   aka                自己手里/副露里是否有这张的赤五 0/1
5   yakuhai            是否役牌（场风/自风/三元；三麻按 rule.north_yakuhai）0/1
6   terminal_honor     是否幺九 / 字牌 0/1
7   own_pond           自己牌河里这张的张数 /4
8   own_last_discard   自己最近一次打的是这张 0/1
9   furiten_wait       自己振听且这张是自己的和牌张 0/1
10  opp1_pond          下家（座位序第一个对手）牌河张数 /4
11  opp2_pond          第二个对手牌河张数 /4
12  opp3_pond          第三个对手牌河张数 /4（三麻恒 0）
13  opp1_last_discard  第一个对手最近一次打的是这张 0/1
14  opp2_last_discard  第二个对手最近一次打的是这张 0/1
15  opp3_last_discard  第三个对手最近一次打的是这张 0/1（三麻恒 0）
16  riichi_danger      对已立直家的危险度（tile_danger 的立直分支取最大）0~1
17  genbutsu           对**每个**立直家都是现物 0/1（无立直家 = 0）
18  opp_meld           这张牌是否在**任意**对手副露里 0/1
19  suji               对已立直家的筋覆盖度：现物 1.0 / 筋 0.5 / 其它 0，取平均 0~1
====================  ======================================================

对手顺序固定为**座位下标升序、去掉自己**（与 ``encode()`` 的
``opp1_pond`` 等块同一约定）。三麻只有两个对手，第 3 个对手的两个通道恒 0；
通道数**固定 20**，三麻/四麻共用同一套网络结构。

全局标量（G = 21，顺序即下标）
------------------------------
0 turn/20 · 1 wall_left/70 · 2 hand_size/14 · 3 shanten/6 · 4 ukeire/40 ·
5 dora_count/6 · 6 aka_count/2 · 7-10 scores(4)/50000 · 11 score_lead ·
12 (kyoku-1)/7 · 13 honba/4 · 14 kyotaku/3 · 15 hands_left/8 · 16 furiten ·
17 dealer_repeat · 18 is_dealer · 19 seat_wind/3 · 20 round_wind(0/1)

路线书写的是「约 20 维」；这里把 ``seat/round wind`` 拆成两个标量，
所以是 21 维。``shanten_onehot(8→取 shanten 值/6)`` 取的是后半句：
一个 0~1 的向听标量（-1 和牌也记为 0）。

跑法::

    .venv/Scripts/python.exe -c "from aimodel.encode_tile import encode_tile; ..."
"""

from __future__ import annotations

import numpy as np

from . import sanma as S
from .encode import visible_counts
from .engine import View, dora_from_index
from .fastshanten import shanten, ukeire_count, waits

# ---------------------------------------------------------------------------
# 通道 / 标量表
# ---------------------------------------------------------------------------

#: ``tiles[34, F]`` 的通道名，顺序即通道号。
TILE_CHANNELS: tuple[str, ...] = (
    "hand",              # 0
    "visible",           # 1
    "remaining",         # 2
    "dora",              # 3
    "aka",               # 4
    "yakuhai",           # 5
    "terminal_honor",    # 6
    "own_pond",          # 7
    "own_last_discard",  # 8
    "furiten_wait",      # 9
    "opp1_pond",         # 10
    "opp2_pond",         # 11
    "opp3_pond",         # 12
    "opp1_last_discard", # 13
    "opp2_last_discard", # 14
    "opp3_last_discard", # 15
    "riichi_danger",     # 16
    "genbutsu",          # 17
    "opp_meld",          # 18
    "suji",              # 19
)

#: 全局标量名，顺序即下标。
GLOBAL_FEATURES: tuple[str, ...] = (
    "turn", "wall_left", "hand_size", "shanten", "ukeire", "dora_count",
    "aka_count", "score0", "score1", "score2", "score3", "score_lead",
    "kyoku", "honba", "kyotaku", "hands_left", "furiten", "dealer_repeat",
    "is_dealer", "seat_wind", "round_wind",
)

F: int = len(TILE_CHANNELS)
G: int = len(GLOBAL_FEATURES)

#: 便捷别名（测试 / 网络侧按这个名字取维度）
NUM_TILE_CHANNELS = F
NUM_GLOBALS = G

# 通道下标（不要手写魔法数字）
_HAND = TILE_CHANNELS.index("hand")
_VISIBLE = TILE_CHANNELS.index("visible")
_REMAINING = TILE_CHANNELS.index("remaining")
_DORA = TILE_CHANNELS.index("dora")
_AKA = TILE_CHANNELS.index("aka")
_YAKUHAI = TILE_CHANNELS.index("yakuhai")
_TERMINAL = TILE_CHANNELS.index("terminal_honor")
_OWN_POND = TILE_CHANNELS.index("own_pond")
_OWN_LAST = TILE_CHANNELS.index("own_last_discard")
_FURITEN_WAIT = TILE_CHANNELS.index("furiten_wait")
_OPP_POND = tuple(TILE_CHANNELS.index(f"opp{i}_pond") for i in (1, 2, 3))
_OPP_LAST = tuple(TILE_CHANNELS.index(f"opp{i}_last_discard") for i in (1, 2, 3))
_DANGER = TILE_CHANNELS.index("riichi_danger")
_GENBUTSU = TILE_CHANNELS.index("genbutsu")
_OPP_MELD = TILE_CHANNELS.index("opp_meld")
_SUJI = TILE_CHANNELS.index("suji")

#: 对手槽位数（三麻只用前 2 个，第 3 个恒 0）
_OPP_SLOTS = 3


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _others(view: View) -> list[int]:
    """对手座位，**按座位下标升序**（与 encode() 的 opp1/2/3 一致）。"""
    return [p for p in range(view.rule.players) if p != view.player]


def _is_yakuhai(view: View, tile: int) -> bool:
    """场风 / 自风 / 三元。三麻的北是否役牌由 rule.north_yakuhai 决定。"""
    if tile in S.DRAGONS:
        return True
    if tile == view.round_wind or tile == view.seat_wind:
        return True
    return bool(getattr(view.rule, "north_yakuhai", False) and tile == S.NORTH)


def _is_terminal_or_honor(tile: int) -> bool:
    return tile >= 27 or tile % 9 in (0, 8)


def _own_aka_flags(view: View) -> np.ndarray:
    """自己手里 / 副露里每种牌是否有赤五（对手牌河不记赤，看不到就不编）。"""
    flags = np.asarray(view.hand_red, dtype=np.float32) > 0
    flags = np.asarray(flags, dtype=bool).copy()
    for m in view.melds:
        if m.aka <= 0:
            continue
        for t in m.tiles:
            # 赤五只可能是 5m/5p/5s（下标 %9 == 4）；吃的话只有中间那张能是赤
            if t % 9 == 4:
                flags[t] = True
    return flags


def _wait_mask(view: View) -> np.ndarray:
    """自己是否在等这张（只在 ``view.furiten`` 为真时调用，省时间）。

    决策点的手牌通常是 14 张（刚摸牌）。听牌张要按**打掉一张之后的 13 张**
    来算：优先去掉 ``view.drawn``（摸切直觉），``drawn`` 缺失时去掉第一张
    非空牌。然后复用 ``fastshanten.waits`` —— 它只在向听 0 时返回和牌张。
    """
    hand = list(view.hand)
    if sum(hand) % 3 == 2:
        drawn = view.drawn
        if drawn is not None and 0 <= int(drawn) < 34 and hand[int(drawn)] > 0:
            hand[int(drawn)] -= 1
        else:
            for t in range(34):
                if hand[t] > 0:
                    hand[t] -= 1
                    break
    mask = np.zeros(34, dtype=bool)
    if sum(hand) % 3 != 1:
        return mask
    n_melds = len(view.melds)
    if shanten(hand, n_melds) != 0:
        return mask
    for t in waits(hand, n_melds, sanma=view.rule.players == 3):
        if hand[t] < 4:
            mask[t] = True
    return mask


def _riichi_danger(
    view: View, tile: int, riichi_opp: list[int], vis: np.ndarray
) -> float:
    """只对**已立直**家的危险度（0~1），取各家最大。

    为什么没有直接调 ``tile_danger(view, tile, opp_riichi=...)``：
    那个函数在算完立直家之后还会把**没立直家**的 open_risk 也取 max 混进来
    （它是生产版的「总危险度」）。通道 16 要的是「对已立直家」这一块，
    通道 17/19 又单独描述现物/筋，混进 open_risk 会让三个通道语义打架。

    这里逐行复刻 ``tile_danger`` 的立直分支（同一套常数、同一个筋折扣），
    单测里在「所有对手都立直」时与 ``tile_danger`` 逐位对齐 —— 万一上游
    改了立直危险度公式，这条测试会红，提醒同步这里。
    """
    if not riichi_opp:
        return 0.0
    # 这张牌本身有多好打（与 tile_danger 完全一致）
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

    best = 0.0
    for q in riichi_opp:
        pond = view.discards[q]
        if tile in pond:
            continue  # 现物
        pset = set(pond)
        dd = d
        for delta in (-3, 3):
            r = tile + delta
            if 0 <= r < 34 and r // 9 == tile // 9 and r in pset:
                dd *= 0.65  # 筋
                break
        if dd > best:
            best = dd
    return min(best, 1.0)


def _suji_coverage(view: View, tile: int, riichi_opp: list[int]) -> float:
    """相对立直家牌河的筋覆盖度：现物 1.0 / 筋 0.5 / 其它 0，取平均。"""
    if not riichi_opp:
        return 0.0
    acc = 0.0
    for q in riichi_opp:
        pond = view.discards[q]
        if tile in pond:
            acc += 1.0
        elif tile < 27:  # 字牌没有筋
            suit = tile // 9
            for r in (tile - 3, tile + 3):
                if 0 <= r < 34 and r // 9 == suit and r in pond:
                    acc += 0.5
                    break
    return acc / len(riichi_opp)


def _count_dora(view: View, dora_types: set[int]) -> int:
    """自己手牌 + 副露 + 拔北里的宝牌张数（拔北按 rule.kita_dora）。"""
    n = 0
    for t in range(34):
        if t in dora_types:
            n += int(view.hand[t])
    for m in view.melds:
        for t in m.tiles:
            if t in dora_types:
                n += 1
    if bool(getattr(view.rule, "kita_dora", False)):
        n += int(view.kita[view.player])
    return n


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def encode_tile(view: View) -> tuple[np.ndarray, np.ndarray]:
    """把观测编码成 ``(tiles[34, F], globals[G])``，均为 float32。

    * ``tiles`` 轴序 **（牌, 通道）**：``tiles[t, c]``。喂网络时转置成
      ``(F, 34)``（Conv1d 要求通道在前）。
    * ``globals`` 是一维标量，**不**塞进 tiles；网络在最后 FC 前 concat。

    三麻/四麻都支持：三麻的第 3 个对手通道恒 0，F 仍为 20。
    """
    n_players = int(view.rule.players)
    if n_players not in (3, 4):
        raise ValueError(f"只支持三麻/四麻，收到 players={n_players}")

    hand = view.hand
    n_melds = len(view.melds)
    sanma = n_players == 3
    others = _others(view)

    vis = np.asarray(visible_counts(view), dtype=np.float32)
    dora_types = {dora_from_index(d, view.rule) for d in view.dora_indicators}
    aka_flags = _own_aka_flags(view)

    # 牌河计数一次算好（34 个牌位 × 每家的牌河长度很小，但别在 34 循环里
    # 反复 list.count —— 那会变成 34 * 3 * pond_len 次扫描）
    pond_counts = np.zeros((n_players, 34), dtype=np.int32)
    pond_last = [-1] * n_players
    for q in range(n_players):
        pond = view.discards[q]
        for t in pond:
            pond_counts[q, int(t)] += 1
        if pond:
            pond_last[q] = int(pond[-1])

    # 对手副露里出现过的牌（只看副露，**不看**对手手牌）
    opp_meld_hit = np.zeros(34, dtype=bool)
    for q in others:
        for m in view.melds_all[q]:
            for t in m.tiles:
                opp_meld_hit[int(t)] = True

    riichi_opp = [q for q in others if view.riichi[q]]
    wait_mask = _wait_mask(view) if bool(view.furiten) else None

    own_player = view.player
    own_pond_counts = pond_counts[own_player]
    own_last = pond_last[own_player]

    tiles = np.zeros((34, F), dtype=np.float32)
    for t in range(34):
        tiles[t, _HAND] = min(int(hand[t]), 4) / 4.0
        tiles[t, _VISIBLE] = min(float(vis[t]), 4.0) / 4.0
        tiles[t, _REMAINING] = max(0.0, 4.0 - float(vis[t])) / 4.0
        tiles[t, _DORA] = 1.0 if t in dora_types else 0.0
        tiles[t, _AKA] = 1.0 if aka_flags[t] else 0.0
        tiles[t, _YAKUHAI] = 1.0 if _is_yakuhai(view, t) else 0.0
        tiles[t, _TERMINAL] = 1.0 if _is_terminal_or_honor(t) else 0.0
        tiles[t, _OWN_POND] = min(int(own_pond_counts[t]), 4) / 4.0
        tiles[t, _OWN_LAST] = 1.0 if t == own_last else 0.0
        tiles[t, _FURITEN_WAIT] = (
            1.0 if (wait_mask is not None and wait_mask[t]) else 0.0
        )

        for slot in range(_OPP_SLOTS):
            if slot >= len(others):
                continue
            q = others[slot]
            tiles[t, _OPP_POND[slot]] = min(int(pond_counts[q, t]), 4) / 4.0
            tiles[t, _OPP_LAST[slot]] = 1.0 if t == pond_last[q] else 0.0

        tiles[t, _DANGER] = _riichi_danger(view, t, riichi_opp, vis)
        tiles[t, _GENBUTSU] = (
            1.0
            if (riichi_opp and all(t in view.discards[q] for q in riichi_opp))
            else 0.0
        )
        tiles[t, _OPP_MELD] = 1.0 if opp_meld_hit[t] else 0.0
        tiles[t, _SUJI] = _suji_coverage(view, t, riichi_opp)

    if sanma:
        # 三麻的 2m..8m 物理上不存在（SANMA_TYPES 只留 1m/9m）。
        # 34 个牌位仍保留（网络结构固定），但这些行的特征必须是 0 ——
        # 否则 remaining 会显示「还剩 4 张」，模型会以为有牌可摸。
        valid = set(S.SANMA_TYPES)
        for t in range(34):
            if t not in valid:
                tiles[t, :] = 0.0

    # ------------------------------------------------------------------
    # 全局标量
    # ------------------------------------------------------------------
    g = np.zeros(G, dtype=np.float32)
    sh = shanten(hand, n_melds)
    uke = ukeire_count(hand, vis, n_melds, sanma=sanma)
    dora_n = _count_dora(view, dora_types)
    aka_n = int(sum(view.hand_red)) + sum(int(m.aka) for m in view.melds)
    scores = [int(s) for s in view.scores]
    scores4 = (scores + [0, 0, 0, 0])[:4]
    best_score = max(scores) if scores else 0

    g[0] = min(max(int(view.turn), 0), 20) / 20.0
    g[1] = min(max(int(view.wall_left), 0), 70) / 70.0
    g[2] = min(max(int(view.hand_size), 0), 14) / 14.0
    g[3] = min(max(int(sh), 0), 6) / 6.0
    g[4] = min(max(int(uke), 0), 40) / 40.0
    g[5] = min(max(int(dora_n), 0), 6) / 6.0
    g[6] = min(max(int(aka_n), 0), 2) / 2.0
    g[7:11] = np.asarray(scores4, dtype=np.float32) / 50000.0
    if scores:
        lead = (scores[own_player] - best_score) / 50000.0
        g[11] = min(max(lead, -1.0), 1.0)
    g[12] = min(max(int(getattr(view, "kyoku", 1)) - 1, 0), 7) / 7.0
    g[13] = min(max(int(getattr(view, "honba", 0)), 0), 4) / 4.0
    g[14] = min(max(int(getattr(view, "kyotaku", 0)), 0), 3) / 3.0
    g[15] = min(max(int(getattr(view, "hands_left", 0)), 0), 8) / 8.0
    g[16] = 1.0 if bool(view.furiten) else 0.0
    g[17] = 1.0 if bool(getattr(view, "dealer_repeat", False)) else 0.0
    g[18] = 1.0 if bool(view.is_dealer) else 0.0
    # 自风：东=0 / 南=1/3 / 西=2/3 / 北=1；场风：东=0 / 南=1
    g[19] = min(max((int(view.seat_wind) - int(S.EAST)) / 3.0, 0.0), 1.0)
    g[20] = min(max((int(view.round_wind) - int(S.EAST)) / 1.0, 0.0), 1.0)

    return tiles, g


__all__ = [
    "TILE_CHANNELS", "GLOBAL_FEATURES", "F", "G",
    "NUM_TILE_CHANNELS", "NUM_GLOBALS", "encode_tile",
]
