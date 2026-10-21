# ============================================================================
# ⚠️ 冻结副本（2026-10-21）—— ``git show HEAD:aimodel/encode.py`` 的逐字拷贝。
#
# 只为准 **486/560** 维老权重（maj-pro 系列 / mac_rl / ELO 历史）服务：
# **bug-for-bug 兼容**，包括 ``visible_counts`` 把「自己的副露 / 自己的拔北」
# 算两遍的旧 bug。改这里 = 老权重的输入分布变了 = ELO 历史不可复现 ✗
#
#   · 不许修 bug、不许加特征、不许同步新布局；
#   · 新特征一律在 ``aimodel/encode.py``；
#   · 按权重维度选 encoder 用 ``aimodel/feature_dispatch.py``。
#
# 本文件当前**不需要改 import**：它只依赖 sanma / engine / fastshanten，
# 没有从 ``.encode`` 反向依赖。
# ============================================================================
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
    "chi": 2,        # 四麻才有「吃」。老版本漏了这个键 -> 引擎
                     # engine.py:877 真的会问 chi，但 train.py:65 的
                     # `if kind in HEAD_SIZE` 判定失败 -> Tracker 里 0 条 chi
                     # 记录（实测 30 场四麻 chi 占 1541/12185 = 12.6% 全丢）。
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


# ── 学出来的危险度（--danger-model）────────────────────────────────
# 手写 `tile_danger` 的已知死穴：没人立直时基本不设防（只有「副露数/巡目」的
# 粗基线）。用天鳳牌谱 + 自对弈数据训一个二分类模型可以显著改善这一块 ——
# 实测（`aimodel/train_danger_selfplay.py`，60/20/20，验证集早停，测试集只碰一次）：
#
#     切片           自对弈        牌谱
#     overall        0.8832        0.6996
#     NO-RIICHI      0.8867        0.6838     <- 手写版的死穴
#     WITH-RIICHI    0.5317        0.5872
#
# 所以自对弈数据把「没人立直」这一片从 0.68 抬到 0.89；而「有人立直」那一片
# 牌谱更好（自对弈里只有 1.6% 的决策处于立直状态）。**两个来源互补**，
# 这也正是保留手写路径做 A/B 的意义。
#
# ⚠️ 输入布局的一致性**已验证**，不必再猜：
#   * `selfplay.py`  记的 `X` 就是 `encode(view)`（同文件 L211）
#   * `logdata.py`   抽的 `danger_X` 也是 `encode(view)`（L82）
#   * `danger_score` 调用的还是同一个 `encode(view)`
#   三处是**同一个函数**，所以布局逐位一致是**构造性成立**的，不是巧合。
#   我在接入时曾担心「View 现算的向量 ≠ 抽取时的向量」，那是多余的谨慎；
#   真正的风险在**人数/维度**（下面 danger_score 的两道校验）。
#
#   两道校验仍然保留，因为它们是防「喂错维度」的安全网 —— 而且这道网
#   当场抓出过一个真 bug：我起初拿模块级 `DIM` 去比，那是**三麻默认 486**，
#   于是合法的四麻模型（560）会被拒绝。

_DANGER_MODEL: dict | None = None


def load_danger_model(path: str) -> None:
    """载入 `--danger-model` 指到的 npz（w/b/mu/sd，均为训练时同一布局）。

    注意**不能**拿模块级的 ``DIM`` 去校验：那是三麻的默认布局（486），
    而四麻是 560。真实校验在 :func:`danger_score` 里按 ``view`` 的人数做 ——
    这里只记下模型期望的维度。
    """
    global _DANGER_MODEL
    data = np.load(path, allow_pickle=True)
    _DANGER_MODEL = {
        "w": data["w"].astype(np.float32),
        "b": float(data["b"]),
        "mu": data["mu"].astype(np.float32),
        "sd": data["sd"].astype(np.float32),
        "dim": int(data["dim"]),
        "players": int(data["players"]),
    }


def danger_model_loaded() -> bool:
    return _DANGER_MODEL is not None


def danger_score(view: View, tile: int) -> float:
    """危险度：有模型且**布局对得上**就用模型，否则退回手写 ``tile_danger``。

    两处校验缺一不可：

    * **人数**：模型是在四麻数据上训的，三麻局面不能喂给它；
    * **维度**：``encode(view)`` 的实际长度必须等于模型期望的维度。

    任何一条不满足就退回手写路径 —— 宁可慢，不可错。
    """
    if _DANGER_MODEL is None:
        return tile_danger(view, tile)
    if view.rule.players != _DANGER_MODEL["players"]:
        return tile_danger(view, tile)
    obs = encode(view)
    if obs.shape[0] != _DANGER_MODEL["dim"]:
        return tile_danger(view, tile)
    x = (obs.astype(np.float32) - _DANGER_MODEL["mu"][0]) / _DANGER_MODEL["sd"][0]
    z = float(x @ _DANGER_MODEL["w"] + _DANGER_MODEL["b"])
    return float(1.0 / (1.0 + np.exp(-max(-30.0, min(30.0, z)))))


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
        # 把已经算好的 s 作为 base 传进去，省掉 useful_tiles 里的重复 shanten。
        # 数值与「不传 base」完全一致（test_fastshanten 有等价性断言）。
        af[t, o["ukeire_after"]] = (
            min(ukeire_count(rest, vis, n_melds, sanma=sanma, base=s), 40) / 40.0
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
# v2：把「逐张牌的通道」和「标量」分开（表示重构的第一步）
# ---------------------------------------------------------------------------
# 为什么要动表示：`action_features` 实测占 `encode`+`action_features` 的 87%
# （1711µs 里 1565µs 是 ukeire_count），而它是**逐候选牌**算的、形状是
# (34, 13) 的扁平表。做成 (通道, 34) 张量之后：
#   * 「某张牌的所有信息」是同一列，卷积/共享权重才顺理成章
#   * 张量形状可以直接喂卷积，也才能谈 GPU
#
# **这个函数只产出表示，不接线** —— `net.py` / `train.py` 仍走老的 `encode()`。
# 接线要等 Mac 那边（它正跑着流水线，中途改特征会让在跑的 RL 基线说不清）。
#
# 张量轴序：``(C, 34)`` —— **通道在前**。第 c 个通道的第 t 个元素 = 第 c 项
# 关于牌 t 的值。通道含义见 :data:`ENCODE_V2_CHANNELS`（顺序即行号）。

#: ``(C, 34)`` 里每个通道的含义，**顺序即通道号**。
#: 写进这里是为了让下一个人不用对着 (16, 34) 猜哪个通道是什么。
ENCODE_V2_CHANNELS: tuple[str, ...] = (
    "hand",            # 自己的手牌计数 /4
    "hand_onehot",     # 手上有这张（>=1）
    "hand_pair",       # 手上两张以上（>=2）
    "hand_triple",     # 手上三张以上（>=3）
    "melds",           # 自己副露里出现的牌 /3
    "own_pond",        # 自己牌河 /4
    "opp1_pond",       # 下家牌河 /4
    "opp2_pond",       # 对家牌河 /4
    "opp3_pond",       # 上家牌河 /4（三麻时这是空通道，见 note）
    "opp1_melds",      # 下家副露 /3
    "opp2_melds",      # 对家副露 /3
    "opp3_melds",      # 上家副露 /3（三麻时空）
    "visible",         # 该玩家能看到的每种牌总数（含自己手牌）/4
    "dora_indicator",  # 宝牌**指示**牌 one-hot
    "dora_tile",       # 宝牌 one-hot
    "yakuhai",         # 见下方 note：这不是逐张牌的项，占位保留以便对齐
)

#: 三麻只有 3 家，`opp3_*` 两个通道恒为 0。
#: **这是有意的**：通道数固定成 16，三麻/四麻用同一套网络结构，
#: 免得再加一个「按人数分叉」的维度（加头/加维都会搞挂存量权重，踩过）。


def _tile_channel_names(players: int) -> list[str]:
    """从 :func:`feature_blocks` 里挑出「宽度 34 且逐张牌有独立值」的块。

    **不写死列表** —— 写死的话，将来有人往 `feature_blocks` 里插一个新块，
    这里会静默错位（通道和含义对不上，而且看不出来）。

    ``yakuhai`` 是宽度 34 但**不是**逐张牌的（它按 5 个役牌种类记数、
    只用了前 5 格），所以这里必须把它排除掉 —— 否则它会占一个通道名、
    而真实的 `yakuhai` 数据既进不了 tile 也进不了 scalar，整块丢失 34 维。
    """
    return [n for n, w in feature_blocks(players)
            if w == 34 and n != "yakuhai"]


#: 三麻只有 3 家，`opp3_*` 两个通道恒为 0 **且布局里没有这两个块**。
#: 通道数固定成 16 是为了三麻/四麻共用一套网络结构（避免按人数分叉）。
_CHANNELS_ABSENT_IN_3P = frozenset({"opp3_pond", "opp3_melds"})


def encode_v2(view: View) -> tuple[np.ndarray, np.ndarray]:
    """观测 -> ``(tile, scalar)``。

    * ``tile``   : ``(C, 34)`` float32，``C == len(ENCODE_V2_CHANNELS)``，
                   通道含义见 :data:`ENCODE_V2_CHANNELS`
    * ``scalar`` : ``(S,)`` float32，剩下那些跟具体哪张牌无关的项
                   （向听 one-hot / 巡目 / 点数 / 立直标记 …）

    **等价性由构造保证**：``tile`` 是从 :func:`encode` 的输出里按块切出来的，
    不是把公式再抄一遍 —— 抄一遍迟早会和 `encode()` 漂移。
    ``scalar`` 同理。所以这个函数和 `encode()` 的关系是「同一次计算的两种排布」，
    由 ``tests/test_encode_v2.py`` 逐通道验证。

    注意：`encode()` 里有两项**不是**逐张牌的（`yakuhai` 5 宽、`shanten_onehot` 8 宽），
    它们落在 `scalar` 里；`yakuhai` 在 `ENCODE_V2_CHANNELS` 里的占位通道恒为 0，
    只是为了让通道号和含义表一一对应、便于阅读。
    """
    players = view.rule.players
    O, dim = layout_for(players)
    x = encode(view)

    names = set(_tile_channel_names(players))
    tile = np.zeros((len(ENCODE_V2_CHANNELS), 34), dtype=np.float32)
    for i, name in enumerate(ENCODE_V2_CHANNELS):
        if name == "yakuhai":
            continue            # 非逐张牌项，占位恒 0
        if name in _CHANNELS_ABSENT_IN_3P and players == 3:
            continue            # 三麻没有第 4 家 -> 布局里没这两个块
        if name in names:
            a = O[name]
            tile[i] = x[a : a + 34]

    # 标量 = 除了「逐张牌那 34 宽」之外的所有块，按 feature_blocks 的顺序拼。
    # 注意 `yakuhai`（宽 34 但非逐张牌）要**留在这里**，它得进 scalar。
    parts: list[np.ndarray] = []
    for n, w in feature_blocks(players):
        if n in names:
            continue
        a = O[n]
        parts.append(np.asarray(x[a : a + w], dtype=np.float32))
    scalar = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return tile, scalar


#: v2 的标量块数量（供 `net.py` 的 meta 用；改布局时这个数会变）
ENCODE_V2_SCALAR_DIM: dict[int, int] = {
    p: sum(w for n, w in feature_blocks(p)
           if n not in set(_tile_channel_names(p)))
    for p in (3, 4)
}


# ---------------------------------------------------------------------------
# 动作掩码
# ---------------------------------------------------------------------------


def option_head_index(kind: str, option) -> int | None:
    """选项 -> 策略头下标；对不上返回 ``None``。

    正常情况下选项就是 :data:`ACTION_TABLE` 里的字面量（``"pass"`` / ``"pon"``
    / ``"chi"`` …）。这里额外容忍 ``"chi:10:12"`` / ``"pon:5p"`` 这种
    「逐吃法 / 带牌信息」的写法 —— 只看冒号前的动作名。

    为什么要容忍：引擎现在给的是字面量，但外部调用方（回放、UI、别的引擎）
    完全可能把**每一种吃法**作为独立选项传进来。老代码在 ``action_mask`` 里
    直接 ``table.index(str(opt))``，遇到这种写法会抛 ``ValueError``；而
    ``MLAgent._head_to_option`` 更糟，找不到就静默返回 0（= pass），
    表现就是「永远不鸣牌」。
    """
    table = ACTION_TABLE[kind]
    s = str(option)
    try:
        return table.index(s)
    except ValueError:
        pass
    try:
        return table.index(s.split(":", 1)[0])
    except ValueError:
        return None


def action_mask(decision: Decision) -> np.ndarray:
    """该决策下每个策略头位置的合法性。"""
    size = HEAD_SIZE[decision.kind]
    m = np.zeros(size, dtype=np.float32)
    if decision.kind == "discard":
        for t in decision.options:
            m[int(t)] = 1.0
    else:
        for opt in decision.options:
            head = option_head_index(decision.kind, opt)
            if head is None:
                raise ValueError(
                    f"{decision.kind} 的选项 {opt!r} 不在动作表 "
                    f"{ACTION_TABLE[decision.kind]} 里")
            m[head] = 1.0
    if m.sum() == 0:  # 兜底，避免 softmax 全零
        m[0] = 1.0
    return m


def action_index(decision: Decision, choice: int) -> int:
    """把「options 里的下标」换算成「策略头里的下标」。"""
    if decision.kind == "discard":
        return int(decision.options[choice])
    head = option_head_index(decision.kind, decision.options[choice])
    if head is None:
        raise ValueError(
            f"{decision.kind} 的选项 {decision.options[choice]!r} 不在动作表 "
            f"{ACTION_TABLE[decision.kind]} 里")
    return head


def action_label(decision: Decision, choice: int) -> str:
    if decision.kind == "discard":
        return S.tile_cn(int(decision.options[choice]))
    return str(decision.options[choice])


__all__ = [
    "DIM", "OFFSET", "FEATURE_BLOCKS", "HEAD_SIZE", "encode", "visible_counts",
    "feature_blocks", "layout_for",
    "encode_v2", "ENCODE_V2_CHANNELS", "ENCODE_V2_SCALAR_DIM",
    "action_mask", "action_index", "action_label",
    "ACTION_DIM", "ACTION_OFFSET", "ACTION_BLOCKS", "action_features", "tile_danger",
]
