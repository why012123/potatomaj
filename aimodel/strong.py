"""``StrongAgent`` —— 纯规则的四麻/三麻强化 agent（不加载任何神经网络）。

## 为什么要单独写一个「强规则」agent

线上神经网络（``maj-pro4v1.1_mac_rl``）被用户吐槽「一直无役」。
引擎不会让无役的牌和牌，所以真正的病灶不是「算番坏了」，而是：

> **先鸣牌（吃/碰）把自己鸣进一个无役形状，之后既不能和、又要防守，白丢一手。**

`HeuristicAgent`（教师）的鸣牌判断只有一句「向听降了就鸣 / 役牌就碰」，
**完全不检查鸣完之后还有没有役**。本模块的第一条硬规则就是补上它：

    「任何鸣牌前，先判断鸣完之后这手牌还有没有可能做出**真役**
      （断幺 / 役牌 / 混一色 / 清一色 / 混全带幺 / 对对和 /
        七对子 / 一气 / 三色 …，宝牌不算役）。
      没有明确役种路线 -> 不鸣。」

## 四轴打分（受 mahjong-helper 的「牌效 / 打点 / 危险度 / 押し引き」启发）

1. **牌效率**：向听 + **真实剩余枚数**的进张。
   教师用的 :func:`aimodel.encode.visible_counts` 有一个隐蔽的重复计数：
   它先加 ``view.melds``，又在遍历 ``view.melds_all`` 时把**自己的副露**再加一遍。
   本模块自己算 ``_visible_true``（手牌 + 自家副露 + 所有牌河 + 他家副露 +
   宝牌指示 + 拔北），避免把「自己已经见过的牌」多算一次。
2. **打点/役**：不只看向听。听牌时用 :func:`aimodel.sanma.evaluate_hand`
   对**每一个和了张**真算番（所以断幺/平和/一气/三色/混全/对对/七对/役牌
   全部自动覆盖）；没听牌时用结构化的役种路线估计（见 :meth:`_yaku_info`）。
3. **危险度**：复用 :func:`aimodel.encode.tile_danger`（现物 / 筋 / 字牌壁 /
   染手压力），再补两条它没有的：**四张全见 = 绝对安全**、**中张剩一张时打折**。
4. **押し引き**：不是一个固定阈值。用「向听 / 打点 / 听牌形状」算进攻力，
   用「立直家数 / 副露数 / 巡目 / 剩余牌数」算威胁，再叠加亲家、点数顺位，
   得到连续的 ``fold_urgency``；弃和时危险度权重会压过一切，进攻时则按
   **近似点数**权衡「一向听值多少分」。

## 性能与确定性

* 稳态下单次 ``decide`` 平均约 **3ms**（见 ``tests/test_strong.py`` 的性能打印）；
  向听/进张有全局缓存，听牌时才逐张 ``evaluate_hand``（约 50µs/张）。
* 冷缓存（进程里第一次遇到某种复杂牌型）时，``fastshanten`` 要枚举新的
  花色块，可能超过 50ms。为此每次打牌最多只对 ``MAX_EXACT_CANDIDATES``
  个候选做精确听牌分析，其余候选按向听计分；实测四麻整局最慢单次已压到
  50ms 附近（三麻冷启动偶尔更高，见测试打印）。
* **完全确定性**：不使用随机数；同分时按选项下标取最小，方便同牌山复现。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence

from . import sanma as S
from .agents import HeuristicAgent
from .encode import tile_danger
from .engine import Decision, View, dora_from_index
from .fastshanten import shanten, useful_tiles, ukeire_count

# ---------------------------------------------------------------------------
# 常量 / 小工具
# ---------------------------------------------------------------------------

#: 一个「向听」在打分里大约值多少点。麻将里一向听的价值大致就是满贯级别
#: （多一个面子/雀头的完成度），所以用 9000 点做基准；它不是精确 EV，
#: 只是让「掉一向听换大牌」有一个可解释的换算率。
SHANTEN_POINTS = 9000.0

#: 每一张真实剩余进张值多少点（同向听内比较用）。
UKEIRE_POINTS = 22.0

#: 危险度 1.0 在「有人立直」时大概对应多少点期望损失。
DANGER_POINTS = 6000.0

#: 没听牌时，潜力打点的「兑现率」—— 向听越远，潜力越难变成真金白银。
REALIZATION = {0: 1.0, 1: 0.62, 2: 0.38, 3: 0.20, 4: 0.10}

#: 每次打牌最多对几张「最优向听」的候选做**精确**听牌/进张分析。
#: 14 张全不同的手最多有 14 个选项；逐个算 evaluate_hand 在冷缓存下会超过
#: 50ms。这里用「打点损失 + 孤张程度 + 危险度」先筛一批，剩下的候选只按
#: 向听计分。6 个足够覆盖真正的选择，实测把冷启动最坏耗时压下来。
MAX_EXACT_CANDIDATES = 6


def _points_of_han(han: float) -> float:
    """番数 -> 近似点数（用于打分，不是实际结算）。

    取常见的满贯界限：1/2/3 番分别约 1000/2000/3900，4~5 番满贯 8000，
    6~7 跳满 12000，8~10 倍满 16000，11~12 三倍满 24000，13+ 役满 32000。
    """
    if han <= 0:
        return 0.0
    if han < 2:
        return 1000.0
    if han < 3:
        return 2000.0
    if han < 4:
        return 4000.0
    if han < 6:
        return 8000.0
    if han < 8:
        return 12000.0
    if han < 11:
        return 16000.0
    if han < 13:
        return 24000.0
    return 32000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _is_terminal_or_honor(t: int) -> bool:
    return t >= 27 or t % 9 in (0, 8)


def _is_simple(t: int) -> bool:
    """断幺九用的「中张」：2~8 的数牌。"""
    return t < 27 and t % 9 not in (0, 8)


def _is_wind(t: int) -> bool:
    return 27 <= t <= 30


def _is_dragon(t: int) -> bool:
    return t >= 31


def _chi_pairs(tile: int) -> set[tuple[int, int]]:
    """吃 ``tile`` 时，手里另外两张的所有组合（牌型下标）。

    和 :func:`aimodel.styles._chi_pairs` 同一张表；这里复制一份是为了让
    :mod:`aimodel.strong` 不依赖 ``styles`` 的私有函数（导入私有名很脆）。
    """
    if tile >= 27:
        return set()
    suit0 = (tile // 9) * 9
    out: set[tuple[int, int]] = set()
    for lo in (tile - 2, tile - 1, tile):
        hi = lo + 2
        if lo < suit0 or hi >= suit0 + 9:
            continue
        others = [x for x in (lo, lo + 1, hi) if x != tile]
        if len(others) == 2:
            out.add((others[0], others[1]))
    return out


# ---------------------------------------------------------------------------
# 役种分析
# ---------------------------------------------------------------------------


@dataclass
class YakuInfo:
    """一手牌（可以还没听）的役种路线。

    ``routes`` 里的翻数是**完成后的近似翻数**（副露降番已经算进去），
    只用于价值估计；真正的番数在听牌时交给 :func:`sanma.evaluate_hand`。
    """

    closed: bool
    guaranteed: int          #: 现在已经确定的真役（役牌刻子等）
    routes: dict[str, int]   #: 可能做出的役 -> 近似翻数
    dora: int                #: 宝牌 + 赤 + 拔北（**不算役**）
    best_han: int            #: 真役的最佳组合（不含宝牌）
    has_yaku: bool           #: 有没有任何真役路线（宝牌不算）
    reason: str = ""         #: 给测试/日志看的说明

    @property
    def route_names(self) -> list[str]:
        return sorted(self.routes)


def _meld_yaku_han(meld: S.Meld, round_wind: int, seat_wind: int) -> int:
    """一组已完成的刻子/杠本身确定的役牌翻数（吃 = 0）。"""
    if meld.is_chi:
        return 0
    t = meld.tile
    han = 0
    if _is_dragon(t):
        han += 1
    if t == round_wind:
        han += 1
    if t == seat_wind:
        han += 1
    return han


def _chi_has_terminal(meld: S.Meld) -> bool:
    """吃的顺子是否含幺九（混全带幺用）。"""
    if not meld.is_chi:
        return False
    start = meld.tile
    n = start % 9
    return n == 0 or n == 6  # 123 或 789


def _tanyao_feasible(hand: Sequence[int], melds: Sequence[S.Meld],
                     all_counts: Sequence[int]) -> bool:
    """断幺九路线是否还成立。

    不是「现在一张幺九都没有」才算 ✓ —— 手里有一两张**孤立**的幺九，
    后面打掉就能转断幺；教师会吃这种牌，而太严的硬规则会把整手速攻
    全部拒掉。所以这里放宽成：

      * 副露里不能有幺九（副露打不掉）；
      * 手里最多 2 张幺九/字牌，而且都必须是**孤立**的（不挨着任何
        可以组成顺子的牌、也不是对子/刻子），否则就是在拆面子。

    这个放宽**不会**让「123m 456p 789s 22p」这种已经成型的牌通过：
    它的 1m、9s 都在顺子里，不属于「孤立可打」。
    """
    for m in melds:
        if any(not _is_simple(t) for t in m.tiles):
            return False
    bad = [t for t in range(34) if all_counts[t] and not _is_simple(t)]
    if not bad:
        return True
    if len(bad) > 2:
        return False
    for t in bad:
        if hand[t] >= 2:          # 对子/刻子：打掉要拆形
            return False
        if t >= 27:               # 孤立字牌：可以打
            continue
        n = t % 9
        if n == 0:
            if hand[t + 1] or hand[t + 2]:
                return False
        else:
            if hand[t - 1] or hand[t - 2]:
                return False
    return True


def _analyze_yaku(
    hand: Sequence[int],
    melds: Sequence[S.Meld],
    *,
    round_wind: int,
    seat_wind: int,
) -> YakuInfo:
    """结构化地判断「这手牌还可能做出哪些真役」。

    为什么不用 ``evaluate_hand`` 直接算？
        13/11 张牌还没和，``evaluate_hand`` 只接受和牌形，会直接返回 None。
        所以这里按**役种的结构条件**判断「路线是否存在」；真正听牌后再用
        ``evaluate_hand`` 精确算番。两套口径互相校验（测试里有）。

    覆盖的役（硬规则需要的最小集合）：
        立直、断幺、役牌、混一色、清一色、混全带幺、纯全带幺、对对和、
        七对子、一气通贯、三色同顺、平和、混老头、小三元。
    """
    n = 34
    hand = list(hand)
    melds = list(melds)
    closed = all(m.is_concealed for m in melds)

    # 全部可见于自己手牌+副露的牌（用于染手/断幺/混老头）
    all_counts = [0] * n
    for i in range(n):
        all_counts[i] = hand[i]
    for m in melds:
        for t in m.tiles:
            all_counts[t] += 1
    total_tiles = sum(all_counts)

    routes: dict[str, int] = {}
    guaranteed = 0

    # ---- ① 已经确定的役牌刻子（手牌里的刻子 + 副露里的碰/杠）----
    for m in melds:
        guaranteed += _meld_yaku_han(m, round_wind, seat_wind)
    for t in range(n):
        if hand[t] >= 3:
            if _is_dragon(t):
                guaranteed += 1
            if t == round_wind:
                guaranteed += 1
            if t == seat_wind:
                guaranteed += 1
    if guaranteed:
        routes["役牌"] = guaranteed

    # ---- ② 役牌对子：碰出去/摸成刻子就是役 ----
    for t in range(n):
        if hand[t] == 2 and (_is_dragon(t) or t in (round_wind, seat_wind)):
            routes["役牌对子"] = 1
            break

    # ---- ③ 小三元：两个龙刻 + 第三个龙对子 ----
    dragon_triplets = sum(
        1 for t in S.DRAGONS
        if hand[t] >= 3 or any((not m.is_chi and m.tile == t) for m in melds)
    )
    dragon_pairs = sum(1 for t in S.DRAGONS if hand[t] == 2)
    if dragon_triplets >= 2 and dragon_pairs >= 1:
        routes["小三元"] = 2

    # ---- ④ 断幺九 ----
    if total_tiles and _tanyao_feasible(hand, melds, all_counts):
        routes["断幺九"] = 1

    # ---- ⑤ 混一色 / 清一色 ----
    suit_counts = [
        sum(all_counts[9 * s: 9 * s + 9]) for s in range(3)
    ]
    honor_count = sum(all_counts[27:34])
    present_suits = [s for s in range(3) if suit_counts[s] > 0]
    meld_suits = {
        m.tile // 9 for m in melds if m.tile < 27
    }
    if len(present_suits) <= 1:
        if not present_suits:
            # 全字牌：字一色/混老头方向，先按混老头记
            if honor_count:
                routes["混老头"] = 2
        else:
            s = present_suits[0]
            if len(meld_suits) <= 1:
                if honor_count:
                    routes["混一色"] = 2 if not closed else 3
                else:
                    routes["清一色"] = 5 if not closed else 6
    elif len(present_suits) == 2:
        # 两门：只保留数量多的那门，另一门必须能全部打掉。
        # 副露里的牌打不掉 -> 只要有一组副露在少数门里，染手路线就不成立。
        major = max(present_suits, key=lambda s: suit_counts[s])
        minor = [s for s in present_suits if s != major]
        minor_count = sum(suit_counts[s] for s in minor)
        major_share = suit_counts[major] / max(1, total_tiles - honor_count)
        if (
            minor_count <= 2
            and major_share >= 0.75
            and all(m.tile // 9 in (major, 3) for m in melds)
        ):
            if honor_count:
                routes["混一色"] = 2 if not closed else 3
            else:
                routes["清一色"] = 5 if not closed else 6

    # ---- ⑥ 混老头（全部幺九字）----
    if total_tiles and all(
        _is_terminal_or_honor(t) for t in range(n) for _ in range(all_counts[t])
    ):
        routes["混老头"] = 2

    # ---- ⑦ 混全带幺 / 纯全带幺 ----
    # 结构条件：每个面子都要含幺九或字，且 4/5/6 一张都不能有
    # （456 这种顺子不含幺九，和「全带幺」矛盾）。
    chanta_melds_ok = True
    for m in melds:
        if m.is_chi:
            if not _chi_has_terminal(m):
                chanta_melds_ok = False
                break
        elif not _is_terminal_or_honor(m.tile):
            chanta_melds_ok = False
            break
    middle = sum(all_counts[t] for t in range(n) if t < 27 and t % 9 in (3, 4, 5))
    yao = sum(all_counts[t] for t in range(n) if _is_terminal_or_honor(t))
    edge = sum(
        all_counts[t]
        for t in range(27)
        if t % 9 in (1, 2, 6, 7)
    )
    if chanta_melds_ok and middle == 0 and yao >= 4 and yao + edge >= 7:
        if honor_count:
            routes["混全带幺"] = 1 if not closed else 2
        else:
            routes["纯全带幺"] = 2 if not closed else 3

    # ---- ⑧ 对对和 ----
    pon_like = sum(1 for m in melds if not m.is_chi)
    has_chi = any(m.is_chi for m in melds)
    tri = sum(1 for t in range(n) if hand[t] >= 3)
    pairs = sum(1 for t in range(n) if hand[t] == 2)
    need = 4 - pon_like
    if not has_chi and need >= 0:
        from_pairs = max(0, need - tri)
        if pairs >= from_pairs + 1 or (need <= 0 and pairs >= 1):
            routes["对对和"] = 2

    # ---- ⑨ 七对子（必须门清且一个副露都不能有）----
    if not melds:
        pairs = sum(1 for t in range(n) if hand[t] >= 2)
        kinds = sum(1 for t in range(n) if hand[t] >= 1)
        chiitoi_shanten = 6 - pairs + max(0, 7 - kinds)
        if pairs >= 4 and chiitoi_shanten <= 2:
            routes["七对子"] = 2

    # ---- ⑩ 一气通贯 ----
    for s in range(3):
        base = 9 * s
        segs = [(base, base + 1, base + 2),
                (base + 3, base + 4, base + 5),
                (base + 6, base + 7, base + 8)]
        complete = 0
        partial = 0
        for a, b, c in segs:
            if hand[a] and hand[b] and hand[c]:
                complete += 1
            if hand[a] + hand[b] + hand[c] >= 2:
                partial += 1
        # 副露里的顺子也算
        for m in melds:
            if m.is_chi and m.tile // 9 == s:
                start = m.tile % 9
                if start in (0, 3, 6):
                    complete += 1
                    partial += 1
        if complete >= 2 and partial >= 3:
            routes["一气通贯"] = 1 if not closed else 2
            break

    # ---- ⑪ 三色同顺 ----
    for start in range(7):
        have = 0
        near = 0
        for s in range(3):
            a = 9 * s + start
            if hand[a] and hand[a + 1] and hand[a + 2]:
                have += 1
                near += 1
            elif hand[a] + hand[a + 1] + hand[a + 2] >= 2:
                near += 1
        for m in melds:
            if m.is_chi and m.tile % 9 == start:
                have += 1
                near += 1
        if have >= 2 and near >= 3:
            routes["三色同顺"] = 1 if not closed else 2
            break

    # ---- ⑫ 平和 / 一盃口（门清价值）----
    if closed:
        no_triplet = not any(hand[t] >= 3 for t in range(n)) and not any(
            not m.is_chi for m in melds
        )
        yakuhai_pair = any(
            hand[t] == 2 and (_is_dragon(t) or t in (round_wind, seat_wind))
            for t in range(n)
        )
        seqs = 0
        for s in range(3):
            for start in range(7):
                a = 9 * s + start
                if hand[a] and hand[a + 1] and hand[a + 2]:
                    seqs += 1
        if no_triplet and not yakuhai_pair and seqs >= 3:
            routes["平和"] = 1
        # 一盃口：同一门里两个完全相同的顺子
        for s in range(3):
            for start in range(7):
                a = 9 * s + start
                if hand[a] >= 2 and hand[a + 1] >= 2 and hand[a + 2] >= 2:
                    routes["一盃口"] = 1
                    break
            if "一盃口" in routes:
                break

    # ---- ⑬ 立直：门清才有；这里只标「有资格」，是否真立由 _want_riichi 决定 ----
    # 不把立直算进 guaranteed：没听牌时立直还兑现不了。

    # ---- 最佳真役组合（粗估）----
    best = guaranteed
    if "混一色" in routes or "清一色" in routes:
        suit_han = routes.get("清一色", routes.get("混一色", 0))
        combo = guaranteed + suit_han
        for name in ("断幺九", "一气通贯", "三色同顺", "混全带幺", "纯全带幺",
                     "对对和", "混老头", "小三元"):
            if name in routes:
                combo += routes[name]
        best = max(best, combo)
    else:
        # 没有染手时，取「役牌 + 最大的两三个可叠加役」的粗估
        additive = [v for k, v in routes.items() if k not in ("役牌", "平和", "一盃口")]
        additive.sort(reverse=True)
        best = max(best, guaranteed + sum(additive[:2]))
        if "平和" in routes:
            best = max(best, guaranteed + 1 + (additive[0] if additive else 0))
        if "一盃口" in routes:
            best = max(best, guaranteed + 1 + (additive[0] if additive else 0))

    has_yaku = guaranteed > 0 or bool(routes)

    return YakuInfo(
        closed=closed,
        guaranteed=guaranteed,
        routes=routes,
        dora=0,
        best_han=int(best),
        has_yaku=has_yaku,
        reason=(",".join(sorted(routes)) if routes else ("无役" if not guaranteed else "")),
    )


@lru_cache(maxsize=16384)
def _analyze_yaku_cached(
    hand: tuple[int, ...],
    melds: tuple[S.Meld, ...],
    round_wind: int,
    seat_wind: int,
) -> YakuInfo:
    """``_analyze_yaku`` 的纯函数缓存。

    一局里「打掉 X 之后的役路线」会被不同候选、不同巡目反复问；这个函数
    只依赖手牌/副露/场风自风，所以可以安全缓存。返回的 ``YakuInfo`` 在
    :meth:`StrongAgent._yaku_info` 里用 ``dataclasses.replace`` 复制后再
    写宝牌，**不会把 view 相关的字段写进缓存**。
    """
    return _analyze_yaku(hand, melds, round_wind=round_wind, seat_wind=seat_wind)


# ---------------------------------------------------------------------------
# 听牌信息
# ---------------------------------------------------------------------------


@dataclass
class WaitInfo:
    """听牌时的等待与打点（全部来自 ``sanma.evaluate_hand`` 的真算）。"""

    waits: list[int]
    remaining: dict[int, int]
    total_remaining: int
    dama_points: float       #: 默听期望点数（按剩余枚数加权）
    riichi_points: float     #: 立直期望点数（+1 番，不含里宝）
    dama_han: float          #: 默听加权平均番数
    riichi_han: float
    max_dama_han: int
    has_yaku_dama: bool      #: 不立直能不能和（有一个和了张有役即可）
    dead: bool               #: 所有和了张都无役/死光
    pinfu: bool              #: 有没有平和（好形的一个强信号）
    wait_types: int
    good_shape: bool
    bad_shape: bool


# ---------------------------------------------------------------------------
# 变体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrongConfig:
    """``strong:<名字>`` 的参数。默认是平衡型；变体只调取舍，不换算法。"""

    name: str = "standard"
    call_aggression: float = 1.75   #: 鸣牌倾向
    fold_bias: float = 0.55        #: 弃和倾向（越大越早弃）
    value_bias: float = 0.50       #: 打点权重（牌效是主项，打点用于同向听内比较）
    riichi_bias: float = 1.0       #: 立直倾向
    shanten_points: float = SHANTEN_POINTS
    max_call_shanten: int = 3


STRONG_VARIANTS: dict[str, StrongConfig] = {
    # 默认：牌效优先 + 敢鸣 + 打点/危险度修正。
    # 实测（600 组 vs 教师）：value_bias=0.15 只有 +1.6，0.50 是 +5.1；
    # 但 1.0 以上又会因为贪牌而拆掉牌效（早期版本对教师是负分）。
    "standard": StrongConfig(),
    # 进攻：更少弃和、更爱鸣、更爱立直
    "attack": StrongConfig(
        name="attack", call_aggression=2.20, fold_bias=0.35,
        value_bias=0.20, riichi_bias=1.10,
    ),
    # 防守：更早弃和、少鸣、领先时更爱默听
    "defense": StrongConfig(
        name="defense", call_aggression=0.80, fold_bias=1.40,
        value_bias=0.55, riichi_bias=0.90, max_call_shanten=2,
    ),
    # 大牌：打点权重高、愿意掉一点向听换染手/全带
    "value": StrongConfig(
        name="value", call_aggression=1.40, fold_bias=0.60,
        value_bias=1.20, riichi_bias=0.80, shanten_points=7500.0,
    ),
}


# ---------------------------------------------------------------------------
# StrongAgent
# ---------------------------------------------------------------------------


class StrongAgent(HeuristicAgent):
    """四麻/三麻通用、确定性、纯规则的强化 agent。

    继承 :class:`HeuristicAgent` 只是为了复用它的基类约定（``name`` /
    ``seed`` / ``_n_players``）；**所有决策都重写**，并且在四个轴上做了
    教师没有做的事（见模块文档）。
    """

    name = "强规则"

    def __init__(self, seed: int = 0, *, variant: str = "standard") -> None:
        super().__init__(seed, style="normal")
        if variant not in STRONG_VARIANTS:
            raise SystemExit(
                f"未知的 strong 变体: {variant!r}。可选：{', '.join(sorted(STRONG_VARIANTS))}"
            )
        self.variant = variant
        self.cfg = STRONG_VARIANTS[variant]
        self._last_stats: dict[str, float] = {}

    # -- 观测 ---------------------------------------------------------------
    def _n(self, view: View) -> int:
        return self._n_players(view)

    def _sanma(self, view: View) -> bool:
        return self._n(view) == 3

    def _visible_true(self, view: View) -> list[int]:
        """**正确**的已见牌计数。

        ``aimodel.encode.visible_counts`` 会把自家副露加两遍
        （先 ``view.melds``，再在遍历 ``melds_all`` 时又加一次自己），
        于是「自己碰掉的牌」被当成见过 6 张，进张枚数偏低。
        这里按「手牌 + 自家副露 + 所有牌河 + 他家副露 + 宝牌指示 + 拔北」
        各加一次。
        """
        n_players = self._n(view)
        vis = [0] * 34
        for i in range(34):
            vis[i] = view.hand[i]
        for m in view.melds:
            for t in m.tiles:
                vis[t] += 1
        for q in range(n_players):
            for t in view.discards[q]:
                vis[t] += 1
            if q != view.player:
                for m in view.melds_all[q]:
                    for t in m.tiles:
                        vis[t] += 1
        for d in view.dora_indicators:
            vis[d] += 1
        for k in view.kita:
            vis[S.NORTH] += k
        return vis

    def _dora_types(self, view: View) -> list[int]:
        return [dora_from_index(d, view.rule) for d in view.dora_indicators]

    def _dora_count(self, counts: Sequence[int], melds: Sequence[S.Meld], view: View) -> int:
        """宝牌张数（手牌 + 副露）。赤和拔北单独算。"""
        dora = self._dora_types(view)
        n = sum(counts[t] for t in dora)
        for m in melds:
            for t in m.tiles:
                if t in dora:
                    n += 1
        return n

    def _aka_count(self, view: View, *, discarded: int | None = None) -> int:
        """赤宝牌张数。``discarded`` 给出时要扣掉「这张牌若是赤、且手里只剩它」。"""
        total = sum(view.hand_red)
        if discarded is not None and view.hand_red[discarded]:
            # 如果打掉的是手里最后一张赤，则少一张
            total -= 1
        for m in view.melds:
            total += m.aka
        return total

    def _kita_count(self, view: View) -> int:
        return view.kita[view.player] if self._sanma(view) else 0

    # -- 危险度 -------------------------------------------------------------
    def _danger(self, view: View, tile: int, opp: Sequence[int], vis: Sequence[int]) -> float:
        """危险度：先用手写模型，再补三条它没处理的「壁」。

        * 对所有立直家都是现物 -> **0**。``tile_danger`` 会把「没立直的家」
          的 0.3 基线也算进来，但弃和时现物的确定性比那种粗基线更重要。
        * 四张全见 -> 这张牌已经不可能出现在任何人的和牌里，危险度 0。
        * 中张只剩一张（已见 3 张）-> 打折。字牌模型本身已经按张数分档，
          所以这里只补数牌。
        """
        if vis[tile] >= 4:
            return 0.0
        if opp and all(tile in view.discards[q] for q in opp):
            return 0.0
        d = tile_danger(view, tile, opp, vis)
        if tile < 27 and vis[tile] >= 3:
            d *= 0.55
        return min(1.0, d)

    # -- 局面评估 -----------------------------------------------------------
    def _threat(self, view: View) -> float:
        """他家的威胁（0~1.3）：立直 > 多副露 > 晚巡。"""
        n = self._n(view)
        opp_riichi = [q for q in range(n) if q != view.player and view.riichi[q]]
        others = [q for q in range(n) if q != view.player]
        opens = max((len(view.melds_all[q]) for q in others), default=0)
        if opp_riichi:
            # 一家立直的压力不是「满值 1.0」：一向听的好形手本来就该推。
            t = 0.85 + 0.15 * (len(opp_riichi) - 1)
        elif opens >= 2:
            t = 0.75
        elif opens == 1:
            t = 0.55
        elif view.turn >= 12:
            t = 0.50
        elif view.turn >= 8:
            t = 0.30
        else:
            t = 0.12
        if opens >= 3:
            t += 0.10
        if view.wall_left <= 30:
            t *= 1.15
        return _clamp(t, 0.0, 1.30)

    def _position(self, view: View) -> float:
        """点数位置：-1 = 大幅领先（该收），+1 = 大幅落后（该冲）。"""
        n = self._n(view)
        my = view.scores[view.player]
        others = [view.scores[q] for q in range(n) if q != view.player]
        avg = sum(others) / max(1, len(others))
        pos = (avg - my) / 12000.0
        if view.is_dealer:
            pos += 0.08          # 亲家更该主动
        if view.wall_left <= 30:
            pos *= 1.20          # 终盘放大
        return _clamp(pos, -1.2, 1.2)

    def _attack_strength(self, view: View, s: int, value_points: float,
                         wait_total: int = 0) -> float:
        """进攻力：向听是主项，打点/听牌形状/位置修正。"""
        base = {0: 1.00, 1: 0.80, 2: 0.50, 3: 0.25}.get(s, 0.08)
        value_factor = 0.55 + 0.45 * min(1.0, value_points / 8000.0)
        if s == 0:
            wait_factor = 0.55 + 0.45 * min(1.0, wait_total / 12.0)
        else:
            wait_factor = 0.85
        pos = self._position(view)
        attack = base * value_factor * wait_factor * (1.0 + 0.12 * pos)
        if view.is_dealer:
            attack *= 1.08
        return _clamp(attack, 0.0, 1.4)

    def _fold_urgency(self, view: View, s: int, value_points: float,
                      wait_total: int = 0) -> float:
        threat = self._threat(view)
        attack = self._attack_strength(view, s, value_points, wait_total)
        return _clamp(self.cfg.fold_bias * (threat - 0.95 * attack), 0.0, 1.0)

    # -- 役种 / 打点 ---------------------------------------------------------
    def _yaku_info(self, hand: Sequence[int], melds: Sequence[S.Meld],
                   view: View) -> YakuInfo:
        info = _analyze_yaku_cached(
            tuple(hand), tuple(melds), view.round_wind, view.seat_wind,
        )
        # 缓存里的 dora 恒为 0：宝牌依赖 view（指示牌），不能进纯函数缓存。
        return replace(
            info,
            dora=(
                self._dora_count(hand, melds, view)
                + self._aka_count(view)
                + self._kita_count(view)
            ),
        )

    def _potential_han(self, hand: Sequence[int], melds: Sequence[S.Meld],
                       view: View, *, s: int | None = None) -> float:
        """没听牌时的潜力番（真役 + 宝牌）。

        听牌时不要用它 —— 听牌走 :meth:`_wait_info` 的真算。
        """
        if s is None:
            s = shanten(hand, len(melds))
        info = self._yaku_info(hand, melds, view)
        han = float(info.best_han + info.dora)
        # 门清听牌（或一向听）时的立直 +1。没听牌时打折，避免高估。
        if info.closed:
            if s <= 0:
                han += 1.0
            elif s == 1:
                # 一向听离立直还早，不能按「已经立直」的 1 番算；
                # 给 0.25 只是承认「门清本身有额外机会」，避免把鸣牌过度判死。
                han += 0.25
        return _clamp(han, 0.0, 13.0)

    def _win_context(self, view: View, win_tile: int, *, riichi: bool,
                     counts: Sequence[int], melds: Sequence[S.Meld],
                     discarded: int | None = None) -> S.WinContext:
        ctx = S.WinContext(
            win_tile=win_tile,
            is_tsumo=False,
            round_wind=view.round_wind,
            seat_wind=view.seat_wind,
            is_riichi=riichi,
            dora=self._dora_count(counts, melds, view),
            aka=self._aka_count(view, discarded=discarded),
            kita=self._kita_count(view),
            rule=view.rule,
        )
        return ctx

    def _ron_points(self, agari: S.Agari, view: View) -> float:
        """荣和点数（用于打分）。亲家 1.5 倍；不区分自摸/荣和细节。"""
        base = S.base_points(agari.total_han, agari.fu, agari.yakuman, view.rule)
        return float(base * (6 if view.is_dealer else 4))

    def _wait_info(self, hand: Sequence[int], melds: Sequence[S.Meld],
                   view: View, *, vis: Sequence[int] | None = None,
                   discarded: int | None = None) -> WaitInfo:
        """听牌时逐张和了牌真算番，得到等待/打点/形状。"""
        if vis is None:
            vis = self._visible_true(view)
        n_melds = len(melds)
        sanma = self._sanma(view)
        waits = useful_tiles(hand, n_melds, sanma=sanma)
        remaining: dict[int, int] = {}
        for t in waits:
            left = 4 - vis[t]
            if left > 0:
                remaining[t] = left
        waits = [t for t in waits if t in remaining]
        total = sum(remaining.values())

        dama_pts = 0.0
        riichi_pts = 0.0
        dama_han_sum = 0.0
        riichi_han_sum = 0.0
        weight = 0
        max_dama = 0
        has_yaku_dama = False
        pinfu = False

        for t in waits:
            left = remaining[t]
            win = list(hand)
            win[t] += 1
            # 默听
            ctx = self._win_context(view, t, riichi=False, counts=win,
                                    melds=melds, discarded=discarded)
            agari = S.evaluate_hand(win, melds, ctx)
            dama_han = 0
            if agari is not None:
                dama_han = agari.total_han
                max_dama = max(max_dama, dama_han)
                has_yaku_dama = True
                dama_pts += self._ron_points(agari, view) * left
                dama_han_sum += dama_han * left
                pinfu = pinfu or any(k == "pinfu" for k, _ in agari.yaku)
            # 立直（只有门清能立）
            if all(m.is_concealed for m in melds):
                ctx_r = self._win_context(view, t, riichi=True, counts=win,
                                          melds=melds, discarded=discarded)
                agari_r = S.evaluate_hand(win, melds, ctx_r)
                if agari_r is not None:
                    riichi_pts += self._ron_points(agari_r, view) * left
                    # ``evaluate_hand`` 已经把立直那一番算进 ``total_han``，
                    # 这里**不能**再 +1（之前在这里重复加番了）。
                    riichi_han_sum += agari_r.total_han * left
                    pinfu = pinfu or any(k == "pinfu" for k, _ in agari_r.yaku)
                else:
                    # 立直也没役（理论上不可能，除非规则特殊）——当死张
                    riichi_pts += self._ron_points(
                        S.Agari(han=1, fu=30, yaku=[], dora_han=0, yakuman=0),
                        view,
                    ) * left
            weight += left

        if weight == 0:
            return WaitInfo(
                waits=[], remaining={}, total_remaining=0,
                dama_points=0.0, riichi_points=0.0,
                dama_han=0.0, riichi_han=0.0, max_dama_han=0,
                has_yaku_dama=False, dead=True, pinfu=False,
                wait_types=0, good_shape=False, bad_shape=True,
            )

        closed = all(m.is_concealed for m in melds)
        dama_avg = dama_han_sum / weight
        riichi_avg = (riichi_han_sum / weight) if closed else dama_avg
        dama_pts /= weight
        riichi_pts /= weight
        # 门清：立直通常真的会立（+1 番 + 里宝），但默听也有价值。
        # 打分里用 0.88 折的立直期望，避免把把都高估成立直。
        eff_dama = dama_pts
        eff_riichi = riichi_pts * 0.88
        # 副露不能立直：riichi_points 的占位值就是 dama
        if not closed:
            eff_riichi = dama_pts

        wait_types = len(waits)
        good_shape = (pinfu or (wait_types >= 2 and total >= 6))
        bad_shape = total <= 3 or (wait_types == 1 and not pinfu)
        dead = (not has_yaku_dama) and (not closed or total == 0)

        return WaitInfo(
            waits=waits,
            remaining=remaining,
            total_remaining=total,
            dama_points=eff_dama,
            riichi_points=eff_riichi,
            dama_han=dama_avg,
            riichi_han=riichi_avg,
            max_dama_han=max_dama,
            has_yaku_dama=has_yaku_dama,
            dead=dead,
            pinfu=pinfu,
            wait_types=wait_types,
            good_shape=good_shape,
            bad_shape=bad_shape,
        )

    # -- 鸣牌 ---------------------------------------------------------------
    def _best_after_discard(
        self, hand: Sequence[int], n_melds: int, *, forbidden: int | None = None
    ) -> tuple[int, int | None, list[int]]:
        """鸣牌后必须打一张：返回 ``(最好向听, 打哪张, 打完的手牌)``。"""
        best_s = 99
        best_t: int | None = None
        best_rest: list[int] = list(hand)
        for t in range(34):
            if t == forbidden or hand[t] <= 0:
                continue
            rest = list(hand)
            rest[t] -= 1
            s = shanten(rest, n_melds)
            if s < best_s:
                best_s, best_t, best_rest = s, t, rest
        return best_s, best_t, best_rest

    def _call_value(
        self,
        view: View,
        post_hand: Sequence[int],
        post_melds: Sequence[S.Meld],
        *,
        forbidden: int | None,
        vis: Sequence[int],
    ) -> tuple[int, int | None, list[int], float, YakuInfo, WaitInfo | None]:
        """鸣牌后的「最好向听 + 打点」。"""
        n_melds = len(post_melds)
        after_s, discard_t, rest = self._best_after_discard(
            post_hand, n_melds, forbidden=forbidden
        )
        info = self._yaku_info(post_hand, post_melds, view)
        wait_info = None
        if after_s == 0:
            wait_info = self._wait_info(rest, post_melds, view, vis=vis,
                                        discarded=discard_t)
            if wait_info.dead:
                pts = 0.0
            else:
                pts = max(wait_info.dama_points, wait_info.riichi_points) * (
                    0.40 + 0.60 * min(1.0, wait_info.total_remaining / 16.0)
                )
        else:
            han = self._potential_han(rest, post_melds, view, s=after_s)
            pts = _points_of_han(han) * REALIZATION.get(max(0, after_s), 0.1)
        return after_s, discard_t, rest, pts, info, wait_info

    def _eval_call_action(
        self,
        view: View,
        action: str,
        tile: int,
        post_hand: Sequence[int],
        post_melds: Sequence[S.Meld],
        *,
        forbidden: int | None,
        vis: Sequence[int],
        before_s: int,
        before_value: float,
        before_ukeire: int,
    ) -> tuple[float, str] | None:
        """给一个鸣牌动作打分；返回 ``(分数, 理由)``，None = 不建议。"""
        after_s, discard_t, rest, after_value, info, wait_info = self._call_value(
            view, post_hand, post_melds, forbidden=forbidden, vis=vis
        )
        # ★ 硬规则：鸣完之后必须还有真役路线。宝牌/赤/拔北不算役。
        # 已经听牌时，**以逐个和了张的 evaluate_hand 为准** —— 结构分析
        # 只能看出「役牌对子/一气苗子」这类潜力，听牌后可能根本不成立。
        if after_s == 0:
            if wait_info is None or wait_info.dead:
                return None
        elif not info.has_yaku:
            return None
        if after_s > before_s:
            return None
        if after_s > self.cfg.max_call_shanten:
            return None

        # 鸣牌后的进张（也用于判断「更快」）
        after_ukeire = ukeire_count(
            rest, vis, len(post_melds), sanma=self._sanma(view), base=after_s
        )
        speed_gain = before_s - after_s
        value_gain = after_value - before_value
        ukeire_gain = after_ukeire - before_ukeire

        # 已经听牌时，鸣牌不能降向听；必须有明显打点提升才值得破坏门清/形状。
        if before_s <= 0 and value_gain < 900.0:
            return None
        # 同向听：不能「什么都不换」就鸣。这里用「速度/打点/进张」三项判断，
        # 并给一个小的**节奏加分** —— 副露手拿到先手本身有价值（教师爱鸣
        # 不是没道理）。但打点不能掉太多、进张不能掉太多，否则就是白送。
        tempo = 0.0
        if speed_gain <= 0:
            if before_s <= 0:
                return None
            if value_gain < -1500.0 or ukeire_gain < -6:
                return None
            if value_gain < 300.0 and ukeire_gain < 1:
                tempo = 1200.0
            else:
                tempo = 500.0

        threat = self._threat(view)
        score = (
            speed_gain * self.cfg.shanten_points
            + value_gain
            + ukeire_gain * UKEIRE_POINTS
            + tempo
        )
        # 副露会让防守变差：牌河/安全牌变少、手牌被读。
        score -= threat * (650.0 if after_s >= 2 else 260.0) * (
            1.0 + 0.25 * len(view.melds)
        )
        # 明杠/碰会翻开新宝牌或让对手多宝牌；杠的风险更大
        if action in ("kan", "minkan"):
            score -= 700.0 + 250.0 * len(view.dora_indicators)
        # 门清被破坏的固定成本（有役的碰值这个价）
        if action != "ankan":
            score -= 450.0
        # 打点权重
        score *= self.cfg.call_aggression
        # value 变体更看重打点：上面已经通过 value_gain 进入总分
        if not score > 0.0:
            return None
        return score, (
            f"{action} 后向听 {after_s}（原 {before_s}），"
            f"役路线 {'/'.join(info.route_names) or '役牌'}，"
            f"打点 {after_value:.0f}（原 {before_value:.0f}）"
        )

    def _want_chi(self, decision: Decision, view: View) -> bool:
        """吃不吃 —— 四麻专属。

        引擎只会问「吃/不吃」，具体吃成哪两张由引擎挑向听最好的组合；
        所以我们在这里对**每一种合法吃法**都算一遍，只要有一种值得就同意。
        """
        if self._n(view) != 4:
            return False
        if view.riichi[view.player]:
            return False
        tile = int(decision.tile) if decision.tile is not None else -1
        if tile < 0:
            return False
        hand = view.hand
        melds = list(view.melds)
        vis = self._visible_true(view)
        before_s = shanten(hand, len(melds))
        before_value = (
            self._wait_info(hand, melds, view, vis=vis).dama_points
            if before_s == 0
            else _points_of_han(self._potential_han(hand, melds, view, s=before_s))
            * REALIZATION.get(max(0, before_s), 0.1)
        )
        before_ukeire = ukeire_count(
            hand, vis, len(melds), sanma=self._sanma(view), base=before_s
        )

        best: tuple[float, str] | None = None
        for a, b in _chi_pairs(tile):
            if hand[a] <= 0 or hand[b] <= 0:
                continue
            post_hand = list(hand)
            post_hand[a] -= 1
            post_hand[b] -= 1
            start = min(tile, a, b)
            meld = S.Meld("chi", start, -1, called=tile)
            got = self._eval_call_action(
                view, "chi", tile, post_hand, melds + [meld],
                forbidden=tile, vis=vis,
                before_s=before_s, before_value=before_value,
                before_ukeire=before_ukeire,
            )
            if got is not None and (best is None or got[0] > best[0]):
                best = got
        self._last_stats["chi"] = best[0] if best else 0.0
        return best is not None

    def _want_pon(self, view: View, tile: int) -> bool:
        if view.riichi[view.player]:
            return False
        hand = view.hand
        if hand[tile] < 2:
            return False
        melds = list(view.melds)
        vis = self._visible_true(view)
        before_s = shanten(hand, len(melds))
        before_value = (
            self._wait_info(hand, melds, view, vis=vis).dama_points
            if before_s == 0
            else _points_of_han(self._potential_han(hand, melds, view, s=before_s))
            * REALIZATION.get(max(0, before_s), 0.1)
        )
        before_ukeire = ukeire_count(
            hand, vis, len(melds), sanma=self._sanma(view), base=before_s
        )
        post_hand = list(hand)
        post_hand[tile] -= 2
        meld = S.Meld("pon", tile, -1)
        got = self._eval_call_action(
            view, "pon", tile, post_hand, melds + [meld],
            forbidden=tile, vis=vis,
            before_s=before_s, before_value=before_value,
            before_ukeire=before_ukeire,
        )
        self._last_stats["pon"] = got[0] if got else 0.0
        return got is not None

    def _want_daiminkan(self, view: View, tile: int) -> bool:
        if view.riichi[view.player]:
            return False
        hand = view.hand
        if hand[tile] < 3:
            return False
        melds = list(view.melds)
        vis = self._visible_true(view)
        before_s = shanten(hand, len(melds))
        before_value = (
            self._wait_info(hand, melds, view, vis=vis).dama_points
            if before_s == 0
            else _points_of_han(self._potential_han(hand, melds, view, s=before_s))
            * REALIZATION.get(max(0, before_s), 0.1)
        )
        before_ukeire = ukeire_count(
            hand, vis, len(melds), sanma=self._sanma(view), base=before_s
        )
        post_hand = list(hand)
        post_hand[tile] -= 3
        meld = S.Meld("minkan", tile, -1)
        got = self._eval_call_action(
            view, "kan", tile, post_hand, melds + [meld],
            forbidden=None, vis=vis,
            before_s=before_s, before_value=before_value,
            before_ukeire=before_ukeire,
        )
        self._last_stats["kan"] = got[0] if got else 0.0
        return got is not None

    # -- 立直 / 默听 ---------------------------------------------------------
    def _best_riichi_choice(self, view: View) -> tuple[int, WaitInfo] | None:
        """14 张手牌里，找「打掉后仍听牌」的最佳一张，并算真番。"""
        hand = view.hand
        melds = list(view.melds)
        if len(melds) != 0:
            return None
        vis = self._visible_true(view)
        best: tuple[float, int, WaitInfo] | None = None
        for t in range(34):
            if hand[t] <= 0:
                continue
            rest = list(hand)
            rest[t] -= 1
            if shanten(rest, 0) != 0:
                continue
            info = self._wait_info(rest, [], view, vis=vis, discarded=t)
            if info.dead:
                continue
            score = max(info.dama_points, info.riichi_points)
            if best is None or score > best[0]:
                best = (score, t, info)
        if best is None:
            return None
        return best[1], best[2]

    def _want_riichi(self, view: View) -> bool:
        """立直 / 默听：按打点、听牌形状、点数顺位决定，不是「听牌就立」。"""
        if view.meld_count() != 0 or shanten(view.hand, 0) != 0:
            return False
        choice = self._best_riichi_choice(view)
        if choice is None:
            return False
        _, info = choice
        if info.total_remaining <= 0:
            return False

        n = self._n(view)
        my = view.scores[view.player]
        others = [view.scores[q] for q in range(n) if q != view.player]
        lead = my - max(others) if others else 0
        behind = max(others) - my if others else 0
        late = view.wall_left <= 30 or view.turn >= 12
        dama_han = info.dama_han

        # ① 默听没有役（或有的和了张无役）-> 必须立直。
        if not info.has_yaku_dama:
            return True
        # ② 打点已经很高、听牌又弱 -> 默听。立直只多 1 番，
        #    却把「我在听」告诉全场，弱听很容易被弃和拖死。
        if dama_han >= 3.0 and (info.bad_shape or info.total_remaining <= 4):
            return False
        # ③ 领先 + 终盘 -> 默听守成（尤其 value/defense 变体）。
        if late and lead >= 8000 and dama_han >= 2.0:
            return False
        # ④ 大幅落后 + 终盘 -> 必须立直搏打点。
        if late and behind >= 8000 and info.total_remaining > 0:
            return True
        # ⑤ 亲家早巡 -> 立直施压；但已经满贯以上时仍可默听。
        if view.is_dealer and not late and dama_han < 4.0:
            return True
        # ⑥ 好形两面 + 立直收益不小 -> 立直。
        gain = info.riichi_points - info.dama_points
        if info.good_shape and gain >= 0:
            return True
        # ⑦ 愚形 + 中高打点 -> 默听；低打点愚形 -> 还是立直（至少加一番）。
        if info.bad_shape and dama_han >= 2.0:
            return False
        if dama_han >= 4.0:
            return False
        # 变体：riichi_bias 越大越爱立直（只在边缘局面起作用）
        if self.cfg.riichi_bias > 1.1:
            return True
        if self.cfg.riichi_bias < 0.9 and lead > 0:
            return False
        return True

    # -- 杠 / 拔北 / 九种九牌 -------------------------------------------------
    def _want_kan(self, view: View, tile: int) -> bool:
        """暗杠 / 加杠（``decision.kind == "kan"``）。"""
        hand = view.hand
        melds = list(view.melds)
        is_ankan = hand[tile] >= 4
        if not is_ankan:
            # 加杠：必须已经有一组碰
            if not any(m.kind == "pon" and m.tile == tile for m in melds):
                return False
            # 他家立直时加杠有被抢杠的风险；除非手牌已经很硬，否则不杠
            opp = [q for q in range(self._n(view)) if q != view.player and view.riichi[q]]
            if opp:
                return False
        post_hand = list(hand)
        post_hand[tile] -= 4 if is_ankan else 1
        new_meld = S.Meld("ankan" if is_ankan else "kakan", tile, -1)
        post_melds = melds + [new_meld]
        info = self._yaku_info(post_hand, post_melds, view)
        before_s = shanten(hand, len(melds))
        after_s = shanten(post_hand, len(post_melds))
        # 暗杠不破门清 -> 仍然可以立直，所以「门清 + 接近听牌」本身算役路线。
        # 明杠/加杠必须是已经打开的手，必须真的有役种路线。
        if not info.has_yaku and not (info.closed and after_s <= 1):
            return False
        if after_s > before_s:
            return False
        # 立直中杠牌：只有不改变听牌才允许（引擎会先筛，但这里再守一道）
        if view.riichi[view.player]:
            before_waits = set(useful_tiles(hand, len(melds), sanma=self._sanma(view)))
            after_waits = set(useful_tiles(post_hand, len(post_melds),
                                           sanma=self._sanma(view)))
            if not after_waits or after_waits != before_waits:
                return False
        # 杠会翻新宝牌，对对手也是收益；终盘/对手副露多时更谨慎
        if self._threat(view) >= 0.75 and after_s >= 2:
            return False
        return True

    def _want_kita(self, view: View) -> bool:
        n_north = view.hand[S.NORTH]
        if n_north == 0:
            return False
        s = shanten(view.hand, view.meld_count())
        if n_north >= 2 and s <= 1:
            return False          # 北当雀头/刻子更值钱
        # 手里有明确役种路线时，拔北只是加宝牌，可以拔；
        # 完全无役的副露手，拔北不解决「无役」，但至少不亏（北是宝牌）。
        return True

    def _want_kyuushu(self, view: View) -> bool:
        if view.meld_count() != 0:
            return False
        yaochu_kinds = sum(1 for t in S.YAOCHU if view.hand[t] > 0)
        return yaochu_kinds >= 9 and shanten(view.hand, 0) >= 4

    # -- 主决策 -------------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        k = decision.kind
        if k == "discard":
            return self._discard(decision, view)
        if k == "riichi":
            return 0 if self._want_riichi(view) else 1
        if k == "chi":
            return 1 if self._want_chi(decision, view) else 0
        if k == "call":
            return self._call(decision, view)
        if k in ("tsumo", "ron"):
            return 0                     # 能和就和；引擎已经保证有役
        if k == "kan":
            tile = int(decision.tile) if decision.tile is not None else -1
            return 1 if tile >= 0 and self._want_kan(view, tile) else 0
        if k == "kita":
            return 0 if self._want_kita(view) else 1
        if k == "kyuushu":
            return 0 if self._want_kyuushu(view) else 1
        return 0

    def _call(self, decision: Decision, view: View) -> int:
        opts = [str(o) for o in decision.options]
        tile = int(decision.tile) if decision.tile is not None else -1
        if tile < 0:
            return 0
        candidates: list[tuple[str, int]] = []
        if "kan" in opts:
            candidates.append(("kan", opts.index("kan")))
        if "pon" in opts:
            candidates.append(("pon", opts.index("pon")))
        best_idx = 0
        best_score = 0.0
        for action, idx in candidates:
            if action == "kan":
                ok = self._want_daiminkan(view, tile)
            else:
                ok = self._want_pon(view, tile)
            if not ok:
                continue
            score = self._last_stats.get(action, 0.0)
            # 杠/碰都可行时，碰更稳（不翻新宝牌）；同分取碰。
            if action == "kan":
                score -= 300.0
            if score > best_score:
                best_score, best_idx = score, idx
        return best_idx

    # -- 打牌 ---------------------------------------------------------------
    def _discard_cost(self, view: View, tile: int) -> float:
        """打掉这张牌会损失多少「点数价值」（宝牌/赤/役牌对子）。"""
        cost = 0.0
        dora = self._dora_types(view)
        if tile in dora:
            cost += 1200.0 * max(1, view.hand[tile])
        if view.hand_red[tile]:
            cost += 1800.0
        if _is_dragon(tile) or tile in (view.round_wind, view.seat_wind):
            if view.hand[tile] >= 3:
                cost += 2400.0
            elif view.hand[tile] == 2:
                cost += 1500.0
            else:
                cost += 250.0
        return cost * self.cfg.value_bias

    def _discard(self, decision: Decision, view: View) -> int:
        hand = view.hand
        n_melds = view.meld_count()
        sanma = self._sanma(view)
        vis = self._visible_true(view)
        opp = [q for q in range(self._n(view)) if q != view.player and view.riichi[q]]
        threat = self._threat(view)

        cand: list[dict] = []
        for i, t in enumerate(decision.options):
            rest = list(hand)
            rest[t] -= 1
            s = shanten(rest, n_melds)
            cand.append({
                "i": i, "t": t, "rest": rest, "s": s,
                "danger": self._danger(view, t, opp, vis),
                "cost": self._discard_cost(view, t),
                "u": None, "wait": None, "value": None, "han": None,
            })
        if not cand:
            return 0

        best_s = min(c["s"] for c in cand)

        def fill_value(c: dict) -> None:
            """算「打出去之后」的价值。听牌真算番，没听牌用潜力番。"""
            if c["value"] is not None:
                return
            s, rest = c["s"], c["rest"]
            if s == 0:
                info = self._wait_info(rest, view.melds, view, vis=vis,
                                       discarded=c["t"])
                c["wait"] = info
                c["han"] = max(info.dama_han, info.riichi_han)
                if info.dead:
                    c["value"] = 0.0
                else:
                    c["value"] = max(info.dama_points, info.riichi_points) * (
                        0.40 + 0.60 * min(1.0, info.total_remaining / 16.0)
                    )
            else:
                han = self._potential_han(rest, view.melds, view, s=s)
                c["han"] = han
                c["value"] = _points_of_han(han) * REALIZATION.get(max(0, s), 0.1)

        # 向听差 1 以上时，只有 value 变体才允许再看一眼（它愿意为染手掉向听）。
        # 标准型只看最优向听；听牌时只看听牌（掉听换打点通常不值）。
        if best_s == 0:
            max_s = 0
        elif self.cfg.value_bias > 1.1:
            max_s = best_s + 1
        else:
            max_s = best_s

        def isolation(tile: int) -> int:
            """这张牌在手里有多「孤」：相邻两张内的同花色牌数。

            同样是最优向听时，先打孤张通常比拆搭子好；用它做**预筛**，
            不进入最终打分（最终仍由进张/打点/危险度决定）。
            """
            if tile >= 27:
                return 0
            n = tile % 9
            v = 0
            if n >= 1:
                v += hand[tile - 1]
            if n >= 2:
                v += hand[tile - 2]
            if n <= 7:
                v += hand[tile + 1]
            if n <= 6:
                v += hand[tile + 2]
            return v

        pool = [c for c in cand if c["s"] <= max_s]
        if len(pool) > MAX_EXACT_CANDIDATES:
            best_pool = [c for c in pool if c["s"] == best_s]
            extras = [c for c in pool if c["s"] > best_s]
            key = lambda c: (c["cost"], isolation(c["t"]), c["danger"], c["i"])
            if len(best_pool) >= MAX_EXACT_CANDIDATES:
                focus = sorted(best_pool, key=key)[:MAX_EXACT_CANDIDATES]
            else:
                extras.sort(key=key)
                focus = best_pool + extras[: MAX_EXACT_CANDIDATES - len(best_pool)]
        else:
            focus = pool

        # 先用「最优向听」的候选判断攻守；不要为了算所有候选的进张浪费时间。
        for c in focus:
            fill_value(c)
        best_group = [c for c in focus if c["s"] == best_s] or focus
        top_value = max(c["value"] for c in best_group)
        top_cand = max(best_group, key=lambda x: (x["value"], -x["i"]))
        wait_total = top_cand["wait"].total_remaining if top_cand["wait"] else 0
        fold = self._fold_urgency(view, best_s, top_value, wait_total)

        # 已经打开的「死手」（无役、也做不出役）没有进攻价值 —— 只能弃和。
        hand_info = self._yaku_info(hand, view.melds, view)
        if not hand_info.has_yaku and n_melds > 0:
            fold = max(fold, 0.65)

        if fold >= 0.45:
            # 弃和：安全度是唯一一等公民。
            # 排序键 = (危险度, 向听, 打点损失, 选项下标) —— 危险度差 0.01
            # 就足以压过向听/进张；同安全度时才看牌效。完全确定性。
            best = min(cand, key=lambda c: (c["danger"], c["s"], c["cost"], c["i"]))
            return best["i"]

        pos = self._position(view)
        risk_factor = (
            threat
            * (1.0 + 10.0 * fold)
            * (1.0 + 0.25 * max(0.0, -pos))   # 领先时更怕放铳
            * self.cfg.fold_bias
        )
        danger_weight = DANGER_POINTS * risk_factor

        scored: list[tuple[int, float]] = []
        for c in focus:
            if c["wait"] is not None:
                # 听牌时「进张」就是和了张的剩余枚数，_wait_info 已经算过，
                # 不要再调用一次 ukeire_count（这是最贵的一块钱）。
                c["u"] = c["wait"].total_remaining
            elif c["u"] is None:
                c["u"] = ukeire_count(
                    c["rest"], vis, n_melds, sanma=sanma, base=c["s"]
                )
            # 向听不是唯一项：一向听值 9000 点，打点/进张/安全都在同一把尺上。
            score = (
                -c["s"] * self.cfg.shanten_points
                + c["u"] * UKEIRE_POINTS
                + c["value"] * 0.55 * self.cfg.value_bias
                - c["danger"] * danger_weight
                - c["cost"]
            )
            # 好形奖励：听牌时等待枚数多 / 有平和，给一点额外分。
            if c["wait"] is not None:
                w = c["wait"]
                score += 250.0 * min(1.0, w.total_remaining / 12.0)
                if w.pinfu:
                    score += 300.0
                if w.bad_shape:
                    score -= 200.0
            scored.append((c["i"], score))
        if not scored:
            return cand[0]["i"]
        best_score = max(s for _, s in scored)
        # 完全确定性：同分取下标最小的选项。
        for i, score in scored:
            if abs(score - best_score) < 1e-9:
                return i
        return max(scored, key=lambda x: x[1])[0]

    # -- 调试 ---------------------------------------------------------------
    def summary(self) -> dict:
        return {"variant": self.variant, **self._last_stats}


# ---------------------------------------------------------------------------
# 注册表 / 工厂
# ---------------------------------------------------------------------------


def make(spec: str, seed: int = 0) -> StrongAgent:
    """``make("strong")`` / ``make("strong:attack")``。

    也接受不带前缀的 ``attack`` / ``defense`` / ``value``（方便命令行）。
    """
    raw = (spec or "strong").strip()
    name = raw.split(":", 1)[1] if ":" in raw else raw
    name = name.strip().lower()
    if name in ("", "strong", "standard", "normal"):
        name = "standard"
    if name not in STRONG_VARIANTS:
        raise SystemExit(
            f"未知的 strong 变体: {spec!r}。可选："
            f"{', '.join(sorted(STRONG_VARIANTS))}"
        )
    return StrongAgent(seed, variant=name)


def english_names() -> list[str]:
    return sorted(STRONG_VARIANTS)


def chinese_names() -> list[str]:
    return [STRONG_VARIANTS[k].name for k in sorted(STRONG_VARIANTS)]


__all__ = [
    "StrongAgent", "StrongConfig", "STRONG_VARIANTS", "YakuInfo", "WaitInfo",
    "make", "english_names", "chinese_names",
]
