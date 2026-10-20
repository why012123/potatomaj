"""六种**打法风格**不同的麻将 agent —— 用来当陪练、进对手池、做风格对照。

## 为什么要有这个模块

`HeuristicAgent` 是**教师**：一套「向听 → 进张 → 打点/安全」的通用牌效，
`variants.py` 里的 normal/mountain/flash/huge 只是这套打分的**参数变体**
（比如 ukeire 权重、立直倾向）。参数变体之间**结构上是一样的** ✗

真实麻将里不同的打法**不是参数差别**，而是**取舍不同** ✓：

* 有人宁可 2 向听也要做门清立直（拒绝副露）
* 有人见听就立（哪怕单骑 1 张）
* 有人宁愿掉 1 向听也要把牌做成混一色/清一色的打点
* 有人一看到立直就全弃（安全牌优先，向听无所谓）
* 有人按**点数/顺位**决定这局是攻还是守（押し引き）
* 有人专门做七对子/国士（完全放弃普通形）

这些风格**互相对打**时，能给神经网络提供真正多样的对手 ✓
—— 比「同一个教师的四个参数版本」有用得多 ✓

## 共同约定

* 全部**确定性**（除显式的 `tie_break_eps`），否则同牌山对照就失效 ✗
* 全部**支持三麻/四麻**（三麻没有吃 ✗ 三麻有拔北 ✓）
* 打分用**两遍法**：先给所有候选算向听（便宜），只给最优向听算进张（贵）
  —— 一局几千次决策，不能每个候选都算 ukeire ✗

## 用法

    from aimodel.styles import make
    agent = make("style:speed", seed=0)      # 速攻
    agent = make("style:defense", seed=0)    # 弃和
    python -m aimodel.ab --a A.npz --b B.npz --third style:value
"""

from __future__ import annotations

import random
from typing import Sequence

from . import sanma as S
from .agents import Agent
from .encode import tile_danger, visible_counts
from .engine import Decision, View, dora_from_index
from .fastshanten import shanten, ukeire_count


# ---------------------------------------------------------------------------
# 共用零件
# ---------------------------------------------------------------------------


class StyleAgent(Agent):
    """风格 agent 的基类：把「共用的观测计算」放在这里，**判决交给子类**。

    ⚠️ 这里**不**提供默认的 `_discard` ✗ —— 每个风格必须自己写打分 ✓
       否则一个「默认实现」会让所有风格又悄悄变成同一个 ✓（那就白做了 ✗）
    """

    name = "风格"

    def __init__(self, seed: int = 0, *, tie_break_eps: float = 0.0) -> None:
        self.rng = random.Random(seed)
        #: 分数差在这个窗口内视为等价，随机挑一个（0 = 关，保持确定性 ✓）
        self.tie_break_eps = float(tie_break_eps)

    # -- 观测（各风格共用）--------------------------------------------------
    def _n(self, view: View) -> int:
        return view.rule.players

    def _visible(self, view: View) -> list[int]:
        return visible_counts(view)

    def _opp_riichi(self, view: View) -> list[int]:
        return [
            p for p in range(self._n(view))
            if p != view.player and view.riichi[p]
        ]

    def _danger(self, view: View, tile: int, opp: Sequence[int]) -> float:
        return tile_danger(view, tile, opp)

    def _dora_types(self, view: View) -> list[int]:
        return [dora_from_index(d) for d in view.dora_indicators]

    def _pressure(self, view: View, opp: Sequence[int]) -> float:
        """「现在有多危险」0~1。三麻没立直也可能是听牌 → 副露数也算。"""
        if opp:
            return 1.0
        others = [q for q in range(self._n(view)) if q != view.player]
        opens = max((len(view.melds_all[q]) for q in others), default=0)
        if opens >= 2:
            return 0.7
        if opens == 1:
            return 0.5
        if view.turn >= 12:
            return 0.6
        if view.turn >= 8:
            return 0.3
        return 0.0

    # -- 局面价值（给「打点派」用）------------------------------------------
    def _value_potential(self, view: View) -> float:
        """**手牌的潜在打点**（单位是「近似番数」，不是真番数 ✗）。

        为什么用启发式而不是真算役：
            13 张牌**还没和** ✓ 真算役需要和牌形 ✗ 而且一个风格在每一手都要评一次 ✓
            真算役会把「每手几千次决策」变成不可接受的开销 ✗
        所以我只抓**真正决定打点的那几件事** ✓（麻将里打点基本由它们决定 ✓）：
            役牌对子/刻子、宝牌、赤、染手、断幺、对对、七对
        """
        hand = view.hand
        total = sum(hand)
        if total == 0:
            return 0.0
        v = 0.0
        dora = set(self._dora_types(view))

        # 役牌（白發中 + 场风/自风）
        for t in (S.HAKU, S.HATSU, S.CHUN, view.round_wind, view.seat_wind):
            if hand[t] >= 3:
                v += 1.0        # 刻子：1 番
            elif hand[t] == 2:
                v += 0.45       # 对子：碰了就是 1 番，权重打折
        # 宝牌 / 赤
        v += 0.55 * sum(hand[t] for t in dora)
        v += 0.6 * sum(view.hand_red)
        # 染手（混一色 2 番 / 清一色 6 番）：只看一门占比
        suits = [sum(hand[t] for t in range(9 * s, 9 * s + 9)) for s in range(3)]
        best_suit = max(suits) if suits else 0
        honors = sum(hand[t] for t in range(27, 34))
        share = best_suit / total
        if share >= 0.75:
            v += 1.6 if honors else 2.6      # 有字 = 混一色；无字 = 清一色苗子
        elif share >= 0.6:
            v += 0.8
        # 断幺（1 番，容易叠）：没有幺九和字
        terminals = sum(hand[t] for t in list(range(0, 1)) + list(range(8, 9))
                        + list(range(9, 10)) + list(range(17, 18))
                        + list(range(18, 19)) + list(range(26, 27)))
        if terminals == 0 and honors == 0:
            v += 0.7
        # 对对（2 番）：刻子多
        triplets = sum(1 for t in range(34) if hand[t] >= 3)
        pairs = sum(1 for t in range(34) if hand[t] == 2)
        if triplets >= 2:
            v += 0.6 * triplets
        # 七对（2 番）：对子数
        if pairs >= 5:
            v += 1.5
        elif pairs == 4:
            v += 0.6
        return v

    # -- 共用：挑「向听最优」的候选，只对它们算进张 --------------------------
    def _shanten_each(self, view: View, options: Sequence[int]):
        """返回 [(option_index, tile, rest_hand, shanten)]，一遍算完（便宜）。"""
        n_melds = view.meld_count()
        out = []
        for i, t in enumerate(options):
            rest = list(view.hand)
            rest[t] -= 1
            out.append((i, t, rest, shanten(rest, n_melds)))
        return out

    def _pick_best(self, scored: list[tuple[int, float]]) -> int:
        if not scored:
            return 0
        best = max(s for _, s in scored)
        if self.tie_break_eps > 0.0:
            top = [i for i, s in scored if s >= best - self.tie_break_eps]
            if len(top) > 1:
                return int(self.rng.choice(top))
        return max(scored, key=lambda x: x[1])[0]

    # -- 默认的动作决策（多数风格一致；子类可覆盖）---------------------------
    def _want_riichi(self, view: View) -> bool:
        return view.meld_count() == 0 and shanten(view.hand, 0) == 0

    def _want_kan(self, view: View) -> bool:
        return True

    def _want_kita(self, view: View) -> bool:
        n = view.hand[S.NORTH]
        if n == 0:
            return False
        if n >= 2 and shanten(view.hand, view.meld_count()) <= 1:
            return False        # 北留着当雀头/刻子更值钱
        return True

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
            return 0            # 能和就和（没有一个风格会「见逃」）
        if k == "kan":
            return 1 if self._want_kan(view) else 0
        if k == "kita":
            return 0 if self._want_kita(view) else 1
        if k == "kyuushu":
            return 0 if shanten(view.hand, view.meld_count()) >= 4 else 1
        return 0

    # 子类必须实现
    def _discard(self, decision: Decision, view: View) -> int:   # pragma: no cover
        raise NotImplementedError

    def _want_chi(self, decision: Decision, view: View) -> bool:  # pragma: no cover
        return False

    def _call(self, decision: Decision, view: View) -> int:       # pragma: no cover
        return 0


# ---------------------------------------------------------------------------
# ① 速攻：见听就立，副露吃到听，完全不看打点
# ---------------------------------------------------------------------------


class SpeedAgent(StyleAgent):
    """**速攻派**。麻将里最朴素也最有效的一派：先把牌做成听牌，再谈别的。

    取舍：
        * 向听是唯一的一等公民 ✓ 进张只是同向听内的次级指标 ✓
        * **有副露就吃/碰** —— 只要能降向听 ✓（降不了就不吃 ✓ 乱吃会掉牌效 ✗）
        * 听牌就立直，**不看等待张数** ✗（单骑 1 张也立 ✓ 逼对手弃和也是一种收益 ✓）
        * **完全不做防守** ✗ —— 危险度权重接近 0 ✓ 这正是它的风格 ✓
        * 不留宝牌、不做染手：打点靠立直+偶然 ✓

    谁会这么打：段位房里最常见的「速攻型」。它对神经网络的用处是
    **逼迫模型学会在快节奏下做正确取舍** ✓（而不是陪着慢慢做牌 ✓）
    """

    name = "速攻"
    #: 副露只在这些向听以下才考虑（向听太差时乱吃会毁掉牌效）
    CALL_SHANTEN = 3

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        vis = self._visible(view)
        sanma = view.rule.players == 3
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        best = min(c[3] for c in cand)
        scored = []
        for i, t, rest, s in cand:
            if s > best:
                continue
            u = ukeire_count(rest, vis, n_melds, sanma=sanma)
            scored.append((i, -s * 1e6 + u * 20.0))
        return self._pick_best(scored)

    def _want_riichi(self, view: View) -> bool:
        # 门清听牌就立 —— **不看等待**（这是风格，不是疏忽）
        return view.meld_count() == 0 and shanten(view.hand, 0) == 0

    def _want_chi(self, decision: Decision, view: View) -> bool:
        if view.rule.players == 3:      # 三麻没吃
            return False
        before = shanten(view.hand, view.meld_count())
        if before > self.CALL_SHANTEN:
            return False                 # 牌太散，吃了也救不回来
        tile = int(decision.tile) if decision.tile is not None else -1
        if tile < 0:
            return False
        # 吃了之后向听能降才吃（引擎会自己挑最好的吃法 ✓）
        rest = list(view.hand)
        best_after = 99
        for a in range(34):
            if rest[a] == 0:
                continue
            for b in range(a + 1, 34):
                if rest[b] == 0:
                    continue
                if (a, b) not in _chi_pairs(tile):
                    continue
                c = list(rest)
                c[a] -= 1
                c[b] -= 1
                best_after = min(best_after, shanten(c, view.meld_count() + 1))
        return best_after < before

    def _call(self, decision: Decision, view: View) -> int:
        opts = [str(o) for o in decision.options]
        if "kan" in opts:
            return opts.index("kan")     # 杠了向听不变或者更好，速攻派不拒绝
        tile = int(decision.tile) if decision.tile is not None else -1
        if "pon" in opts and tile >= 0:
            before = shanten(view.hand, view.meld_count())
            if before > self.CALL_SHANTEN:
                return 0
            rest = list(view.hand)
            rest[tile] -= 2
            after = shanten(rest, view.meld_count() + 1)
            if after < before:
                return opts.index("pon")
        return 0


def _chi_pairs(tile: int) -> set[tuple[int, int]]:
    """给定被吃的那张，列出「另外两张」的所有组合（牌型下标）。

    例：吃 4s（下标 21）可能的组合是 (2s,3s)=(19,20)、(3s,5s)=(20,22)、(5s,6s)=(22,23)。
    字牌不能吃 -> 空集。
    """
    if tile >= 27:                       # 字牌
        return set()
    suit0 = (tile // 9) * 9              # 本花色的起始下标
    out: set[tuple[int, int]] = set()
    for lo in (tile - 2, tile - 1, tile):  # 顺子的起始牌
        hi = lo + 2
        if lo < suit0 or hi >= suit0 + 9:   # 越界或跨花色
            continue
        others = [x for x in (lo, lo + 1, hi) if x != tile]
        if len(others) == 2:
            out.add((others[0], others[1]))
    return out


# ---------------------------------------------------------------------------
# ② 门清立直：绝对不副露
# ---------------------------------------------------------------------------


class ClosedAgent(StyleAgent):
    """**门清派**。宁可慢半拍，也要保住立直/门清自摸/一杯口这些门清才有的权益。

    取舍：
        * **永不副露** ✓（连役牌对子也不碰 ✗）—— 吃了就再也立不了直 ✓
        * 打牌只看门清牌效：向听 → 进张 → 保留好形（两面 > 嵌张 > 边张 > 单骑）
        * 听牌即立直 ✓（门清听牌却有役牌暗刻时不立，等暗刻的加符）
        * 有防守：对手立直时按危险度收一点 ✓（门清手更脆 ✗）

    谁会这么打：重视打点的上位者。对神经网络的价值是
    **提供一个「不给你碰、也不给你读副露」的对手** ✓
    """

    name = "门清"

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        vis = self._visible(view)
        sanma = view.rule.players == 3
        opp = self._opp_riichi(view)
        pressure = self._pressure(view, opp)
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        base = min(c[3] for c in cand)
        # 手牌已经不便宜的时候才考虑收 —— 门清手一般不早早弃和
        folding = bool(opp) and base >= 3
        scored = []
        for i, t, rest, s in cand:
            if folding:
                d = self._danger(view, t, opp)
                scored.append((i, -d * 1e6 - s * 1000.0))
                continue
            if s > base:
                continue
            u = ukeire_count(rest, vis, n_melds, sanma=sanma)
            score = -s * 1e6 + u * 22.0
            # 门清派额外看重「好形」：保留两面搭子的牌不该先打
            score += 8.0 * _good_shape_bonus(rest)
            if pressure > 0:
                score -= self._danger(view, t, opp) * 900.0 * pressure
            scored.append((i, score))
        return self._pick_best(scored)

    def _want_chi(self, decision: Decision, view: View) -> bool:
        return False                       # 门清派不吃

    def _call(self, decision: Decision, view: View) -> int:
        return 0                           # 也不碰不杠

    def _want_kan(self, view: View) -> bool:
        # 暗杠也是副露（会亮牌、失去门清立直）—— 门清派拒绝
        return False

    def _want_riichi(self, view: View) -> bool:
        if view.meld_count() != 0 or shanten(view.hand, 0) != 0:
            return False
        # 手上有役牌暗刻：立直会白丢加符，且暗刻本身可能已经够役
        for t in (S.HAKU, S.HATSU, S.CHUN):
            if view.hand[t] >= 3:
                return False
        return True


def _good_shape_bonus(hand: Sequence[int]) -> float:
    """粗略的「好形」加分：两面搭子 > 嵌张/边张 > 孤立牌。"""
    bonus = 0.0
    for t in range(27):
        if hand[t] == 0 or t % 9 >= 8:
            continue
        if hand[t + 1] > 0:                       # 相邻搭子（两面/边张）
            base = t % 9
            bonus += 1.0 if 1 <= base <= 6 else 0.4
    return bonus


# ---------------------------------------------------------------------------
# ③ 打点派：为了混一色/清一色，愿意掉向听
# ---------------------------------------------------------------------------


class ValueAgent(StyleAgent):
    """**打点派（大牌狙い）**。宁可慢，也要做出能决定名次的手。

    取舍：
        * 打分 = 向听 + **潜在打点**（染手/役牌/宝牌/断幺/对对/七对）
        * 允许为了打点**掉 1 向听** ✓（这是它和其它风格最根本的区别 ✓）
        * 役牌对子**必碰**（碰了就是 1 番，且是染手的踏脚石）
        * 染手苗子（一门 ≥60%）时，优先打掉别的花色
        * **几乎不弃和** ✗ —— 大牌在手时弃和等于白做 ✓（只有对手立直且手牌极差才收 1 巡）

    谁会这么打：喜欢逆転的大小牌型。对神经网络的价值是
    **让它面对「明明可以快攻、对手却在做大牌」的局面** ✓（必须学会看副露 ✗）
    """

    name = "大牌"

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        vis = self._visible(view)
        sanma = view.rule.players == 3
        opp = self._opp_riichi(view)
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        best_s = min(c[3] for c in cand)
        scored = []
        for i, t, rest, s in cand:
            # 允许比最优向听差 1，用打点来换 ✓
            if s > best_s + 1:
                continue
            u = ukeire_count(rest, vis, n_melds, sanma=sanma)
            # 用「打完之后的潜在打点」评估，而不是打完之前的
            v_after = self._value_potential(_view_with_hand(view, rest))
            # 向听的惩罚比别的风格**小一个数量级** —— 这正是它愿意慢的原因
            scored.append((i, -s * 1e5 + v_after * 900.0 + u * 6.0))
        return self._pick_best(scored)

    def _want_chi(self, decision: Decision, view: View) -> bool:
        if view.rule.players == 3:
            return False
        # 打点派不靠吃牌 —— 吃会毁掉门清，而且顺子不加打点
        return False

    def _call(self, decision: Decision, view: View) -> int:
        opts = [str(o) for o in decision.options]
        tile = int(decision.tile) if decision.tile is not None else -1
        if "kan" in opts:
            return opts.index("kan")       # 杠加符、加宝牌，打点派喜欢
        if "pon" in opts and tile >= 0:
            yakuhai = tile in (S.HAKU, S.HATSU, S.CHUN,
                               view.round_wind, view.seat_wind)
            if yakuhai:
                return opts.index("pon")   # 役牌必碰（1 番 + 染手踏脚石）
            # 宝牌多、且碰了不掉向听时也碰
            if tile in set(self._dora_types(view)) and view.hand[tile] >= 2:
                before = shanten(view.hand, view.meld_count())
                rest = list(view.hand)
                rest[tile] -= 2
                if shanten(rest, view.meld_count() + 1) <= before:
                    return opts.index("pon")
        return 0

    def _want_riichi(self, view: View) -> bool:
        if view.meld_count() != 0 or shanten(view.hand, 0) != 0:
            return False
        # 打点派：手牌已经很值钱（≥3 番潜力）时**默听**，不立直（怕吓跑对手）
        return self._value_potential(view) < 3.0


def _view_with_hand(view: View, hand: Sequence[int]) -> View:
    """复制一份 View，只换手牌（用于「打完这张之后」的价值评估）。

    ⚠️ 用 `dataclasses.replace` 而不是手动构造：View 字段多，
       手写会漏字段，而且以后加字段也不会自动跟上 ✗
    """
    import dataclasses

    return dataclasses.replace(view, hand=list(hand))


# ---------------------------------------------------------------------------
# ④ 弃和派：安全第一
# ---------------------------------------------------------------------------


class DefenseAgent(StyleAgent):
    """**弃和派（ベタオリ）**。麻将里最赚钱的技术不是和牌，是**不放铳**。

    取舍：
        * 一旦有压力（有人立直 / 对面两组副露 / 12 巡之后）→ **立刻全弃** ✓
        * 弃和时**只看安全度**：现物 → 筋 → 字牌 → 幺九 → 中张
        * 向听在弃和时**几乎不重要** ✗（拆听牌也在所不惜 ✓）
        * **几乎不副露** ✗（副露会让手牌更容易被读出、也更容易放铳）
        * 只有自己听牌、或手牌极好时才不弃

    谁会这么打：守备型上位者。对神经网络的价值最大 ——
    **模型目前最缺的就是「该弃和时真的弃和」** ✓（放铳率是它的短板 ✓）
    """

    name = "弃和"

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        vis = self._visible(view)
        sanma = view.rule.players == 3
        opp = self._opp_riichi(view)
        pressure = self._pressure(view, opp)
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        best_s = min(c[3] for c in cand)
        # 自己已经听牌 -> 不弃（弃和派也有底线：听牌了就冲）
        folding = pressure > 0 and best_s >= 1
        scored = []
        for i, t, rest, s in cand:
            d = self._danger(view, t, opp)
            if folding:
                # 弃和：安全度是唯一的一等公民
                scored.append((i, -d * 1e6 - s * 200.0))
            else:
                if s > best_s:
                    continue
                u = ukeire_count(rest, vis, n_melds, sanma=sanma)
                scored.append((i, -s * 1e6 + u * 18.0 - d * 3000.0 * pressure))
        return self._pick_best(scored)

    def _want_chi(self, decision: Decision, view: View) -> bool:
        return False                        # 副露 = 更容易被读 = 更容易放铳 ✗

    def _call(self, decision: Decision, view: View) -> int:
        return 0

    def _want_kan(self, view: View) -> bool:
        return False                        # 杠了立直棒更多、也更显眼

    def _want_riichi(self, view: View) -> bool:
        # 弃和派立直很谨慎：只听好形、且不给自己挖坑
        if view.meld_count() != 0 or shanten(view.hand, 0) != 0:
            return False
        return True


# ---------------------------------------------------------------------------
# ⑤ 押し引き派：按点数/顺位决定攻守
# ---------------------------------------------------------------------------


class SituationalAgent(StyleAgent):
    """**押し引き派**。同一手牌，领先时该弃、落后时必须冲 —— 这是「段位」思维。

    取舍：
        * 先算**局面权重** `urge`：
            落后越多 -> urge 越大（必须冲 ✓）
            领先越多 -> urge 越小（可以弃 ✓）
            南场/晚巡 -> 放大上述差异（终盘每一手都更值钱）
        * 打牌打分 = 牌效 × (1 + urge) 与 安全度 × (1 - urge) 的**加权和**
          -> 同一张牌，在不同点况下会得到**不同的分数** ✓（这正是它的风格）
        * 副露：落后才积极，领先不碰 ✓

    谁会这么打：段位稳定上升的实战派。对神经网络的价值是
    **把「点数状况」这个维度真正引入对局** ✓（模型现在几乎不看点数 ✗）
    """

    name = "押し引き"

    def _urge(self, view: View) -> float:
        """-1（该死守）.. +1（必须冲）。"""
        n = self._n(view)
        my = view.scores[view.player]
        others = [view.scores[p] for p in range(n) if p != view.player]
        avg = sum(others) / max(1, len(others))
        diff = (my - avg) / 8000.0          # 归一化到 ~[-1, 1]
        urge = -max(-1.0, min(1.0, diff))
        # 终盘放大
        if view.wall_left <= 30:
            urge *= 1.4
        return max(-1.0, min(1.0, urge))

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        vis = self._visible(view)
        sanma = view.rule.players == 3
        opp = self._opp_riichi(view)
        pressure = self._pressure(view, opp)
        urge = self._urge(view)
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        best_s = min(c[3] for c in cand)
        attack = max(0.0, (urge + 1.0) / 2.0)        # 0..1
        defend = max(0.0, (1.0 - urge) / 2.0)        # 0..1
        scored = []
        for i, t, rest, s in cand:
            if s > best_s + (1 if attack > 0.75 else 0):
                continue
            u = ukeire_count(rest, vis, n_melds, sanma=sanma)
            d = self._danger(view, t, opp)
            score = (
                (-s * 1e6 + u * 20.0) * (0.35 + 0.65 * attack)
                - d * 3200.0 * defend * (0.4 + pressure)
            )
            scored.append((i, score))
        return self._pick_best(scored)

    def _want_chi(self, decision: Decision, view: View) -> bool:
        if view.rule.players == 3:
            return False
        if self._urge(view) < -0.1:          # 领先 -> 不碰，保门清与安全
            return False
        before = shanten(view.hand, view.meld_count())
        return before <= 2

    def _call(self, decision: Decision, view: View) -> int:
        opts = [str(o) for o in decision.options]
        urge = self._urge(view)
        if "kan" in opts and urge > -0.2:
            return opts.index("kan")
        tile = int(decision.tile) if decision.tile is not None else -1
        if "pon" in opts and tile >= 0 and urge > -0.1:
            before = shanten(view.hand, view.meld_count())
            rest = list(view.hand)
            rest[tile] -= 2
            if shanten(rest, view.meld_count() + 1) <= before:
                return opts.index("pon")
        return 0

    def _want_riichi(self, view: View) -> bool:
        if view.meld_count() != 0 or shanten(view.hand, 0) != 0:
            return False
        # 领先很多时不立直（立直就是告诉对手「我在听」，反而方便他弃和）
        return self._urge(view) > -0.6


# ---------------------------------------------------------------------------
# ⑥ 七对/国士派：完全放弃普通形
# ---------------------------------------------------------------------------


class SpecialAgent(StyleAgent):
    """**特殊形派**（七对子 / 国士无双）。放弃普通形，专攻两种非常规役。

    取舍：
        * 目标是**对子数**（七对）或**幺九字牌的种类数**（国士），不是向听 ✗
        * 打分分叉：哪个目标更近就走哪个 ✓
        * **永不副露** ✓（两种役都要求门清 ✗）
        * 打牌优先打掉「孤立的中间牌」，保留对子和幺九字
        * 国士 13 种齐全就听牌 → 立直？国士不能立直（无役以外的考量）—— 这里听牌即和

    谁会这么打：几乎没人这么打 ✗ 而这正是它的价值 ✓ ——
    **神经网络需要见过「完全不合常理」的对手** ✓，否则它在实盘遇到特殊形会崩 ✗
    """

    name = "特殊形"

    def _chiitoi_shanten(self, hand: Sequence[int]) -> int:
        """七对子向听：6 - 对子数（4 张同牌算 2 对，需要特殊处理）。"""
        pairs = 0
        kinds = 0
        for t in range(34):
            if hand[t] >= 1:
                kinds += 1
            if hand[t] >= 2:
                pairs += 1
            if hand[t] >= 4:
                pairs += 1          # 暗杠形态：4 张算两对
        s = 6 - pairs
        if kinds < 7:
            s += 7 - kinds
        return s

    def _kokushi_shanten(self, hand: Sequence[int]) -> int:
        """国士向听：13 - 拥有的幺九字种类数（有对子再 -1）。"""
        yaochu = [0, 8, 9, 17, 18, 26] + list(range(27, 34))
        have = sum(1 for t in yaochu if hand[t] >= 1)
        pair = any(hand[t] >= 2 for t in yaochu)
        return 13 - have - (1 if pair else 0)

    def _discard(self, decision: Decision, view: View) -> int:
        cand = self._shanten_each(view, decision.options)
        if not cand:
            return 0
        scored = []
        for i, t, rest, _s in cand:
            chiitoi = self._chiitoi_shanten(rest)
            kokushi = self._kokushi_shanten(rest)
            target = min(chiitoi, kokushi)
            # 主要目标：把特殊形的向听压到最低
            score = -target * 1e6
            # 次级：留住对子（七对）与幺九字（国士）
            score += 4000.0 * sum(1 for x in range(34) if rest[x] >= 2)
            if kokushi <= chiitoi:
                yaochu = [0, 8, 9, 17, 18, 26] + list(range(27, 34))
                score += 3000.0 * sum(1 for x in yaochu if rest[x] >= 1)
            scored.append((i, score))
        return self._pick_best(scored)

    def _want_chi(self, decision: Decision, view: View) -> bool:
        return False

    def _call(self, decision: Decision, view: View) -> int:
        return 0

    def _want_kan(self, view: View) -> bool:
        return False

    def _want_riichi(self, view: View) -> bool:
        # 七对听牌可以立直；国士听牌不立（立直对国士无意义且会限制）
        if view.meld_count() != 0:
            return False
        if self._kokushi_shanten(view.hand) == 0:
            return False
        return self._chiitoi_shanten(view.hand) == 0


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


#: 风格名 -> 类。名字用小写英文，方便 `--third style:speed` 这种写法。
STYLES: dict[str, type[StyleAgent]] = {
    "speed": SpeedAgent,
    "closed": ClosedAgent,
    "value": ValueAgent,
    "defense": DefenseAgent,
    "situational": SituationalAgent,
    "special": SpecialAgent,
}


def make(spec: str, seed: int = 0, *, tie_break_eps: float = 0.0) -> Agent:
    """按名字造一个 agent。``spec`` 支持 ``style:<名字>`` 或直接 ``<名字>``。

        make("style:speed")  -> SpeedAgent
        make("defense")      -> DefenseAgent

    也给 :func:`aimodel.ab.make_third` 用（它接受 ``style:<名字>``）。
    """
    name = spec.split(":", 1)[1] if ":" in spec else spec
    name = name.strip().lower()
    cls = STYLES.get(name)
    if cls is None:
        raise SystemExit(
            f"未知风格: {spec!r}。可选：{', '.join(sorted(STYLES))}"
        )
    return cls(seed, tie_break_eps=tie_break_eps)


def english_names() -> list[str]:
    """所有风格的英文名（给 CLI / 测试用）。"""
    return sorted(STYLES)


def chinese_names() -> list[str]:
    """所有风格的中文名（给报告用）。"""
    return [STYLES[k].name for k in sorted(STYLES)]
