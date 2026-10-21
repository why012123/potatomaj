"""三种 agent：随机 / 启发式教师 / 神经网络。

* :class:`RandomAgent`  —— 基线，用来确认「引擎跑得通」和「模型确实学到了东西」
* :class:`HeuristicAgent` —— **教师**：向听 -> 进张 -> 打点/安全度 的经典牌效，
  再加上立直/副露/拔北/降り（弃和）的规则。行为克隆就是学它
* :class:`MLAgent`      —— 神经网络策略（可采样、可贪婪），并把每次决策记进
  :class:`Tracker`，供行为克隆 / PPO 使用
* :class:`HumanAgent`   —— 终端里手动打牌（``play.py --human``）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import sanma as S
from .encode import (
    ACTION_DIM,
    HEAD_SIZE,
    tile_danger,
    danger_model_loaded,
    danger_score,
    action_features,
    action_index,
    action_label,
    action_mask,
    encode,
    option_head_index,
    visible_counts,
)
from .engine import Decision, View, dora_from_index
from .fastshanten import shanten, ukeire_count, useful_tiles


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------


class Agent:
    """agent 只需要实现 :meth:`decide`：给一个 :class:`Decision` 返回选项下标。"""

    name = "agent"

    @staticmethod
    def _n_players(view: View) -> int:
        """这一桌几家。三麻 3 / 四麻 4。

        **不要写死 3** —— 历史 bug：``_opp_riichi`` / ``_pressure`` /
        ``show_table`` 里的对手范围曾经写死 ``range(3)``，四麻的第 4 家（P3）
        被完全忽略，导致教师看不见 P3 的立直（实测 649 次「场上有立直」的
        四麻决策里 38.7% 漏看立直家，同一张 5m 危险度 0.62 被算成 0.08）。
        """
        n = getattr(getattr(view, "rule", None), "players", 0)
        if n:
            return int(n)
        return len(view.scores)  # 兜底：手写 View 没带 rule 时从点数表推

    def reset(self, seed: int = 0) -> None:  # pragma: no cover - 可选钩子
        pass

    def decide(self, decision: Decision, view: View) -> int:
        raise NotImplementedError

    def summary(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# 随机
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    name = "随机"

    def __init__(self, seed: int = 0, riichi_rate: float = 0.5, call_rate: float = 0.3) -> None:
        self.rng = random.Random(seed)
        self.riichi_rate = riichi_rate
        self.call_rate = call_rate

    def decide(self, decision: Decision, view: View) -> int:
        k = decision.kind
        if k == "discard":
            return self.rng.randrange(len(decision.options))
        if k in ("tsumo", "ron"):
            return 0  # 能和就和，这样基线不至于太离谱
        if k == "riichi":
            return 0 if self.rng.random() < self.riichi_rate else 1
        if k == "kita":
            return 0 if self.rng.random() < 0.5 else 1
        if k == "kan":
            return 1
        if k == "call":
            return 0 if self.rng.random() > self.call_rate else self.rng.randrange(1, len(decision.options))
        if k == "kyuushu":
            return 1
        return 0


# ---------------------------------------------------------------------------
# 启发式教师
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherStyle:
    """教师的口味。四个模型各有各的教师，克隆出来的底子才是对的方向。"""

    name: str = "normal"
    #: 向听 >= 这个值、又有压力时，转入全面弃和
    fold_shanten: int = 3
    #: 危险度权重的倍数（越大越怕死）
    pressure_scale: float = 1.0
    #: 进张权重（越大越贪速度）
    ukeire_weight: float = 20.0
    #: 宝牌/赤/役牌这些「打点」的权重（越大越贪大牌）
    value_weight: float = 1.0
    #: 允许副露的手牌向听上限（越大越爱叫）
    call_shanten: int = 3
    #: 立不立直
    riichi: bool = True
    #: 拔不拔北
    kita: bool = True
    #: 只在自己名字时用（日志）
    cn: str = "标准"


#: 四套教师口味
STYLES: dict[str, TeacherStyle] = {
    "normal": TeacherStyle(),
    # 防守：更早弃和、把危险度看得更重、少叫牌
    # （第一版是 2/2.0/2，实测和「标准」的差别太小，所以再拉开）
    "mountain": TeacherStyle(
        name="mountain", fold_shanten=2, pressure_scale=3.0, ukeire_weight=16.0,
        call_shanten=1, cn="防守",
    ),
    # 速攻：几乎不弃和、进张优先、疯狂叫牌、不在意打点
    "flash": TeacherStyle(
        name="flash", fold_shanten=5, pressure_scale=0.35, ukeire_weight=32.0,
        value_weight=0.25, call_shanten=5, cn="速攻",
    ),
    # 大牌：宝牌/赤/役牌权重大、宁可门清做大的、少叫牌
    # 打点权重再拉高、进张权重压低（宁可慢一手也要牌大）、基本不副露
    "huge": TeacherStyle(
        name="huge", fold_shanten=3, pressure_scale=1.2, ukeire_weight=10.0,
        value_weight=6.5, call_shanten=1, cn="大牌",
    ),
}


class HeuristicAgent(Agent):
    """牌效 + 基础防守的教师。

    打分（越大越好）::

        正常时: -向听*1e6 + 进张枚数*20 - 打出去会损失的价值 - 危险度*权重
        弃和时: -危险度*1e6 - 向听*1000 + 进张枚数

    「危险度」是一个很粗的模型（现物 0、字牌/幺九低、中张高、筋牌打折），
    够教师用；真正的读牌留给神经网络自己去学。
    """

    name = "教师"

    def __init__(
        self,
        seed: int = 0,
        *,
        style: str | TeacherStyle = "normal",
        fold: bool = True,
        declare_riichi: bool | None = None,
        call_pon: bool = True,
        take_kita: bool | None = None,
        tie_break_eps: float = 0.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.style = STYLES[style] if isinstance(style, str) else style
        self.fold = fold
        self.declare_riichi = (
            self.style.riichi if declare_riichi is None else declare_riichi
        )
        self.call_pon = call_pon
        self.take_kita = self.style.kita if take_kita is None else take_kita
        #: 近似等价候选的 tie-break 随机化窗口（P1-1）。
        #: 0.0 = 关（默认，保持教师确定性，避免搅动对拍基线）。
        #: 设成比如 1.0 就表示「分数差在 1.0 以内视为等价，随机挑一个」。
        #: **注意**：`self.rng` 以前从建好就没被用过，这里才真正用上它。
        self.tie_break_eps = float(tie_break_eps)
        self._vis_cache: tuple | None = None

    # -- 工具 ---------------------------------------------------------------
    def _visible(self, view: View) -> list[int]:
        return visible_counts(view)

    def _opp_riichi(self, view: View) -> list[int]:
        return [
            p for p in range(self._n_players(view))
            if p != view.player and view.riichi[p]
        ]

    def _danger(self, view: View, tile: int, opp_riichi: Sequence[int]) -> float:
        """危险度：默认仍走手写 :func:`aimodel.encode.tile_danger`。

        若用 ``--danger-model`` 载入了学出来的模型，且该模型**能安全地用**，
        则改走 :func:`aimodel.encode.danger_score` —— 教师和网络始终看同一份
        危险度，所以两条路不会各说各话。

        ``danger_score`` 在「输入布局尚未验证一致」时会自己退回手写路径，
        所以这里不需要再判断一次；保留手写路径本身就是为了 A/B。
        """
        if danger_model_loaded():
            return danger_score(view, tile)
        return tile_danger(view, tile, opp_riichi)

    def _pressure(self, view: View, opp_riichi: Sequence[int]) -> float:
        """现在有多少「别乱打」的压力（0~1）::

        立直 = 1.0；对面有两组副露 = 0.7；晚巡 = 0.6 …
        三麻只有三家、牌又快，没立直不等于没听牌 —— 上一版就是这里瞎了，
        被默听清一色打了 12000。
        """
        if opp_riichi:
            return 1.0
        others = [q for q in range(self._n_players(view)) if q != view.player]
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

    def _discard_cost(self, view: View, tile: int) -> float:
        """打掉这张牌会损失多少价值（宝牌 / 赤 / 役牌）。"""
        cost = 0.0
        dora_types = [dora_from_index(d) for d in view.dora_indicators]
        if tile in dora_types:
            cost += 350.0
        if view.hand_red[tile]:
            cost += 500.0
        if tile in (S.HAKU, S.HATSU, S.CHUN):
            cost += 400.0 if view.hand[tile] >= 2 else 120.0
        elif tile in (view.round_wind, view.seat_wind):
            cost += 350.0 if view.hand[tile] >= 2 else 80.0
        return cost

    # -- 主决策 -------------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        k = decision.kind
        if k == "discard":
            return self._discard(decision, view)
        if k == "riichi":
            return 0 if (self.declare_riichi and self._want_riichi(view)) else 1
        if k == "call":
            return self._call(decision, view)
        if k == "chi":
            # 选项顺序是 ACTION_TABLE["chi"] = ("pass", "chi")：
            # options[0] = 过、options[1] = 吃。老代码写成
            # `0 if self._want_chi(view) else 1` —— **正好反了**，
            # 「想吃的时候过、想过的时候吃」，教师给出的 chi 标签全是反的。
            # 实测（engine.py:877 + Decision("chi", ["pass","chi"])）：
            # 恒返回 1 时 12 场 599 次吃，恒返回 0 时 0 次 —— 1 才是吃。
            return 1 if self._want_chi(view) else 0
        if k in ("tsumo", "ron"):
            return 0
        if k == "kita":
            return 0 if (self.take_kita and self._want_kita(view)) else 1
        if k == "kan":
            return 1
        if k == "kyuushu":
            return 0 if shanten(view.hand, view.meld_count()) >= 4 else 1
        return 0

    def _discard(self, decision: Decision, view: View) -> int:
        n_melds = view.meld_count()
        hand = view.hand
        sanma = view.rule.players == 3
        base_s = shanten(hand, n_melds)
        vis = self._visible(view)
        opp_riichi = self._opp_riichi(view)
        pressure = self._pressure(view, opp_riichi)
        folding = self.fold and bool(opp_riichi) and base_s >= self.style.fold_shanten

        # 第一遍：只算向听（便宜），拿到最优向听
        cand = []
        for i, t in enumerate(decision.options):
            rest = list(hand)
            rest[t] -= 1
            cand.append((i, t, rest, shanten(rest, n_melds)))
        if not cand:
            return 0
        best_s = min(c[3] for c in cand)

        # 第二遍：只给「并列最优向听」的候选算进张（贵），其余直接淘汰。
        # 弃和时反过来：先看危险度，再看向听。
        scored: list[tuple[int, float]] = []
        for i, t, rest, s in cand:
            if folding:
                score = -self._danger(view, t, opp_riichi) * 1_000_000 - s * 1_000
                if s == best_s:
                    score += ukeire_count(rest, vis, n_melds, sanma=sanma)
                scored.append((i, score))
            else:
                if s > best_s:
                    continue
                u = ukeire_count(rest, vis, n_melds, sanma=sanma)
                score = (
                    -s * 1_000_000
                    + u * self.style.ukeire_weight
                    - self._discard_cost(view, t) * self.style.value_weight
                )
                if pressure > 0:
                    weight = (
                        (400.0 if s <= 1 else 2500.0 if s == 2 else 6000.0)
                        * pressure
                        * self.style.pressure_scale
                    )
                    score -= self._danger(view, t, opp_riichi) * weight
                scored.append((i, score))

        if not scored:
            return cand[0][0]
        best_score = max(sc for _, sc in scored)
        # tie-break 随机化（P1-1）：两个候选**向听、进张都一样**时，
        # 老代码取第一个 -> 教师是确定性的 -> BC 学出「一个确定性策略的三份镜像」。
        # 默认关（tie_break_eps=0.0）以免搅动别人的对拍基线。
        if self.tie_break_eps > 0.0:
            top = [i for i, sc in scored if sc >= best_score - self.tie_break_eps]
            if len(top) > 1:
                return int(self.rng.choice(top))
        return max(scored, key=lambda x: x[1])[0]

    def _want_riichi(self, view: View) -> bool:
        n_melds = view.meld_count()
        if n_melds > 0:
            return False
        return shanten(view.hand, 0) == 0

    def _want_chi(self, view: View) -> bool:
        """吃不吃：手上向听还行、又没有更好的碰，就吃。

        三麻没吃，这个只在四麻会被问到。
        """
        if not self.call_pon:
            return False
        return shanten(view.hand, view.meld_count()) <= self.style.call_shanten

    def _want_kita(self, view: View) -> bool:
        n = view.hand[S.NORTH]
        if n == 0:
            return False
        s = shanten(view.hand, view.meld_count())
        if n >= 2 and s <= 1:
            return False  # 北当雀头 / 刻子更值钱
        return True

    def _call(self, decision: Decision, view: View) -> int:
        opts = [str(o) for o in decision.options]
        if not self.call_pon:
            return 0
        tile = int(decision.tile) if decision.tile is not None else -1
        n_melds = view.meld_count()
        before = shanten(view.hand, n_melds)

        if "kan" in opts:
            return opts.index("kan")

        if "pon" in opts and tile >= 0:
            rest = list(view.hand)
            rest[tile] -= 2
            after = shanten(rest, n_melds + 1)
            is_yakuhai = tile in (S.HAKU, S.HATSU, S.CHUN) or tile in (
                view.round_wind, view.seat_wind,
            )
            limit = self.style.call_shanten
            if is_yakuhai and after <= limit:
                return opts.index("pon")
            if 0 <= after < before and before <= limit:
                return opts.index("pon")
        return 0


# ---------------------------------------------------------------------------
# 神经网络 agent
# ---------------------------------------------------------------------------


@dataclass
class Tracker:
    """把每次决策记下来，训练时再统一组装成 minibatch。"""

    kinds: list[str] = field(default_factory=list)
    X: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logps: list[float] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    actfeats: list[np.ndarray | None] = field(default_factory=list)
    #: 「吃成什么样」：值 = 该局面合法吃法按 (a,b) 升序的序号；-1 = 无关/没吃
    chipats: list[int] = field(default_factory=list)

    def record(
        self, kind: str, x: np.ndarray, mask: np.ndarray, action: int, logp: float,
        player: int, label: str = "", actfeat: np.ndarray | None = None,
        chipat: int = -1,
    ) -> None:
        self.kinds.append(kind)
        self.X.append(x)
        self.masks.append(mask)
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.players.append(int(player))
        self.labels.append(label)
        self.actfeats.append(actfeat)
        self.chipats.append(int(chipat))

    def __len__(self) -> int:
        return len(self.X)

    def clear(self) -> None:
        self.kinds.clear()
        self.X.clear()
        self.masks.clear()
        self.actions.clear()
        self.logps.clear()
        self.players.clear()
        self.labels.clear()
        self.actfeats.clear()
        self.chipats.clear()

    def _chipat_aligned(self) -> "np.ndarray":
        """把 chipats 对齐到 len(X) —— 并**大声报**不一致 ✗

        2026-10-21 实测：chipat 有 89,142 条而 X 只有 86,327 条 ✗
        原因还没定 ✗ 但**绝不能让错长的列进数据集** ✗
        -> 截断/补齐，并且在 stderr 报一次 ✓（不静默 ✗）
        """
        import sys

        import numpy as np

        n = len(self.X)
        arr = np.full(n, -1, dtype=np.int8)
        m = min(n, len(self.chipats))
        if m:
            arr[:m] = np.asarray(self.chipats[:m], dtype=np.int8)
        if len(self.chipats) != n:
            print(
                f"  ⚠️ chipats 与 X 长度不一致 ✗：{len(self.chipats)} vs {n} "
                f"-> 已按 {n} 对齐（多出的丢掉、缺的补 -1）",
                file=sys.stderr,
            )
        return arr

    def arrays(self) -> dict:
        import numpy as np

        max_size = 34
        mask = np.zeros((len(self.X), max_size), dtype=np.float32)
        for i, m in enumerate(self.masks):
            mask[i, : len(m)] = m
        # 动作特征用 float16 存，236k 条决策也只有 ~170MB
        actfeat = np.zeros((len(self.X), 34, ACTION_DIM), dtype=np.float16)
        for i, af in enumerate(self.actfeats):
            if af is not None:
                actfeat[i] = af
        return {
            "X": np.stack(self.X).astype(np.float32),
            "mask": mask,
            "actfeat": actfeat,
            "action": np.asarray(self.actions, dtype=np.int64),
            "logp": np.asarray(self.logps, dtype=np.float32),
            "kind": np.asarray(self.kinds),
            "player": np.asarray(self.players, dtype=np.int64),
            #: 「吃成什么样」的标签（-1 = 不是吃/没吃）✓ 独立一列，不动 action ✓
            "chipat": self._chipat_aligned(),
        }


def legal_chi_pairs(hand: Sequence[int], tile: int) -> list[tuple[int, int]]:
    """吃 ``tile`` 时，手里**真有**的「另外两张」组合，按 ``(a,b)`` 升序 ✓

    ⚠️ 这个顺序就是数据侧 ``chipat`` 的编码基准 ✗
    （``tenhou_bc._legal_chi_pairs``，见 docs/chisel-数据侧-2026-10-21.md）
    推理端**必须**用同一顺序，否则 slot k 训练/推理语义错位 ✗
    两张实现用 ``test_chisel.py`` 逐 tile × 随机手牌对拍 ✓

    为什么这里不直接 import ``tenhou_bc._legal_chi_pairs``：
    ``tenhou_bc`` -> ``logcheck`` -> ``agents`` 会形成**循环 import** ✗
    而且它 import 一次要 ~1s（拉进 mjai / logcheck / rewards）✓
    运行时每个 chi 决策都付不起这个钱 ✗
    所以照抄这份**唯一**的实现，并用测试钉住两边一致 ✓

    与 ``engine._chi_pairs`` 的等价性：引擎按 ``start`` 从小到大遍历
    （``n-2, n-1, n``），每个 ``start`` 产出的 ``(a,b)`` 内部也是升序 ✓
    所以它天然就是 ``(a,b)`` 升序 ✓ 不需要（也不能）改引擎 ✓
    """
    if tile < 0 or tile >= 27:          # 字牌 / 非法牌 -> 不能吃
        return []
    suit0 = (tile // 9) * 9
    out: list[tuple[int, int]] = []
    for lo in (tile - 2, tile - 1, tile):
        hi = lo + 2
        if lo < suit0 or hi >= suit0 + 9:   # 越界 / 跨花色
            continue
        others = [x for x in (lo, lo + 1, hi) if x != tile]
        if all(hand[x] > 0 for x in others):
            out.append((others[0], others[1]))
    return sorted(out)


class MLAgent(Agent):
    """用神经网络的策略头决策；策略头没有的种类退化成随机。"""

    name = "神经网络"

    def __init__(
        self,
        net,
        seed: int = 0,
        greedy: bool = False,
        tracker: Tracker | None = None,
        fallback: Agent | None = None,
        temperature: float = 1.0,
        *,
        style: str | TeacherStyle = "normal",
        use_chisel: bool = True,
    ) -> None:
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy
        self.tracker = tracker
        # 兜底教师必须跟着**这个模型自己的风格**走。老代码恒为
        # HeuristicAgent(seed)（normal 风格），于是 flash / huge 模型
        # 遇到没有策略头的决策时，吃牌规则和 normal 一模一样 ——
        # 风格在这个分支上等于没生效（P0-2）。
        self.fallback = fallback or HeuristicAgent(seed, style=style)
        self.temperature = temperature
        self.counts: dict[str, int] = {}
        #: 「吃成什么样」开关（消融用）✓
        #: False -> 与加 chisel 之前**完全一致**：chi 头说吃就吃，
        #:         具体用哪两张交给引擎的向听启发式 ✓
        #: True  -> 只有当拿到「逐吃法」选项（``chi:<a>:<b>`` / ``decision.pairs``）
        #:         时才由 chisel 选 ✓ 拿不到（stock engine 只有二值）时
        #:         **仍然走原路径** ✗ 不 veto、不改变吃不吃 ✓
        #:         —— veto 会同时改「吃不吃」，破坏单变量消融（见交接说明）✗
        self.use_chisel = bool(use_chisel)
        #: 上一手 chisel 的选择（诊断 / UI 用）：
        #: ``{"pair": (a,b), "slot": k, "legal": [...], "logits": [...]}``
        self.last_chisel: dict | None = None

    def _chisel_active(self) -> bool:
        """chisel 头**可不可用**：开关开着 + 权重真的训过 + 有接口 ✓

        老权重的 ``chisel_trained=False``（net.py 里 load 时标好）✓
        没训过 chisel 的权重（例如纯教师 BC）也不会启用 ✓
        这两种情况都退回「引擎按向听挑吃法」= 加头之前的行为 ✓
        """
        return bool(
            self.use_chisel
            and getattr(self.net, "chisel_trained", False)
            and hasattr(self.net, "chisel_logits")
        )

    def _head_to_option(self, decision: Decision, head: int) -> int | None:
        """策略头下标 -> ``decision.options`` 下标；对不上返回 ``None``。

        **不能找不到就返回 0**。以前这里 `return 0`，而 ``call`` / ``chi`` 的
        选项表第一项是 ``"pass"`` —— 只要选项的写法有一点对不上（比如外部
        引擎把「每一种吃法」作为独立选项传进来），网络选中的「吃」就会
        静默变成「过」，表现正是「永远不鸣牌」，而且不报错、日志里也看不出来。

        现在对不上就交给 :meth:`decide` 走兜底教师（教师按规则判断），
        至少不会把「鸣」悄悄吞成「过」。
        """
        if decision.kind == "discard":
            try:
                return decision.options.index(head)
            except ValueError:
                return None
        return next(
            (i for i, o in enumerate(decision.options)
             if option_head_index(decision.kind, o) == head),
            None,
        )

    # -- chisel（吃成什么样）------------------------------------------------
    def _chisel_pick(self, x: np.ndarray, decision: Decision, view: View):
        """chisel 在**合法吃法**里挑一对；返回 ``(slot, pair, legal)`` 或 None。

        合法吃法由 :func:`legal_chi_pairs` 给出（与数据侧
        ``tenhou_bc._legal_chi_pairs`` 逐项相同，都是 ``(a,b)`` 升序）✓
        chisel 的 3 个 slot **只在合法数量内**比较 ✓ 非法 slot 直接不参与
        argmax（训练时没有「有几种合法吃法」这一列，推理时**有**）✓
        """
        tile = getattr(decision, "tile", None)
        if tile is None:
            return None
        legal = legal_chi_pairs(view.hand, int(tile))
        if not legal:
            return None
        h2 = self.net.trunk(x[None, :])
        z = np.asarray(self.net.chisel_logits(h2), dtype=np.float64)[0, : len(legal)]
        slot = int(np.argmax(z))
        return slot, legal[slot], legal

    def _pair_option_index(self, decision: Decision, pair: tuple[int, int]) -> int | None:
        """在 ``decision.options`` 里找「正是这一对」的选项下标。

        支持两种引擎写法（都已实测）：
          · ``["pass", "chi:15:16", "chi:20:22"]`` —— 从选项串里解析 (a,b) ✓
          · ``Decision.pairs = [(15,16), (20,22)]`` 且 options 与之对齐 ✓
            （字段可能还没合并进来 -> 用 ``getattr`` 兼容 ✓）
        都没有（stock engine 的 ``["pass","chi"]``）-> ``None`` ✓
        """
        opts = list(decision.options)
        # ① 选项串 chi:<a>:<b> —— 最可靠，直接按值匹配
        for i, o in enumerate(opts):
            s = str(o)
            if s.split(":", 1)[0] != "chi":
                continue
            parts = s.split(":")
            if len(parts) < 3:
                continue
            try:
                ab = (int(parts[1]), int(parts[2]))
            except ValueError:
                continue
            if ab == pair:
                return i
        # ② Decision.pairs（顺序与 options[1:] 对齐）
        pairs = getattr(decision, "pairs", None)
        if pairs:
            pl = [tuple(int(v) for v in p) for p in pairs]
            if pair in pl:
                j = pl.index(pair)
                # 无字符串信息时只能假定 options = [pass] + pairs
                if len(opts) == len(pl) + 1:
                    return 1 + j
        return None

    def decide(self, decision: Decision, view: View) -> int:
        kind = decision.kind
        if kind not in HEAD_SIZE or kind not in getattr(self.net, "kinds", []):
            return self.fallback.decide(decision, view)
        x = encode(view)
        mask = action_mask(decision)
        size = HEAD_SIZE[kind]
        actfeat = action_features(view) if kind == "discard" else None
        if self.temperature != 1.0:
            head, logp, _ = self._sample_tempered(x, kind, mask[:size], actfeat)
        else:
            head, logp, _ = self.net.act(
                x, kind, mask[:size], self.rng, greedy=self.greedy, actfeat=actfeat
            )
        # 「吃成什么样」：只在 **chi 头已经决定吃** 之后才轮到它 ✓
        # chipat 只在「引擎真的会吃我们选的那一对」时才记 ✓
        # 拿不到逐吃法选项时保持 -1 ✗ 不能把「心想的 pair」当成实际标签 ✗
        chipat = -1
        choice = None
        if kind == "chi" and head == 1 and self._chisel_active():
            got = self._chisel_pick(x, decision, view)
            if got is not None:
                slot, pair, legal = got
                opt = self._pair_option_index(decision, pair)
                self.last_chisel = {
                    "pair": pair, "slot": slot, "legal": list(legal),
                    # honored=False = stock engine 只有二值选项，
                    # 这一手**实际**还是引擎按向听挑的 ✓（诊断时别误读）
                    "honored": opt is not None,
                }
                if opt is not None:
                    # 引擎给了逐吃法选项 -> chisel 的 pair 就是真正吃出去的那一对 ✓
                    choice = opt
                    chipat = slot
                    self.counts["chi_chisel"] = self.counts.get("chi_chisel", 0) + 1
                else:
                    # stock engine 只有 ["pass","chi"] -> **无法指定 pair** ✗
                    # 走原路径：chi 头说吃就吃，引擎按向听挑 ✓
                    # 这里**不 veto** ✓（veto 会同时改「吃不吃」，不是单变量消融）
                    self.counts["chi_pair_engine"] = (
                        self.counts.get("chi_pair_engine", 0) + 1
                    )
        if choice is None:
            choice = self._head_to_option(decision, head)
        if choice is None:
            # 网络选中的动作在本次合法选项里不存在（选项写法/动作空间不一致）。
            # 不记进 Tracker（那一手的头下标没法对应到选项），交给兜底教师。
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return self.fallback.decide(decision, view)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.tracker is not None:
            self.tracker.record(
                kind, x, mask[:size], head, logp, view.player,
                action_label(decision, choice), actfeat,
                chipat=chipat,
            )
        return choice

    def _sample_tempered(self, x: np.ndarray, kind: str, mask: np.ndarray, actfeat=None):
        p = self.net.policy(x, kind, mask, actfeat).astype(np.float64)
        p = np.power(np.maximum(p, 1e-12), 1.0 / self.temperature)
        p = p / p.sum()
        head = int(self.rng.choice(len(p), p=p))
        return head, float(np.log(max(p[head], 1e-12))), p

    def summary(self) -> dict:
        return {"决策数": dict(self.counts)}


# ---------------------------------------------------------------------------
# 人类玩家（终端）
# ---------------------------------------------------------------------------


class HumanAgent(Agent):
    """在终端里手动打牌。"""

    name = "人类"

    def __init__(self, input_fn=input, print_fn=print, seed: int = 0) -> None:
        self.input_fn = input_fn
        self.print_fn = print_fn

    #: 三麻座位名（自己 + 两家）
    SEAT_CN = ("自己", "下家", "上家")
    #: 四麻座位名（自己 / 下家 / 对家 / 上家）—— 四麻有 4 家，
    #: 老代码只有 3 个名字，第 4 家会 IndexError 或顶掉「上家」。
    SEAT_CN4 = ("自己", "下家", "对家", "上家")

    @classmethod
    def seat_cn(cls, players: int) -> tuple[str, ...]:
        return cls.SEAT_CN4 if players == 4 else cls.SEAT_CN

    def show_table(self, view: View) -> None:
        p = view.player
        n = self._n_players(view)
        names = self.seat_cn(n)
        others = [q for q in range(n) if q != p]
        self.print_fn(f"\n  ── 第 {view.turn} 巡 · 剩余 {view.wall_left} 张 ──")
        self.print_fn(f"  你的手牌   {S.hand_str(view.hand)}")
        if view.melds:
            self.print_fn("  你的副露   " + " ".join(str(m) for m in view.melds))
        self.print_fn(f"  你的牌河   {self._pond(view.discards[p])}")
        for q in others:
            # 用「相对自己的第几家」取名，别拿绝对座位号去查表
            tag = names[(q - p) % n]
            flag = " 立直!" if view.riichi[q] else ""
            self.print_fn(
                f"  {tag}牌河   {self._pond(view.discards[q])}"
                + (f"   副露 {[str(m) for m in view.melds_all[q]]}" if view.melds_all[q] else "")
                + flag
            )
        dora = [S.tile_cn(d) for d in view.dora_indicators]
        self.print_fn(
            f"  宝牌指示 {dora}"
            f"   点数 " + " ".join(f"P{q}:{view.scores[q]}" for q in range(n))
        )
        n_melds = view.meld_count()
        sh = shanten(view.hand, n_melds)
        ups = useful_tiles(view.hand, n_melds)
        self.print_fn(
            f"  向听 {sh}"
            + (f"   进张 {[S.tile_cn(t) for t in ups]}" if ups and sh <= 2 else "")
        )

    @staticmethod
    def _pond(tiles) -> str:
        return " ".join(S.tile_cn(t) for t in tiles) if tiles else "（空）"

    def decide(self, decision: Decision, view: View) -> int:
        opts = decision.options
        kind = decision.kind
        if kind == "discard":
            self.show_table(view)
            labels = [S.tile_cn(int(t)) for t in opts]
        else:
            labels = [{"yes": "是", "no": "否", "pass": "过", "pon": "碰", "kan": "杠"}.get(
                str(o), str(o)) for o in opts]
        ask = {
            "discard": "打哪张",
            "riichi": "要立直吗",
            "call": "要不要鸣牌",
            "ron": "要荣和吗",
            "tsumo": "要自摸吗",
            "kita": "要拔北吗",
            "kan": "要杠吗",
            "kyuushu": "要九种九牌流局吗",
        }.get(kind, "请选择")
        self.print_fn(f"  {ask}?  " + "  ".join(f"[{i}]{n}" for i, n in enumerate(labels)))
        while True:
            try:
                raw = self.input_fn("  > ").strip()
            except EOFError:  # 非交互环境（测试）直接选第一个
                return 0
            if raw.isdigit() and 0 <= int(raw) < len(opts):
                return int(raw)
            self.print_fn("  输入序号")


def make_agents(kind: str, seed: int = 0, players: int = 3, **kw) -> list[Agent]:
    """按名字造 ``players`` 个 agent（评测 / 演示共用）。

    四麻要 4 个 —— 老版本写死 ``range(3)``，四麻调用方只能自己再补一个。
    """
    n = int(players)
    if kind == "random":
        return [RandomAgent(seed + i) for i in range(n)]
    if kind == "teacher":
        return [HeuristicAgent(seed + i, **kw) for i in range(n)]
    raise ValueError(f"未知 agent 类型: {kind}")


__all__ = [
    "Agent", "RandomAgent", "HeuristicAgent", "MLAgent", "HumanAgent", "Tracker",
    "TeacherStyle", "STYLES", "make_agents",
]
