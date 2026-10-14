"""三麻对局引擎：摸打、副露、拔北、立直、流局、算点。

设计要点
--------

* 引擎只管**规则**和**流程**，决策全部交给 ``agents`` 里的 agent；
  同一套引擎既能跑随机 / 启发式 / 神经网络，也能人机对战。
* agent 拿到的 :class:`View` 只包含**该玩家能看到的信息**（自己的手牌、
  三家牌河、宝牌指示牌、点数…），不是上帝视角 —— 免得训练出会作弊的模型。
* 每次决策包成 :class:`Decision`（种类 + 合法选项），神经网络按种类取对应的
  策略头，动作掩码就是「合法选项」。

一巡的流程::

    摸牌 -> 自摸? -> 九种九牌? -> 拔北? -> 杠? -> 立直? -> 打牌
         -> 他家荣和? -> 他家碰/杠? -> 下一家（或副露者直接打牌 / 岭上补牌）

牌山：三麻 108 张 = 每人 13 张（39）+ 王牌 14 张 + 摸牌区 55 张。
杠 / 拔北 的岭上牌从摸牌区尾部取（等价于现实里从牌山尾补牌），
所以「还能摸几张」始终是 ``尾部指针 - 头部指针``。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import sanma as S
from .fastshanten import shanten, useful_tiles

N_PLAYERS = 3  # 兼容旧代码用的默认值；实际人数看 rule.players

#: 决策种类 -> 固定动作表（神经网络按这个顺序建头）
ACTION_TABLE: dict[str, tuple[str, ...]] = {
    "discard": (),  # 34 张牌型，用动态掩码
    "tsumo": ("yes", "no"),
    "ron": ("yes", "no"),
    "riichi": ("yes", "no"),
    "call": ("pass", "pon", "kan"),
    "chi": ("pass", "chi"),
    "kita": ("yes", "no"),
    "kan": ("pass", "kan"),
    "kyuushu": ("yes", "no"),
}


def build_wall(rng: random.Random, rule: S.Rule) -> list[tuple[int, int]]:
    """构造并洗好整副牌（三麻 108 张 / 四麻 136 张），返回 ``[(牌型, 是否赤5)]``。"""
    wall: list[tuple[int, int]] = []
    for t in S.tile_types(rule.players):
        wall.extend([(t, 0)] * 4)
    if rule.aka:
        for t in S.aka_types(rule.players):
            for i, (tt, red) in enumerate(wall):
                if tt == t and red == 0:
                    wall[i] = (t, 1)
                    break
    rng.shuffle(wall)
    return wall


def dora_from_index(i: int, rule: S.Rule = S.DEFAULT_RULE) -> int:
    """宝牌指示牌下标 -> 宝牌下标（转调 :func:`aimodel.sanma.dora_from_index`）。"""
    return S.dora_from_index(i, rule)


# ---------------------------------------------------------------------------
# 观测 / 决策
# ---------------------------------------------------------------------------


@dataclass
class View:
    """某个玩家在某个决策点能看到的全部信息（不含别人的手牌和牌山）。"""

    player: int
    hand: list[int]  # 34 张暗牌计数（含刚摸到的牌）
    drawn: int | None  # 刚摸到的牌型；不是摸牌后的决策就是 None
    melds: list[S.Meld]
    hand_red: list[int]
    discards: list[list[int]]  # 三家牌河
    melds_all: list[list[S.Meld]]
    kita: list[int]
    scores: list[int]
    dora_indicators: list[int]
    round_wind: int
    seat_wind: int
    wall_left: int
    turn: int  # 第几巡（从 1 开始）
    riichi: list[bool]
    ippatsu: list[bool]
    is_dealer: bool
    last_discard: tuple[int, int] | None  # (打牌者, 牌型)
    rule: S.Rule

    @property
    def hand_size(self) -> int:
        return sum(self.hand)

    def hand_str(self) -> str:
        return S.hand_str(self.hand)

    def meld_count(self) -> int:
        return len(self.melds)


@dataclass
class Decision:
    """一次决策请求。``options`` 是合法动作，agent 返回下标。"""

    kind: str  # discard | tsumo | ron | riichi | call | kita | kan | kyuushu
    options: list  # discard 时是牌型下标，其余是字符串
    tile: int | None = None  # 相关的牌（被和 / 被碰 / 被杠的那张）

    def __len__(self) -> int:
        return len(self.options)


# ---------------------------------------------------------------------------
# 玩家
# ---------------------------------------------------------------------------


@dataclass
class Player:
    seat: int
    hand: list[int] = field(default_factory=lambda: [0] * 34)
    red: list[int] = field(default_factory=lambda: [0] * 34)
    melds: list[S.Meld] = field(default_factory=list)
    #: 打牌历史：被碰走的牌也算（振听判定用它）
    discards: list[int] = field(default_factory=list)
    #: 牌河：被碰走的牌会被拿走，所以这里和 discards 不完全一样
    pond: list[int] = field(default_factory=list)
    riichi: bool = False
    double_riichi: bool = False
    #: 立直宣言牌（宣布立直之后打出的那一张）。前端要横放标「立直」，
    #: 靠轮询「看到立直那一刻」去猜会漏 —— 必须由引擎记下来。
    riichi_tile: int | None = None
    ippatsu: bool = False
    riichi_discards: int = 0
    kita: int = 0
    drawn: int | None = None
    passed_ron: bool = False
    kuikae_tile: int = -1  # 刚碰的牌，下一张不能打（喰い替え禁止）

    @property
    def hand_size(self) -> int:
        return sum(self.hand)

    @property
    def menzen(self) -> bool:
        return all(m.is_concealed for m in self.melds)

    def dora_tiles(self) -> list[int]:
        """手牌 + 副露 + 拔北 的全部牌型（算宝牌用）。"""
        out = [i for i in range(34) for _ in range(self.hand[i])]
        for m in self.melds:
            out.extend(m.tiles)
        out.extend([S.NORTH] * self.kita)
        return out

    def aka_count(self) -> int:
        return sum(self.red) + sum(m.aka for m in self.melds)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class HandResult:
    kind: str  # tsumo | ron | draw | kyuushu | abort
    winner: int = -1  # 头跳（双响时是离放铳者最近的一家）
    loser: int = -1
    agari: S.Agari | None = None
    deltas: tuple[int, ...] = ()
    tenpai: tuple[bool, ...] = ()
    turns: int = 0
    note: str = ""
    winners: tuple[int, ...] = ()  # 双响 / 三响时的全部和了者
    agaris: tuple = ()  # 与 winners 对应的 Agari

    @property
    def is_win(self) -> bool:
        return self.kind in ("tsumo", "ron")

    def describe(self) -> str:
        if self.kind == "tsumo":
            return f"自摸 {self.agari.describe() if self.agari else ''}"
        if self.kind == "ron":
            multi = "双响 " if len(self.winners) > 1 else ""
            return f"{multi}荣和 {self.agari.describe() if self.agari else ''}"
        if self.kind == "draw":
            return f"流局（听牌 {''.join('T' if t else '-' for t in self.tenpai)}）"
        if self.kind == "kyuushu":
            return "九种九牌流局"
        return "途中流局"


@dataclass
class Stats:
    """一个座位的对局统计。"""

    hands: int = 0
    tsumo: int = 0
    ron: int = 0
    deal_in: int = 0
    draws: int = 0
    riichi: int = 0
    kita: int = 0
    calls: int = 0
    win_han: list[int] = field(default_factory=list)
    win_score: list[int] = field(default_factory=list)
    deal_in_score: list[int] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)

    def merge(self, other: "Stats") -> None:
        for f in ("hands", "tsumo", "ron", "deal_in", "draws", "riichi", "kita", "calls"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        for f in ("win_han", "win_score", "deal_in_score", "turns"):
            getattr(self, f).extend(getattr(other, f))

    def summary(self) -> dict:
        n = max(1, self.hands)
        return {
            "局数": self.hands,
            "和了率": (self.tsumo + self.ron) / n,
            "自摸率": self.tsumo / n,
            "荣和率": self.ron / n,
            "放铳率": self.deal_in / n,
            "流局率": self.draws / n,
            "立直率": self.riichi / n,
            "副露率": self.calls / n,
            "拔北数": self.kita,
            "平均翻数": (sum(self.win_han) / len(self.win_han)) if self.win_han else 0.0,
            "平均和了点": (sum(self.win_score) / len(self.win_score)) if self.win_score else 0.0,
            "平均放铳点": (sum(self.deal_in_score) / len(self.deal_in_score)) if self.deal_in_score else 0.0,
            "平均巡目": (sum(self.turns) / len(self.turns)) if self.turns else 0.0,
        }


@dataclass
class MatchResult:
    scores: list[int]
    ranks: list[int]  # 0 = 一位
    hands: list[HandResult]
    stats: list[Stats]
    start_scores: list[int]
    seed: int = 0
    #: 段位场最终得分（千点）：(点数-返し点)/1000 + 顺位点 + 额外段位分
    rank_points: list[float] = field(default_factory=list)
    #: 铜币收支（只用于展示）
    coins: list[float] = field(default_factory=list)

    @property
    def deltas(self) -> list[int]:
        # 注意：这里是 MatchResult（不是 Game），没有 self.n，用 scores 的长度
        return [
            self.scores[i] - self.start_scores[i] for i in range(len(self.scores))
        ]

    def rank_point(self, p: int) -> float:
        return self.rank_points[p] if self.rank_points else float(self.deltas[p])


#: 顺位计算在 sanma 层，这里转出去方便老代码继续用
ranks_of = S.ranks_of


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class Game:
    """一场三麻（默认东风战：东 1 / 东 2 / 东 3，含连荘）。"""

    def __init__(
        self,
        agents: Sequence,
        seed: int = 0,
        rule: S.Rule | None = None,
        rounds: int | None = None,
        dealer: int = 0,
        round_wind: int = S.EAST,
        scores: Sequence[int] | None = None,
        max_hands: int = 32,
        on_event: Callable[[str], None] | None = None,
        on_between: Callable[[int], None] | None = None,
    ) -> None:
        self.agents = list(agents)
        # 人数从规则里来：三麻 3 家 / 四麻 4 家
        self.rule = rule or S.rule_for(len(agents))
        self.n = self.rule.players
        if len(agents) != self.n:
            raise ValueError(f"这条规则要 {self.n} 个 agent，收到 {len(agents)} 个")
        self.rng = random.Random(seed)
        self.seed = seed
        # 东风战：三麻 3 局（东1~东3）、四麻 4 局（东1~东4）
        self.rounds = rounds if rounds is not None else self.n
        self.round_wind = round_wind
        self.start_scores = (
            list(scores) if scores else [self.rule.start_score] * self.n
        )
        self.scores = list(self.start_scores)
        self.dealer = dealer
        self.max_hands = max_hands
        self.on_event = on_event
        #: 结构化动作钩子（导出牌谱 / 回放 / 对拍用；见 :meth:`_act`）
        self.on_action: Callable[[dict], None] | None = None
        #: 灌入的牌山。设了它 :meth:`play_hand` 就会用这副牌，而不是随机发。
        self.rigged_wall: list[tuple[int, int]] | None = None
        #: 局间回调（打完一局、开下一局之前调用；可以阻塞等玩家确认）
        self.on_between = on_between
        self.hands: list[HandResult] = []
        self.stats = [Stats() for _ in range(self.n)]
        self.sticks = 0
        #: 最近一次和牌（回放 / UI 标红铳牌用），每局开头会被清掉
        self.last_agari: dict | None = None

    # -- 日志 ---------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.on_event is not None:
            self.on_event(msg)

    def _act(self, kind: str, **kw) -> None:
        """**结构化**动作钩子（和 ``on_event`` 的文本日志并行）。

        文本日志是给人看的，这个是给程序用的：导出牌谱（MJAI 格式）、
        逐手回放、以及「把真实牌谱喂回引擎对拍」都要靠它。

        事件种类：``start_hand`` / ``draw`` / ``discard`` / ``riichi`` /
        ``call``（吃碰杠）/ ``kita`` / ``win`` / ``ryukyoku``
        """
        if self.on_action is not None:
            self.on_action({"kind": kind, **kw})

    def _mark_agari(self, winner: int, tile: int, loser: int | None = None) -> None:
        """记下「谁和了、和的哪张、谁放铳」——回放要拿这个把铳牌标红。"""
        self._act("win", winner=winner, tile=tile, loser=loser, tsumo=loser is None)
        self.last_agari = {
            "winner": winner,
            "loser": loser,
            "tile": tile,
            "tsumo": loser is None,
        }

    # -- 一场 ---------------------------------------------------------------
    def play(self) -> MatchResult:
        dealer = self.dealer
        seat_rounds = 0
        while len(self.hands) < self.max_hands:
            # 局间钩子：可以在这里挂住等玩家点「下一局」（传进去的是已打完的局数）
            if self.on_between is not None and self.hands:
                self.on_between(len(self.hands))
            res = self.play_hand(dealer)
            if len(res.deltas) != self.n:  # 兜底：万一哪种结果忘了带
                res.deltas = (0,) * self.n
            self.hands.append(res)
            for p in range(self.n):
                self.scores[p] += res.deltas[p]
                self.stats[p].hands += 1
                self.stats[p].turns.append(res.turns)

            if any(s < self.rule.bust_score for s in self.scores):
                self._log("🈳 有人被击飞，对局结束")
                break

            renchan = (
                dealer in (res.winners or (res.winner,))
                or res.kind in ("kyuushu", "abort")
                or (res.kind == "draw" and res.tenpai[dealer])
            )
            if not renchan:
                dealer = (dealer + 1) % self.n
                seat_rounds += 1
                if seat_rounds >= self.rounds:
                    if self.round_wind == S.SOUTH:
                        break  # 南入 / 半荘戦 打完
                    if not (
                        self.rule.sudden_death
                        and self.rounds <= self.n
                        and max(self.scores) < self.rule.return_score
                    ):
                        break  # 东风战结束（一位已到返し点）
                    # 雀魂官方规则：オーラス时一位未满返し点就继续南入
                    self.round_wind = S.SOUTH
                    seat_rounds = 0
                    self._log("── 一位未满返し点，南入")
            # 南入中一旦有人达到返し点，对局立刻结束
            if (
                self.round_wind == S.SOUTH
                and self.rounds <= self.n
                and max(self.scores) >= self.rule.return_score
            ):
                self._log("── 已有人达到返し点，对局结束")
                break
        ranks = S.ranks_of(self.scores)
        return MatchResult(
            scores=list(self.scores),
            ranks=ranks,
            hands=self.hands,
            stats=self.stats,
            start_scores=list(self.start_scores),
            seed=self.seed,
            rank_points=S.rank_points(self.scores, self.rule),
            coins=S.coin_delta(self.scores, self.rule),
        )

    # -- 一局 ---------------------------------------------------------------
    def play_hand(self, dealer: int) -> HandResult:
        self._setup(dealer)
        self.last_agari = None   # 每局开头清掉，免得回放标错铳牌
        p = dealer
        need_draw = True
        rinshan = False
        turn = 1

        while True:
            self.turn = turn
            pl = self.players[p]
            riichi_now = False

            if need_draw:
                if self.wall_left() <= 0:
                    return self._exhaustive_draw(turn)
                tile, red = self._draw_rinshan() if rinshan else self._draw()
                pl.hand[tile] += 1
                pl.red[tile] += red
                pl.drawn = tile
                pl.passed_ron = False
                self._log(
                    f"  [{self._seat_name(p)}] {'岭上' if rinshan else '摸'} "
                    f"{S.tile_cn(tile)}（剩 {self.wall_left()}）"
                )
                self._act("draw", player=p, tile=tile, red=red, rinshan=rinshan)

                if self._is_winning(pl):
                    res = self._offer_tsumo(p, turn, rinshan=rinshan)
                    if res is not None:
                        return res

                # 九种九牌：只能在自己第一次摸牌、且全场无人副露时
                if (
                    self.rule.kyuushu_kyuuhai
                    and not self.any_call
                    and len(pl.discards) == 0
                    and self._yaochu_kinds(pl) >= 9
                    and self._ask_choice(p, "kyuushu", ["yes", "no"]) == 0
                ):
                    self._log(f"  [{self._seat_name(p)}] 九种九牌，途中流局")
                    return HandResult(
                        kind="kyuushu", deltas=(0,) * self.n, turns=turn, note="九种九牌"
                    )

                # 拔北（可以连续拔）
                if (
                    self.rule.kita_dora
                    and pl.hand[S.NORTH] > 0
                    and self._kita_allowed(pl)
                    and self._ask_choice(p, "kita", ["yes", "no"], tile=S.NORTH) == 0
                ):
                    self._do_kita(p)
                    self.any_call = True
                    # 拔出来的北是可以被荣和的（不算抢杠，没役也不能荣和）
                    res = self._after_discard(p, S.NORTH, turn, from_kita=True)
                    if res is not None:
                        return res
                    rinshan = True
                    continue

                # 暗杠 / 加杠
                kan_tile = self._kan_options(pl)
                if (
                    kan_tile is not None
                    and self._ask_choice(p, "kan", ["pass", "kan"], tile=kan_tile) == 1
                ):
                    was_kakan = any(m.kind == "pon" and m.tile == kan_tile for m in pl.melds)
                    self._do_kan(p, kan_tile)
                    self.any_call = True
                    if self._register_kan(p):
                        self._log("  🌀 四槓散了，途中流局")
                        return HandResult(
                            kind="abort", deltas=(0,) * self.n, turns=turn, note="四槓散了"
                        )
                    if was_kakan:
                        res = self._chankan_check(kan_tile, p, turn)
                        if res is not None:
                            return res
                    rinshan = True
                    continue

                # 立直（宣布后这次打牌不算摸切）
                if self._can_riichi(p) and self._ask_choice(p, "riichi", ["yes", "no"]) == 0:
                    self._do_riichi(p)
                    riichi_now = True

            # 打牌
            tile = self._choose_discard(p, force_tsumogiri=False if riichi_now else None)
            self._finish_discard(p, tile, turn)

            res = self._after_discard(p, tile, turn)
            if res is not None:
                return res
            caller, is_kan = self._check_call(tile, p)
            if self._pending_abort:
                note = self._pending_abort
                self._pending_abort = None
                self._log(f"  🌀 {note}，途中流局")
                return HandResult(kind="abort", deltas=(0,) * self.n, turns=turn, note=note)
            if caller is not None:
                p = caller
                need_draw = is_kan
                rinshan = is_kan
                turn += 1
                continue
            p = (p + 1) % self.n
            if p == dealer:
                turn += 1
            need_draw = True
            rinshan = False

    # -- 初始化 -------------------------------------------------------------
    def _setup(self, dealer: int) -> None:
        # 先清掉上一局的铳牌标记，否则新一局第一帧会挂着上一局的「荣和」
        self.last_agari = None
        #: 灌入的牌山（对拍真实牌谱用）。给了就用它，不给就按 seed 随机发。
        #: 顺序：前 ``hand_size × n`` 张按座位依次发牌（P0 的 13 张、P1 的、P2 的），
        #: 之后是牌山摸牌顺序，最后 ``rule.dead_wall`` 张是王牌。
        wall = (
            list(self.rigged_wall) if self.rigged_wall else build_wall(self.rng, self.rule)
        )
        self.players = [Player(seat=i) for i in range(self.n)]
        dead = self.rule.dead_wall
        self.live = wall[: len(wall) - dead]
        self.dead = wall[len(wall) - dead :]
        self._pos = 0
        self._tail = len(self.live)
        self.dora_slots = 1
        self.dealer = dealer
        self.last_discard = None
        self.turn = 0
        self.any_call = False  # 有人鸣牌 / 拔北后，九种九牌与天和地和都失效
        self._pending_abort = None
        self.total_kans = 0
        self.kan_players: set[int] = set()
        for pl in self.players:
            for _ in range(self.rule.hand_size):
                t, red = self._draw()
                pl.hand[t] += 1
                pl.red[t] += red
        self._log(
            f"── 第 {len(self.hands) + 1} 局  亲家 {self._seat_name(dealer)}"
            f"  宝牌指示 {S.tile_cn(self._indicator(0))}"
        )
        self._act(
            "start_hand",
            hand_no=len(self.hands) + 1,
            dealer=dealer,
            round_wind=self.round_wind,
            dora=[self._indicator(k) for k in range(self.dora_slots)],
            hands=[list(pl.hand) for pl in self.players],
            reds=[list(pl.red) for pl in self.players],
            scores=list(self.scores),
            sticks=self.sticks,
        )

    def _draw(self) -> tuple[int, int]:
        t, red = self.live[self._pos]
        self._pos += 1
        return t, red

    def _draw_rinshan(self) -> tuple[int, int]:
        self._tail -= 1
        return self.live[self._tail]

    def wall_left(self) -> int:
        return self._tail - self._pos

    def _indicator(self, i: int, ura: bool = False) -> int:
        idx = i * 2 + (1 if ura else 0)
        if idx < len(self.dead):
            return self.dead[idx][0]
        return self.dead[0][0]

    # -- 工具 ---------------------------------------------------------------
    def _seat_name(self, p: int) -> str:
        return f"P{p}({S.tile_cn(self.seat_wind(p))})"

    # -- 吃（只有四麻有）----------------------------------------------------
    def _chi_pairs(self, p: int, tile: int, from_p: int) -> list[tuple[int, int]]:
        """``p`` 能不能吃 ``from_p`` 打的这张；返回可以用的两种手牌组合。

        只能吃**上家**（也就是上一家）打的牌；字牌没有顺子。
        """
        if self.n != 4 or tile >= 27 or from_p != (p - 1) % self.n:
            return []
        hand = self.players[p].hand
        n = tile % 9
        base = tile - n
        out: list[tuple[int, int]] = []
        for start in (n - 2, n - 1, n):
            if not (0 <= start <= 6):
                continue
            a, b = base + start, base + start + 1
            c = base + start + 2
            if tile not in (a, b, c):
                continue
            need = [t for t in (a, b, c) if t != tile]
            if hand[need[0]] > 0 and hand[need[1]] > 0:
                out.append((need[0], need[1]))
        return out

    def seat_wind(self, p: int) -> int:
        """自风：三麻 东/南/西，四麻 东/南/西/北。"""
        table = (S.EAST, S.SOUTH, S.WEST) if self.n == 3 else (S.EAST, S.SOUTH, S.WEST, S.NORTH)
        return table[(p - self.dealer) % self.n]

    def _yaochu_kinds(self, pl: Player) -> int:
        return sum(1 for i in S.YAOCHU if pl.hand[i] > 0)

    def _is_winning(self, pl: Player) -> bool:
        need = (4 - len(pl.melds)) * 3 + 2
        return pl.hand_size == need and shanten(pl.hand, len(pl.melds)) < 0

    def _kita_allowed(self, pl: Player) -> bool:
        """立直后拔北必须不改变听牌。"""
        return True if not pl.riichi else self._shape_keeps_wait(pl, S.NORTH)

    def _shape_keeps_wait(self, pl: Player, tile: int) -> bool:
        n = len(pl.melds)
        before = set(useful_tiles(pl.hand, n))
        c = list(pl.hand)
        c[tile] -= 1
        return set(useful_tiles(c, n)) == before

    def _kan_options(self, pl: Player) -> int | None:
        for m in pl.melds:
            if m.kind == "pon" and pl.hand[m.tile] >= 1:
                if not pl.riichi or self._shape_keeps_wait(pl, m.tile):
                    return m.tile
        for t in S.tile_types(self.n):
            if pl.hand[t] >= 4 and (not pl.riichi or self._shape_keeps_wait(pl, t)):
                return t
        return None

    def _can_riichi(self, p: int) -> bool:
        pl = self.players[p]
        return (
            not pl.riichi
            and pl.menzen
            and self.scores[p] >= self.rule.riichi_stick
            and self.wall_left() > 0
            and pl.hand_size == (4 - len(pl.melds)) * 3 + 2
            and shanten(pl.hand, len(pl.melds)) == 0
        )

    # -- 动作 ---------------------------------------------------------------
    def _register_kan(self, p: int) -> bool:
        """登记一次杠；返回是否触发四槓散了。"""
        self.total_kans += 1
        self.kan_players.add(p)
        return (
            self.rule.abort_four_kans
            and self.total_kans >= 4
            and len(self.kan_players) >= 2
        )

    def _do_kita(self, p: int) -> None:
        pl = self.players[p]
        pl.hand[S.NORTH] -= 1
        if pl.red[S.NORTH] and pl.hand[S.NORTH] < pl.red[S.NORTH]:
            pl.red[S.NORTH] -= 1
        pl.kita += 1
        self.stats[p].kita += 1
        if self.rule.kita_flips_dora and self.dora_slots < self.rule.dora_slots:
            self.dora_slots += 1
        for q in self.players:
            q.ippatsu = False
        self._log(f"  [{self._seat_name(p)}] 拔北（累计 {pl.kita}）")
        self._act("kita", player=p)

    def _do_kan(self, p: int, tile: int) -> None:
        pl = self.players[p]
        for i, m in enumerate(pl.melds):
            if m.kind == "pon" and m.tile == tile:
                aka = m.aka
                if pl.red[tile] and pl.hand[tile] < pl.red[tile]:
                    pl.red[tile] -= 1
                    aka += 1
                pl.hand[tile] -= 1
                pl.melds[i] = S.Meld("kakan", tile, m.from_player, aka)
                self._log(f"  [{self._seat_name(p)}] 加杠 {S.tile_cn(tile)}")
                self._act("call", player=p, call="kakan", tile=tile,
                          from_player=p, consumed=[tile, tile, tile])
                break
        else:
            pl.hand[tile] -= 4
            aka = 0
            while pl.red[tile] and pl.hand[tile] < pl.red[tile]:
                pl.red[tile] -= 1
                aka += 1
            pl.melds.append(S.Meld("ankan", tile, -1, aka))
            self._log(f"  [{self._seat_name(p)}] 暗杠 {S.tile_cn(tile)}")
            self._act("call", player=p, call="ankan", tile=tile,
                      from_player=p, consumed=[tile] * 4)
        for q in self.players:
            q.ippatsu = False
        if self.dora_slots < self.rule.dora_slots:
            self.dora_slots += 1

    def _do_riichi(self, p: int) -> None:
        pl = self.players[p]
        pl.riichi = True
        pl.double_riichi = len(pl.discards) == 0 and not self.any_call
        pl.ippatsu = True
        pl.riichi_discards = len(pl.discards)
        self.scores[p] -= self.rule.riichi_stick
        self.sticks += 1
        self.stats[p].riichi += 1
        self._log(f"  [{self._seat_name(p)}] 立直！{'（两立直）' if pl.double_riichi else ''}")
        self._act("riichi", player=p, double=pl.double_riichi)

    def _choose_discard(self, p: int, force_tsumogiri: bool | None = None) -> int:
        pl = self.players[p]
        if force_tsumogiri is None:
            force_tsumogiri = pl.riichi
        if force_tsumogiri and pl.drawn is not None:
            return pl.drawn
        options = [t for t in S.tile_types(self.n) if pl.hand[t] > 0]
        if pl.kuikae_tile in options and len(options) > 1:
            options = [t for t in options if t != pl.kuikae_tile]
        return options[self._ask_choice(p, "discard", options)]

    def _finish_discard(self, p: int, tile: int, turn: int) -> None:
        pl = self.players[p]
        # 刚宣布立直后的第一张打牌 = 立直宣言牌
        if pl.riichi and pl.riichi_tile is None:
            pl.riichi_tile = tile
        pl.hand[tile] -= 1
        if pl.red[tile] and pl.hand[tile] < pl.red[tile]:
            pl.red[tile] -= 1
        pl.discards.append(tile)
        pl.pond.append(tile)
        pl.drawn = None
        pl.kuikae_tile = -1
        self.last_discard = (p, tile)
        if pl.riichi and pl.ippatsu and len(pl.discards) > pl.riichi_discards + 1:
            pl.ippatsu = False
        self._log(f"  [{self._seat_name(p)}] 打 {S.tile_cn(tile)}")
        self._act("discard", player=p, tile=tile, tsumogiri=(pl.drawn is None),
                  riichi=pl.riichi)

    def _after_discard(
        self, p: int, tile: int, turn: int, from_kita: bool = False
    ) -> HandResult | None:
        """打牌（或拔北）之后：他家荣和。

        雀魂 / 天凤三麻都是**双荣和**（无头跳）：多家和牌时每家都按满额收，
        放铳者要赔给每一家；供托与立直棒归离放铳者最近的那家（头跳）。
        """
        winners: list[tuple[int, S.Agari]] = []
        for k in range(1, self.n):
            q = (p + k) % self.n
            pl = self.players[q]
            if pl.passed_ron:
                continue
            c = list(pl.hand)
            c[tile] += 1
            if sum(c) != (4 - len(pl.melds)) * 3 + 2 or shanten(c, len(pl.melds)) >= 0:
                continue
            if self._is_furiten(q):
                continue
            if self._ask_choice(q, "ron", ["yes", "no"], tile=tile) == 1:
                pl.passed_ron = True
                continue
            ctx = self._win_context(q, tile, tsumo=False, turn=turn)
            ctx.is_houtei = (not from_kita) and self.wall_left() == 0
            agari = S.evaluate_hand(c, pl.melds, ctx)
            if agari is None:
                pl.passed_ron = True
                continue
            winners.append((q, agari))
            if not self.rule.double_ron:
                break
        if not winners:
            return None
        if len(winners) == 1:
            q, agari = winners[0]
            return self._settle_ron(q, p, tile, agari, turn)
        return self._settle_multi_ron(winners, p, tile, turn)

    def _is_furiten(self, p: int) -> bool:
        pl = self.players[p]
        waits = set(useful_tiles(pl.hand, len(pl.melds)))
        return any(t in waits for t in pl.discards)

    def _check_call(self, tile: int, from_p: int) -> tuple[int | None, bool]:
        """碰 / 明杠 / 吃。返回 ``(副露者, 是否需要岭上补牌)``。

        优先级按麻将规矩来：先问一圈碰 / 杠（谁先要谁拿），都没人要才轮到
        上家吃 —— 不能让吃抢在后面的碰前面。
        """
        for k in range(1, self.n):
            p = (from_p + k) % self.n
            pl = self.players[p]
            if pl.riichi:
                continue
            can_pon = pl.hand[tile] >= 2
            can_kan = pl.hand[tile] >= 3 and self.wall_left() > 0
            if not (can_pon or can_kan):
                continue
            opts = ["pass"] + (["pon"] if can_pon else []) + (["kan"] if can_kan else [])
            act = opts[self._ask_choice(p, "call", opts, tile=tile)]
            if act == "pass":
                continue
            self._clear_ippatsu()
            self.any_call = True
            self.stats[p].calls += 1
            if act == "pon":
                self._do_pon(p, tile, from_p)
                return p, False
            self._do_minkan(p, tile, from_p)
            if self._register_kan(p):
                self._pending_abort = "四槓散了"
            return p, True

        # 没人碰/杠 -> 再看上家能不能吃
        chi_p = (from_p + 1) % self.n
        if not self.players[chi_p].riichi:
            pairs = self._chi_pairs(chi_p, tile, from_p)
            if pairs and self._ask_choice(chi_p, "chi", ["pass", "chi"], tile=tile) == 1:
                # 多种吃法时挑向听最好的那个
                best, best_s = pairs[0], 99
                for a, b in pairs:
                    c = list(self.players[chi_p].hand)
                    c[a] -= 1
                    c[b] -= 1
                    v = shanten(c, len(self.players[chi_p].melds) + 1)
                    if v < best_s:
                        best, best_s = (a, b), v
                self._clear_ippatsu()
                self.any_call = True
                self.stats[chi_p].calls += 1
                self._do_chi(chi_p, tile, from_p, best[0], best[1])
                return chi_p, False
        return None, False

    def _clear_ippatsu(self) -> None:
        for q in self.players:
            q.ippatsu = False

    def _do_chi(self, p: int, tile: int, from_p: int, a: int, b: int) -> None:
        pl = self.players[p]
        pl.hand[a] -= 1
        pl.hand[b] -= 1
        start = min(tile, a, b)
        self._take_from_pond(from_p, tile)
        pl.melds.append(S.Meld("chi", start, from_p, called=tile))
        pl.kuikae_tile = tile
        self._log(f"  [{self._seat_name(p)}] 吃 {S.tile_cn(start)}"
                  f"{S.tile_cn(start + 1)}{S.tile_cn(start + 2)}")
        self._act("call", player=p, call="chi", tile=start, called=tile,
                  from_player=from_p, consumed=[a, b])

    def _take_from_pond(self, p: int, tile: int) -> None:
        """被碰/杠走的牌从牌河里拿掉（但打牌历史保留，振听照算）。"""
        pond = self.players[p].pond
        for i in range(len(pond) - 1, -1, -1):
            if pond[i] == tile:
                del pond[i]
                return

    def _do_pon(self, p: int, tile: int, from_p: int) -> None:
        pl = self.players[p]
        pl.hand[tile] -= 2
        aka = 0
        while pl.red[tile] and pl.hand[tile] < pl.red[tile]:
            pl.red[tile] -= 1
            aka += 1
        self._take_from_pond(from_p, tile)
        pl.melds.append(S.Meld("pon", tile, from_p, aka))
        pl.kuikae_tile = tile  # 喰い替え禁止：下一张不能打同种牌
        self._log(f"  [{self._seat_name(p)}] 碰 {S.tile_cn(tile)}")
        self._act("call", player=p, call="pon", tile=tile, from_player=from_p,
                  consumed=[tile, tile])

    def _do_minkan(self, p: int, tile: int, from_p: int) -> None:
        pl = self.players[p]
        pl.hand[tile] -= 3
        aka = 0
        while pl.red[tile] and pl.hand[tile] < pl.red[tile]:
            pl.red[tile] -= 1
            aka += 1
        self._take_from_pond(from_p, tile)
        pl.melds.append(S.Meld("minkan", tile, from_p, aka))
        if self.dora_slots < self.rule.dora_slots:
            self.dora_slots += 1
        self._log(f"  [{self._seat_name(p)}] 明杠 {S.tile_cn(tile)}")
        self._act("call", player=p, call="kan", tile=tile, from_player=from_p,
                  consumed=[tile, tile, tile])

    def _chankan_check(self, tile: int, from_p: int, turn: int) -> HandResult | None:
        """抢杠：加杠的那张牌被别人荣和。"""
        for k in range(1, self.n):
            p = (from_p + k) % self.n
            pl = self.players[p]
            c = list(pl.hand)
            c[tile] += 1
            if sum(c) != (4 - len(pl.melds)) * 3 + 2 or shanten(c, len(pl.melds)) >= 0:
                continue
            if self._ask_choice(p, "ron", ["yes", "no"], tile=tile) == 1:
                continue
            ctx = self._win_context(p, tile, tsumo=False, turn=turn)
            ctx.is_chankan = True
            ctx.is_houtei = False
            agari = S.evaluate_hand(c, pl.melds, ctx)
            if agari is None:
                continue
            self._log("  ⚡ 抢杠！")
            return self._settle_ron(p, from_p, tile, agari, turn)
        return None

    def _offer_tsumo(self, p: int, turn: int, rinshan: bool) -> HandResult | None:
        pl = self.players[p]
        ctx = self._win_context(p, pl.drawn, tsumo=True, turn=turn)
        ctx.is_rinshan = rinshan
        ctx.is_haitei = self.wall_left() == 0
        agari = S.evaluate_hand(pl.hand, pl.melds, ctx)
        if agari is None:
            return None
        if self._ask_choice(p, "tsumo", ["yes", "no"]) == 1:
            return None
        return self._settle_tsumo(p, agari, turn)

    # -- 和牌上下文 ---------------------------------------------------------
    def _win_context(self, p: int, tile: int, tsumo: bool, turn: int) -> S.WinContext:
        pl = self.players[p]
        dora_types = [
            dora_from_index(self._indicator(i), self.rule) for i in range(self.dora_slots)
        ]
        tiles = pl.dora_tiles()
        dora = sum(tiles.count(d) for d in dora_types)
        ura = 0
        if pl.riichi:
            ura_types = [
                dora_from_index(self._indicator(i, ura=True), self.rule)
                for i in range(self.dora_slots)
            ]
            ura = sum(tiles.count(d) for d in ura_types)
        first = tsumo and len(pl.discards) == 0 and not self.any_call
        return S.WinContext(
            win_tile=tile,
            is_tsumo=tsumo,
            round_wind=self.round_wind,
            seat_wind=self.seat_wind(p),
            is_riichi=pl.riichi,
            is_double_riichi=pl.double_riichi,
            is_ippatsu=pl.ippatsu and pl.riichi,
            is_tenhou=first and p == self.dealer,
            is_chiihou=first and p != self.dealer,
            dora=dora,
            ura=ura,
            aka=pl.aka_count(),
            kita=pl.kita,
            rule=self.rule,
        )

    # -- 结算 ---------------------------------------------------------------
    def _settle_tsumo(self, p: int, agari: S.Agari, turn: int) -> HandResult:
        pay = S.settle(
            agari, is_dealer=p == self.dealer, is_tsumo=True, winner=p, loser=None,
            rule=self.rule, dealer_seat=self.dealer, n_players=self.n,
        )
        deltas = list(pay.deltas)
        self._take_sticks(deltas, p)
        st = self.stats[p]
        st.tsumo += 1
        st.win_han.append(agari.total_han)
        st.win_score.append(deltas[p])
        # 自摸和的那张就是自己摸进来的牌
        self._mark_agari(p, self.players[p].drawn if self.players[p].drawn is not None else -1)
        self._log(f"  🀄 [{self._seat_name(p)}] 自摸 {agari.describe()}  {pay.detail}")
        return HandResult(
            kind="tsumo", winner=p, agari=agari, deltas=tuple(deltas), turns=turn,
            winners=(p,), agaris=(agari,),
        )

    def _settle_multi_ron(
        self, winners: list[tuple[int, S.Agari]], from_p: int, tile: int, turn: int
    ) -> HandResult:
        """双响 / 三响：每家各按满额结算，供托归头跳。"""
        deltas = [0] * self.n
        head = winners[0][0]
        detail = []
        for q, agari in winners:
            pay = S.settle(
                agari, is_dealer=q == self.dealer, is_tsumo=False, winner=q,
                loser=from_p, rule=self.rule, dealer_seat=self.dealer,
                n_players=self.n,
            )
            for i in range(self.n):
                deltas[i] += pay.deltas[i]
            st = self.stats[q]
            st.ron += 1
            st.win_han.append(agari.total_han)
            detail.append(f"{self._seat_name(q)} {agari.describe()} {pay.detail}")
        if self.sticks:
            deltas[head] += self.sticks * self.rule.riichi_stick
            self.sticks = 0
        for q, _ in winners:
            self.stats[q].win_score.append(deltas[q])
        self.stats[from_p].deal_in += 1
        self.stats[from_p].deal_in_score.append(-deltas[from_p])
        self._log(
            f"  🀄 多家和牌！{' ｜ '.join(detail)}"
            f"（放铳 {self._seat_name(from_p)}）"
        )
        return HandResult(
            kind="ron", winner=head, loser=from_p, agari=winners[0][1],
            deltas=tuple(deltas), turns=turn,
            winners=tuple(q for q, _ in winners), agaris=tuple(a for _, a in winners),
        )

    def _settle_ron(self, p: int, from_p: int, tile: int, agari: S.Agari, turn: int) -> HandResult:
        pay = S.settle(
            agari, is_dealer=p == self.dealer, is_tsumo=False,
            winner=p, loser=from_p, rule=self.rule, dealer_seat=self.dealer,
            n_players=self.n,
        )
        deltas = list(pay.deltas)
        self._take_sticks(deltas, p)
        st = self.stats[p]
        st.ron += 1
        st.win_han.append(agari.total_han)
        st.win_score.append(deltas[p])
        self.stats[from_p].deal_in += 1
        self.stats[from_p].deal_in_score.append(-deltas[from_p])
        self._mark_agari(p, tile, from_p)
        self._log(
            f"  🀄 [{self._seat_name(p)}] 荣和 {S.tile_cn(tile)}"
            f"（放铳 {self._seat_name(from_p)}）{agari.describe()}  {pay.detail}"
        )
        return HandResult(
            kind="ron", winner=p, loser=from_p, agari=agari, deltas=tuple(deltas), turns=turn,
            winners=(p,), agaris=(agari,),
        )

    def _take_sticks(self, deltas: list[int], winner: int) -> None:
        if self.sticks:
            deltas[winner] += self.sticks * self.rule.riichi_stick
            self.sticks = 0

    def _exhaustive_draw(self, turn: int) -> HandResult:
        tenpai = [
            shanten(self.players[p].hand, len(self.players[p].melds)) == 0
            for p in range(self.n)
        ]
        deltas = [0] * self.n
        n_t = sum(tenpai)
        if 0 < n_t < self.n:
            gain = self.rule.noten_penalty // n_t
            pay = self.rule.noten_penalty // (self.n - n_t)
            for p in range(self.n):
                deltas[p] = gain if tenpai[p] else -pay
        for p in range(self.n):
            self.stats[p].draws += 1
        self._log(f"  🌊 流局（听牌 {''.join('T' if t else '-' for t in tenpai)}）")
        self._act("ryukyoku", tenpai=list(tenpai), deltas=list(deltas))
        return HandResult(kind="draw", deltas=tuple(deltas), tenpai=tuple(tenpai), turns=turn)

    # -- 与 agent 的接口 -----------------------------------------------------
    def view_for(self, p: int) -> View:
        pl = self.players[p]
        return View(
            player=p,
            hand=list(pl.hand),
            drawn=pl.drawn,
            melds=list(pl.melds),
            hand_red=list(pl.red),
            discards=[list(q.pond) for q in self.players],
            melds_all=[list(q.melds) for q in self.players],
            kita=[q.kita for q in self.players],
            scores=list(self.scores),
            dora_indicators=[self._indicator(i) for i in range(self.dora_slots)],
            round_wind=self.round_wind,
            seat_wind=self.seat_wind(p),
            wall_left=self.wall_left(),
            turn=self.turn + 1,
            riichi=[q.riichi for q in self.players],
            ippatsu=[q.ippatsu for q in self.players],
            is_dealer=(p == self.dealer),
            last_discard=self.last_discard,
            rule=self.rule,
        )

    def _ask_choice(self, p: int, kind: str, options: list, tile: int | None = None) -> int:
        view = self.view_for(p)
        i = self.agents[p].decide(Decision(kind, options, tile), view)
        if not isinstance(i, int) or not (0 <= i < len(options)):
            i = 0
        return i


def play_match(
    agents: Sequence,
    seed: int = 0,
    rule: S.Rule = S.DEFAULT_RULE,
    rounds: int = 3,
    on_event: Callable[[str], None] | None = None,
) -> MatchResult:
    """跑一整场（东风战）。"""
    return Game(agents, seed=seed, rule=rule, rounds=rounds, on_event=on_event).play()
