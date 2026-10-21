"""ELO 混战池：让不同模型、不同版本互相打，打出一个天梯。

规则（按需求定的，都能在 UI 里调）：

* 每个参战者初始 **500 分**
* 一场 = 一局三麻 / 四麻（3 或 4 个人上桌）
* 多人大逃杀式的 ELO：桌上每两个人之间都算一次对局，名次靠前就是赢
* **低于 0 分保底为 0**，不出局（一直在池子里打）
* 匹配时**桌上分差不超过 200 分**（可关掉严格模式放宽）

为什么要多人 ELO 而不是简单「赢了加分」：一桌三个人的实力差会互相影响，
两两结算才能让「跟强者打得接近」和「虐菜赢」得到不同的分数。

跑法::

    python -m aimodel.arena --matches 200              # 命令行跑 200 场，打印天梯
    python -m aimodel.arena --players 4 --matches 100  # 四麻池
    python -m aimodel.arena --reset                    # 清空战绩重来
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import external, variants
from .agents import Agent, HeuristicAgent, MLAgent, RandomAgent
from . import sanma as S
from .engine import Game
from .net import Net

STATE_FILE = Path(__file__).resolve().parent / "arena_state.json"

#: 参战者的来源：版本目录里的模型 + 两个内置基线
BUILTIN = ("teacher", "random")

#: 官方天梯里加外部参战者的配置开关（**默认不设 = 关闭** ✓）
EXTERNAL_ENV = "POTATOMAJ_ARENA_EXTERNAL"
EXTERNAL_TIMEOUT_ENV = "POTATOMAJ_ARENA_EXTERNAL_TIMEOUT"
EXTERNAL_REUSE_ENV = "POTATOMAJ_ARENA_REUSE_EXTERNAL"


def externals_from_env() -> list[str]:
    """读 ``$POTATOMAJ_ARENA_EXTERNAL``（逗号 / 分号 / 空格分隔）。

    默认没设这个变量 -> 返回空列表 -> 天梯阵容和以前**一模一样** ✓
    """
    raw = os.environ.get(EXTERNAL_ENV, "").strip()
    if not raw:
        return []
    return [x for x in re.split(r"[,;\s]+", raw) if x]


def normalize_externals(externals: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """把外部 AI 名字归一成 spec key；不认识的名字**明确报错**。"""
    if not externals:
        return []
    if isinstance(externals, str):
        externals = [externals]
    out: list[str] = []
    for raw in externals:
        key = external.normalize_key(str(raw).strip())
        if key is None:
            raise ValueError(
                f"不认识的外部 AI：{raw!r}；可用：{external.spec_help()}"
            )
        if key not in out:
            out.append(key)
    return out


#: **初始分**（三麻/四麻同）。每期开局都从这里重新起算 ✓
#: ⚠️ 回填脚本 ``tools/elo_reseason.py`` 也读这一个常量 ✗ 不许另写一份数字 ✗
INITIAL_RATING = 500.0

#: ELO 的期望分底数：``10 ** ((对手分 - 自己分) / 400)``
ELO_SCALE = 400.0


@dataclass
class Participant:
    """一个参战者（某个模型的某个版本，或者教师 / 随机基线）。"""

    key: str  #: 唯一标识，例如 "maj-flashv1.2#2"（同一个模型的第 2 号参赛者）
    label: str  #: 显示名
    players: int  #: 3 或 4
    kind: str  #: model / teacher / random / strong / external
    source: str = ""  #: 权重来源（同一个模型派出的多个参赛者共用一份权重）
    version: str = ""  #: 模型版本（基线为空）
    style: str = ""  #: 风格（基线为空）
    rating: float = INITIAL_RATING
    games: int = 0
    wins: int = 0  #: 一位次数
    last: int = 0  #: 末位次数
    score_term: float = 0.0  #: 累计「点数段位分」
    place_term: float = 0.0  #: 累计「额外段位分」
    hands: int = 0  #: 参与过的局数（算和了率的分母）
    agari: int = 0  #: 和了次数（自摸 + 荣和）
    deal_in: int = 0  #: 放铳次数
    ranks: list[int] = field(default_factory=list)
    floor_hits: int = 0  #: 有多少次被打到 0 分（保底触底次数）
    external_failures: int = 0  #: 外部 AI 超时/崩溃/降级的次数（kind="external"）
    matches: list[dict] = field(default_factory=list)  #: 逐场明细（点开看战绩用）
    history: list[float] = field(default_factory=list)  #: 分数曲线
    #: 外部参战者（Mortal 等）：spec key（``mortal-akagi``），空 = 不是外部
    external: str = ""
    external_games: int = 0  #: 外部进程上过桌的场数
    external_failed: int = 0  #: 其中「崩了 / 超时 / 非法动作」的场数
    external_errors: int = 0
    external_timeouts: int = 0
    external_crashes: int = 0
    external_bad_actions: int = 0
    external_last_error: str = ""
    #: 启动时 ``external.check_spec`` 的结果（不启动进程；只用于「装没装」展示）
    external_available: bool = True
    external_reason: str = ""

    def reset_for_season(self) -> None:
        """**开新一期**：分数回到初始分，本期成绩清零。

        为什么要单独一个方法：官方天梯改成**每期独立**了 —— 每期从
        :data:`INITIAL_RATING` 重新起算，跨期不可比 ✓
        （用户 2026-10-21 实测：rating 946 -> 1209 -> 1462 -> 1680 一路累积 ✗）

        ⚠️ 这里只清「**本期**」的东西 ✗ —— 服务级的累计进度 ``Arena.done_total``
           不归它管 ✓（配额/调度看那个 ✓ 两个计数器绝不能混 ✗）
        """
        self.rating = INITIAL_RATING
        self.games = 0
        self.wins = 0
        self.last = 0
        self.score_term = 0.0
        self.place_term = 0.0
        self.hands = 0
        self.agari = 0
        self.deal_in = 0
        self.ranks = []
        self.floor_hits = 0
        self.external_failures = 0
        self.matches = []
        self.history = []
        # 外部参战者（Mortal 等）的**本场/本期**统计也要归零 ✓
        # ⚠️ ``external``（spec key）/ ``external_available`` / ``external_reason``
        #    **不能清** ✗ —— 那是「装没装」的启动事实，不是本期成绩 ✓
        self.external_games = 0
        self.external_failed = 0
        self.external_errors = 0
        self.external_timeouts = 0
        self.external_crashes = 0
        self.external_bad_actions = 0
        self.external_last_error = ""

    @property
    def alive(self) -> bool:
        #: **不出局**：0 分保底，掉到 0 就停在 0，继续在池子里打
        return True

    @property
    def at_floor(self) -> bool:
        return self.rating <= 0.0

    @property
    def avg_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    @property
    def first_rate(self) -> float:
        """一位率。"""
        return self.wins / self.games if self.games else 0.0

    @property
    def agari_rate(self) -> float:
        """和了率（每局和牌的比例）。"""
        return self.agari / self.hands if self.hands else 0.0

    @property
    def deal_in_rate(self) -> float:
        return self.deal_in / self.hands if self.hands else 0.0


#: 早期权重的显示名（看名字不知道是啥，直接写清楚）
LOOSE_LABELS = {
    "policy_bc4": "bc4（纯克隆基线·5000局）",
    "policy": "policy（v1 早期·带强化学习）",
}


class ExternalSeatAgent(Agent):
    """天梯里的一个**外部进程座位** —— 核心是「崩了也不许炸整期」。

    设计上的三道保险：

    1. :class:`~aimodel.external.ExternalMJaiAgent` 自己带 ``fallback``：
       超时 / 进程退出 / 非法动作 -> 该座位当场改用教师出牌；
    2. 这一层再包 ``try/except Exception``：连协议桥自己写错（IndexError 之类）
       也只会让**这一局**降级，不会把异常抛进 ``Game.play()``；
    3. 进程死了 / 降级过 -> 本场结束后关掉，下一场重建一个干净进程；
       没失败且 spec 支持 ``reuse_process`` -> 用 ``begin_game()`` 复用
       （Mortal 重发 ``start_game`` 即可重置，省掉每场 ~2s 的 torch 启动）。

    ``failed`` / 各类计数是**本场**的；场结束后由 :meth:`Arena.run_match`
    累加进 :class:`Participant`，并写进 ``matches.jsonl`` 的 ``external_failed``。
    """

    #: 给 :func:`aimodel.external.attach` 认：它和 ExternalMJaiAgent 一样有 feed()
    external_agent = True
    name = "外部AI"

    def __init__(
        self,
        participant: Participant,
        *,
        timeout: float = 8.0,
        max_restarts: int = 1,
        reuse: bool = True,
    ) -> None:
        self.participant = participant
        self.spec_key = participant.external or participant.source or participant.key
        self.timeout = float(timeout)
        self.max_restarts = max(0, int(max_restarts))
        try:
            self.spec = external.get_spec(self.spec_key)
        except external.ExternalAIError:
            self.spec = None
        self.reuse = bool(reuse and getattr(self.spec, "reuse_process", False))

        self._agent: external.ExternalMJaiAgent | None = None
        self._fallback: Agent = HeuristicAgent(0)
        self._seed = 0
        self._player_id = 0
        self._seen_restarts = 0

        #: 本场统计（每场 begin_match() 清零）
        self.decisions = 0
        self.errors = 0
        self.timeouts = 0
        self.crashes = 0
        self.bad_actions = 0
        self.failed = False
        self.failure_reason = ""
        self.failure_kind = ""

    # -- 生命周期 -----------------------------------------------------------
    def _make(self) -> None:
        """建外部 agent（不启动进程；第一次 feed 时才起）。失败只降级。"""
        try:
            self._agent = external.ExternalMJaiAgent(
                self.spec_key,
                seed=self._seed,
                player_id=self._player_id,
                players=4,
                timeout=self.timeout,
                start_timeout=max(60.0, self.timeout * 3),
                fallback=self._fallback,
                max_restarts=self.max_restarts,
                on_error=self._on_error,
                allow_missing=True,
            )
            self._seen_restarts = int(getattr(self._agent, "_restarts", 0))
        except Exception as exc:                       # noqa: BLE001
            self._agent = None
            self._record(exc)

    def _on_error(self, _agent: object, exc: BaseException) -> None:
        self._record(exc)

    def _record(self, exc: BaseException) -> None:
        if not self.failed:
            self.failed = True
            self.failure_reason = f"{type(exc).__name__}: {exc}"
            self.failure_kind = type(exc).__name__
        self.errors += 1
        if isinstance(exc, external.ExternalAITimeout):
            self.timeouts += 1
        elif isinstance(exc, external.ExternalAICrashed):
            self.crashes += 1
        elif isinstance(exc, external.ExternalAIProtocolError):
            self.bad_actions += 1

    def _health(self) -> None:
        """进程偷偷重启过（Mortal 自己 recover）也要从下一手起改走教师。"""
        ag = self._agent
        if ag is None:
            return
        if ag.degraded and not self.failed:
            self._record(external.ExternalAIError(
                ag.degrade_reason or "外部 AI 已降级"))
            return
        restarts = int(getattr(ag, "_restarts", 0))
        if restarts > self._seen_restarts and not self.failed:
            self._seen_restarts = restarts
            self._record(external.ExternalAIError(
                f"外部进程重启了 {restarts} 次（已重放本局事件，仍按失败计）"))

    def begin_match(self, seed: int, player_id: int) -> None:
        self._seed = int(seed)
        self._player_id = int(player_id)
        self._fallback = HeuristicAgent(self._seed)
        self.decisions = 0
        self.errors = 0
        self.timeouts = 0
        self.crashes = 0
        self.bad_actions = 0
        self.failed = False
        self.failure_reason = ""
        self.failure_kind = ""

        if self._agent is not None:
            if self.reuse:
                try:
                    self._agent.begin_game(player_id=player_id, seed=seed)
                    self._seen_restarts = int(getattr(self._agent, "_restarts", 0))
                except Exception as exc:               # noqa: BLE001
                    self._record(exc)
                    self.close()
            else:
                self.close()
        if self._agent is None and not self.failed:
            self._make()

    def end_match(self) -> None:
        """本场收尾：失败过 / 不支持复用 -> 关进程；否则留着下一场复用。"""
        if self._agent is None:
            return
        if self.failed or not self.reuse or self._agent.degraded:
            self.close()

    def close(self) -> None:
        ag, self._agent = self._agent, None
        if ag is not None:
            try:
                ag.close()
            except Exception:                          # noqa: BLE001
                pass

    def mark_failed(self, exc: BaseException) -> None:
        """整局级异常（比如 Game.play 因外部座位炸了）——记失败并关进程。"""
        self._record(exc)
        self.close()

    @property
    def player_id(self) -> int:
        return self._player_id

    # -- Agent 接口 ---------------------------------------------------------
    def decide(self, decision, view) -> int:
        self.decisions += 1
        if self.failed or self._agent is None:
            return int(self._fallback.decide(decision, view))
        try:
            ret = int(self._agent.decide(decision, view))
            n = len(decision.options)
            if not (0 <= ret < n):
                raise external.ExternalAIProtocolError(
                    f"外部 AI 返回的下标 {ret} 越界（这一手有 {n} 个合法选项）")
            self._health()
            return ret
        except Exception as exc:                       # noqa: BLE001
            self._record(exc)
            return int(self._fallback.decide(decision, view))

    def feed(self, action: dict) -> None:
        if self.failed or self._agent is None:
            return
        try:
            self._agent.feed(action)
            self._health()
        except Exception as exc:                       # noqa: BLE001
            self._record(exc)

    def summary(self) -> dict:
        return {
            "spec": self.spec_key,
            "decisions": self.decisions,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "crashes": self.crashes,
            "bad_actions": self.bad_actions,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "reuse": self.reuse,
        }


def loose_participants(players: int | None = None, copies: int = 3) -> list[Participant]:
    """把 ``weights/`` 里**不符合 maj-* 命名**的权重也收进来（比如 policy_bc4）。

    人数从网络维度推断：特征 486 = 三麻、560 = 四麻。
    """
    from .encode import layout_for
    from .net import Net

    out: list[Participant] = []
    for f in sorted(variants.WEIGHTS_DIR.glob("*.npz")):
        stem = f.stem
        if stem.startswith("maj-") or stem.endswith(("_bcbase", "_prev")):
            continue
        try:
            net = Net.load(f)
        except Exception:                       # noqa: BLE001
            continue
        n = 3 if net.dim < 500 else 4
        if players and n != players:
            continue
        if net.dim != layout_for(n)[1]:
            continue
        label = LOOSE_LABELS.get(stem, stem)
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"{stem}#{i}" if copies > 1 else stem,
                label=f"{label}{suffix}", source=stem, players=n, kind="model",
                version="早期", style="—",
            ))
    return out


def collect(
    players: int | None = None,
    include: list[str] | None = None,
    copies: int = 3,
    externals: str | list[str] | None = None,
) -> list[Participant]:
    """把所有可用的模型（含各版本）+ 内置基线 + **可选的外部 AI** 收进来。

    **每个模型版本派出 ``copies`` 个独立参赛者**，各记各的分。
    它们用的是同一份权重，但座位、牌运、对手组合都不一样 ——
    分开记才能看出「同一个模型自己的方差有多大」，而不是被平均掉。

    ``externals`` 为空 = **和以前完全一样**（不碰外部进程）✓
    """
    out: list[Participant] = []
    seen: set[str] = set()

    def add(spec: variants.ModelSpec) -> None:
        if not spec.path.exists() or (players and spec.players != players):
            return
        if spec.name in seen:
            return
        seen.add(spec.name)
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"{spec.name}#{i}" if copies > 1 else spec.name,
                label=f"{spec.name}{suffix}",
                source=spec.name,
                players=spec.players, kind="model",
                version=spec.version, style=variants.STYLE_CN.get(spec.style, spec.style),
            ))

    # ★ 注册表里的（当前版本）+ 盘上所有带版本号的权重 ✓
    #   ⚠️ 这两段被我删过一次 ✗ —— 用 s[:start]+s[end:] 做拼接时，
    #      把「_want 那段的开头」到「loose 那段的开头」之间的东西全删了 ✓
    #      症状是**池子里只剩教师和随机**，一个模型都没有 ✗
    #      （靠 `--models` 试跑时看到「2 个参战者」才发现 ✓ —— 改完必须试跑 ✓）
    for spec in variants.all_specs():          # 当前版本
        add(spec)
    for f in sorted(variants.WEIGHTS_DIR.glob("maj-*v*.npz")):
        if f.stem.endswith(("_bcbase", "_prev")):
            continue
        try:
            add(variants.get(f.stem))
        except KeyError:
            continue

    # 不在命名规则里的早期权重（policy_bc4 / policy 这些）也拉进来当参照
    for p in loose_participants(players, copies):
        if p.source in seen or (players and p.players != players):
            continue
        seen.add(p.source)
        out.extend(p if isinstance(p, list) else [p])
    if include:
        keys = {k.lower() for k in include}
        out = [p for p in out if p.source.lower() in keys]
    for n in (3, 4):
        if players and n != players:
            continue
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"teacher{n}#{i}", label=f"教师（{n}麻基线）{suffix}",
                source=f"teacher{n}", players=n, kind="teacher",
            ))
            out.append(Participant(
                key=f"random{n}#{i}", label=f"随机（{n}麻基线）{suffix}",
                source=f"random{n}", players=n, kind="random",
            ))
        # 规则型强化 agent（aimodel/strong.py）：和教师/随机一样是**基线**
        # 而不是权重模型，所以不进 variants 注册表、也不吃 copies。
        out.append(Participant(
            key=f"strong{n}", label=f"规则·强（{n}麻）",
            source="strong", players=n, kind="strong", style="规则",
        ))

    # 外部 AI（Mortal 等）：**默认不进来**；调用方显式给 externals 才加。
    # 名字解析 / 安装检查都在这做；没装也照加（比赛时降级教师并记
    # external_failed），这样「装坏了」不会让整期开不了，但能在页面上看见。
    for key in normalize_externals(externals):
        sp = external.get_spec(key)
        if players and sp.players != players:
            raise ValueError(
                f"外部 AI {sp.title} 只支持 {sp.players} 麻，当前天梯是 {players} 麻；"
                f"请用 --players {sp.players}（外部 AI 目前只有四麻）"
            )
        ok, why = external.check_spec(sp)
        out.append(Participant(
            key=f"external:{sp.key}",
            label=sp.title,
            players=sp.players,
            kind="external",
            source=sp.key,
            version="外部",
            style="外部 AI",
            external=sp.key,
            external_available=bool(ok),
            external_reason=why,
        ))
    return out


# ---------------------------------------------------------------------------
# ELO 结算的**纯函数**部分（不碰 Participant，可单独回声/回填）
# ---------------------------------------------------------------------------


def elo_place_terms(ratings: list[float], ranks: list[int], k: float) -> list[float]:
    """多人大逃杀 ELO 的「**额外段位分**」增量（一局，纯函数）。

    桌上**每两个人之间**都算一次对局：名次靠前 = 赢 1、靠后 = 赢 0、并列 = 0.5，
    再乘上「期望胜率」``1 / (1 + 10 ** ((对手分 - 自己分) / 400))``，最后除以
    ``n - 1`` 摊回单人对局的量级。这就是 :meth:`Arena._settle` 里那段两重循环。

    ⚠️ **抽出来是为了「两条路径必一致」** ✗ 不是为了好看 ✗
       —— 实时跑局（``run_match`` -> ``_settle``）和回填（``tools/elo_reseason.py``
       用 ``matches.jsonl`` 里的 ``ranks`` 重算）**必须走同一份算法** ✓
       否则重算出来的分和真打出来的分对不上，而两边都不会报错（静默错 ✗）
    """
    n = len(ratings)
    terms = [0.0] * n
    if n < 2:
        return terms                     # 1 个人没法两两结算（也别除 0 ✗）
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if i == j:
                continue
            s = (1.0 if ranks[i] < ranks[j]
                 else (0.0 if ranks[i] > ranks[j] else 0.5))
            e = 1.0 / (1.0 + 10 ** ((ratings[j] - ratings[i]) / ELO_SCALE))
            acc += s - e
        terms[i] = k / (n - 1) * acc
    return terms


def point_terms(scores: list[int], start_score: int, divisor: float) -> list[float]:
    """「**点数段位分**」增量：``(最终点数 - 初始点数) / divisor``。

    ``divisor`` 为 0 就是关掉点数项（只按名次算）✓
    """
    if not divisor:
        return [0.0] * len(scores)
    return [(s - start_score) / divisor for s in scores]


class Arena:
    """一张天梯榜 + 匹配 + ELO 结算。"""

    def __init__(
        self,
        players: int = 3,
        k: float = 24.0,
        max_gap: float = 200.0,
        strict: bool = True,
        seed: int = 0,
        participants: list[Participant] | None = None,
        copies: int = 3,
        divisor: float = 500.0,
        models: list[str] | None = None,
        externals: str | list[str] | None = None,
        external_timeout: float | None = None,
        external_max_restarts: int | None = None,
        external_reuse: bool | None = None,
    ) -> None:
        self.copies = copies
        #: 点数段位分的除数：``(最终点数 - 初始点数) / divisor``。
        #: 越小点数项越强势（500 时一局能到 ±90，会把名次项完全压过去）；
        #: 设 0 就是关掉点数项，只按名次算。
        self.divisor = divisor
        self.players = players
        self.k = k
        self.max_gap = max_gap
        self.strict = strict
        self.seed = seed
        #: 外部参战者（spec key）。None = 看环境变量；[] = 明确一个都不要。
        if participants is not None and externals is None:
            externals = [p.external for p in participants
                         if p.kind == "external" and p.external]
        self.externals = normalize_externals(
            externals_from_env() if externals is None else externals)
        if external_timeout is None:
            raw = os.environ.get(EXTERNAL_TIMEOUT_ENV, "").strip()
            try:
                external_timeout = float(raw) if raw else 8.0
            except ValueError:
                external_timeout = 8.0
        self.external_timeout = max(0.1, float(external_timeout))
        if external_max_restarts is None:
            external_max_restarts = 1
        self.external_max_restarts = max(0, int(external_max_restarts))
        if external_reuse is None:
            raw = os.environ.get(EXTERNAL_REUSE_ENV, "").strip().lower()
            external_reuse = raw not in ("0", "false", "no", "off")
        self.external_reuse = bool(external_reuse)
        self.participants = (
            participants if participants is not None
            else collect(players, include=models, copies=copies,
                         externals=self.externals)
        )
        self.n_matches = 0
        #: **服务级累计**场次（跨期只增不减 ✓）。与 :attr:`n_matches` 分开 ✗：
        #:   · ``n_matches`` = **本期**场次序号（派生牌山种子 + 页面「本期进度」）
        #:   · ``done_total`` = 这个池子从建起来到现在一共跑了多少场（配额/调度看它）
        #:   混在一起就会出现「一期 4000 局，页面写 48000」那种口径错 ✗
        self.done_total = 0
        self.log: list[str] = []
        self._nets: dict[str, Net] = {}
        self._rng = random.Random(seed)
        #: 每个外部参战者一个可复用座位（跨场保留进程；失败后自动重建）
        self._external_seats: dict[str, ExternalSeatAgent] = {}
        #: 只加不改的汇总计数（``/api/arena`` 里能看到）
        self.external_matches = 0
        self.external_failed = 0
        self.external_decisions = 0

    # -- 持久化 -------------------------------------------------------------
    def save(self, path: Path = STATE_FILE) -> None:
        data = {
            "players": self.players, "k": self.k, "max_gap": self.max_gap,
            "n_matches": self.n_matches, "seed": self.seed, "copies": self.copies,
            "divisor": self.divisor,
            "done_total": self.done_total,
            "externals": list(self.externals),
            "external_timeout": self.external_timeout,
            "external_max_restarts": self.external_max_restarts,
            "external_reuse": self.external_reuse,
            "external_matches": self.external_matches,
            "external_failed": self.external_failed,
            "external_decisions": self.external_decisions,
            "ts": time.time(),
            "participants": [asdict(p) for p in self.participants],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "Arena | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        parts = [Participant(**p) for p in d.get("participants", [])]
        exts = d.get("externals")
        if exts is None:
            exts = [p.external for p in parts
                    if p.kind == "external" and p.external]
        a = cls(players=d.get("players", 3), k=d.get("k", 24.0),
                max_gap=d.get("max_gap", 200.0), participants=parts,
                copies=d.get("copies", 3), seed=d.get("seed", 0),
                divisor=d.get("divisor", 500.0), externals=exts,
                external_timeout=d.get("external_timeout", 8.0),
                external_max_restarts=d.get("external_max_restarts", 1),
                external_reuse=d.get("external_reuse", True))
        a.n_matches = d.get("n_matches", 0)
        # 老格式的 arena.json 没有 done_total（那时 n_matches 本身就是累计值）✓
        a.done_total = int(d.get("done_total", d.get("n_matches", 0)) or 0)
        a.external_matches = d.get("external_matches", 0)
        a.external_failed = d.get("external_failed", 0)
        a.external_decisions = d.get("external_decisions", 0)
        return a

    def close_externals(self) -> None:
        """关掉所有外部进程（重置/收尾用；幂等）。"""
        for seat in self._external_seats.values():
            seat.close()
        self._external_seats.clear()


    # -- 期次边界 -----------------------------------------------------------
    def new_season(self) -> None:
        """**开新一期**：rating 回到初始分、本期成绩清零、本期场次序号归零。

        ⚠️ ``done_total`` **不清零** ✗ —— 它是服务级累计进度（配额/调度看它 ✓）
        ⚠️ ``n_matches`` **要清零** ✓ —— 它是**本期**场次序号：
           1) 派生牌山种子 ``seed_of(index)``；2) 页面「本期进度」。
           不清零的话下一期的进度会从上一期的数字接着数 ✗（口径错 ✗）
        """
        for p in self.participants:
            p.reset_for_season()
        self.n_matches = 0
        self.log.clear()

    def keyed(self) -> dict[str, Participant]:
        """``key -> Participant`` 的查找表（回填/续跑按 key 找人）。"""
        return {p.key: p for p in self.participants}

    def settle_ranks(self, group: list[Participant], ranks: list[int],
                     scores: list[int], start_score: int | None = None) -> None:
        """按**已有的名次/点数**结算一局：**不重打牌**，只跑 ELO 更新。

        给「从 ``matches.jsonl`` 回算」用（``tools/elo_reseason.py``、重启续跑重建）。

        ⚠️ 内部走的就是 :meth:`_settle`（和 ``run_match()`` 同一条路径 ✓）——
           两条路径必然一致 ✓ 这是「重算 == 真打」的地基 ✓
        """
        start = (S.rule_for(self.players).start_score
                 if start_score is None else start_score)
        self._settle(list(group), list(ranks), list(scores), None, start)
        self.n_matches += 1
        self.done_total += 1

    def replay_ranks(self, records: Iterable[dict], *, fresh: bool = True,
                     before_settle=None) -> int:
        """把一串 ``{group, ranks, scores}`` 逐局结算回来 —— **不重打牌**。

        用途（两处共用同一条路径 ✓）：
          · 回填历史：``tools/elo_reseason.py`` 按新口径重算各期
          · 重启续跑：某期只落了 ``matches.jsonl``、没有 ``arena.json`` 时重建分数

        ``fresh=True`` 先 :meth:`new_season`（本期从 :data:`INITIAL_RATING` 重开 ✓）。
        ``before_settle(group, ranks, scores)``：结算**之前**回调一次 ——
        回填脚本用它取「这一桌当时的分数」（算对手强度用 ✓），不改结算结果 ✓
        返回真正结算了几局。
        """
        if fresh:
            self.new_season()
        by_key = self.keyed()
        n = 0
        for rec in records:
            keys = [str(k) for k in (rec.get("group") or [])]
            if not keys:
                continue
            group: list[Participant] = []
            for k in keys:
                p = by_key.get(k)
                if p is None:
                    # 名单对不上就**炸** ✗ —— 悄悄跳过会算出一份「少了几个人」的假表 ✗
                    raise KeyError(f"对局里有名单外的参战者：{k!r}")
                group.append(p)
            ranks = list(rec.get("ranks") or [])
            if len(ranks) != len(group):
                raise ValueError(f"ranks 长度 {len(ranks)} != 人数 {len(group)}")
            scores = list(rec.get("scores") or [])
            if len(scores) != len(group):
                # 早期存档可能没存点数 -> 点数项按 0 记（名次项照常 ✓）
                scores = [0] * len(group)
            if before_settle is not None:
                before_settle(group, ranks, scores)
            self.settle_ranks(group, ranks, scores)
            n += 1
        return n

    # -- 匹配 ---------------------------------------------------------------
    #: 上场数低于池内均值的这个比例 -> **强制**给它一张桌（即使超 max_gap）
    QUOTA_MIN_FRAC = 0.5
    #: 被 max_gap 完全排除的人（比如分数拉得太开）-> 低于均值这个比例就强制
    #: 不能只在 50% 才救：那样长期份额会卡在均值的 50%，达不到「±20%」的验收。
    #: 取 0.85：即使随机波动也只会在 80% 附近，留出余量。
    ISOLATED_MIN_FRAC = 0.85

    def pool(self) -> list[Participant]:
        return sorted([p for p in self.participants if p.alive], key=lambda p: -p.rating)

    @staticmethod
    def _spread(group: list[Participant]) -> float:
        return group[-1].rating - group[0].rating

    def _group_around(self, alive: list[Participant], p: Participant) -> list[Participant]:
        """包含 ``p`` 的 n 人桌里**分差最小**的那张（超 max_gap 也接受）。

        用于「强制配额」和「整池没有合法桌」两种兜底 —— 保证分数拉得再开
        也不会有人被永久饿死。
        """
        n = self.players
        best: tuple[tuple[float, int], list[Participant]] | None = None
        for i in range(len(alive) - n + 1):
            group = alive[i : i + n]
            if not any(x is p for x in group):
                continue
            key = (self._spread(group), sum(x.games for x in group))
            if best is None or key < best[0]:
                best = (key, group)
        return list(best[1]) if best is not None else []

    def pick(self) -> list[Participant] | None:
        """挑一桌。

        ⚠️ **2026-10-21 修掉的致命 bug**：旧逻辑先算 ``min_spread``，只从
        「分差最小」的那批桌里挑场数少的。后果是分数拉开的人（例如
        ``maj-pro4v1.1_mac_rl`` 1680 分，和最近的人差 178.9）只要恰好不在
        「分差最小」的桌里，就**永远进不了候选** —— 一期 4000 局只上了 80 局
        （别人 889），冠军被「谁先拉开谁冻结在高位」这个 bug 系统性污染。

        新判据（用户定的）：

        1. ``spread <= max_gap`` 只是**准入条件**，不是排序键；
        2. 准入的桌里**只挑总场数最少的**（同总场数随机）；
        3. 配额：上场数 < 池内均值 50% 的人 -> **强制**给一张桌（超 max_gap 也要给）；
           被 max_gap 完全排除的人放宽到 85% 就强制 —— 否则长期份额会卡在 50%，
           达不到验收要求的「均值 ±20%」；
        4. 整个池子一张合法桌都没有时 -> 给场数最少的人凑一张，**天梯不许卡死**。

        ⚠️ **依赖语义**：``Participant.games`` 必须是**本期内场数**（每期从 0 开始）。
           每期重置由赛季生命周期负责；如果哪天改回跨期累积，这里的「谁上场少」
           会被前几期的历史带偏 —— 公平性逻辑本身没错，但输入语义必须对。
        """
        alive = sorted(self.pool(), key=lambda p: p.rating)
        n = self.players
        if len(alive) < n:
            return None

        # 合法桌：连续 n 人、分差 ≤ max_gap（老规则的**准入**部分保留）
        legal: list[list[Participant]] = []
        for i in range(len(alive) - n + 1):
            group = alive[i : i + n]
            if self._spread(group) <= self.max_gap + 1e-9:
                legal.append(group)

        avg_games = sum(p.games for p in alive) / len(alive)
        in_legal: set[int] = set()
        for group in legal:
            in_legal.update(id(p) for p in group)

        forced: list[Participant] | None = None
        if avg_games > 0:
            need: list[Participant] = []
            for p in alive:
                frac = p.games / avg_games
                if p.games == 0 or frac < self.QUOTA_MIN_FRAC:
                    need.append(p)
                elif id(p) not in in_legal and frac < self.ISOLATED_MIN_FRAC:
                    # 分数太高/太低，任何合法桌都装不下它 —— 必须定期放进来，
                    # 否则它会一直冻结在当前分数。
                    need.append(p)
            if need:
                # 最落后的人优先（同场数时低分优先，和旧行为一致）
                p = min(need, key=lambda x: (x.games, x.rating))
                forced = self._group_around(alive, p)
        if forced is None and not legal:
            # 一张合法桌都没有：不能直接返回 None（严格模式会让整期停摆）
            p = min(alive, key=lambda x: (x.games, x.rating))
            forced = self._group_around(alive, p)
        if forced:
            return forced

        if legal:
            # **只看谁上场少**；分差已经作为准入过滤过了
            total = lambda g: sum(p.games for p in g)  # noqa: E731
            fewest = min(total(g) for g in legal)
            cands = [g for g in legal if total(g) == fewest]
            return self._rng.choice(cands) if len(cands) > 1 else cands[0]

        if self.strict:
            return None
        # 放宽：拿分差最小的那一组，但别总是同样几个人
        i = min(range(len(alive) - n + 1),
                key=lambda i: alive[i + n - 1].rating - alive[i].rating)
        group = alive[i : i + n]
        self._rng.shuffle(group)
        return group

    # -- 出牌的人 -----------------------------------------------------------
    def _agent(self, p: Participant, seed: int, seat: int = 0) -> Agent:
        if p.kind == "teacher":
            return HeuristicAgent(seed)
        if p.kind == "random":
            return RandomAgent(seed)
        if p.kind == "strong":
            from .strong import make as make_strong

            return make_strong("strong", seed)
        if p.kind == "external":
            # 天梯的外部座位走 run_match -> ExternalSeatAgent（崩溃隔离 + 进程复用）；
            # 这里绝不能直连裸 ExternalMJaiAgent，否则一处异常就会炸掉整期。
            raise ValueError(
                f"{p.label} 是外部参战者，要走 run_match 的 ExternalSeatAgent 分支")
        src = p.source or p.key          # 同一个模型派出的多个参赛者共用权重
        net = self._nets.get(src)
        if net is None:
            net = Net.load(variants.WEIGHTS_DIR / f"{src}.npz")
            self._nets[src] = net
        return MLAgent(net, seed=seed, greedy=True)

    # -- 打一场 -------------------------------------------------------------
    def seed_of(self, index: int) -> int:
        """第 ``index`` 场的随机种子。**只跟种子基数和场次序号有关** ——
        同一套参战者、同一个顺序，重跑就得到一模一样的牌。"""
        return self.seed + index * 7919

    def _external_seat(self, p: Participant, seed: int, player_id: int) -> ExternalSeatAgent:
        seat = self._external_seats.get(p.key)
        if seat is None:
            seat = ExternalSeatAgent(
                p, timeout=self.external_timeout,
                max_restarts=self.external_max_restarts,
                reuse=self.external_reuse,
            )
            self._external_seats[p.key] = seat
        seat.begin_match(seed, player_id)
        return seat

    def _agents_for(self, group: list[Participant],
                    seed: int) -> tuple[list[Agent], list[ExternalSeatAgent]]:
        agents: list[Agent] = []
        seats: list[ExternalSeatAgent] = []
        for i, p in enumerate(group):
            aseed = seed * self.players + i
            if p.kind == "external":
                seat = self._external_seat(p, aseed, i)
                agents.append(seat)
                seats.append(seat)
            else:
                agents.append(self._agent(p, aseed))
        return agents, seats

    def _fallback_agents(self, group: list[Participant], seed: int) -> list[Agent]:
        """整局级重跑用：外部座位一律换成教师（同种子，确定性）。"""
        return [
            HeuristicAgent(seed * self.players + i)
            if p.kind == "external" else self._agent(p, seed * self.players + i)
            for i, p in enumerate(group)
        ]

    def run_match(self) -> dict | None:
        group = self.pick()
        if group is None:
            return None
        seed = self.seed_of(self.n_matches)
        agents, seats = self._agents_for(group, seed)
        game = Game(agents, seed=seed, rounds=self.players)
        if seats:
            external.attach(game)

        fallback_error: Exception | None = None
        try:
            try:
                r = game.play()
            except Exception as exc:                    # noqa: BLE001
                if not seats:
                    raise
                # 最后一道保险：连 ExternalSeatAgent 都没兜住（比如协议桥自己
                # 有 bug），整局用「外部座位=教师」的确定性阵容重跑一次。
                # **绝不把异常抛给整期调度器** —— 一期 4000 局不能因为
                # Mortal 崩一次就全废。
                fallback_error = exc
                for s in seats:
                    s.mark_failed(exc)
                r = Game(self._fallback_agents(group, seed),
                         seed=seed, rounds=self.players).play()
                self.log.append(
                    f"⚠️ 外部座位整局异常，已用教师重跑 seed={seed}："
                    f"{type(exc).__name__}: {exc}")

            failed = fallback_error is not None or any(s.failed for s in seats)
            ranks = list(r.ranks)
            start_score = S.rule_for(self.players).start_score
            self._settle(group, ranks, r.scores, r, start_score)
            self.n_matches += 1
            self.done_total += 1

            # 外部座位的「本场计数」-> 参战者累计 + 全局累计
            for s in seats:
                p = group[s.player_id]
                p.external_games += 1
                p.external_errors += s.errors
                p.external_timeouts += s.timeouts
                p.external_crashes += s.crashes
                p.external_bad_actions += s.bad_actions
                if s.errors:
                    p.external_failures += max(1, s.errors)
                if s.failure_reason:
                    p.external_last_error = s.failure_reason
                if s.failed:
                    p.external_failed += 1
            if seats:
                self.external_matches += 1
                self.external_decisions += sum(s.decisions for s in seats)
                if failed:
                    self.external_failed += 1

            text = "  ".join(
                f"{p.label} {r.scores[i]}({ranks[i] + 1}位"
                f" {(r.scores[i] - start_score) / self.divisor if self.divisor else 0:+.0f})"
                for i, p in enumerate(group)
            )
            if failed:
                text += "  ⚠️ external_failed"
            self.log.append(f"[{self.n_matches}] 🌱{seed}  {text}")
            del self.log[:-200]
            res = {
                "group": [p.key for p in group], "ranks": ranks,
                "scores": list(r.scores), "seed": seed,
            }
            if seats:
                # 只加字段：没有外部参战者的局**一个字段都不多**（老存档兼容 ✓）
                res.update({
                    "external": True,
                    "external_failed": bool(failed),
                    "external_specs": [s.spec_key for s in seats],
                    "external_seats": [s.player_id for s in seats],
                })
            return res
        finally:
            for s in seats:
                s.end_match()

    def _settle(self, group: list[Participant], ranks: list[int], scores: list[int],
                r=None, start_score: int = 35000) -> None:
        """一局结束给每个人加分 = **额外段位分 + 点数段位分**。

        * **额外段位分**：多人大逃杀 ELO，桌上两两结算，名次靠前算赢
        * **点数段位分**：``(最终点数 - 初始点数) / 500``，可正可负 ——
          这样「同样第 2 名，赢 2 万分」和「小输 3 千」不该是同一个分

        ⚠️ 两项增量分别来自 :func:`elo_place_terms` / :func:`point_terms` ✗
           不许在这里另写一份公式 ✗ —— 回填脚本和实时跑局必须**同一份算法** ✓
        """
        n = len(group)
        place = elo_place_terms([p.rating for p in group], ranks, self.k)
        points = point_terms(scores, start_score, self.divisor)
        for i, p in enumerate(group):
            before = p.rating
            place_term = place[i]                              # 额外段位分
            score_term = points[i]                             # 点数段位分
            raw = p.rating + place_term + score_term
            if raw <= 0.0:                       # 0 分保底，不出局
                p.floor_hits += 1
                raw = 0.0
            p.rating = raw
            p.score_term += score_term
            p.place_term += place_term
            p.games += 1
            if r is not None:                      # 和了率 / 放铳率
                st = r.stats[i]
                p.hands += st.hands
                p.agari += st.tsumo + st.ron
                p.deal_in += st.deal_in
            p.ranks.append(ranks[i])
            if ranks[i] == 0:
                p.wins += 1
            if ranks[i] == n - 1:
                p.last += 1
            p.history.append(round(p.rating, 1))
            del p.history[:-300]
            p.matches.append({
                "n": self.n_matches + 1,
                "seed": self.seed_of(self.n_matches),
                "rank": ranks[i] + 1,
                "score": scores[i],
                "pts": round(score_term, 1),
                "place": round(place_term, 1),
                "delta": round(place_term + score_term, 1),
                "rating": round(p.rating, 1),
                "opponents": [q.label for k, q in enumerate(group) if k != i],
            })
            del p.matches[:-200]
            if raw == 0.0 and before > 0:
                self.log.append(f"⬇️ {p.label} 触底 0 分（保底，不出局）")

    # -- 批量跑 -------------------------------------------------------------
    def run(self, matches: int, verbose: bool = True, on_match=None) -> int:
        done = 0
        for _ in range(matches):
            res = self.run_match()
            if res is None:
                if verbose:
                    print("⚠️ 池子里已经没有分差 ≤ "
                          f"{self.max_gap:.0f} 的组合了（活着 {len(self.pool())} 个）")
                break
            done += 1
            if on_match:
                on_match(res)
            if verbose and done % 20 == 0:
                print(f"  跑了 {done} 场…")
        return done

    # -- 报表 ---------------------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        for p in sorted(self.participants, key=lambda x: -x.rating):
            rows.append({
                "名": p.label, "模型": p.source or p.key, "类型": p.kind,
                "人数": p.players,
                "版本": p.version, "风格": p.style,
                "分数": round(p.rating, 1), "场数": p.games, "一位": p.wins,
                "末位": p.last, "平均顺位": round(p.avg_rank + 1, 2) if p.ranks else 0,
                "一位率": round(p.first_rate, 4), "和了率": round(p.agari_rate, 4),
                "放铳率": round(p.deal_in_rate, 4), "局数": p.hands,
                "点数分": round(p.score_term, 1), "名次分": round(p.place_term, 1),
                "触底": p.floor_hits,
                "外部失败次数": p.external_failures,
                "状态": "0分保底" if (p.at_floor and p.games) else "存活",
                "曲线": p.history[-120:],
                # 只加字段：老页面照常读，新页面/接口能看见外部参战者状态
                "外部": p.external,
                "外部场数": p.external_games,
                "外部失败": p.external_failed,
                "外部错误": p.external_errors,
                "外部超时": p.external_timeouts,
                "外部崩溃": p.external_crashes,
                "外部非法动作": p.external_bad_actions,
                "外部可用": p.external_available,
                "外部原因": p.external_reason,
                "外部最后错误": p.external_last_error,
            })
        return rows

    def player_detail(self, key: str) -> dict | None:
        """某个参赛者的完整战绩（点开看历史用）。"""
        p = next((x for x in self.participants if x.key == key or x.label == key), None)
        if p is None:
            return None
        return {
            "key": p.key, "label": p.label, "模型": p.source or p.key,
            "类型": p.kind,
            "版本": p.version, "风格": p.style, "人数": p.players,
            "分数": round(p.rating, 1), "场数": p.games,
            "一位": p.wins, "末位": p.last,
            "一位率": round(p.first_rate, 4), "和了率": round(p.agari_rate, 4),
            "放铳率": round(p.deal_in_rate, 4),
            "平均顺位": round(p.avg_rank + 1, 2) if p.ranks else 0,
            "名次分": round(p.place_term, 1), "点数分": round(p.score_term, 1),
            "局数": p.hands, "触底": p.floor_hits,
            "外部失败次数": p.external_failures,
            "状态": "0分保底" if (p.at_floor and p.games) else "存活",
            "曲线": p.history,
            "对局": list(reversed(p.matches)),     # 新的在前
            "外部": p.external,
            "外部场数": p.external_games,
            "外部失败": p.external_failed,
            "外部错误": p.external_errors,
            "外部超时": p.external_timeouts,
            "外部崩溃": p.external_crashes,
            "外部非法动作": p.external_bad_actions,
            "外部可用": p.external_available,
            "外部原因": p.external_reason,
            "外部最后错误": p.external_last_error,
        }

    def print_table(self) -> None:
        print(f"\n{'名次':<5}{'参战者':<22}{'版本':<7}{'风格':<6}{'分数':>8}"
              f"{'场数':>6}{'一位':>6}{'末位':>6}{'均位':>7}  状态")
        for i, r in enumerate(self.table(), 1):
            print(f"{i:<5}{r['名']:<22}{r['版本']:<7}{r['风格']:<6}{r['分数']:>8}"
                  f"{r['场数']:>6}{r['一位']:>6}{r['末位']:>6}{r['平均顺位']:>7}  {r['状态']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ELO 混战池")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--k", type=float, default=24.0, help="ELO 的 K 值（越大分数动得越快）")
    ap.add_argument("--max-gap", type=float, default=200.0, help="一桌允许的最大分差")
    ap.add_argument("--loose", action="store_true", help="分差不够就放宽（默认严格）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子基数：第 n 场的种子 = seed + n×7919（同一种子重跑得到同样的牌）")
    ap.add_argument("--divisor", type=float, default=500.0,
                    help="点数段位分的除数（越小点数项越强；0 = 关掉点数项）")
    ap.add_argument("--models", nargs="*", default=None,
                    help="只让这几个模型参赛（默认全部）✓ 直接接到 collect 的 "
                         "include 上。**要写带版本号的正名**（如 maj-normal4v1.2），"
                         "因为 include 是精确匹配 source ✓。"
                         "例：--models maj-normal4v1.2 maj-pro4v1.1")
    ap.add_argument("--copies", type=int, default=3,
                    help="每个模型版本派几个参赛者（默认 3，各记各的分）")
    ap.add_argument("--external", nargs="*", default=None,
                    help="把外部 mjai AI 加进天梯（如 mortal-akagi）；"
                         "不传 = 看 $POTATOMAJ_ARENA_EXTERNAL，再没有就关闭")
    ap.add_argument("--external-timeout", type=float, default=None,
                    help="外部 AI 等一手牌的超时秒数（默认 8；可用 "
                         "$POTATOMAJ_ARENA_EXTERNAL_TIMEOUT 覆盖）")
    ap.add_argument("--no-external-reuse", action="store_true",
                    help="每场重启外部进程（默认 Mortal 复用进程以省 ~2s/场）")
    ap.add_argument("--reset", action="store_true",
                    help="清空战绩，全部回到 500 分（0 分保底，不出局）")
    ap.add_argument("--save", action="store_true", help="跑完把状态存下来")
    args = ap.parse_args(argv)

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("已清空天梯")

    want_ext = normalize_externals(
        externals_from_env() if args.external is None else args.external)
    arena = None if args.reset else Arena.load()
    if arena is not None:
        have_ext = [p.external for p in arena.participants
                    if p.kind == "external"]
        if arena.players != args.players or (
                (args.external is not None or externals_from_env())
                and sorted(have_ext) != sorted(want_ext)):
            arena = None
    if arena is None:
        arena = Arena(players=args.players, k=args.k, max_gap=args.max_gap,
                      strict=not args.loose, seed=args.seed, copies=args.copies,
                      divisor=args.divisor, models=args.models,
                      externals=want_ext,
                      external_timeout=args.external_timeout,
                      external_reuse=not args.no_external_reuse)
        print(f"新建 {args.players} 麻池：{len(arena.participants)} 个参战者"
              f"（每个模型 {args.copies} 个），"
              f"初始 500 分，分差上限 {args.max_gap:.0f}，0 分保底（不出局）")
        if want_ext:
            print(f"  外部参战者：{', '.join(want_ext)}"
                  f"（超时 {arena.external_timeout:.1f}s，"
                  f"{'复用进程' if arena.external_reuse else '每场重启' }）")
    arena.k, arena.max_gap, arena.strict = args.k, args.max_gap, not args.loose

    t0 = time.perf_counter()
    done = arena.run(args.matches)
    dt = time.perf_counter() - t0
    arena.print_table()
    print(f"\n跑了 {done} 场，用时 {dt:.0f}s（{dt / max(1, done):.1f}s/场）")
    alive = len(arena.pool())
    print(f"存活 {alive} / {len(arena.participants)}")
    if args.save:
        arena.save()
        print(f"已保存到 {STATE_FILE}")


if __name__ == "__main__":
    main()
