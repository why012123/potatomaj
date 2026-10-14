"""一张牌桌 = 一个引擎线程 + 一个异步事件队列。

线程模型（**这条最关键，不串局的根本**）::

    asyncio 事件循环（aiohttp）
        └─ WebSocket 收发、房间列表、静态资源
              │  submit_action()  ->  Pending.answer()      （线程安全）
              │  loop.call_soon_threadsafe(push_state)      （回调到循环）
              ▼
        引擎线程（每个房间一条）
              └─ Game.play()  阻塞式规则内核，原样复用 aimodel/engine.py
                    ├─ HumanSeat.decide()  -> 挂住等人，超时教师托管
                    └─ AISeat.decide()     -> 线程池推理 + 节拍

房间之间**没有任何共享可变状态**：每个 Room 有自己的 Game、自己的座位表、
自己的事件序号。全局只有两样东西是共享的 —— 只读的模型权重缓存，
和 AI 推理线程池（带并发上限）。

服务端权威：客户端发的每条动作都先过 :meth:`Room.submit_action`，
用「当前待决策的合法选项」校验，越界的直接拒掉，不改任何状态。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from aimodel import sanma as S
from aimodel.engine import Game

from . import agents as AG
from . import protocol as P
from . import tiles as T
from .config import Config
from .recording import AsyncWriter, GameRecorder, ProfileStore

log = logging.getLogger("maj.room")
gamelog = logging.getLogger("maj.game")

#: 房间状态机
PHASE_LOBBY = "lobby"
PHASE_PLAYING = "playing"
PHASE_HAND_END = "hand_end"
PHASE_FINISHED = "finished"


class RoomError(Exception):
    """可以直接回给客户端的错误。"""


# ---------------------------------------------------------------------------
# 座位 / 连接
# ---------------------------------------------------------------------------


@dataclass
class Seat:
    seat: int
    uid: str
    name: str
    kind: str = "human"          # human | ai | waiting
    ai: str | None = None
    token: str = ""              # 座位凭证（重连用）
    host: bool = False
    online: bool = False
    ready: bool = False
    next_ack: bool = False       # 是否点了「下一局」
    timeouts: int = 0
    ai_degraded: bool = False
    ai_degrade_reason: str = ""
    #: 玩家自己关了「本局记录」-> 这个座位一条都不落盘
    no_record: bool = False
    declared_rank: str = ""
    main_mode: str = ""

    def public(self) -> dict:
        return {
            "seat": self.seat,
            "name": self.name,
            "kind": self.kind,
            "ai": self.ai,
            "host": self.host,
            "online": self.online,
            "ready": self.ready,
            "next_ack": self.next_ack,
            "ai_degraded": self.ai_degraded,
            "ai_degrade_reason": self.ai_degrade_reason,
            "no_record": self.no_record,
            "declared_rank": self.declared_rank,
            "main_mode": self.main_mode,
        }


@dataclass
class Conn:
    """一条 WebSocket 连接。``seat < 0`` 表示观战。"""

    id: str
    seat: int
    #: 观战者**当前生效**的全亮状态（只有纯 AI 房才可能为真）
    reveal: bool = False
    #: 观战者**想要**的全亮状态：请求时可能房间还不满足条件，
    #: 等条件满足（真人退光了）再自动生效 —— 愿望和现状要分开存。
    reveal_want: bool = False
    spectator: bool = True
    ws: Any = None
    uid: str = ""
    name: str = ""
    joined_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 房间
# ---------------------------------------------------------------------------


class Room:
    def __init__(
        self,
        room_id: str,
        cfg: Config,
        *,
        loop: asyncio.AbstractEventLoop,
        name: str = "",
        players: int = 4,
        host_uid: str = "",
        host_name: str = "房主",
        rounds: int = 0,
        ai_delay: float | None = None,
        seats: Iterable[tuple[str, str]] | None = None,
        spectator_reveal: bool = True,
        writer: AsyncWriter | None = None,
        profiles: ProfileStore | None = None,
    ) -> None:
        self.id = room_id
        self.cfg = cfg
        self.loop = loop
        self.name = name or f"房间 {room_id}"
        self.players = 4 if players == 4 else 3
        self.rounds = rounds or self.players
        self.ai_delay = cfg.ai_delay if ai_delay is None else ai_delay
        self.spectator_reveal = spectator_reveal

        self.rule = S.rule_for(self.players)
        self.phase = PHASE_LOBBY
        self.created_at = time.time()
        self.last_active = time.time()
        self.seq = 0

        self.lock = threading.RLock()
        self.seats: list[Seat | None] = [None] * self.players
        self.conns: dict[str, Conn] = {}
        self.chat: list[dict] = []
        self.log_lines: deque = deque(maxlen=400)
        self.last_event: dict | None = None
        self.result_view: dict | None = None
        self.match_result: dict | None = None

        self.game: Game | None = None
        self.human_seats: dict[int, AG.HumanSeat] = {}
        self.ai_seats: dict[int, AG.AISeat] = {}
        self._agents: list[Any] = []
        self.scores_before: list[int] = [self.rule.start_score] * self.players
        self.sticks_before = 0
        #: 已经落盘到第几局（一局可能被记两次，靠这个记账去重）
        self._hand_recorded = 0
        #: 牌河里每张牌是不是赤 5（引擎不带这个信息，房间自己记）
        self._pond_red: list[list[int]] = [[] for _ in range(self.players)]
        self._hand_started = False
        #: 引擎报了「和了 / 流局」，等安全点去落盘
        self._pending_hand_end = False

        self._stop = threading.Event()
        self._between_evt = threading.Event()
        self._thread: threading.Thread | None = None
        #: 广播合并：最小间隔 + 是否已排队 + 窗口内是否又有新动作
        self._push_lock = threading.Lock()
        self._push_scheduled = False
        self._push_pending = False
        self._push_last = 0.0
        self._push_min_gap = 1.0 / 30.0
        self.closed = False
        self.finished_at: float | None = None
        self.error: str | None = None

        #: 决策级记录（可关；关了就一条不写）
        self.writer = writer
        self.profiles = profiles
        self.recorder: GameRecorder | None = None

        # 房主先占 0 号位（纯 AI 房时 host_uid 传空串，这里就跳过）
        host_seated = False
        if host_uid:
            self.seats[0] = Seat(0, host_uid, host_name[:16] or "房主", host=True,
                                 token=_token(), ready=True)
            host_seated = True
        # 再来填 AI。**0 号位已经被房主占了就跳过那一格** —— 否则
        # 「房主 + 全 AI」会把房主挤到最后一个空位上去。
        given = list(seats or [])
        for i, (kind, ai) in enumerate(given):
            if i >= self.players:
                break
            if host_seated and i == 0:
                continue
            if kind == "ai":
                self.seats[i] = Seat(i, f"ai:{ai}:{i}", _ai_label(ai), kind="ai",
                                     ai=ai, ready=True)
        for i in range(self.players):
            if self.seats[i] is None:
                self.seats[i] = Seat(i, f"wait:{i}", "等待中", kind="waiting",
                                     ai="teacher", ready=True)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def human_count(self) -> int:
        return sum(1 for s in self.seats if s and s.kind == "human")

    @property
    def ai_count(self) -> int:
        return sum(1 for s in self.seats if s and s.kind == "ai")

    @property
    def dealt(self) -> bool:
        """这一局发牌了没有。

        ``Game`` 在 ``_setup()`` 里才建 ``players``；引擎线程刚起来、还没
        发牌的那一小段时间里，任何去摸 ``game.players`` 的代码都会炸。
        所有对外暴露局面的地方都要先问这一句。
        """
        g = self.game
        return g is not None and bool(getattr(g, "players", None))

    @property
    def spectator_ok(self) -> bool:
        """能不能开全亮牌观战。

        **只有桌上一张真人也没有**（纯 AI 房）才允许 —— 否则观战的人可以
        把别家手牌念给正在打的人听。这是服务端的判断，客户端说什么都不算。
        """
        if not self.spectator_reveal:
            return False
        with self.lock:
            return all(s is None or s.kind != "human" for s in self.seats)

    def can_start(self) -> bool:
        return self.phase == PHASE_LOBBY and self._thread is None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """开打。引擎线程从这里开始跑。

        **幂等**：已经开了就当成功 —— 客户端重连、双击、网络重发都可能
        把 ``room.start`` 送两次，不该报错。

        空座位由内置教师顶着（真人随时可以在开局前接过去）。
        """
        with self.lock:
            if self._thread is not None:
                return
            for s in self.seats:
                if s and s.kind == "waiting":
                    s.kind = "ai"
                    s.ai = s.ai or "teacher"
                    s.name = _ai_label(s.ai)
            # **先把 agent 建出来再改状态**：模型加载失败之类的错误要能
            # 原样抛给调用方，而不是留下一个「phase=playing 但引擎没跑」的死房间。
            self.recorder = self._open_recorder()
            seat_agents = self._build_agents()
            log.info("房间 %s 开局：记录器=%s，座位=%s", self.id,
                     "开" if self.recorder else "关",
                     [type(a).__name__ for a in seat_agents])
            if len(seat_agents) != self.players:
                raise RoomError(f"座位没配齐（{len(seat_agents)}/{self.players}），开不了桌")
            self._agents = seat_agents
            self.phase = PHASE_PLAYING
            self._thread = threading.Thread(target=self._run, name=f"room-{self.id}",
                                            daemon=True)
        self._thread.start()
        self.broadcast_later()

    def close(self, reason: str = "") -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self._stop.set()
        self._between_evt.set()
        for hs in list(self.human_seats.values()):
            hs.cancel(reason)
        log.info("房间 %s 关闭 %s", self.id, f"（{reason}）" if reason else "")

    # ------------------------------------------------------------------
    # 引擎
    # ------------------------------------------------------------------
    def _build_agents(self) -> list[Any]:
        """按座位表造 agent（真人座位 / AI 座位）。"""
        seat_agents: list[Any] = []
        seed0 = abs(hash(self.id)) % 100000
        for i in range(self.players):
            s = self.seats[i]
            if s is None:
                continue
            if s.kind == "human":
                hs = AG.HumanSeat(
                    i,
                    timeout=self.cfg.human_timeout,
                    fallback=AG.HeuristicAgent(seed=seed0 + i * 977),
                    on_pending=self._on_pending,
                    on_resolved=self._on_resolved,
                    recorder=self.recorder,
                )
                self.human_seats[i] = hs
                seat_agents.append(hs)
            else:
                ai = AG.AISeat(
                    i, s.ai or "teacher",
                    workers=self.cfg.ai_workers,
                    delay=self.ai_delay,
                    seed=seed0 + i * 7919,
                    on_error=self._on_ai_error,
                    recorder=self.recorder,
                )
                self.ai_seats[i] = ai
                s.ai_degraded = ai.degraded
                s.ai_degrade_reason = ai.degrade_reason
                seat_agents.append(ai)
        return seat_agents

    def _open_recorder(self) -> GameRecorder | None:
        """开一份决策级记录。**任何失败都只是没有记录，对局照打。**"""
        if self.writer is None or self.profiles is None:
            return None
        try:
            seats = []
            for i, s in enumerate(self.seats):
                if s is None:
                    continue
                uid = s.uid if s.kind == "human" else f"ai:{s.ai or 'teacher'}"
                prof = self.profiles.get(
                    uid, name=s.name, kind=("ai" if s.kind == "ai" else "human")
                )
                seats.append({
                    "seat": i,
                    "uid": uid,
                    "name": s.name,
                    "kind": s.kind,
                    "ai": s.ai,
                    "host": s.host,
                    "elo": round(prof.elo, 2),
                    "games": prof.games,
                    "no_record": s.no_record,
                    "declared_rank": s.declared_rank,
                    "main_mode": s.main_mode,
                    "profile": prof.as_dict(),
                })
            return GameRecorder(
                writer=self.writer,
                profiles=self.profiles,
                room_id=self.id,
                room_name=self.name,
                players=self.players,
                rounds=self.rounds,
                seats=seats,
                opt_out=[s.seat for s in self.seats if s and s.no_record],
                seed=abs(hash(self.id)) % (2**31 - 1),
                ai_delay=self.ai_delay,
            )
        except Exception:  # noqa: BLE001
            log.exception("房间 %s 初始化记录器失败（本局不记录）", self.id)
            return None

    def set_no_record(self, seat: int, on: bool) -> None:
        """玩家切换「本局不记录」。**开了就真的不记。**"""
        with self.lock:
            s = self.seats[seat] if 0 <= seat < self.players else None
            if s is None:
                return
            s.no_record = bool(on)
            if self.recorder is not None:
                if on:
                    self.recorder.opt_out.add(seat)
                else:
                    self.recorder.opt_out.discard(seat)
        log.info("房间 %s 座位 %d 记录开关 -> %s", self.id, seat, "关" if on else "开")

    def _run(self) -> None:
        gamelog.info("=== 房间 %s 开打 人数=%d 局数=%d AI节拍=%.1fs ===",
                     self.id, self.players, self.rounds, self.ai_delay)
        try:
            game = Game(
                self._agents,
                seed=abs(hash(self.id)) % (2**31 - 1),
                rule=self.rule,
                rounds=self.rounds,
                on_event=self._on_engine_log,
                on_between=self._between,
            )
            # 引擎的结构化动作流（``aimodel/engine.py`` 的 ``_act``）。
            # **一个钩子就够**：牌桌渲染、动画、决策级记录都挂它，
            # 不需要在 engine.py 里再加任何东西。
            game.on_action = self._on_action
            with self.lock:
                self.game = game
            self._drain_events()
            self.broadcast_later()
            result = game.play()
            self._record_hand_end(game)      # 补最后一局
            with self.lock:
                self.match_result = P.match_result_view(result)
                self.phase = PHASE_FINISHED
                self.finished_at = time.time()
            gamelog.info("=== 房间 %s 对局结束 scores=%s ===", self.id, result.scores)
            if self.recorder is not None:
                try:
                    deltas = self.recorder.end_match(result)
                    gamelog.info("[%s] ELO 变化 %s", self.id,
                                 [round(d, 1) for d in deltas])
                except Exception:  # noqa: BLE001
                    log.exception("房间 %s 记录对局结果失败", self.id)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不能让服务端崩
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("房间 %s 引擎异常", self.id)
            if self.recorder is not None:
                try:
                    self.recorder.abort(self.error)
                except Exception:  # noqa: BLE001
                    pass
            with self.lock:
                self.phase = PHASE_FINISHED
                self.finished_at = time.time()
            self._push_event(None, {"type": "error",
                                    "message": f"引擎异常：{self.error}"})
        finally:
            with self.lock:
                self.phase = PHASE_FINISHED
                self.finished_at = self.finished_at or time.time()
            for s in self.seats:
                if s:
                    s.next_ack = False
            self.broadcast_later()

    # -- 引擎回调（都在引擎线程里）--------------------------------------
    def _on_engine_log(self, msg: str) -> None:
        gamelog.info("[%s] %s", self.id, msg)
        with self.lock:
            self.log_lines.append({"at": round(time.time(), 2), "text": msg.rstrip()})

    def _on_action(self, ev: dict) -> None:
        """引擎结构化动作流（引擎线程）。

        引擎侧的 ``_act`` 把每个动作都发出来（吃碰杠 / 摸打 / 立直 / 拔北 /
        和了 / 流局），这里做三件事：

        1. **记牌河的赤 5**：引擎的 ``Player`` 只有牌型下标，赤 5 身份丢了
           （``pond_red`` 那套在训练侧的版本里没有）。所以在这里、也就是
           ``discard`` 事件刚发出的那一刻，从 ``pl.red[tile]`` 现读一次 ——
           此刻手牌与赤 5 计数**刚被扣掉**，``red > hand`` 就说明打的正是赤 5。
        2. 攒 ``room.last_event``（前端拿它做动画提示）。
        3. **每个动作推一帧** —— 否则纯 AI 房（没有真人座位、不会走
           ``_on_pending`` 那条路）整局只在开头结尾各推一次，
           观战的人看到的就是「牌桌卡住了」。``broadcast_later`` 自带 30fps 合并。
        """
        kind = ev.get("kind")
        if kind == "start_hand":
            self._pond_red = [[] for _ in range(self.players)]
            self._hand_started = True
            self._begin_hand_record(ev)
        elif kind == "discard":
            g = self.game
            seat = ev.get("player")
            tile = ev.get("tile")
            players = getattr(g, "players", None) if g is not None else None
            if (players and isinstance(seat, int) and isinstance(tile, int)
                    and 0 <= seat < self.players and 0 <= tile < 34):
                pl = players[seat]
                # 此刻手牌与赤 5 计数刚被扣掉：手牌里同名牌比赤 5 少，
                # 说明打出去的那张正是赤 5。
                is_red = 1 if (pl.red[tile] > pl.hand[tile]) else 0
                while len(self._pond_red) <= seat:
                    self._pond_red.append([])
                self._pond_red[seat].append(is_red)
        elif kind in ("win", "ryukyoku"):
            # 和了 / 流局：盘面还完整，趁现在把这一局记下来
            self._pending_hand_end = True
        with self.lock:
            self.last_event = ev
        self.broadcast_later()

    def _drain_events(self) -> None:
        """（引擎线程的安全点）把该落的盘落掉。

        ``_act`` 是同步回调，所以这里不用排空队列 —— 只是给房间一个
        「引擎此刻不在规则计算中间」的时机去读盘面。
        """
        g = self.game
        if g is not None and self._pending_hand_end:
            self._record_hand_end(g)

    def _on_pending(self, pending: AG.Pending) -> None:
        # 引擎线程 -> 事件循环：立刻让客户端看到「轮到你了 + 倒计时」
        self.broadcast_later()

    def _on_resolved(self, pending: AG.Pending, answer: int | None,
                     timed_out: bool) -> None:
        if timed_out:
            with self.lock:
                s = self.seats[pending.seat]
                if s:
                    s.timeouts += 1
            gamelog.info("[%s] 座位 %d 思考超时，教师托管", self.id, pending.seat)

    def _on_ai_error(self, seat: int, reason: str) -> None:
        with self.lock:
            s = self.seats[seat]
            if s:
                s.ai_degraded = True
                s.ai_degrade_reason = reason
        self._push_event(None, {
            "type": "notice", "level": "warn", "seat": seat,
            "message": f"{seat} 号位的 AI 出问题了（{reason}），已降级为内置教师",
        })
        self.broadcast_later()

    def _begin_hand_record(self, ev: dict) -> None:
        rec = self.recorder
        g = self.game
        if rec is None or g is None:
            return
        try:
            rec.begin_hand(len(g.hands) + 1, {
                "dealer": ev.get("dealer"),
                "round_wind": ev.get("round_wind"),
                "dora": ev.get("dora"),
                "hands": ev.get("hands"),
                "scores": list(g.scores),
                "sticks": g.sticks,
            })
        except Exception:  # noqa: BLE001
            log.exception("记录 hand_start 失败")

    def _record_hand_end(self, game: Game) -> None:
        """把已经打完、还没落盘的局记下来（四家手牌 + 和牌详情）。

        **对局本身绝不能因为这一步失败而受影响** —— 整段包在 try 里。
        用 ``len(game.hands)`` 与 ``_hand_recorded`` 对账：一局可能在
        ``play_hand`` 里好几个地方 return（自摸 / 荣和 / 流局 / 九种九牌），
        逐个埋钩子容易漏，谁没记就补谁。
        """
        self._pending_hand_end = False
        rec = self.recorder
        # 牌河赤 5 的记录交给记录器一份副本（Player 上没这个字段）
        if rec is not None:
            rec.pond_red = {i: list(row) for i, row in enumerate(self._pond_red)}
        if rec is None:
            return
        total = len(game.hands)
        while self._hand_recorded < total:
            idx = self._hand_recorded
            self._hand_recorded += 1
            try:
                h = game.hands[idx]
                detail = (P.hand_result_view(game) if idx == total - 1 else None)
                rec.end_hand(hand_no=idx + 1, game=game, result=h, detail=detail)
            except Exception:  # noqa: BLE001
                log.exception("房间 %s 记录第 %d 局失败", self.id, idx + 1)

    # -- 局间 -----------------------------------------------------------
    def _between(self, done: int) -> None:
        """一局打完、下一局开打之前。

        引擎线程在这里停下来等：所有真人点「下一局」，或者超时自动继续。
        """
        if self._stop.is_set():
            return
        with self.lock:
            self.phase = PHASE_HAND_END
            for s in self.seats:
                if s:
                    s.next_ack = False
            self.scores_before = list(self.game.scores) if self.game else self.scores_before
            self.sticks_before = self.game.sticks if self.game else 0
        gamelog.info("[%s] 第 %d 局结束，等玩家确认下一局（最多 %.0fs）",
                     self.id, done, self.cfg.between_timeout)
        self.broadcast_later()

        deadline = time.time() + self.cfg.between_timeout
        while not self._stop.is_set():
            with self.lock:
                pending_humans = [
                    s for s in self.seats
                    if s and s.kind == "human" and s.online and not s.next_ack
                ]
            if not pending_humans:
                break
            remain = deadline - time.time()
            if remain <= 0:
                gamelog.info("[%s] 等下一局超时，自动继续", self.id)
                break
            self._between_evt.wait(min(0.25, remain))
            self._between_evt.clear()

        with self.lock:
            self.phase = PHASE_PLAYING
            for s in self.seats:
                if s:
                    s.next_ack = False
        self.broadcast_later()

    def ack_next(self) -> None:
        with self.lock:
            for c in self.conns.values():
                if 0 <= c.seat < len(self.seats):
                    s = self.seats[c.seat]
                    if s:
                        s.next_ack = True
        self._between_evt.set()
        self.broadcast_later()

    # ------------------------------------------------------------------
    # 连接 / 座位
    # ------------------------------------------------------------------
    def add_conn(
        self, conn_id: str, ws: Any, *, uid: str, name: str,
        seat: int | None = None, token: str = "", spectator: bool = False,
        reveal: bool = False,
    ) -> Conn:
        """挂一条连接到房间。

        **入座规则**（顺序很重要）：

        1. 纯 AI 房 -> 一律当观战者。邀请链接是 ``/room/XXXX``，谁点都
           不该把 AI 挤掉；而且只有这样房间才保持「没有真人」，
           全亮牌才允许。
        2. uid / 凭证对得上 -> 回自己的座位（刷新、断线重连）。
        3. 明确指定了空位 / AI 位 -> 坐过去。
        4. 还有空位 -> 随便坐。
        5. 都没了 -> 观战。
        """
        with self.lock:
            if len(self.conns) >= self.cfg.max_conns_per_room:
                raise RoomError("这个房间连接数满了")
            # 入座前的「能不能全亮」——加了自己之后会变，所以先算。
            ok_before = all(s is None or s.kind != "human" for s in self.seats)
            if ok_before:
                spectator = True
            target = -1
            # 2) 回自己的座位（uid 或凭证对得上）。说了要观战就绝不占座 ——
            #    纯 AI 观战房的房主本来就在 0 号位留了个坑。
            if not spectator:
                for i, s in enumerate(self.seats):
                    if s is None or s.kind != "human":
                        continue
                    if s.uid == uid or (token and s.token == token):
                        target = i
                        break
            # 3) 指定座位（只有空位 / AI 位能让）
            if target < 0 and not spectator and seat is not None \
                    and 0 <= seat < self.players:
                s = self.seats[seat]
                assert s is not None
                if s.kind in ("waiting", "ai") and self._can_take_ai_seat(seat):
                    target = self._seat_to_human(seat, uid, name)
            # 4) 随便找个位置
            if target < 0 and not spectator:
                target = self._free_seat()
                if target >= 0:
                    s = self.seats[target]
                    if s is not None and s.kind != "human":
                        target = self._seat_to_human(target, uid, name)
            if target < 0:
                c = Conn(conn_id, -1, reveal=False, spectator=True, ws=ws,
                         uid=uid, name=name)
                c.reveal_want = bool(reveal)
                c.reveal = bool(reveal and ok_before)
                self.conns[conn_id] = c
                log.info("房间 %s 新增观战连接 %s（%s）", self.id, conn_id, name)
                return c

            s = self.seats[target]
            assert s is not None
            s.online = True
            s.disconnected_at = 0.0
            c = Conn(conn_id, target, reveal=False, spectator=False, ws=ws,
                     uid=uid, name=name)
            self.conns[conn_id] = c
            # 同一个座位只留最新一条连接
            for cid, other in list(self.conns.items()):
                if cid != conn_id and other.seat == target and other.ws is not ws:
                    other.seat = -1
                    other.spectator = True
            log.info("房间 %s 连接 %s 入座 %d（%s）", self.id, conn_id, target, name)
            self.last_active = time.time()
            return c

    def _seat_to_human(self, seat: int, uid: str, name: str) -> int:
        """把一个 AI / 空座位换成真人。"""
        s = self.seats[seat]
        assert s is not None
        s.uid = uid
        s.name = (name or "玩家")[:16]
        s.kind = "human"
        s.ai = None
        s.online = True
        s.token = _token()
        s.ai_degraded = False
        s.ai_degrade_reason = ""
        return seat

    def _can_take_ai_seat(self, seat: int) -> bool:
        """AI 顶着的座位能不能让给真人。

        规矩：**只在对局开头**（还没人打过牌、还在第 1 巡）能让。
        否则中途换人会拿到一手已经打了半局的牌，说不清楚。
        """
        s = self.seats[seat]
        if s is None or s.online:
            return False
        g = self.game
        if g is None or not _started(g, seat):
            return True
        return False

    def _free_seat(self) -> int:
        """找人位：优先空着的「等待中」，其次可以接管的 AI 座位。"""
        for i, s in enumerate(self.seats):
            if s is not None and s.kind == "waiting" and not s.online:
                return i
        for i, s in enumerate(self.seats):
            if s is not None and s.kind == "ai" and self._can_take_ai_seat(i):
                return i
        return -1

    def remove_conn(self, conn_id: str) -> None:
        now = time.time()
        with self.lock:
            c = self.conns.pop(conn_id, None)
            if c is None or c.seat < 0:
                return
            if any(o.seat == c.seat for o in self.conns.values()):
                return   # 同座位还有别的连接
            s = self.seats[c.seat]
            if s is not None and s.kind == "human":
                s.online = False
                s.disconnected_at = now
                log.info("房间 %s 座位 %d 掉线（%s）", self.id, c.seat, s.name)
        self.broadcast_later()

    def set_reveal(self, conn: Conn, want: bool) -> tuple[bool, str]:
        """观战者请求全亮牌。

        **有真人在座一律拒绝**（判断依据是 :attr:`spectator_ok`，在服务端）——
        客户端说破天也没用。请求会被记进 ``reveal_want``：万一人退光了，
        下一帧自动生效，不用再点一次。
        """
        if conn.seat >= 0:
            return False, "你在座位上，不能全亮"
        if want and not self.spectator_ok:
            return False, "桌上有真人玩家，不能全亮牌"
        conn.reveal_want = bool(want)
        conn.reveal = bool(want)
        return True, ""

    def seat_of(self, uid: str, token: str = "") -> int | None:
        """这个 uid / 凭证现在坐哪一格（没坐就 None）。"""
        with self.lock:
            for s in self.seats:
                if s is None or s.kind != "human":
                    continue
                if s.uid == uid or (token and s.token == token):
                    return s.seat
        return None

    def reconnect_seat(self, uid: str, token: str) -> int | None:
        return self.seat_of(uid, token)

    # ------------------------------------------------------------------
    # 动作（服务端权威）
    # ------------------------------------------------------------------
    def submit_action(
        self, conn: Conn, *, kind: str, index: int | None = None, tile: str | None = None
    ) -> tuple[bool, str]:
        if conn.seat < 0:
            return False, "观战不能出牌"
        hs = self.human_seats.get(conn.seat)
        if hs is None:
            return False, "这个座位不是真人"
        pending = hs.pending()
        if pending is None:
            return False, "现在没轮到你"
        if kind and pending.kind != kind:
            return False, f"现在要处理的是「{P.KIND_CN.get(pending.kind, pending.kind)}」"

        idx = index
        if idx is None and tile:
            tid = str(tile).strip().lower()
            if pending.kind == "discard":
                for i, t in enumerate(pending.options):
                    if T.index_to_id(int(t)) == tid:
                        idx = i
                        break
            else:
                for i, o in enumerate(pending.options):
                    if str(o).lower() == tid:
                        idx = i
                        break
        if idx is None:
            return False, "缺少 index 或 tile"
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return False, "index 必须是整数"
        if not (0 <= idx < len(pending.options)):
            return False, f"选项越界：{idx}（0~{len(pending.options) - 1}）"
        ok, err = hs.submit(idx)
        if ok:
            self.last_active = time.time()
        return ok, err

    def chat_msg(self, conn: Conn, text: str) -> tuple[bool, str]:
        if not self.cfg.allow_chat:
            return False, "本服务已关闭聊天"
        text = (text or "").strip()
        if not text:
            return False, "空消息"
        text = text[: self.cfg.max_chat_len]
        who = conn.name or (self.seats[conn.seat].name if conn.seat >= 0 else "观战")
        with self.lock:
            self.chat.append({"at": round(time.time(), 2), "seat": conn.seat,
                              "name": who, "text": text, "system": False})
            if len(self.chat) > 100:
                del self.chat[: len(self.chat) - 100]
        self.broadcast_later()
        return True, ""

    def system_msg(self, text: str) -> None:
        with self.lock:
            self.chat.append({"at": round(time.time(), 2), "seat": -1, "name": "",
                              "text": text, "system": True})
            if len(self.chat) > 100:
                del self.chat[: len(self.chat) - 100]
        self.broadcast_later()

    # ------------------------------------------------------------------
    # 广播
    # ------------------------------------------------------------------
    def _push_event(self, seat: int | None, event: dict) -> None:
        """给指定座位（None = 所有人）推一条即时事件。"""
        for c in list(self.conns.values()):
            if seat is None or c.seat == seat:
                _send(c, event)

    def broadcast_later(self) -> None:
        """从引擎线程安全地请求一次状态广播。

        这里会做**合并**：引擎跑得快的时候（AI 节拍 0），一次动作可能连着
        触发好几次广播请求；全部照发只会把客户端和网络刷爆，而且后一帧
        本来就把前一帧覆盖了。所以合并窗口内只保留一次，窗口结束时补发。
        """
        if self.closed and self.phase != PHASE_FINISHED:
            return
        now = time.time()
        with self._push_lock:
            if self._push_scheduled:
                self._push_pending = True
                return
            wait = self._push_min_gap - (now - self._push_last)
            self._push_scheduled = True
        try:
            if wait <= 0:
                self.loop.call_soon_threadsafe(self._push_now)
            else:
                self.loop.call_later(wait, self._push_now)
        except RuntimeError:
            with self._push_lock:
                self._push_scheduled = False

    def _push_now(self) -> None:
        """（事件循环线程）合并窗口到期，真正推一帧。"""
        with self._push_lock:
            again = self._push_pending
            self._push_pending = False
            if again:
                self._push_last = time.time()
                try:
                    self.loop.call_later(self._push_min_gap, self._push_now)
                    return
                except RuntimeError:
                    pass
            self._push_scheduled = False
            self._push_last = time.time()
        self.push_state()

    def push_state(self) -> None:
        """（事件循环线程）给每条连接推一份**他自己的**快照。

        **必须同步执行、不许 await**：这里逐条连接构造座位视角，中间一旦
        让出控制权，就可能出现「A 房快照建到一半，房间被换成 B 房」这种脏读。
        所有 IO 都通过连接的发送队列异步出去。
        """
        conns = list(self.conns.values())
        if not conns:
            return
        with self.lock:
            self.seq += 1
            seq = self.seq
            now = time.time()
            pending = None
            for _st, hs in self.human_seats.items():
                p = hs.pending()
                if p is not None:
                    pending = p
                    break
            base = self._room_info()
            seats_info = [s.public() if s else {} for s in self.seats]
            result = self.result_view
            if result is None and self.game is not None and self.game.hands:
                if self.phase in (PHASE_HAND_END, PHASE_FINISHED):
                    result = P.hand_result_view(self.game)
                    self.result_view = result
            log_lines = [e["text"] for e in self.log_lines]
            chat = list(self.chat)
            sok = self.spectator_ok
            spectator_count = sum(1 for c in conns if c.seat < 0)
            phase = self.phase
            finished = phase == PHASE_FINISHED
            game = self.game
            scores_before = list(self.scores_before)
            sticks_before = self.sticks_before
            last_event = self.last_event
            match = self.match_result
            err = self.error
            next_acks = [bool(s.next_ack) for s in self.seats if s]
            dealt = self.dealt

        for c in conns:
            # 观战者的全亮只在**当下**允许时才生效：桌上有真人的话，
            # 就算他之前请求过也已经自动失效（真人可能刚坐下）。
            if c.seat < 0:
                c.reveal = bool(c.reveal_want and sok)
            pv = pending if (pending is not None and c.seat == pending.seat) else (
                pending if c.seat < 0 else None
            )
            try:
                payload = P.build_state(
                    game,
                    viewer=c.seat,
                    reveal=c.reveal,
                    room=base,
                    seats=self._seats_for_viewer(seats_info, c),
                    phase=phase,
                    seq=seq,
                    spectator_ok=sok,
                    scores_before=scores_before,
                    sticks_before=sticks_before,
                    ready=dealt,
                    pending=pv,
                    last_event=last_event,
                    finished=finished,
                    result=result,
                    log=log_lines,
                    chat=chat,
                    spectator_count=spectator_count,
                    now=now,
                )
            except Exception:  # noqa: BLE001
                # **一帧拼不出来绝不能把整桌搞哑**：记一笔，推个最小快照，
                # 让客户端至少还能看到「出错了」，而不是永远转圈。
                log.exception("房间 %s 构造快照失败（seq=%d, seat=%d）",
                              self.id, seq, c.seat)
                payload = {
                    "t": "state", "v": P.PROTOCOL_VERSION, "seq": seq, "now": now,
                    "phase": phase, "room": base,
                    "viewer": {"seat": c.seat, "spectator": c.seat < 0,
                               "reveal": False, "name": c.name},
                    "round": None, "seats": [], "actions": None, "last": None,
                    "result": None, "log": [], "chat": [], "spectator_ok": sok,
                    "spectator_count": spectator_count, "finished": finished,
                }
            if match and finished:
                payload["match"] = match
            if err:
                payload["error"] = err
            payload["room"]["next_acks"] = next_acks
            payload["room"]["dealt"] = dealt
            _send(c, payload)

    def _seats_for_viewer(self, seats_info: list[dict], conn: Conn) -> list[dict]:
        """按座位生成座位视图 —— **别家的手牌在这里就不写进去**。"""
        out: list[dict] = []
        # 引擎刚起、这一局还没发牌的时候 ``Game.players`` 还不存在，
        # 这时候按座位表画空桌就行，绝不能去摸引擎内部属性。
        if not self.dealt:
            for s in seats_info:
                seat = int(s.get("seat", 0))
                out.append({
                    "seat": seat, "name": s.get("name", "?"),
                    "kind": s.get("kind", "waiting"), "ai": s.get("ai"),
                    "online": s.get("online", False),
                    "score": self.rule.start_score, "score_delta": 0,
                    "hand_count": 0, "hand": None, "drawn": None,
                    "melds": [], "pond": [], "pond_red": [], "riichi": False,
                    "riichi_tile": None, "kita": 0, "seat_wind": "?",
                    "is_dealer": False,
                    "rel": P.rel_label(conn.seat, seat, self.players),
                    "connected": s.get("online", False), "menzen": True,
                    "host": s.get("host", False), "double_riichi": False,
                    "riichi_ippatsu": False, "ippatsu": False,
                    "ai_degraded": s.get("ai_degraded", False),
                    "ai_degrade_reason": s.get("ai_degrade_reason", ""),
                    "no_record": s.get("no_record", False),
                    "declared_rank": s.get("declared_rank", ""),
                    "main_mode": s.get("main_mode", ""),
                })
            return out
        reveal = bool(conn.reveal and self.spectator_ok)
        for i in range(self.players):
            out.append(P.seat_view(
                self.game, i, viewer=conn.seat, reveal=reveal,
                info=seats_info[i], scores_before=self.scores_before,
                pond_red=(self._pond_red[i] if i < len(self._pond_red) else None),
            ))
        return out

    def _room_info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "players": self.players,
            "mode": "ai" if self.human_count == 0 else (
                "mixed" if self.ai_count else "human"),
            "rounds": self.rounds,
            "ai_delay": self.ai_delay,
            "human_timeout": self.cfg.human_timeout,
            "seats": [s.public() if s else {} for s in self.seats],
            "max_conns": self.cfg.max_conns_per_room,
            "spectator_reveal": self.spectator_reveal,
            "start_score": self.rule.start_score,
            "return_score": self.rule.return_score,
            "created_at": round(self.created_at, 2),
            "record": bool(self.cfg.record),
            "rules": {
                "aka": self.rule.aka,
                "kita": self.rule.kita_dora,
                "chi": self.players == 4,
                "noten_penalty": self.rule.noten_penalty,
                "double_ron": self.rule.double_ron,
                "start_score": self.rule.start_score,
                "return_score": self.rule.return_score,
            },
        }

    def summary(self) -> dict:
        """房间列表用的一行。"""
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "players": self.players,
                "phase": self.phase,
                "humans": self.human_count,
                "ais": self.ai_count,
                "spectators": sum(1 for c in self.conns.values() if c.seat < 0),
                "conns": len(self.conns),
                "ai_delay": self.ai_delay,
                "spectator_ok": self.spectator_ok,
                "hand_no": (len(self.game.hands) + 1) if self.game else 0,
                "rounds": self.rounds,
                "created_at": round(self.created_at, 2),
                "seats": [
                    {
                        "seat": s.seat, "name": s.name, "kind": s.kind,
                        "ai": s.ai, "online": s.online, "host": s.host,
                        "ai_degraded": s.ai_degraded,
                    }
                    for s in self.seats if s
                ],
            }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _started(game: Game, seat: int) -> bool:
    """这一局是不是已经真打起来了（有人摸过 / 打过 / 鸣过）。"""
    players = getattr(game, "players", None)
    if not players or seat >= len(players):
        return False
    if game.hands:
        return True
    return any(
        p.discards or p.melds or p.riichi or p.kita or p.drawn is not None
        for p in players
    )


def _send(conn: Conn, payload: dict) -> None:
    """把一条 JSON 推给这条连接。

    aiohttp 的 ``send_str`` 是 **协程**：一个连不上的客户端会让它挂住，
    要是这里逐个 ``await``，一个慢客户端就把整个房间的广播拖死。
    所以这里只做两件事 —— 序列化、塞进连接自己的发送队列；
    真正的 ``await`` 在 :meth:`webapp.server.app.WSClient.writer` 里。
    """
    ws = conn.ws
    if ws is None:
        return
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        log.exception("房间推送序列化失败")
        return
    outbox = getattr(ws, "_maj_outbox", None)
    if outbox is None:
        try:
            ws.send_json(payload)     # 兜底（单测里塞的假 ws）
        except Exception:  # noqa: BLE001
            pass
        return
    if outbox.full():
        try:
            outbox.get_nowait()       # 丢最旧的，保证「最新状态」一定发得出去
        except Exception:  # noqa: BLE001
            pass
    try:
        outbox.put_nowait(text)
    except Exception:  # noqa: BLE001
        pass


def _token() -> str:
    import secrets

    return secrets.token_urlsafe(18)


def _ai_label(model: str | None) -> str:
    """AI 座位显示什么名字 —— 用中文风格标签（标准 / 防守 / 速攻 / 大牌）。"""
    from aimodel import variants

    if not model:
        return "AI·教师"
    if model in ("teacher", "heuristic"):
        return "AI·教师"
    if model == "random":
        return "AI·随机"
    try:
        spec = variants.get(model)
    except KeyError:
        return f"AI·{model}"
    return f"AI·{variants.STYLE_CN.get(spec.style, spec.style)}{spec.version}"
