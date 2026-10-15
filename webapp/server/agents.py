"""座位适配：真人座位 / AI 座位。

两类座位对引擎来说都只是一个 :meth:`decide` 方法，但实现完全不同：

* :class:`HumanSeat` —— 引擎线程挂住，等 WebSocket 那边把「我要打这张」
  送过来；超时交给内置教师托管（**绝不把整桌卡死**）。
* :class:`AISeat` —— 把推理丢进全局线程池。模型是纯 numpy 的，一次推理
  几毫秒到几十毫秒；放线程池是为了即使某个模型卡住，也不会把 asyncio
  事件循环拖住（服务端权威 + 实时推送的前提）。

**节拍**（``delay``）在引擎线程里睡，不占线程池的坑 —— 观战看得清，
又不至于把 worker 全占满。
"""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from aimodel import variants
from aimodel.agents import Agent, HeuristicAgent, MLAgent, RandomAgent
from aimodel.engine import Decision, View

log = logging.getLogger("maj.agents")


# ---------------------------------------------------------------------------
# 全局推理线程池
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def pool(workers: int) -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="maj-ai")
            log.info("AI 推理线程池启动，workers=%d", workers)
        return _pool


def shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


# 权重缓存：同一个模型被多个房间用到时不用反复读盘（~2MB 一个）
_net_lock = threading.Lock()
_nets: dict[str, Any] = {}


def load_net(name: str):
    """按名字载入权重（带缓存）。

    用**宽松模式** ``Net.load(..., strict=False)``：权重缺决策头时把缺的头
    零初始化、并记进 ``net.untrained_heads``（别名 ``missing_heads``），
    而不是直接拒绝。

    为什么会这样：训练侧给网络加了 ``chi``（吃）头，仓库里 26 个存量权重
    都是 8 头。严格模式下一个都载不进来 —— 平台上「选任何模型」都只会
    降级成教师。宽松模式下：

    * 三麻权重**完全不受影响**（三麻没有吃）；
    * 四麻只有「要不要吃」这一项退化，而那一手会被交给内置教师
      （见 :meth:`AISeat.decide`），**绝不让零初始化的头参与决策**。

    等训练侧重训完（带 chi 头的新权重），这里不用改任何代码就会自动用上。
    """
    from aimodel.net import Net

    with _net_lock:
        if name in _nets:
            return _nets[name]
    path = variants.WEIGHTS_DIR / f"{name}.npz"
    net = Net.load(path, strict=False)
    with _net_lock:
        _nets.setdefault(name, net)
        return _nets[name]


def net_missing_heads(net: Any) -> tuple[str, ...]:
    """这个网络缺哪些决策头（没缺就是空元组）。

    训练侧用 ``untrained_heads`` / ``missing_heads`` 两个名字都记过，
    这里都认。
    """
    miss = getattr(net, "untrained_heads", None)
    if miss:
        return tuple(sorted(miss))
    return tuple(getattr(net, "missing_heads", ()) or ())


def probe_model(name: str, players: int | None = None) -> tuple[bool, str]:
    """试着载一次并**真跑一下前向**，返回 ``(能不能用, 原因)``。

    ``players`` 很关键：``encode()`` 的特征维度是**跟人数走**的
    （三麻 486 维、四麻 560 维，因为对手的手牌/副露/牌河块数不同）。
    拿三麻的 486 维去喂四麻权重，第一层 matmul 就会炸 —— 这就是
    「四麻模型一直降级成教师」的真原因。
    """
    try:
        spec = variants.get(name)
    except KeyError:
        return False, "注册表里没有这个模型"
    n = players or spec.players
    try:
        net = load_net(spec.name)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    want = int(getattr(net, "dim", 0) or 0)
    got = _encode_dim(n)
    if want and want != got:
        return False, (f"特征维度对不上：权重 {want} 维，当前 encode({n}人) "
                       f"{got} 维 —— 表示改过了，必须重训")
    missing = net_missing_heads(net)
    if missing:
        return True, f"未训练 {'/'.join(missing)}（这些动作交给内置教师）"
    return True, ""


@lru_cache(maxsize=8)
def _encode_dim(players: int) -> int:
    """当前代码在 N 人局下的特征维度（缓存，别每次都造引擎）。

    维度**跟人数走**（三麻 486 / 四麻 560，对手的牌河/副露块数不同），
    所以不能写死一个常数。
    """
    from aimodel import sanma as S
    from aimodel.agents import HeuristicAgent
    from aimodel.encode import encode
    from aimodel.engine import Game

    # 座位 0 是「被观察的自己」，后面才是对手 —— 引擎要的是**整桌**的
    # agent（人数必须和规则一致），少一个就报「这条规则要 4 个 agent」。
    agents = [HeuristicAgent(999)] + [HeuristicAgent(1000 + i)
                                      for i in range(players - 1)]
    game = Game(agents, seed=99, rule=S.rule_for(players), rounds=1)
    game.play_hand(0)
    return int(len(encode(game.view_for(0))))


def audit_at_startup() -> dict:
    """启动自检：把全部权重试加载一遍，把「缺头 / 载不进来」列进日志。

    训练侧提供的 :func:`aimodel.variants.audit_weights` 就是干这个的，
    直接用它（别自己再写一套，两边口径会漂）。
    """
    try:
        report = variants.audit_weights(verbose=False)
    except Exception:  # noqa: BLE001
        log.exception("权重自检失败（不影响开服）")
        return {}
    ok = report.get("ok") or []
    failed = report.get("failed") or {}
    untrained = report.get("untrained") or {}
    log.info("权重自检：可用 %d 个，载入失败 %d 个，缺决策头 %d 个",
             len(ok), len(failed), len(untrained))
    for name, err in list(failed.items())[:10]:
        log.warning("  载入失败：%s —— %s", name, str(err)[:120])
    for name, heads in list(untrained.items())[:10]:
        log.info("  缺决策头：%s —— %s（这些动作走内置教师）", name, heads)
    return report


#: 训练/评测用的内部对照产物，不给玩家选
_INTERNAL_SUFFIX = ("_bcbase", "_prev")


_MODEL_CACHE: list[dict] | None = None


def available_models(refresh: bool = False) -> list[dict]:
    """所有有权重的模型（含中文风格标签），给前端下拉用。

    「模型可选全部现有模型」—— 直接把 ``weights/*.npz`` 扫一遍，连同历史
    版本（v1 / v1.1 / v1.2）一起给出来，只滤掉内部对照产物。

    每个模型都会**实际试着载一次**（结果缓存），并如实标出：

    * ``usable``    能不能用（载得进来就是能用）
    * ``missing_heads``  缺哪些决策头（缺的动作会交给内置教师）
    * ``note``      给玩家看的一句话说明
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not refresh:
        return _MODEL_CACHE
    out: dict[str, dict] = {}

    def add(name: str) -> None:
        if name in out or name.endswith(_INTERNAL_SUFFIX):
            return
        try:
            spec = variants.get(name)
        except KeyError:
            return
        usable, why = probe_model(spec.name, spec.players)
        net = _nets.get(spec.name)
        missing = list(net_missing_heads(net)) if net is not None else []
        out[name] = {
            "name": spec.name,
            "players": spec.players,
            "style": spec.style,
            "style_cn": variants.STYLE_CN.get(spec.style, spec.style),
            "cn": spec.cn,
            "version": spec.version,
            "usable": usable,
            "missing_heads": missing,
            "note": ("" if usable and not missing else why),
        }

    if variants.WEIGHTS_DIR.exists():
        for f in sorted(variants.WEIGHTS_DIR.glob("*.npz")):
            add(f.stem)
    # 注册表里有、但权重文件不在的也标出来（前端会置灰）
    for spec in variants.all_specs():
        if spec.name not in out:
            out[spec.name] = {
                "name": spec.name,
                "players": spec.players,
                "style": spec.style,
                "style_cn": variants.STYLE_CN.get(spec.style, spec.style),
                "cn": spec.cn,
                "version": spec.version,
                "usable": False,
                "missing_heads": [],
                "missing": True,
                "note": "没有权重文件",
            }
    rows = list(out.values())
    rows.sort(key=lambda m: (m["players"], m["style"], m["version"]))
    _MODEL_CACHE = rows
    return rows


def make_builtin_agent(spec: str, seed: int) -> Agent:
    """内置座位：``teacher`` / ``random`` / ``policy``。"""
    s = (spec or "teacher").strip().lower()
    if s in ("random", "r"):
        return RandomAgent(seed)
    if s in ("policy", "ml", "net"):
        try:
            return MLAgent(load_net("policy"), seed=seed, greedy=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("policy 权重载入失败（%r），退回教师", exc)
            return HeuristicAgent(seed)
    return HeuristicAgent(seed)


# ---------------------------------------------------------------------------
# 待决策
# ---------------------------------------------------------------------------


@dataclass
class Pending:
    """一次等人回答的决策。"""

    seat: int
    kind: str
    options: list
    tile: int | None
    deadline: float
    timeout: float
    view: View
    riichi_ok: set[int] = field(default_factory=set)
    drawn: int | None = None
    fallback: str = ""
    _done: threading.Event = field(default_factory=threading.Event)
    _answer: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def answer(self, index: int) -> bool:
        """投答案。已经答过了返回 False（幂等）。"""
        with self._lock:
            if self._answer is not None:
                return False
            self._answer = int(index)
        self._done.set()
        return True

    def wait(self, timeout: float) -> int | None:
        self._done.wait(timeout)
        with self._lock:
            return self._answer

    @property
    def answered(self) -> bool:
        with self._lock:
            return self._answer is not None


# ---------------------------------------------------------------------------
# 真人座位
# ---------------------------------------------------------------------------


class HumanSeat(Agent):
    """真人：把决策挂出去等 WebSocket 送答案。"""

    name = "真人"

    def __init__(
        self,
        seat: int,
        *,
        timeout: float,
        fallback: Agent | None = None,
        on_pending=None,
        on_resolved=None,
        recorder: Any = None,
    ) -> None:
        self.seat = seat
        self.timeout = timeout
        self.fallback = fallback or HeuristicAgent(seed=seat * 7919)
        self.on_pending = on_pending
        self.on_resolved = on_resolved
        self.recorder = recorder
        self._lock = threading.Lock()
        self._pending: Pending | None = None
        self.timeouts = 0
        self.answers = 0
        self.last_fallback = ""

    # -- 引擎线程调用 -------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        timeout = self.timeout
        # 荣和 / 自摸 / 鸣牌这类「不回应就过」的场合，等太久没意义，
        # 给一段短窗口就够（真人往往在几秒内点）。
        if decision.kind in ("ron", "tsumo", "call", "chi", "kan", "kita", "kyuushu"):
            timeout = min(timeout, max(6.0, timeout * 0.35))
        pending = Pending(
            seat=self.seat,
            kind=decision.kind,
            options=list(decision.options),
            tile=decision.tile,
            deadline=time.time() + timeout,
            timeout=timeout,
            view=view,
            drawn=view.drawn,
            riichi_ok=_riichi_ok_options(decision, view),
        )
        if self.recorder is not None:
            self.recorder.note_turn_start(self.seat, decision.kind)
        with self._lock:
            self._pending = pending
        if self.on_pending is not None:
            self.on_pending(pending)
        ans = pending.wait(timeout)
        with self._lock:
            self._pending = None
        timed_out = ans is None
        if timed_out:
            # 超时托管：交给教师，而不是随便打一张
            self.timeouts += 1
            self.last_fallback = decision.kind
            ans = int(self.fallback.decide(decision, view))
        else:
            self.answers += 1
        self._record(decision, view, int(ans), "timeout" if timed_out else "human")
        if self.on_resolved is not None:
            self.on_resolved(pending, None if timed_out else int(ans), timed_out)
        return int(ans)

    def _record(self, decision: Decision, view: View, index: int, source: str) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.record_decision(
                seat=self.seat, kind=decision.kind, options=list(decision.options),
                tile=decision.tile, index=index, view=view, source=source,
            )
        except Exception:  # noqa: BLE001 - 记录失败绝不能影响对局
            log.exception("决策记录失败（座位 %d）", self.seat)

    # -- 事件循环调用 -------------------------------------------------------
    def submit(self, index: int) -> tuple[bool, str]:
        with self._lock:
            p = self._pending
        if p is None:
            return False, "现在没有轮到你决策"
        if not isinstance(index, int) or isinstance(index, bool) \
                or not (0 <= index < len(p.options)):
            return False, f"选项越界：{index}"
        if not p.answer(index):
            return False, "这次决策已经结束了"
        return True, ""

    def pending(self) -> Pending | None:
        with self._lock:
            return self._pending

    def cancel(self, reason: str = "取消") -> None:
        """房间关掉时把挂住的决策放掉（防止引擎线程永远等下去）。"""
        with self._lock:
            p = self._pending
        if p is not None:
            p.answer(0)


def _riichi_ok_options(decision: Decision, view: View) -> set[int]:
    """哪些出牌选项「打了还能听牌」——前端用来高亮可立直的牌。

    只对 ``discard`` 有意义；**这是提示，不是合法性判断**，
    真正的立直合法性由引擎的 ``_can_riichi`` 决定。
    """
    if decision.kind != "discard":
        return set()
    out: set[int] = set()
    try:
        from aimodel.fastshanten import shanten

        n_melds = len(view.melds)
        for i, t in enumerate(decision.options):
            rest = list(view.hand)
            rest[int(t)] -= 1
            if shanten(rest, n_melds) == 0:
                out.add(i)
    except Exception:  # noqa: BLE001 - 提示算不出来就算了，不影响出牌
        return set()
    return out


# ---------------------------------------------------------------------------
# AI 座位
# ---------------------------------------------------------------------------


class AISeat(Agent):
    """AI：推理进线程池 + 可调节拍 + 异常降级。"""

    name = "AI"

    def __init__(
        self,
        seat: int,
        model: str,
        *,
        workers: int,
        delay: float = 0.0,
        seed: int = 0,
        on_error=None,
        recorder: Any = None,
    ) -> None:
        self.seat = seat
        self.model = model
        self.workers = workers
        self.delay = delay
        self.seed = seed
        self.on_error = on_error
        self.recorder = recorder
        self.degraded = False
        self.degrade_reason = ""
        self.errors = 0
        self.decisions = 0
        self._missing: tuple[str, ...] = ()
        self.teacher = HeuristicAgent(seed)
        self.agent: Agent = self._build(model)
        #: 这个模型缺哪些决策头（缺的那些动作直接交给教师）
        self.missing_heads: tuple[str, ...] = getattr(self, "_missing", ())

    # -- 构建 / 降级 --------------------------------------------------------
    def _build(self, model: str) -> Agent:
        name = (model or "").strip().lower()
        if not name or name in ("teacher", "random"):
            return make_builtin_agent(name or "teacher", self.seed)
        try:
            spec = variants.get(name)
            net = load_net(spec.name)
            self._missing = net_missing_heads(net)
            if self._missing:
                log.warning(
                    "座位 %d 的模型 %s 缺决策头 %s —— 这些动作交给内置教师"
                    "（其余动作照常用模型）",
                    self.seat, name, list(self._missing),
                )
            # **开局前先真跑一次推理**：权重可能是按旧特征维度训的
            # （encode.py 一改表示，存量权重的第一层就对不上了）。
            # 那时候每次决策都会在线程池里抛 ValueError、再降级 ——
            # 桌子不会崩，但一手牌要多花几十毫秒，而且日志会被刷满。
            # 在这里撞一次墙，就能**整局**直接用教师，并且如实标出来。
            self._probe(net, spec.players)
            return MLAgent(net, seed=self.seed, greedy=True, fallback=self.teacher)
        except Exception as exc:  # noqa: BLE001 - 模型坏了不能崩整桌
            self._missing = ()
            self.degraded = True
            self.degrade_reason = f"{type(exc).__name__}: {exc}"
            log.warning("座位 %d 的模型 %s 载入失败，降级为内置教师：%s",
                        self.seat, name, self.degrade_reason)
            if self.on_error is not None:
                self.on_error(self.seat, self.degrade_reason)
            return self.teacher

    def _probe(self, net: Any, players: int) -> None:
        """确认权重和**当前代码**的表示配套。

        两个都会踩的坑：

        1. ``View`` 的字段会随训练侧改动增删（加 ``hand_red`` 那次就把这里
           撞坏了，结果所有模型在开局探测时被判「载入失败」、整桌降级成教师）。
           所以**不自己 ``View(...)``**，改用真的引擎造一副局面。
        2. ``encode()`` 的特征维度**跟人数走**（三麻 486 / 四麻 560）。
           拿三麻维度去喂四麻权重，第一层 matmul 就炸 —— 这就是
           「四麻模型一直静默降级」的真原因。
        """
        want = int(getattr(net, "dim", 0) or 0)
        got = _encode_dim(players)
        if want and want != got:
            raise ValueError(
                f"特征维度对不上：这个权重是 {want} 维，当前代码在 {players} 人局下"
                f"编出 {got} 维 —— 表示改过了，该权重必须重训"
            )

    def _degrade(self, exc: BaseException) -> None:
        self.errors += 1
        if not self.degraded:
            self.degraded = True
            self.degrade_reason = f"{type(exc).__name__}: {exc}"
            log.exception("座位 %d 推理异常，降级为内置教师", self.seat)
            if self.on_error is not None:
                self.on_error(self.seat, self.degrade_reason)
        self.agent = self.teacher

    # -- 决策 ---------------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        if self.recorder is not None:
            self.recorder.note_turn_start(self.seat, decision.kind)
        if self.delay > 0:
            # 节拍睡在引擎线程里：观战看得清，又不占线程池
            time.sleep(self.delay)
        self.decisions += 1
        agent = self.agent
        source = "ai"
        if decision.kind in self.missing_heads:
            # 这个决策头没训过（旧权重缺 chi 那类）—— 直接交给教师，
            # 绝不让零初始化的头参与决策。
            index = int(self.teacher.decide(decision, view))
            source = "teacher_no_head"
        else:
            try:
                fut: Future = pool(self.workers).submit(_safe_decide, agent,
                                                        decision, view)
                try:
                    index = int(fut.result(timeout=20.0))
                except Exception as exc:  # noqa: BLE001 - 超时 / 推理炸了
                    self._degrade(exc)
                    index = int(self.teacher.decide(decision, view))
                    source = "teacher_fallback"
            except Exception as exc:  # noqa: BLE001 - 线程池都没提交上去
                self._degrade(exc)
                index = int(self.teacher.decide(decision, view))
                source = "teacher_fallback"
        if self.recorder is not None:
            try:
                self.recorder.record_decision(
                    seat=self.seat, kind=decision.kind, options=list(decision.options),
                    tile=decision.tile, index=index, view=view, source=source,
                )
            except Exception:  # noqa: BLE001
                log.exception("决策记录失败（座位 %d）", self.seat)
        return index

    def summary(self) -> dict:
        return {
            "model": self.model,
            "decisions": self.decisions,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "errors": self.errors,
        }


def _safe_decide(agent: Agent, decision: Decision, view: View) -> int:
    """线程池里跑的那一下。异常往上抛，由 :class:`AISeat` 统一降级。"""
    idx = agent.decide(decision, view)
    if not isinstance(idx, int) or isinstance(idx, bool) \
            or not (0 <= idx < len(decision.options)):
        log.warning("agent %s 对 %s 返回非法下标 %r，按 0 处理",
                    getattr(agent, "name", "?"), decision.kind, idx)
        return 0
    return idx


def random_seed() -> int:
    return random.randrange(1, 2**31 - 1)
