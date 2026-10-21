"""akochan —— **外部 C++ 引擎**座位（四麻）。

为什么单独一个模块
------------------
``aimodel/agents.py`` 里的座位（随机 / 教师 / 神经网络）都是**纯 numpy 进程内**
推理。akochan 完全不是一回事：

* 它是个 **Windows 可执行文件**（``system.exe``），每次问它都得**起一个进程**；
* 它只认**文件**：把「这局到当前为止的全部 MJAI 事件」写成一行一个 JSON 的文件，
  再告诉它「我是几号座」「问到第几行」；
* 它把整个牌谱**重放**一遍，然后回答「轮到我时我打哪张」。

所以这里做三件事：

1. :class:`AkochanEngine` —— 子进程封装（写临时文件 -> 起进程 -> 读最后一行 JSON
   -> 删临时文件）。并发多局用 **uuid** 文件名，失败/超时**一定清理**。
2. :func:`build_events` —— ``View``（本引擎在某个决策点看到的全部信息）
   -> **标准 MJAI** 事件流（四麻）。这是最容易错的一段，见下面的「坑」。
3. :class:`ExternalAkochanAgent` —— 跟 :class:`aimodel.agents.Agent` 同接口的
   座位：``decide(decision, view) -> int``。失败**一律退回教师**，绝不卡死对局。

调用协议（实测）
----------------
::

    cd <akochan 目录>
    system.exe mjai_log <牌谱文件> <玩家 id 0-3> <行号 0-based>

stdout = 该牌谱前 N 行事件的回显 + 一行 ``calculating review`` + **最后一行**是 JSON：

::

    [{"moves": [{"actor": 0, "pai": "1m", "tsumogiri": true, "type": "dahai"}],
      "review": {"pt_exp_after": 0.0144, ...}}, ...]

``review`` 数组按 ``pt_exp_total`` **从大到小**排好，``review[0]`` 就是它的推荐。
``moves`` 可能有多步（「立直 + 打牌」是两条；「碰 + 打牌」也是两条）。

⚠️ **延迟是双峰的**（这是实现的关键）：
    * 最后一行正好是「玩家 id 要做决定」的点 -> **~0.4 秒**，有 ``moves``；
    * 不是决定点（它会去算完整 review）-> **~10 秒**，没有 ``moves``，还会撞
      ``set_selector`` 的 assert。
所以**只在该玩家真的要动的时候查询**（本模块的事件流保证最后一行一定是
「自己的摸牌」或「他家的打牌/加杠」）。就算写错了，也有超时兜底。

⚠️ **OpenMP 线程数**：``system.exe`` 里 ``cal_kyoku_end_pt_exp`` 有
``#pragma omp parallel``。本机实测 **16 线程以上会随机访问越界崩溃**
（``0xC0000005`` / boost vector assert），8 线程稳定（20/20 次成功，均 0.40s）。
所以子进程强制 ``OMP_NUM_THREADS=8``（可用 :data:`DEFAULT_OMP_THREADS` 改）。

⚠️ **赤五的写法**：akochan 只认标准 MJAI 的 ``5mr`` / ``5pr`` / ``5sr``，
**不认**本项目内部习惯的 ``0m``（``hai_str_to_int`` 会 assert）。
本模块的序列化一律写 ``5mr``；解析它的 ``moves`` 时两种写法都认
（走 :func:`aimodel.mjai.tile_from_mjai`）。

序列化的几个坑（读过 akochan 源码 ``share/types.cpp`` / ``ai_src/selector.cpp``）
--------------------------------------------------------------------------------
* **只要当前这一局**。``get_game_state`` 从**最后一个** ``start_kyoku`` 开始重放，
  前面几局完全不用写。
* **别人的手牌内容没人看**。``get_hai_visible_all`` 只统计
  「四家牌河 + 副露的 consumed + 宝牌指示牌」，``get_nokori_hai_num`` 只减
  **自己**的手牌。所以别家 ``tehais`` 凑满 13 张合法牌名即可；
  **只有自己的手牌必须精确等于 View**。
* **被叫走的牌仍要留在打牌者的牌河里**：``go_next_state`` 处理 ``pon/chi``
  **不会**把它从 ``kawa`` 拿掉（``dahai`` 先 push 进 kawa，``pon`` 只减自己的手牌）。
  所以「被碰走的那张」必须照样发一条 ``dahai``，否则 ``hai_visible_all`` 少一张。
* **立直**：``reach`` 必须在**宣言牌之前**、``reach_accepted`` 在之后，
  这样 ``kawa[i].is_reach`` 才落在宣言牌上（``get_sute_before_reach_flag``
  要用）。``reach_accepted`` 会自己 ``score -= 1000; kyoutaku += 1``，
  而本引擎的 ``View.scores`` **已经扣过**这 1000，所以 ``start_kyoku.scores``
  要**加回去**（每个立直家 +1000），否则会重复扣。
* **宝牌指示牌**：第一张放 ``start_kyoku.dora_marker``，其余用 ``dora`` 事件补
  （杠出来的）。位置无所谓 —— 可见牌和期望值都用最终列表。
* **摸切**：别家一律「摸什么打什么」。``count_tsumo_num_all`` 有 ``<= 70`` 的
  硬断言，而且它被用来估「牌山还剩几张」；摸切最省事又自洽。
* **自己的手牌怎么凑**：先按真实顺序把「自己的打牌 / 副露 / 被叫走的牌」
  铺出来，摸牌一律写成「摸到就是马上要打的那张」（摸切），杠另有 2 张摸牌；
  最后**反推** ``start_kyoku.tehais``：``H0 = H_final - 摸牌 + 打牌 + 副露消耗``。
  对一局真实的四麻，这个和**必然正好 13 张**；不是就说明序列化有 bug ->
  抛 :class:`SerializeError` -> 座位退回教师（不会喂垃圾给 akochan）。

不做的
------
* **三麻**：akochan 的 MJAI 回放按 4 家写死（``haipai_array`` 必须是 4 个）。
  本模块遇到三麻直接抛 :class:`UnsupportedView`。
* **天梯**：用户明确不做（4000 局/期，每次 0.4 秒起进程跑不动）。
* **吃哪两张**：本引擎的 ``chi`` 决策只有 pass/chi 两项，具体吃哪两张由引擎
  自己挑（``engine._do_chi`` 里按向听最优），这里只回答「吃不吃」。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import mjai
from . import sanma as S
from .agents import Agent, HeuristicAgent
from .engine import Decision, View

log = logging.getLogger("maj.akochan")

#: 平台里这个模型的名字（``variants.EXTERNAL`` 注册的就是它）
MODEL_NAME = "akochan"

#: 四麻的东（本项目内部字牌下标 27..33）
_EAST = S.EAST

#: 允许被「外部引擎座位」回答的决策种类（三麻的 ``kita`` 不在此列）
SUPPORTED_KINDS = frozenset({
    "discard", "tsumo", "ron", "riichi", "call", "chi", "kan", "kyuushu",
})

#: 子进程超时（秒）。正常决策点实测 0.34~1.4 秒；写错成非决策点会 ~10 秒后崩，
#: 所以超时是**防线**而不是常态。
DEFAULT_TIMEOUT = 6.0

#: OpenMP 线程数。16+ 会随机崩（见模块文档），8 稳定。
DEFAULT_OMP_THREADS = 8


# ---------------------------------------------------------------------------
# 牌名 / 下标
# ---------------------------------------------------------------------------
#
# akochan 内部的下标（``hai_str_to_int``）：
#     1m..9m = 1..9      5mr = 10
#     1p..9p = 11..19    5pr = 20
#     1s..9s = 21..29    5sr = 30
#     E/S/W/N/P/F/C = 31..37
# 也就是**赤五单独占一个下标**。我们内部是 34 种牌 + 单独的赤计数，
# 所以序列化里所有「自己的手牌 / 副露消耗 / 摸打」都统一换到这个 38 下标空间，
# 反推初始手牌时才不会把赤五算错。

def _tile_name(tile: int, red: bool = False) -> str:
    """牌型下标（34 种）-> 标准 MJAI 牌名。赤五写 ``5mr`` 而不是 ``0m``。"""
    if tile >= 27:
        return mjai.HONOR_MJAI[tile]
    suit = "mps"[tile // 9]
    n = tile % 9 + 1
    if red and n == 5:
        return f"5{suit}r"
    return f"{n}{suit}"


def _ak_index(tile: int, red: bool = False) -> int:
    """牌型下标（34 种）+ 是否赤 -> akochan 的 38 下标。"""
    if tile >= 27:
        return 31 + (tile - 27)
    base = (tile // 9) * 10
    n = tile % 9 + 1
    if n == 5 and red:
        return base + 10
    return base + n


def _ak_base(ak: int) -> int:
    """akochan 的 38 下标 -> 牌型下标（34 种）。

    赤五单独占下标：10 = 5mr、20 = 5pr、30 = 5sr。
    （这里曾经写成 ``(ak // 10) * 9 + 4``，把 20 算成 5s、30 算成「白」，
    结果序列化把红五写成别的牌 —— 反推手牌时对不上。）
    """
    if ak >= 31:
        return 27 + (ak - 31)
    if ak in (10, 20, 30):
        return (ak // 10 - 1) * 9 + 4
    return (ak // 10) * 9 + (ak % 10) - 1


def _ak_red(ak: int) -> bool:
    return ak in (10, 20, 30)


def _ak_str(ak: int) -> str:
    return _tile_name(_ak_base(ak), _ak_red(ak))


def _ak_counts(hand: Sequence[int], reds: Sequence[int]) -> list[int]:
    """34 种计数 + 赤计数 -> 38 下标计数。"""
    out = [0] * 38
    for t in range(min(34, len(hand))):
        n = int(hand[t])
        r = int(reds[t]) if t < len(reds) else 0
        r = max(0, min(n, r))
        out[_ak_index(t, True)] += r
        out[_ak_index(t, False)] += n - r
    return out


# ---------------------------------------------------------------------------
# 找安装目录 / 可用性
# ---------------------------------------------------------------------------

def _candidate_roots() -> list[Path]:
    """按优先级列出可能的 akochan 安装目录。"""
    out: list[Path] = []
    for env in ("POTATOMAJ_AKOCHAN_HOME", "AKOCHAN_HOME"):
        v = os.environ.get(env)
        if v:
            out.append(Path(v))
    here = Path(__file__).resolve()
    for up in (2, 1):
        if len(here.parents) <= up:
            continue
        base = here.parents[up]
        out.append(base / "_data" / "akochan" / "reviewer" / "akochan")
        out.append(base / "data" / "akochan" / "reviewer" / "akochan")
    # 本机实测路径（部署时用 POTATOMAJ_AKOCHAN_HOME 覆盖即可）
    out.append(Path(r"E:\WHYidea\_data\akochan\reviewer\akochan"))
    return out


def _exe_name() -> str:
    return "system.exe" if os.name == "nt" else "system"


def locate_root(explicit: str | Path | None = None) -> tuple[Path | None, str]:
    """找到 ``system.exe`` + ``setup_mjai.json`` 都在的目录。

    返回 ``(目录, 原因)``；找不到时目录是 ``None``，原因是给玩家看的一句话。
    **显式传了目录就只认它**（不做「猜路径」兜底）—— 免得「指错了却偷偷用了
    另一份 akochan」，排查时看不出问题。
    """
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    else:
        cands.extend(_candidate_roots())
    seen: set[str] = set()
    for c in cands:
        try:
            c = Path(c).expanduser().resolve()
        except OSError:
            continue
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        if (c / _exe_name()).is_file() and (c / "setup_mjai.json").is_file():
            return c, ""
    if explicit:
        return None, f"{explicit} 不是 akochan 安装目录（要 system.exe + setup_mjai.json）"
    return None, ("没找到 akochan（靠 system.exe + setup_mjai.json 认）；"
                  "把 POTATOMAJ_AKOCHAN_HOME 指到它的安装目录")


def available(root: str | Path | None = None) -> tuple[bool, str]:
    """``(能不能用, 原因)``。只查文件，不起进程（起进程的是 :meth:`probe`）。"""
    path, why = locate_root(root)
    if path is None:
        return False, why
    return True, ""


# ---------------------------------------------------------------------------
# 子进程封装
# ---------------------------------------------------------------------------

class AkochanError(RuntimeError):
    """akochan 出问题（起不来 / 崩了 / 输出看不懂）。座位会退回教师。"""


class AkochanUnavailable(AkochanError):
    """没装 / 目录不对。"""


class AkochanTimeout(AkochanError):
    """超过 :data:`DEFAULT_TIMEOUT` 还没回答（通常意味着问错了行）。"""


def _parse_review(stdout: str) -> list | None:
    """从 stdout 里把最后一行 review JSON 抠出来。

    stdout 前面是「牌谱回显」（每行一个对象）+ 一行 ``calculating review``；
    最后一行是 ``[...]``。所以**从后往前找第一个以 ``[`` 开头的行**。
    """
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, list):
            return data
    return None


class AkochanEngine:
    """``system.exe mjai_log`` 的封装。**线程安全**（每次查询用独立临时文件）。"""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        omp_threads: int = DEFAULT_OMP_THREADS,
        runner: Callable[[list[str], float], tuple[int, str, str]] | None = None,
    ) -> None:
        self.root, self.why = locate_root(root)
        self.exe: Path | None = (
            (self.root / _exe_name()) if self.root is not None else None
        )
        self.timeout = float(timeout)
        self.omp_threads = int(omp_threads)
        #: 测试注入点：``runner(args, timeout) -> (rc, stdout, stderr)``。
        #: 注入了就不起真进程（负向测试靠它模拟「超时 / 没有 moves / 进程不存在」）。
        self._runner = runner
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "queries": 0,          # 真发出去的查询
            "ok": 0,               # 有 moves 的
            "no_moves": 0,         # 跑通了但 review 里没有 moves（问错了行）
            "timeouts": 0,
            "errors": 0,
            "temp_files": 0,       # 一共写了多少个临时文件（用来证明清理）
            "seconds": 0.0,
            "max_seconds": 0.0,
            "last_temp": "",       # 最近一次临时文件路径（测试断言「已删」）
        }

    # -- 状态 ---------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self._runner is not None:
            return True
        return bool(self.root is not None and self.exe is not None
                    and self.exe.is_file())

    def probe(self, *, live: bool = False, timeout: float = 3.0) -> tuple[bool, str]:
        """平台「这个模型能不能选」的判断。``live=True`` 会真起一次进程。"""
        if not self.available:
            return False, self.why or "akochan 不可用"
        if not live:
            return True, ""
        try:
            moves = self.query(_probe_events(), 0, timeout=timeout)
        except AkochanError as exc:
            return False, f"试跑失败：{exc}"
        if not moves:
            return False, "试跑没有给出 moves"
        return True, ""

    # -- 查询 ---------------------------------------------------------------
    def query(
        self,
        events: Sequence[dict],
        seat: int,
        *,
        timeout: float | None = None,
    ) -> list[dict]:
        """问一次：「这个牌谱的最后一行轮到我，我走哪步」。

        ``events`` 的最后一条必须是**自己的摸牌**或**他家的打牌/加杠**。
        返回 ``review[0]["moves"]``（推荐走法，标准 MJAI 事件列表）。
        没有 ``moves`` 时返回 ``[]``（**不是异常** —— 调用方按「问错了行」兜底）。
        """
        if not self.available:
            raise AkochanUnavailable(self.why or "akochan 不可用")
        events = list(events)
        if not events:
            raise AkochanError("空事件流")
        seat = int(seat)
        if not 0 <= seat <= 3:
            raise AkochanError(f"座位号越界: {seat}")

        tmp = Path(tempfile.gettempdir()) / (
            f"akochan-{os.getpid()}-{uuid.uuid4().hex}.json"
        )
        payload = "".join(
            json.dumps(e, ensure_ascii=True, separators=(",", ":")) + "\n"
            for e in events
        )
        to = float(self.timeout if timeout is None else timeout)
        t0 = time.monotonic()
        wrote = False
        rc, out, err = -1, "", ""
        try:
            tmp.write_text(payload, encoding="utf-8", newline="")
            wrote = True
            with self._lock:
                self.stats["last_temp"] = str(tmp)
                self.stats["temp_files"] += 1
            args = [str(self.exe), "mjai_log", str(tmp), str(seat),
                    str(len(events) - 1)]
            rc, out, err = self._run(args, to)
        finally:
            # ⚠️ 临时文件必须清理（成功、超时、崩溃都走这里）
            try:
                tmp.unlink()
            except OSError:
                pass
        dt = time.monotonic() - t0
        with self._lock:
            st = self.stats
            st["queries"] += 1
            st["seconds"] += dt
            st["max_seconds"] = max(st["max_seconds"], dt)
        if rc != 0:
            with self._lock:
                self.stats["errors"] += 1
            detail = (err or out or "").strip().splitlines()
            tail = detail[-1][:200] if detail else ""
            raise AkochanError(f"system.exe 退出码 {rc}" + (f"：{tail}" if tail else ""))
        review = _parse_review(out)
        if review is None:
            with self._lock:
                self.stats["errors"] += 1
            raise AkochanError("输出里没有 review JSON")
        moves: list[dict] = []
        if review and isinstance(review[0], dict):
            m = review[0].get("moves")
            if isinstance(m, list):
                moves = [x for x in m if isinstance(x, dict)]
        with self._lock:
            if moves:
                self.stats["ok"] += 1
            else:
                self.stats["no_moves"] += 1
        return moves

    def _run(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        try:
            if self._runner is not None:
                return self._runner(args, timeout)
            env = dict(os.environ)
            if self.omp_threads > 0:
                env["OMP_NUM_THREADS"] = str(self.omp_threads)
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NO_WINDOW", 0x08000000)
            proc = subprocess.run(
                args,
                cwd=str(self.root) if self.root is not None else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            with self._lock:
                self.stats["timeouts"] += 1
            raise AkochanTimeout(f"system.exe 超时（>{timeout:.1f}s）") from exc
        except OSError as exc:
            raise AkochanUnavailable(f"起不来 system.exe：{exc}") from exc
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    # -- 展示 ---------------------------------------------------------------
    def summary(self) -> dict:
        with self._lock:
            st = dict(self.stats)
            st["root"] = str(self.root) if self.root else None
            st["timeout"] = self.timeout
            st["omp_threads"] = self.omp_threads
            st["available"] = self.available
            return st


def _probe_events() -> list[dict]:
    """一次「第一局第一巡亲家摸牌」的最小合法牌谱（自检用，固定不随机）。"""
    deck: list[str] = []
    for suit in "mps":
        for n in range(1, 10):
            deck.extend([f"{n}{suit}"] * 4)
    deck.extend(["E"] * 4 + ["S"] * 4 + ["W"] * 4 + ["N"] * 4
                + ["P"] * 4 + ["F"] * 4 + ["C"] * 4)
    hands = [deck[i * 13:(i + 1) * 13] for i in range(4)]
    return [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"],
         "aka_flag": True, "kyoku_first": 4},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": deck[53],
         "kyoku": 1, "honba": 0, "kyoutaku": 0, "oya": 0,
         "scores": [25000, 25000, 25000, 25000], "tehais": hands},
        {"type": "tsumo", "actor": 0, "pai": deck[52]},
    ]


# ---------------------------------------------------------------------------
# 本局状态跟踪（跨决策累积；View 是快照，缺的只有「第几局 / 本场 / 谁是宣言牌」）
# ---------------------------------------------------------------------------

@dataclass
class _HandTracker:
    """记住「这是第几局、几本场、立直宣言牌是哪张」。

    ``View`` 里没有 ``kyoku`` / ``honba``，也没有「这张是不是立直宣言牌」。
    这两样只能靠**跨决策观察**补出来（座位对象在整局里是同一个）。
    """

    kyoku: int = 1
    honba: int = 0
    prev_dealer: int | None = None
    prev_round_wind: int | None = None
    prev_turn: int = 0
    prev_total: int = 0
    prev_dora: int = 0
    riichi: dict[int, bool] = field(default_factory=dict)
    decl: dict[int, int] = field(default_factory=dict)
    prev_pond: tuple[int, ...] = ()
    hands: int = 0
    _init: bool = False
    _fp: tuple | None = None

    @staticmethod
    def fingerprint(view: View, seat: int) -> tuple:
        """同一个决策点会被问好几次（自摸/立直/杠/打牌）—— 用这个去重。"""
        melds = tuple(
            tuple((m.kind, m.tile, m.from_player, m.called) for m in ms)
            for ms in (view.melds_all or [])
        )
        return (
            int(seat), int(view.turn), tuple(int(x) for x in view.scores),
            tuple(tuple(int(x) for x in d) for d in view.discards),
            melds, tuple(int(x) for x in view.dora_indicators),
            tuple(bool(x) for x in view.riichi),
            tuple(int(x) for x in view.hand), int(view.drawn if view.drawn is not None else -1),
            tuple(int(x) for x in view.hand_red),
        )

    def observe(self, view: View, seat: int) -> None:
        fp = self.fingerprint(view, seat)
        if fp == self._fp:
            return
        self._fp = fp
        # 只对四麻有意义；三麻的 View 只有 3 家，这里写成 ``len(...)`` 免得越界
        n = min(4, len(getattr(view, "riichi", ()) or ())) or 4
        try:
            dealer = (seat - (int(view.seat_wind) - _EAST)) % 4
        except Exception:  # noqa: BLE001
            dealer = 0
        total = sum(len(d) for d in view.discards)
        dora_n = len(view.dora_indicators)
        riichi_now = {q: bool(view.riichi[q]) for q in range(n)}
        if not self._init:
            self._init = True
            self.prev_dealer = dealer
            self.prev_round_wind = int(view.round_wind)
            self.prev_turn = int(view.turn)
            self.prev_total = total
            self.prev_dora = dora_n
            self.riichi = riichi_now
            for q in range(n):
                if self.riichi.get(q):
                    # 中途接手的座位：只能猜「最近一张」是宣言牌
                    self.decl[q] = max(0, len(view.discards[q]) - 1)
            self.prev_pond = tuple(len(d) for d in view.discards)
            self.hands = 1
            return
        new_hand = (
            int(view.turn) < self.prev_turn
            or int(view.round_wind) != self.prev_round_wind
            or dora_n < self.prev_dora
            or (int(view.turn) <= 1 and self.prev_total >= 4
                and total < self.prev_total)
        )
        if new_hand:
            self.hands += 1
            if int(view.round_wind) != self.prev_round_wind:
                self.kyoku, self.honba = 1, 0
            elif dealer == self.prev_dealer:
                self.honba += 1          # 连荘
            else:
                self.kyoku = min(4, self.kyoku + 1)
                self.honba = 0
            self.riichi = riichi_now
            self.decl = {}
            for q in range(n):
                if self.riichi.get(q):
                    self.decl[q] = max(0, len(view.discards[q]) - 1)
        else:
            for q in range(n):
                if riichi_now.get(q) and not self.riichi.get(q, False):
                    # 刚立直：
                    #  * 别家 —— 宣言牌已经进牌河（最后一张）
                    #  * 自己 —— ``_do_riichi`` 之后才问打牌，正常情况下宣言牌
                    #    **还没打出去**；但如果两次观察之间隔了一手（比如这一手
                    #    没问到自己），牌河里可能已经有了，所以要拿上一次的
                    #    牌河长度比一比。
                    grew = (len(self.prev_pond) > q
                            and len(view.discards[q]) > self.prev_pond[q])
                    self.decl[q] = (max(0, len(view.discards[q]) - 1) if grew
                                    else len(view.discards[q]))
                self.riichi[q] = bool(riichi_now.get(q))
        self.prev_dealer = dealer
        self.prev_round_wind = int(view.round_wind)
        self.prev_turn = int(view.turn)
        self.prev_total = total
        self.prev_dora = dora_n
        self.prev_pond = tuple(len(d) for d in view.discards)


class SerializeError(RuntimeError):
    """View -> MJAI 失败（座位会退回教师，不把锅甩给对局）。"""


class UnsupportedView(SerializeError):
    """三麻 / 字段不认识。"""


# ---------------------------------------------------------------------------
# View -> MJAI
# ---------------------------------------------------------------------------

@dataclass
class _Discard:
    """牌河里的一条打牌。"""

    tile: int
    pond_index: int                             # View.discards 里的下标；-1 = 被叫走的
    call: tuple[int, Any] | None = None         # (谁叫的, 那组副露)
    reach: bool = False


def _is_call(meld: Any) -> bool:
    """这组副露是不是「叫了别人的牌」（碰 / 吃 / 明杠）。"""
    return str(getattr(meld, "kind", "")) in ("pon", "chi", "minkan")


def _is_self_kan(meld: Any) -> bool:
    return str(getattr(meld, "kind", "")) in ("ankan", "kakan")


_CALL_TYPE = {"pon": "pon", "chi": "chi", "minkan": "daiminkan",
              "ankan": "ankan", "kakan": "kakan"}


def _meld_event(actor: int, meld: Any) -> dict:
    """一组副露 -> 标准 MJAI 事件（含赤五）。

    ``consumed`` 里放「从这家人自己手里拿出来的牌」；被叫的那张在 ``pai``。
    赤五：``Meld.aka`` 只统计**从自己手里**拿出来的赤（见 ``engine._do_pon``），
    所以赤五放到 ``consumed`` 里；``hai_visible_all`` 也只统计 ``consumed``。
    """
    kind = str(meld.kind)
    ev: dict[str, Any] = {"type": _CALL_TYPE[kind], "actor": int(actor)}
    if kind in ("pon", "minkan", "kakan"):
        ev["pai"] = _tile_name(int(meld.tile))
    elif kind == "chi":
        called = int(getattr(meld, "called", -1))
        if called < 0:
            called = int(meld.tile) + 1          # 不知道就取中间那张
        ev["pai"] = _tile_name(called)
    if kind != "ankan":
        target = int(getattr(meld, "from_player", -1))
        if target >= 0:
            ev["target"] = target
    if kind == "chi":
        called = int(getattr(meld, "called", -1))
        if called < 0:
            called = int(meld.tile) + 1
        tiles = [t for t in (int(meld.tile), int(meld.tile) + 1, int(meld.tile) + 2)
                 if t != called]
        if len(tiles) != 2:                      # 兜底：被吃的就是最小那张
            tiles = [int(meld.tile) + 1, int(meld.tile) + 2]
        ev["consumed"] = [_tile_name(t) for t in tiles]
    else:
        # pon 是「手里 2 张 + 被叫 1 张」；明杠 3；暗杠 4；加杠（叠在碰上的 3 张）
        n = {"pon": 2, "minkan": 3, "ankan": 4, "kakan": 3}[kind]
        ev["consumed"] = [_tile_name(int(meld.tile))] * n
    aka = int(getattr(meld, "aka", 0) or 0)
    base = int(meld.tile)
    if aka and base < 27 and base % 9 == 4:
        red_name = "5" + "mps"[base // 9] + "r"
        cons = list(ev.get("consumed") or [])
        for i, name in enumerate(cons):
            try:
                t, red = mjai.tile_from_mjai(name)
            except ValueError:
                continue
            if t == base and not red:
                cons[i] = red_name
                aka -= 1
                if aka <= 0:
                    break
        ev["consumed"] = cons
    return ev


def _hand_names(counts38: Sequence[int]) -> list[str]:
    return [_ak_str(ak) for ak in range(38) for _ in range(int(counts38[ak]))]


def _meld_events(actor: int, meld: Any) -> list[dict]:
    """一组副露 -> MJAI 事件（**加杠要拆成两条**）。

    ⚠️ 加杠（kakan）必须**先有一条 pon**：akochan 的 ``go_next_state`` 处理
    ``kakan`` 时调 :cpp:func:`replace_pon_to_kakan`，它只在 ``fuuro`` 里
    **找已有的碰**去替换 —— 找不到就什么也不做 ✗ 结果这组杠在它眼里
    根本不存在（``fuuro`` 少一组、可见牌也少）。所以这里先补发那条碰。
    """
    if str(getattr(meld, "kind", "")) == "kakan":
        tile = int(meld.tile)
        target = int(getattr(meld, "from_player", -1))
        pre = {"type": "pon", "actor": int(actor), "pai": _tile_name(tile),
               "consumed": [_tile_name(tile)] * 2}
        if target >= 0:
            pre["target"] = target
        return [pre, _meld_event(actor, meld)]
    return [_meld_event(actor, meld)]


def _akochan_consumed(meld: Any) -> list[str]:
    """akochan 回放时**真正从手牌里扣掉**的牌（跟 ``go_next_state`` 一致）。

    关键差别：``kakan`` 只扣 ``pai`` **一张**（另外两张在更早的 ``pon`` 里
    已经扣过了）。本函数给「反推初始手牌 / 别家起手牌总账」用。
    """
    kind = str(getattr(meld, "kind", ""))
    tile = int(getattr(meld, "tile", 0))
    if kind == "kakan":
        tiles = [tile]
    elif kind == "chi":
        return [str(x) for x in (_meld_event(0, meld).get("consumed") or [])]
    else:
        n = {"pon": 2, "minkan": 3, "ankan": 4}.get(kind, 0)
        tiles = [tile] * n
    return [_tile_name(t) for t in tiles]


class _Serializer:
    """一次性的 View -> MJAI 序列化器。"""

    def __init__(
        self,
        view: View,
        seat: int,
        tracker: _HandTracker,
        *,
        kind: str = "",
        tile: int | None = None,
        held_is_rinshan: bool = False,
    ) -> None:
        self.view = view
        self.seat = int(seat)
        self.tracker = tracker
        self.kind = str(kind)
        self.tile = int(tile) if tile is not None else None
        self.held_is_rinshan = bool(held_is_rinshan)
        self.dealer = 0
        self.melds_all: list[list] = []
        self.n = 4

    # -- 公共入口 -----------------------------------------------------------
    def build(self) -> list[dict]:
        view, seat = self.view, self.seat
        n = int(getattr(getattr(view, "rule", None), "players", 0)
                or len(view.scores) or 4)
        if n == 3:
            raise UnsupportedView("akochan 接入目前只做四麻")
        if n != 4 or len(view.discards) != 4:
            raise UnsupportedView("akochan 接入目前只做四麻")
        self.n = 4
        if len(view.hand) < 34 or len(view.hand_red) < 34:
            raise UnsupportedView("View 字段不完整")
        try:
            self.dealer = (seat - (int(view.seat_wind) - _EAST)) % 4
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedView(f"无法从 View 推出亲家：{exc}") from exc
        self.melds_all = [list(ms) for ms in view.melds_all]
        if len(self.melds_all) != 4:
            raise UnsupportedView("melds_all 家数不对")
        dora = [int(x) for x in view.dora_indicators] or [4]
        riichi_cost = int(getattr(getattr(view, "rule", None), "riichi_stick", 1000) or 1000)
        scores = [int(s) for s in view.scores]
        for q in range(4):
            if view.riichi[q]:
                # reach_accepted 会再扣一次 1000；View 已经扣过 -> 先加回来
                scores[q] += riichi_cost
        self.head: list[dict] = [
            {"type": "start_game", "names": [f"P{q}" for q in range(4)],
             "aka_flag": True, "kyoku_first": 4},
            {"type": "start_kyoku",
             "bakaze": mjai.WIND_MJAI.get(int(view.round_wind), "E"),
             "kyoku": int(self.tracker.kyoku), "honba": int(self.tracker.honba),
             "kyotaku": 0, "oya": int(self.dealer), "scores": scores,
             "dora_marker": _tile_name(dora[0]),
             "tehais": [[] for _ in range(4)]},
        ]
        for d in dora[1:]:
            self.head.append({"type": "dora", "dora_marker": _tile_name(d)})
        play = self._play()
        events = self.head + play
        h0 = self._initial_hand(events)
        events[1]["tehais"] = self._tehais(h0)
        return events

    # -- 事件流 -------------------------------------------------------------
    def _queues(self) -> list[list[_Discard]]:
        view = self.view
        queues: list[list[_Discard]] = [
            [_Discard(int(t), i) for i, t in enumerate(view.discards[q])]
            for q in range(4)
        ]
        # 「刚刚叫的牌」：现在是**叫牌之后那张打牌**（drawn is None），
        # 那么被叫走的牌一定是**最后一次打牌**，必须插在牌河末尾 ——
        # 否则模拟会把它当成「很早以前叫的」，让它后面还有牌可打，
        # 反推初始手牌就会多出一张（实测 14 张）。
        force_end: set[int] = set()
        if self.kind == "discard" and view.drawn is None and view.melds:
            last = view.melds[-1]
            if _is_call(last):
                force_end.add(id(last))
        # 被别家叫走的牌：先塞回打牌者的队列（顺序只能估，见模块文档）
        for q in range(4):
            melds = self.melds_all[q]
            for j, m in enumerate(melds):
                if not _is_call(m):
                    continue
                target = int(getattr(m, "from_player", -1))
                if target < 0 or target >= 4 or target == q:
                    continue
                called = int(getattr(m, "called", -1))
                tile = called if (str(m.kind) == "chi" and called >= 0) else int(m.tile)
                if id(m) in force_end:
                    pos = len(queues[target])
                else:
                    frac = (j + 0.5) / max(1, len(melds))
                    pos = int(round(frac * len(queues[target])))
                pos = max(0, min(len(queues[target]), pos))
                queues[target].insert(pos, _Discard(tile, -1, call=(q, m)))
        # 立直宣言牌
        for q in range(4):
            if not view.riichi[q]:
                continue
            idx = self.tracker.decl.get(q, -1)
            for ent in queues[q]:
                if ent.pond_index == idx:
                    ent.reach = True
                    break
        return queues

    def _take_pending(self, queues: list[list[_Discard]]):
        """叫牌/荣和决策：把「正在被考虑的那张打牌」从牌河队列里拎出来，最后再发。"""
        if self.kind not in ("call", "chi", "ron"):
            return None, -1
        view, seat = self.view, self.seat
        cand_players: list[int] = []
        lp = view.last_discard
        if lp is not None and 0 <= int(lp[0]) < 4 and int(lp[0]) != seat:
            cand_players.append(int(lp[0]))
        cand_players.extend(q for q in range(4) if q != seat and q not in cand_players)
        for who in cand_players:
            q = queues[who]
            for i in range(len(q) - 1, -1, -1):
                ent = q[i]
                if ent.pond_index < 0:
                    continue
                if self.tile is not None and ent.tile != self.tile:
                    continue
                del q[i]
                return ent, who
        return None, -1

    def _emitted_meld(self, events: list[dict], q: int, ptr: list[int], m: Any) -> None:
        """按副露表顺序补发到 ``m`` 为止（被插队时保证顺序不乱）。"""
        ms = self.melds_all[q]
        while ptr[q] < len(ms) and ms[ptr[q]] is not m:
            cur = ms[ptr[q]]
            if _is_self_kan(cur):
                events.append({"type": "tsumo", "actor": q,
                               "pai": _tile_name(int(cur.tile))})
                events.extend(_meld_events(q, cur))
                events.append({"type": "tsumo", "actor": q,
                               "pai": _tile_name(int(cur.tile))})
            else:
                events.extend(_meld_events(q, cur))
            ptr[q] += 1

    def _play(self) -> list[dict]:
        view, seat = self.view, self.seat
        queues = self._queues()
        pending, pending_from = self._take_pending(queues)
        pending_meld: Any = None
        if pending is None and self.kind == "ron" and self.tile is not None:
            # 抢杠：最后一条应该是「对家的加杠」
            for who in range(4):
                if who == seat:
                    continue
                ms = self.melds_all[who]
                for i, m in enumerate(ms):
                    if str(getattr(m, "kind", "")) == "kakan" \
                            and int(getattr(m, "tile", -1)) == self.tile:
                        pending_meld = m
                        self.melds_all[who] = ms[:i] + ms[i + 1:]
                        break
                if pending_meld is not None:
                    break
        ptr = [0] * 4
        events: list[dict] = []
        actor = self.dealer
        need_draw = True
        guard = 0
        while True:
            guard += 1
            if guard > 4000:
                raise SerializeError("事件模拟超过 4000 步（不该发生）")
            q = actor
            if not queues[q]:
                if all(not queues[x] for x in range(4)):
                    break
                nxt = (q + 1) % 4
                if nxt == actor:
                    break
                actor, need_draw = nxt, True
                continue
            d = queues[q].pop(0)
            kan = self._next_self_kan(q, ptr)
            drew = bool(need_draw)
            if drew:
                events.append({"type": "tsumo", "actor": q,
                               "pai": _tile_name(int(kan.tile) if kan is not None
                                                 else d.tile)})
            if d.reach:
                events.append({"type": "reach", "actor": q})
            if kan is not None:
                events.extend(_meld_events(q, kan))
                ptr[q] += 1
                if drew:
                    # 岭上摸牌：摸到马上要打的那张（摸切）
                    events.append({"type": "tsumo", "actor": q,
                                   "pai": _tile_name(d.tile)})
            events.append({"type": "dahai", "actor": q,
                           "pai": _tile_name(d.tile),
                           "tsumogiri": drew})
            if d.reach:
                events.append({"type": "reach_accepted", "actor": q})
            need_draw = True
            if d.call is not None:
                caller, m = d.call
                self._emitted_meld(events, caller, ptr, m)
                events.append(_meld_event(caller, m))
                ptr[caller] += 1
                actor = int(caller)
                need_draw = str(getattr(m, "kind", "")) == "minkan"
            else:
                actor = (q + 1) % 4
        # 剩下的自家杠（后面已经没牌可打了）：照真实补两张摸牌；
        # 如果这次决策就是「杠完补牌」，岭上牌就是 View 里的 drawn，不能再补一张。
        for q in range(4):
            ms = self.melds_all[q]
            while ptr[q] < len(ms) and _is_self_kan(ms[ptr[q]]):
                m = ms[ptr[q]]
                held = bool(self.held_is_rinshan and q == seat)
                events.append({"type": "tsumo", "actor": q,
                               "pai": _tile_name(int(m.tile))})
                events.extend(_meld_events(q, m))
                if not held:
                    events.append({"type": "tsumo", "actor": q,
                                   "pai": _tile_name(int(m.tile))})
                ptr[q] += 1
        # 还剩下的副露（顺序被估歪了，但状态必须齐）
        for q in range(4):
            ms = self.melds_all[q]
            while ptr[q] < len(ms):
                events.extend(_meld_events(q, ms[ptr[q]]))
                ptr[q] += 1
        if pending_meld is not None:
            who = int(getattr(pending_meld, "from_player", -1))
            events.extend(_meld_events(who if who >= 0 else 0, pending_meld))
        if pending is not None:
            events.append({"type": "dahai", "actor": pending_from,
                           "pai": _tile_name(pending.tile),
                           "tsumogiri": False})
        # 立直宣言牌**还没打出去**（现在这次打牌就是宣言牌 —— 引擎先
        # ``_do_riichi`` 再问打牌，所以 View 里 ``riichi=True`` 但牌河里没有它）。
        # 真实顺序是「reach -> 打宣言牌 -> reach_accepted」，这里那张牌还不存在，
        # 就把 reach/reach_accepted 补在最后：akochan 的状态（已扣 1000、
        # 已在立直中、必须保持听牌）跟真实一致，它给出的打牌**不会破听**。
        for q in range(4):
            if not view.riichi[q]:
                continue
            if self.tracker.decl.get(q, -1) >= len(view.discards[q]):
                events.append({"type": "reach", "actor": q})
                events.append({"type": "reach_accepted", "actor": q})
        # 自家的「本次摸牌」（打牌/立直/杠/自摸和了这些决策都在这一摸之后）
        if self.kind not in ("call", "chi", "ron"):
            drawn = view.drawn
            if drawn is not None:
                # ⚠️ 摸到的是不是赤五：View 只有「牌型 + 手里有几张赤」，
                # 手里这个牌型有赤 -> 就当这张是赤。反推初始手牌时
                # （``H0 = H_final - 摸牌 + ...``）这样才不会算出 -1 张。
                red = bool(
                    int(drawn) < len(view.hand_red)
                    and int(view.hand_red[int(drawn)]) > 0
                )
                events.append({"type": "tsumo", "actor": seat,
                               "pai": _tile_name(int(drawn), red)})
        return events

    def _next_self_kan(self, q: int, ptr: Sequence[int]):
        ms = self.melds_all[q]
        if ptr[q] < len(ms) and _is_self_kan(ms[ptr[q]]):
            return ms[ptr[q]]
        return None

    # -- 初始手牌 -----------------------------------------------------------
    def _initial_hand(self, events: Sequence[dict]) -> list[int]:
        """反推 ``start_kyoku.tehais[seat]``。

        ``H0 = H_final - 摸牌 + 打牌 + 副露消耗``（全用 38 下标，赤五单算）。
        真实对局下这个和的牌数**必然正好 13**；不是就说明事件流自相矛盾。
        """
        view, seat = self.view, self.seat
        h0 = _ak_counts(view.hand, view.hand_red)
        for ev in events:
            if not isinstance(ev, dict):
                continue
            t = ev.get("type")
            if ev.get("actor") != seat:
                continue
            if t == "tsumo":
                tile, red = mjai.tile_from_mjai(str(ev.get("pai") or ""))
                h0[_ak_index(tile, red)] -= 1
            elif t == "dahai":
                tile, red = mjai.tile_from_mjai(str(ev.get("pai") or ""))
                h0[_ak_index(tile, red)] += 1
            elif t in ("pon", "chi", "daiminkan", "ankan", "kakan"):
                # ⚠️ 跟 akochan 的 go_next_state 保持一致：kakan 只从手牌里
                # 扣 pai **一张**（另外两张在更早的 pon 事件里扣过了）
                names = ([ev.get("pai")] if t == "kakan"
                         else list(ev.get("consumed") or []))
                for name in names:
                    if not name:
                        continue
                    tile, red = mjai.tile_from_mjai(str(name))
                    h0[_ak_index(tile, red)] += 1
        total = sum(h0)
        if total != 13:
            raise SerializeError(
                f"反推的初始手牌有 {total} 张（应为 13）—— 事件流不自洽")
        if any(v < 0 for v in h0):
            bad = [(_ak_str(i), v) for i, v in enumerate(h0) if v < 0]
            raise SerializeError(f"反推初始手牌出现负数：{bad[:6]}")
        for base in range(34):
            n = h0[_ak_index(base)]
            if base < 27 and base % 9 == 4:
                n += h0[_ak_index(base, True)]
            if n > 4:
                raise SerializeError(
                    f"初始手牌里 {_tile_name(base)} 有 {n} 张（>4）")
        return h0

    def _tehais(self, h0: Sequence[int]) -> list[list[str]]:
        """四家的起手牌。只有自己的要精确；别家凑满 13 张合法牌名即可。

        ⚠️ 别家的手牌虽然**没人读**（见模块文档），但**牌数必须守恒**：
        实测把「4 张 5pr」这种不存在的牌发给别家，akochan 会**静默返回空
        review**（``rc=0``、``[]``）—— 座位就会一直兜底。所以这里严格按
        「136 张总账」发：自己的手牌 + 全部打牌 + 副露 consumed + 宝牌指示牌
        之外的牌才发给别家，且每种牌（含赤五）不超过 4 张、赤五每色最多 1 张。
        """
        view = self.view
        out: list[list[str]] = [[] for _ in range(4)]
        out[self.seat] = _hand_names(h0)
        used = [0] * 38
        for ak in range(38):
            used[ak] += int(h0[ak])
        for q in range(4):
            for tile in view.discards[q]:
                used[_ak_index(int(tile))] += 1
            for m in self.melds_all[q]:
                for name in _meld_event(q, m).get("consumed") or []:
                    t, r = mjai.tile_from_mjai(str(name))
                    used[_ak_index(t, r)] += 1
        for d in view.dora_indicators:
            used[_ak_index(int(d))] += 1
        # 池子：每种牌 4 张（赤五算在 4 张里，只在没被用掉时给）
        pool: list[int] = []
        for base in range(34):
            red_ak = _ak_index(base, True) if (base < 27 and base % 9 == 4) else None
            red_used = used[red_ak] if red_ak is not None else 0
            left = 4 - used[_ak_index(base)] - red_used
            if left <= 0:
                continue
            if red_ak is not None and red_used == 0:
                pool.append(red_ak)
                left -= 1
            pool.extend([_ak_index(base)] * left)
        # 固定顺序（不用随机，保证序列化可缓存、可复现），但别是「1m1m1m1m2m...」
        pool = [pool[i] for i in (lambda n: sorted(range(n), key=lambda i: (i * 37) % n))(len(pool))]
        pos = 0
        for q in range(4):
            if q == self.seat:
                continue
            hand: list[str] = []
            # 这家副露里「从自己手里拿出来」的牌必须在起手牌里
            for m in self.melds_all[q]:
                for name in _meld_event(q, m).get("consumed") or []:
                    if len(hand) < 13:
                        hand.append(str(name))
            while len(hand) < 13 and pos < len(pool):
                hand.append(_ak_str(pool[pos]))
                pos += 1
            while len(hand) < 13:            # 池子不够（极端情况）随便补齐
                hand.append(_ak_str(1 + (q * 13 + len(hand) * 7) % 37))
            out[q] = hand[:13]
        return out


def build_events(
    view: View,
    seat: int,
    tracker: _HandTracker | None = None,
    *,
    kind: str = "",
    tile: int | None = None,
    held_is_rinshan: bool = False,
) -> list[dict]:
    """``View`` -> 标准 MJAI 事件流（当前这一局，最后一条是当前决策点）。

    ``kind`` / ``tile`` 是本次决策的种类和相关牌（``engine.Decision`` 的字段）。
    叫牌/荣和类决策（``call`` / ``chi`` / ``ron``）时，最后一条必须是
    **他家刚打出的那张**，所以会把它从牌河里拎出来放到最后。
    """
    tracker = tracker if tracker is not None else _HandTracker()
    n = int(getattr(getattr(view, "rule", None), "players", 0)
            or len(getattr(view, "scores", ())) or 4)
    if n != 4:
        # 先拒绝，再碰 tracker（三麻的 View 只有 3 家，observe 里按 4 家写）
        raise UnsupportedView("akochan 接入目前只做四麻")
    tracker.observe(view, int(seat))
    ser = _Serializer(view, int(seat), tracker, kind=str(kind), tile=tile,
                      held_is_rinshan=held_is_rinshan)
    return ser.build()


# ---------------------------------------------------------------------------
# akochan 的 moves -> 本引擎的选项下标
# ---------------------------------------------------------------------------

def move_to_index(decision: Decision, moves: Sequence[dict] | None) -> int | None:
    """把 akochan 的推荐（``moves``）翻译成本引擎的 ``decision.options`` 下标。

    ``moves`` 是 ``review[0]["moves"]``：一条或多条 MJAI 事件。映射规则：

    ==========  ==================================================
    决策        怎么判
    ==========  ==================================================
    discard     找 ``dahai`` 的 ``pai``，对照 ``options``（牌型下标）
    riichi      第一条是 ``reach`` -> 立直
    tsumo/ron   有 ``hora`` -> 和
    call        有 ``pon`` / ``daiminkan`` -> 碰 / 明杠
    chi         有 ``chi`` -> 吃
    kan         有 ``ankan`` / ``kakan`` 且牌对得上 -> 杠
    kyuushu     有 ``kyushukyuhai`` -> 九种九牌
    ==========  ==================================================

    对不上（或没有 moves）返回 ``None``，调用方走兜底。
    """
    if not moves:
        return None
    opts = list(decision.options)
    types = [str(m.get("type") or "") for m in moves]
    kind = str(decision.kind)

    def pick(name: str) -> int | None:
        try:
            return opts.index(name)
        except ValueError:
            return None

    if kind == "discard":
        for m in reversed(moves):
            if str(m.get("type")) != "dahai":
                continue
            try:
                tile, _red = mjai.tile_from_mjai(str(m.get("pai") or ""))
            except ValueError:
                return None
            if tile in opts:
                return opts.index(tile)
            return None
        return None
    if kind == "riichi":
        return pick("yes") if types[:1] == ["reach"] else pick("no")
    if kind in ("tsumo", "ron"):
        return pick("yes") if "hora" in types else pick("no")
    if kind == "kyuushu":
        return pick("yes") if "kyushukyuhai" in types else pick("no")
    if kind == "call":
        if "pon" in types and "pon" in opts:
            return pick("pon")
        if "daiminkan" in types and "kan" in opts:
            return pick("kan")
        return pick("pass")
    if kind == "chi":
        return pick("chi") if "chi" in types else pick("pass")
    if kind == "kan":
        for m in moves:
            if str(m.get("type")) not in ("ankan", "kakan"):
                continue
            name = m.get("pai")
            if not name and m.get("consumed"):
                name = m["consumed"][0]
            try:
                tile, _red = mjai.tile_from_mjai(str(name or ""))
            except ValueError:
                continue
            if decision.tile is not None and tile != int(decision.tile):
                return pick("pass")
            return pick("kan")
        return pick("pass")
    return None


# ---------------------------------------------------------------------------
# 平台座位
# ---------------------------------------------------------------------------

class ExternalAkochanAgent(Agent):
    """一个普通 :class:`aimodel.agents.Agent` 座位：``decide`` 里问 akochan。

    * 每个**决策点**只查一次（自摸/立直/杠/打牌是四次 ``decide``，同一状态
      会复用上一次的 ``moves``）；
    * 叫牌之后的那张打牌（最后一条是**自己**的碰/吃，akochan 不接受）用
      它在上一次 fuuro 选择里**一起给出**的那张 ``dahai``；
    * 任何失败（没装 / 超时 / 崩了 / 没有 moves / 序列化不自洽 / 三麻）
      都退回 :class:`~aimodel.agents.HeuristicAgent`，连教师也炸才返回 0
      （``options[0]`` 在所有决策里都是合法动作）。
    """

    name = "akochan"

    def __init__(
        self,
        seed: int = 0,
        *,
        root: str | Path | None = None,
        engine: AkochanEngine | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        fallback: Agent | None = None,
        tracker: _HandTracker | None = None,
        enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.engine = engine if engine is not None else AkochanEngine(
            root, timeout=timeout)
        self.fallback = fallback if fallback is not None else HeuristicAgent(self.seed)
        self.tracker = tracker if tracker is not None else _HandTracker()
        self.enabled = bool(enabled)
        self.decisions = 0
        self.cached = 0
        self.fallbacks = 0
        self._cache_key: str | None = None
        self._cache_moves: list[dict] = []
        self._post_call: dict[str, Any] | None = None
        self._rinshan_armed = False
        self._rinshan_key: tuple | None = None
        self._last_error = ""

    # -- 内部 ---------------------------------------------------------------
    def _query(self, events: list[dict], seat: int) -> list[dict]:
        key = hashlib.sha1(
            json.dumps(events, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key == self._cache_key:
            self.cached += 1
            return list(self._cache_moves)
        moves = self.engine.query(events, seat)
        self._cache_key = key
        self._cache_moves = list(moves)
        return list(moves)

    def _fallback(self, decision: Decision, view: View, reason: str) -> int:
        self.fallbacks += 1
        self._last_error = reason
        log.warning("akochan 座位兜底（%s）：%s", decision.kind, reason)
        try:
            idx = int(self.fallback.decide(decision, view))
            if not isinstance(idx, bool) and 0 <= idx < len(decision.options):
                return idx
        except Exception:  # noqa: BLE001 - 教师也炸就返回 0
            log.exception("akochan 兜底的教师也抛异常")
        return 0

    # -- 接口 ---------------------------------------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        self.decisions += 1
        seat = int(getattr(view, "player", 0))
        kind = str(decision.kind)
        if not self.enabled:
            return self._fallback(decision, view, "座位被禁用")
        if kind not in SUPPORTED_KINDS:
            return self._fallback(decision, view, f"不支持的决策 {kind}")
        try:
            n = int(getattr(getattr(view, "rule", None), "players", 0)
                    or len(view.scores))
        except Exception:  # noqa: BLE001
            n = 0
        if n != 4:
            return self._fallback(decision, view, "akochan 接入只做四麻")
        if not self.engine.available:
            return self._fallback(decision, view, self.engine.why or "akochan 不可用")

        # 本次 View 是不是「杠完补牌」？答案在「杠决策之后的第一批决策」里。
        fp = _HandTracker.fingerprint(view, seat)
        if self._rinshan_armed:
            self._rinshan_key = fp
            self._rinshan_armed = False
        held_rinshan = self._rinshan_key is not None and fp == self._rinshan_key
        if self._rinshan_key is not None and fp != self._rinshan_key:
            self._rinshan_key = None
            held_rinshan = False

        # 碰/吃之后的那张打牌：akochan 最后一条不能是「自己的副露」，
        # 它在上一次 fuuro 选择里已经把要打的那张一起给了。
        # （只有「刚叫完、还没摸牌」的那次打牌才是这种情况：drawn is None）
        if kind == "discard" and self._post_call is not None:
            if view.drawn is None:
                tile = int(self._post_call.get("tile", -1))
                self._post_call = None
                opts = list(decision.options)
                if tile in opts:
                    return opts.index(tile)
                return self._fallback(decision, view,
                                      "叫牌后要打的那张不在合法选项里")
            # 状态对不上（已经摸过牌了）：别用旧的缓存，正常查询
            self._post_call = None

        try:
            events = build_events(
                view, seat, self.tracker, kind=kind,
                tile=decision.tile, held_is_rinshan=held_rinshan,
            )
        except SerializeError as exc:
            return self._fallback(decision, view, f"序列化失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            log.exception("akochan 序列化异常")
            return self._fallback(
                decision, view, f"序列化异常：{type(exc).__name__}: {exc}")

        try:
            moves = self._query(events, seat)
        except AkochanTimeout as exc:
            return self._fallback(decision, view, str(exc))
        except AkochanError as exc:
            return self._fallback(decision, view, str(exc))
        except Exception as exc:  # noqa: BLE001 - 任何意外都不能卡住牌桌
            log.exception("akochan 查询异常")
            return self._fallback(decision, view, f"{type(exc).__name__}: {exc}")

        idx = move_to_index(decision, moves)
        if idx is None or not (0 <= idx < len(decision.options)):
            return self._fallback(
                decision, view,
                "akochan 没有给出可用的 moves" if not moves
                else "moves 和合法选项对不上",
            )

        # 记下「叫牌后要打的那张」：碰/吃 的 moves 是 [副露, 打牌]
        if kind in ("call", "chi"):
            for j, m in enumerate(moves):
                if str(m.get("type")) not in ("pon", "chi"):
                    continue
                for nxt in moves[j + 1:]:
                    if str(nxt.get("type")) != "dahai":
                        continue
                    try:
                        tile, _red = mjai.tile_from_mjai(str(nxt.get("pai") or ""))
                    except ValueError:
                        break
                    self._post_call = {"tile": tile, "kind": kind}
                    break
                break
        # 记下「杠完补牌」：下一个决策点的 View 里那张摸牌是岭上牌
        made_kan = (kind == "kan" and idx == 1) or (
            kind == "call" and str(decision.options[idx]) == "kan")
        if made_kan:
            self._rinshan_armed = True
            self._rinshan_key = None
        return idx

    def reset(self, seed: int = 0) -> None:  # pragma: no cover - 可选钩子
        self._cache_key = None
        self._cache_moves = []
        self._post_call = None
        self._rinshan_armed = False
        self._rinshan_key = None
        self.tracker = _HandTracker()
        if hasattr(self.fallback, "reset"):
            self.fallback.reset(seed)

    def summary(self) -> dict:
        out = {
            "model": MODEL_NAME,
            "decisions": self.decisions,
            "cached": self.cached,
            "fallbacks": self.fallbacks,
            "last_error": self._last_error,
        }
        out.update(self.engine.summary())
        return out


def make_agent(seed: int = 0, **kw: Any) -> ExternalAkochanAgent:
    """平台 ``AISeat`` 用这个建座位。"""
    kw.setdefault("timeout", float(os.environ.get("AKOCHAN_TIMEOUT", DEFAULT_TIMEOUT)))
    return ExternalAkochanAgent(seed=seed, **kw)


__all__ = [
    "MODEL_NAME", "SUPPORTED_KINDS", "DEFAULT_TIMEOUT", "DEFAULT_OMP_THREADS",
    "AkochanEngine", "AkochanError", "AkochanUnavailable", "AkochanTimeout",
    "ExternalAkochanAgent", "SerializeError", "UnsupportedView",
    "_HandTracker", "build_events", "move_to_index", "make_agent", "available",
    "locate_root",
]
