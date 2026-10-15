"""决策级对局记录 + 玩家水平画像。

为什么要有这一层
----------------

**只有决策级数据能训练**。「谁第几名、多少点」这种结果数据训练不出东西来；
真正有用的是「在什么局面下、引擎给了哪些合法动作、他选了哪一个、想了多久」。

所以这里对**每一个决策点**都记一条 ``decision``；一局结束时记一条 ``hand``
（四家手牌全亮 + 和牌详情，危险度模型的正样本来源）；一场结束时记 ``match``
（顺位 / ELO / 画像）。

工程约束
--------

* **异步写、失败只打日志**：记录绝不能影响对局，更不能把桌子搞崩。
  写盘在一个独立线程里做，队列满了就丢最旧的，坏盘 / 只读目录都只是打日志。
* **按天分文件**，写 gzip 流（一行一条 JSON）。决策数据涨得很快，不压缩
  几个月就是几十 GB。**必须定期 flush** —— gzip 不 flush 的话磁盘上一直是 0 字节。
* **容量上限 + 轮转**：超过 ``max_bytes`` 换下一个分片，超过
  ``max_total_bytes`` 删最老的那天。
* **不筛选**：弱玩家的数据一条都不丢。筛选是不可逆的，标准留给训练时定。
* **隐私**：玩家 ID 是随机匿名串（``p_xxxxxxxx``），不是昵称 / 邮箱；
  玩家可以选「本局不记录」，选了就**真的**不落盘。
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import queue
import random
import string
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("maj.record")

#: 记录格式版本（字段有变动就 +1，方便以后按版本读）
SCHEMA = 1

_ALNUM = string.ascii_lowercase + string.digits


def anon_id() -> str:
    """匿名玩家 ID：``p_`` + 8 位随机串。**不掺任何真实信息。**"""
    return "p_" + "".join(random.choice(_ALNUM) for _ in range(8))


# ---------------------------------------------------------------------------
# 玩家画像
# ---------------------------------------------------------------------------


@dataclass
class PlayerProfile:
    """一个玩家的累计统计 + ELO。

    ELO **必须和 AI 用完全同一套公式**（同样的 K 值、同样的点数分除数），
    这样人类分数和模型分数就在同一个量纲上：说「我打赢了 maj-flashv1.1」
    就是一句校准过的水平读数。
    """

    uid: str
    name: str = ""
    kind: str = "human"          # human | ai
    games: int = 0
    wins: int = 0                # 一位次数
    rank_sum: float = 0.0        # 顺位之和（1 = 一位）
    hands: int = 0               # 打过的局数（一局 = 一庄）
    agari: int = 0
    tsumo: int = 0
    ron: int = 0
    deal_in: int = 0
    draw: int = 0
    riichi: int = 0
    kita: int = 0
    calls: int = 0
    score_sum: int = 0
    elo: float = 1500.0
    games_since_elo: int = 0
    declared_rank: str = ""
    main_mode: str = ""
    updated_at: float = 0.0

    @property
    def avg_rank(self) -> float:
        return self.rank_sum / self.games if self.games else 0.0

    @property
    def first_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def agari_rate(self) -> float:
        return self.agari / self.hands if self.hands else 0.0

    @property
    def deal_in_rate(self) -> float:
        return self.deal_in / self.hands if self.hands else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.games if self.games else 0.0

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "kind": self.kind,
            "games": self.games,
            "wins": self.wins,
            "avg_rank": round(self.avg_rank, 4),
            "first_rate": round(self.first_rate, 4),
            "hands": self.hands,
            "agari": self.agari,
            "tsumo": self.tsumo,
            "ron": self.ron,
            "agari_rate": round(self.agari_rate, 4),
            "deal_in": self.deal_in,
            "deal_in_rate": round(self.deal_in_rate, 4),
            "draw": self.draw,
            "riichi": self.riichi,
            "kita": self.kita,
            "calls": self.calls,
            "avg_score": round(self.avg_score, 1),
            "elo": round(self.elo, 1),
            "declared_rank": self.declared_rank,
            "main_mode": self.main_mode,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerProfile":
        p = cls(uid=str(d.get("uid") or anon_id()))
        for k in ("name", "kind", "declared_rank", "main_mode"):
            setattr(p, k, str(d.get(k) or ""))
        for k in ("games", "wins", "hands", "agari", "tsumo", "ron", "deal_in",
                  "draw", "riichi", "kita", "calls", "score_sum", "games_since_elo"):
            try:
                setattr(p, k, int(d.get(k) or 0))
            except (TypeError, ValueError):
                setattr(p, k, 0)
        for k in ("rank_sum", "elo", "updated_at"):
            try:
                setattr(p, k, float(d.get(k) or 0.0))
            except (TypeError, ValueError):
                setattr(p, k, 0.0)
        if not p.elo:
            p.elo = 1500.0
        return p


# ---------------------------------------------------------------------------
# ELO：人和 AI 用同一套
# ---------------------------------------------------------------------------

ELO_K_NEW = 40.0
ELO_K = 24.0
ELO_K_MASTER = 16.0
#: 前多少局算「新手期」
ELO_PROVISIONAL = 15
#: ELO 起点，人也一样
ELO_START = 1500.0
#: 点数分除数：把「终局点数差」折成「等效名次差」的量纲。
#: 30000 点差 ≈ 一个顺位，这是日麻圈常用的换算尺度。
ELO_POINT_DIVISOR = 30000.0


def _k_factor(profile: PlayerProfile) -> float:
    if profile.games < ELO_PROVISIONAL:
        return ELO_K_NEW
    if profile.elo >= 2000:
        return ELO_K_MASTER
    return ELO_K


def elo_update(
    profiles: list[PlayerProfile], scores: list[int], ranks: list[int]
) -> list[float]:
    """按一局的结果更新 ELO，返回每人这一局的变化量。

    做法是**成对比较**（日麻常用的「顺位 ELO」）：每两个人之间算一次
    「谁赢了」，赢了拿分、输了丢分；比分里再掺一点点终局点数，
    让「同样二位但差 20000 点」能区分开。

    人和 AI 走的是同一个函数，所以分数可比。成对比较天然零和。
    """
    n = len(profiles)
    if n < 2:
        return [0.0] * n
    deltas = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = profiles[i], profiles[j]
            exp_i = 1.0 / (1.0 + 10 ** ((pj.elo - pi.elo) / 400.0))
            if ranks[i] < ranks[j]:
                actual = 1.0
            elif ranks[i] > ranks[j]:
                actual = 0.0
            else:
                actual = 0.5
            diff = (scores[i] - scores[j]) / ELO_POINT_DIVISOR
            actual = min(1.0, max(0.0, actual + diff * 0.15))
            k = (_k_factor(pi) + _k_factor(pj)) / 2.0
            d = k * (actual - exp_i)
            deltas[i] += d
            deltas[j] -= d
    scale = 1.0 / (n - 1)
    return [d * scale for d in deltas]


def apply_match(profile: PlayerProfile, *, rank: int, score: int, delta: float,
                stats: dict | None = None) -> None:
    """把一局的结果并进画像。"""
    profile.games += 1
    profile.rank_sum += rank + 1
    profile.score_sum += score
    if rank == 0:
        profile.wins += 1
    profile.elo += delta
    profile.games_since_elo += 1
    profile.updated_at = time.time()
    st = stats or {}
    if st:
        profile.hands += int(st.get("hands", 0))
        profile.tsumo += int(st.get("tsumo", 0))
        profile.ron += int(st.get("ron", 0))
        profile.agari += int(st.get("tsumo", 0)) + int(st.get("ron", 0))
        profile.deal_in += int(st.get("deal_in", 0))
        profile.draw += int(st.get("draws", 0))
        profile.riichi += int(st.get("riichi", 0))
        profile.kita += int(st.get("kita", 0))
        profile.calls += int(st.get("calls", 0))


# ---------------------------------------------------------------------------
# 画像表
# ---------------------------------------------------------------------------


class ProfileStore:
    """玩家画像表。内存里一份，定期 + 退出时落盘。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self._p: dict[str, PlayerProfile] = {}
        self._dirty = False

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            with self.lock:
                for uid, d in (data.get("players") or {}).items():
                    self._p[uid] = PlayerProfile.from_dict({**d, "uid": uid})
            log.info("载入 %d 个玩家画像（%s）", len(self._p), self.path)
        except Exception:  # noqa: BLE001
            log.exception("画像表载入失败（当成空的继续跑）")

    def save(self) -> None:
        with self.lock:
            if not self._dirty:
                return
            blob = {
                "schema": SCHEMA,
                "saved_at": time.time(),
                "players": {uid: p.as_dict() for uid, p in self._p.items()},
            }
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            log.exception("画像表写盘失败")

    def get(self, uid: str, *, name: str = "", kind: str = "human") -> PlayerProfile:
        with self.lock:
            p = self._p.get(uid)
            if p is None:
                p = PlayerProfile(uid=uid, name=name, kind=kind, elo=ELO_START)
                self._p[uid] = p
                self._dirty = True
            elif name and p.name != name:
                p.name = name
                self._dirty = True
            return p

    def touch(self) -> None:
        with self.lock:
            self._dirty = True

    def snapshot(self, uid: str) -> dict:
        with self.lock:
            p = self._p.get(uid)
            return p.as_dict() if p else PlayerProfile(uid=uid).as_dict()

    def all(self) -> list[dict]:
        with self.lock:
            return [p.as_dict() for p in self._p.values()]


# ---------------------------------------------------------------------------
# 异步写盘
# ---------------------------------------------------------------------------


class AsyncWriter:
    """后台线程写 gzip JSONL。

    * 队列有上限，满了**丢最旧的**并计数（宁可丢数据也不能拖住对局）
    * 任何异常只打日志，不往上抛
    * 按天分文件：``games-YYYY-MM-DD.jsonl.gz``
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        prefix: str = "games",
        max_bytes: int = 256 << 20,
        max_total_bytes: int = 8 << 30,
        flush_interval: float = 2.0,
        queue_max: int = 20000,
    ) -> None:
        self.dir = out_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.max_total_bytes = max_total_bytes
        self.flush_interval = flush_interval
        self.queue: queue.Queue[tuple[str, dict] | None] = queue.Queue(maxsize=queue_max)
        self.dropped = 0
        self.written = 0
        self.errors = 0
        self.bytes_written = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fh: gzip.GzipFile | None = None
        self._fh_day = ""
        self._fh_part = 0
        self._fh_bytes = 0
        self._disabled = False
        self._flushed_at = 0.0

    # -- 生命周期 -------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"maj-record-{self.prefix}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 6.0) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout)
        self._close()

    def put(self, kind: str, rec: dict) -> None:
        if self._disabled:
            return
        try:
            self.queue.put_nowait((kind, rec))
        except queue.Full:
            self.dropped += 1
            if self.dropped % 500 == 1:
                log.warning("记录队列满了，已丢弃 %d 条", self.dropped)

    # -- 线程 -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=self.flush_interval)
            except queue.Empty:
                self._maybe_roll(time.time())
                self._maybe_flush()
                continue
            if item is None:
                break
            kind, rec = item
            try:
                self._write(kind, rec)
            except Exception:  # noqa: BLE001 - 记录失败绝不能影响对局
                self.errors += 1
                if self.errors <= 5:
                    log.exception("写记录失败（第 %d 次）", self.errors)
            self._maybe_flush()
        # 收尾：把队列里剩下的都写掉
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            try:
                self._write(item[0], item[1])
            except Exception:  # noqa: BLE001
                self.errors += 1

    def _maybe_flush(self, force: bool = False) -> None:
        """定期把 gzip 缓冲刷到磁盘。

        **必须做这件事**：gzip 是流式压缩，不 flush 的话文件在磁盘上一直是
        0 字节（只有关文件才落盘），运维和验证脚本都会以为「什么都没写」。
        """
        if self._fh is None:
            return
        now = time.time()
        if not force and now - self._flushed_at < 1.0:
            return
        try:
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass
        self._flushed_at = now

    def flush(self) -> None:
        """把缓冲区刷到磁盘（验证 / 运维用）。"""
        self._maybe_flush(force=True)

    def _write(self, kind: str, rec: dict) -> None:
        """落一条记录。

        **``kind`` 是保留字段**（记录类型），业务字段一律别用这个名字 ——
        ``hand`` 记录里「本局怎么结束的」叫 ``result_kind``。
        """
        if "kind" in rec:
            raise ValueError("记录里不能自带 kind 字段（保留给记录类型）")
        rec = {"schema": SCHEMA, "kind": kind, **rec}
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
        raw = line.encode("utf-8")
        self._ensure_file(time.time())
        if self._fh is None:
            raise OSError(f"记录文件打不开：{self.dir}")
        self._fh.write(raw)
        self._fh_bytes += len(raw)
        self.bytes_written += len(raw)
        self.written += 1
        if self._fh_bytes >= self.max_bytes:
            self._roll_part()

    def _ensure_file(self, now: float) -> None:
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if self._fh is not None and day == self._fh_day:
            return
        self._close()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh_day = day
        self._fh_part = 0
        self._fh = self._open(day, 0)
        self._enforce_capacity()

    def _open(self, day: str, part: int) -> gzip.GzipFile | None:
        suffix = "" if part == 0 else f".{part:03d}"
        path = self.dir / f"{self.prefix}-{day}{suffix}.jsonl.gz"
        try:
            return gzip.open(path, "ab", compresslevel=6)
        except Exception:  # noqa: BLE001
            log.exception("打不开记录文件 %s", path)
            return None

    def _roll_part(self) -> None:
        self._maybe_flush(force=True)
        self._close()
        self._fh_part += 1
        self._fh_bytes = 0
        self._fh = self._open(self._fh_day, self._fh_part)

    def _maybe_roll(self, now: float) -> None:
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if self._fh is not None and day != self._fh_day:
            self._ensure_file(now)

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None

    def _enforce_capacity(self) -> None:
        """超过总量上限就删最老的文件。"""
        try:
            files = sorted(self.dir.glob(f"{self.prefix}-*.jsonl.gz"),
                           key=lambda p: p.name)
            total = sum(p.stat().st_size for p in files)
            while total > self.max_total_bytes and len(files) > 1:
                victim = files.pop(0)
                sz = victim.stat().st_size
                victim.unlink(missing_ok=True)
                total -= sz
                log.warning("记录目录超限，删除最老的分片 %s（%.1f MB）",
                            victim.name, sz / 1e6)
        except Exception:  # noqa: BLE001
            log.exception("清理记录分片失败")

    def stats(self) -> dict:
        return {
            "written": self.written,
            "dropped": self.dropped,
            "errors": self.errors,
            "queued": self.queue.qsize(),
            "bytes": self.bytes_written,
            "dir": str(self.dir),
        }


# ---------------------------------------------------------------------------
# 局面快照
# ---------------------------------------------------------------------------


def tile_id(index: int) -> str:
    from . import tiles as T

    return T.index_to_id(index)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """宽容取属性。

    ``View`` / ``Player`` 的字段会随训练侧改动而增减，而记录器是**只读**的
    旁路，不该因为上游加了个字段就整条记录失败。
    """
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


def _meld_json(m: Any) -> dict:
    return {
        "kind": _get(m, "kind", "?"),
        "tile": tile_id(_get(m, "tile", 0)),
        "from": _get(m, "from_player", -1),
        "aka": _get(m, "aka", 0),
    }


def snapshot_view(view: Any) -> dict:
    """把引擎的 :class:`~aimodel.engine.View` 压成可落盘的局面快照。

    **这是决策级数据的核心**：有了它 + 合法动作 + 实际选择，这份数据就能
    直接拿去训练（模仿学习 / 离线 RL）。

    ``View`` 的字段是**跨版本**的（训练侧会加字段），所以这里一律用
    :func:`_get` 取、取不到就退回一个合理的默认值 —— 记录器绝不能因为
    上游加了个字段就把整条决策丢掉（这个坑踩过一次：少了 ``discards_red``
    直接让所有 decision 记录报 AttributeError，一条都没落盘）。
    """
    discards = _get(view, "discards", [])
    return {
        "seat": _get(view, "player", 0),
        "hand": [tile_id(t) for t in range(34) for _ in range(view.hand[t])],
        "hand_counts": list(view.hand),
        "hand_red": list(_get(view, "hand_red", [0] * 34)),
        "drawn": (tile_id(view.drawn) if view.drawn is not None else None),
        "melds": [_meld_json(m) for m in _get(view, "melds", [])],
        "ponds": [[tile_id(t) for t in row] for row in discards],
        "ponds_red": [list(r) for r in (_get(view, "discards_red", []) or [])],
        "melds_all": [[_meld_json(m) for m in row]
                      for row in _get(view, "melds_all", [])],
        "kita": list(_get(view, "kita", [])),
        "scores": list(_get(view, "scores", [])),
        "dora_indicators": [tile_id(d) for d in _get(view, "dora_indicators", [])],
        "round_wind": tile_id(_get(view, "round_wind", 27)),
        "seat_wind": tile_id(_get(view, "seat_wind", 27)),
        "wall_left": _get(view, "wall_left", 0),
        "turn": _get(view, "turn", 0),
        "riichi": list(_get(view, "riichi", [])),
        "ippatsu": list(_get(view, "ippatsu", [])),
        "is_dealer": view.is_dealer,
        "last_discard": (
            [view.last_discard[0], tile_id(view.last_discard[1])]
            if view.last_discard else None
        ),
        "players": len(_get(view, "scores", []) or [1]),
        "aka": bool(getattr(_get(view, "rule", None), "aka", False)),
        "kita_rule": bool(getattr(_get(view, "rule", None), "kita_dora", False)),
    }


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    """读一个 gzip JSONL 分片，**容忍被截断的尾巴**。

    分片是**追加**写的（``gzip.open(path, "ab")``），gzip 的追加模式会往同一个
    文件里接一个新的 gzip 成员（member）。这带来两个必须处理的现实：

    1. 进程被强杀时，最后一个成员是**残缺**的；
    2. 一个好文件里可能有**很多个成员**。

    三个坑，全踩过：

    * :func:`gzip.open` 会在**读第一行之前**就抛
      ``EOFError: Compressed file ended before the end-of-stream marker``，
      于是整个文件里已经写好的记录一起读不出来；
    * 换成「自己管 ``GzipFile``、出错就 break」也**不对** —— 异常抛出时，
      解出来的那几百 KB 已经在这个调用的返回值里了，直接 break 等于丢掉；
    * 用 ``decompressobj`` 在**同一个缓冲区里原地搬移** ``unused_data``
      更是错上加错：成员边界一旦算错，后面全部解不出来，而错误是**静默**的
      （实测：2.5MB / 5498 条的分片读回来 0 条，而且不报任何错）。

    所以这里用最笨也最稳的办法：**一个成员一个成员地解，每个成员解完就
    把它的全部输出切行**。成员用 ``unused_data`` 定位，不自己算偏移。
    """
    if not path.exists():
        return
    try:
        data = path.read_bytes()
    except OSError:
        return
    pos = 0
    total = len(data)
    while pos < total:
        d = zlib.decompressobj(31)          # 31 = gzip 容器
        # **一块一块喂**，每块出来就立刻切行 —— 不能 `decompress(整段)`：
        # 那样一遇到坏字节就整段抛异常，之前解出来的全部丢掉。
        # 分块喂的话，坏点之前的记录一定能保住（实测：5500 条的分片里，
        # 坏点在中间时能救回大半）。
        i = pos
        buf = b""
        while i < total:
            piece = data[i : i + (1 << 16)]
            i += len(piece)
            try:
                out = d.decompress(piece)
            except zlib.error:
                out = b""
                i = total                    # 这个成员废了，跳到下一个
            buf += out
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl]
                buf = buf[nl + 1 :]
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
            if d.eof or d.unused_data:
                break
        if buf.strip():
            try:
                yield json.loads(buf)
            except json.JSONDecodeError:
                pass
        if not d.unused_data:
            # 没有下一个成员了（读完了，或者被截断 / 损坏在这里）
            return
        pos = total - len(d.unused_data)
        if pos <= 0:                         # 保险：不许死循环
            return


@dataclass
class SeatTiming:
    started_at: float = 0.0
    kind: str = ""


class GameRecorder:
    """一场对局的记录器。

    生命周期跟着 :class:`~webapp.server.table.Room` 走：

    * 每次有人决策 -> :meth:`record_decision`（引擎线程里调，只入队，不写盘）
    * 引擎发 ``hand_start`` -> :meth:`begin_hand`
    * 一局结束 -> :meth:`end_hand`（四家手牌全亮 + 和牌详情）
    * 一场结束 -> :meth:`end_match`（顺位 / ELO / 画像）

    ``opt_out`` 里的座位**一条都不记**（玩家自己选的「本局不记录」）。
    """

    def __init__(
        self,
        *,
        writer: AsyncWriter,
        profiles: ProfileStore,
        room_id: str,
        room_name: str,
        players: int,
        rounds: int,
        seats: list[dict],
        opt_out: Iterable[int] = (),
        seed: int = 0,
        ai_delay: float = 0.0,
    ) -> None:
        self.writer = writer
        self.profiles = profiles
        self.room_id = room_id
        self.room_name = room_name
        self.players = players
        self.rounds = rounds
        self.seed = seed
        self.ai_delay = ai_delay
        self.opt_out = {int(s) for s in opt_out}
        self.seats = seats
        self.game_uid = f"{room_id}-{int(time.time() * 1000):x}"
        self.started_at = time.time()
        self.hand_no = 0
        self.decisions = 0
        self.hands_recorded = 0
        self.timings: dict[int, SeatTiming] = {}
        self.elo_before = {s["seat"]: float(s.get("elo", ELO_START)) for s in seats}
        #: 牌河里每张牌是不是赤 5（房间在 discard 事件里现读现攒，
        #: 兜底写成一份副本 —— 引擎的 Player 不带这个信息）
        self.pond_red: dict[int, list[int]] = {}
        self._closed = False
        self._write("game_start", {
            "game_uid": self.game_uid,
            "room": self.room_id,
            "room_name": self.room_name,
            "players": self.players,
            "rounds": self.rounds,
            "seed": self.seed,
            "ai_delay": self.ai_delay,
            "at": self.started_at,
            "opt_out": sorted(self.opt_out),
            "seats": self.seats,
        })

    def _write(self, kind: str, rec: dict) -> None:
        rec.setdefault("game_uid", self.game_uid)
        rec.setdefault("room", self.room_id)
        rec.setdefault("at", time.time())
        self.writer.put(kind, rec)

    def seat_info(self, seat: int) -> dict:
        for s in self.seats:
            if s["seat"] == seat:
                return s
        return {"seat": seat, "name": f"P{seat}", "uid": "", "kind": "?"}

    def noting(self, seat: int) -> bool:
        """这个座位要不要记。玩家选了「不记录」就返回 False。"""
        return seat not in self.opt_out

    # -- 决策点 ---------------------------------------------------------
    def note_turn_start(self, seat: int, kind: str) -> None:
        self.timings[seat] = SeatTiming(started_at=time.time(), kind=kind)

    def record_decision(
        self,
        *,
        seat: int,
        kind: str,
        options: list,
        tile: int | None,
        index: int,
        view: Any,
        source: str = "agent",
    ) -> None:
        if not self.noting(seat) or self._closed:
            return
        t = self.timings.get(seat)
        now = time.time()
        elapsed_ms = int((now - t.started_at) * 1000) if t and t.started_at else None
        info = self.seat_info(seat)
        # 选项**原样**存：出牌是牌型下标，其余是字符串（引擎给什么就存什么）
        opts = []
        for o in options:
            if isinstance(o, int):
                opts.append({"index": len(opts), "tile": tile_id(o), "raw": o})
            else:
                opts.append({"index": len(opts), "action": str(o)})
        chosen = opts[index] if 0 <= index < len(opts) else None
        self._write("decision", {
            "hand_no": self.hand_no,
            "seat": seat,
            "player": info.get("uid", ""),
            "name": info.get("name", ""),
            "kind_player": info.get("kind", "?"),
            "ai": info.get("ai"),
            "decision": kind,
            "target_tile": (tile_id(tile) if tile is not None else None),
            "options": opts,
            "choice_index": index,
            "choice": chosen,
            "elapsed_ms": elapsed_ms,
            "source": source,
            "profile": self.profiles.snapshot(info.get("uid", "")),
            "view": snapshot_view(view),
        })
        self.decisions += 1

    # -- 一局 -----------------------------------------------------------
    def begin_hand(self, hand_no: int, extra: dict | None = None) -> None:
        self.hand_no = hand_no
        self._write("hand_start", {"hand_no": hand_no, **(extra or {})})

    def end_hand(self, *, hand_no: int, game: Any, result: Any,
                 detail: dict | None) -> None:
        """一局结束：**四家手牌 + 和牌详情全部落盘**。

        这是危险度模型的正样本来源 —— 谁打出的哪张牌被谁荣和了、多少钱，
        都能对上。
        """
        if self._closed:
            return
        hands = []
        for i, pl in enumerate(getattr(game, "players", [])):
            info = self.seat_info(i)
            hand_counts = list(_get(pl, "hand", [0] * 34))
            hands.append({
                "seat": i,
                "player": info.get("uid", ""),
                "name": info.get("name", ""),
                "kind": info.get("kind", "?"),
                "ai": info.get("ai"),
                "recorded": self.noting(i),
                "hand": [tile_id(t) for t in range(34)
                         for _ in range(hand_counts[t])],
                "hand_counts": hand_counts,
                "hand_red": list(_get(pl, "red", [0] * 34)),
                "melds": [_meld_json(m) for m in _get(pl, "melds", [])],
                "pond": [tile_id(t) for t in _get(pl, "pond", [])],
                # Player 没有 pond_red（那是给网页牌桌的字段），所以这里
                # 用房间自己攒的牌河赤 5 记录；没有就退回全 0。
                "pond_red": list(self.pond_red.get(i, [])),
                "kita": _get(pl, "kita", 0),
                "riichi": bool(_get(pl, "riichi", False)),
                "riichi_tile": (tile_id(_get(pl, "riichi_tile"))
                                if _get(pl, "riichi_tile") is not None else None),
                "score": game.scores[i] if i < len(game.scores) else 0,
            })
        agari = []
        for q, a in zip(getattr(result, "winners", ()) or (),
                        getattr(result, "agaris", ()) or ()):
            if a is None:
                continue
            agari.append({
                "seat": q,
                "player": self.seat_info(q).get("uid", ""),
                "yaku": [{"key": k, "han": v} for k, v in a.yaku],
                "dora_detail": dict(a.dora_detail),
                "han": a.han,
                "dora_han": a.dora_han,
                "total_han": a.total_han,
                "fu": a.fu,
                "yakuman": a.yakuman,
            })
        self._write("hand", {
            "hand_no": hand_no,
            # 「本局怎么结束的」叫 result_kind —— ``kind`` 是记录类型的保留字段
            "result_kind": result.kind,
            "kind_cn": result.describe(),
            "winner": result.winner,
            "winners": list(result.winners or ()),
            "loser": result.loser,
            "deltas": list(result.deltas),
            "tenpai": list(result.tenpai),
            "turns": result.turns,
            "note": result.note,
            "scores_after": list(game.scores),
            "sticks": game.sticks,
            "dora_indicators": [tile_id(game._indicator(i))
                                for i in range(game.dora_slots)],
            "ura_indicators": [tile_id(game._indicator(i, ura=True))
                               for i in range(game.dora_slots)],
            "wall_left": game.wall_left(),
            "hands": hands,
            "agari": agari,
            "detail": detail,
        })
        self.hands_recorded += 1

    # -- 一场 -----------------------------------------------------------
    def end_match(self, result: Any) -> list[float]:
        """对局结束：结算、更新 ELO、落 ``match`` 记录。返回 ELO 变化量。

        ``stats`` 里放的是引擎的**原始累计计数**，不是四舍五入过的比率 ——
        这样画像的「放铳率」和逐局记录能一一对应。
        """
        if self._closed:
            return [0.0] * self.players
        infos = [self.seat_info(i) for i in range(self.players)]
        raw_stats = []
        for i in range(self.players):
            st = result.stats[i] if i < len(result.stats) else None
            raw_stats.append({
                "hands": int(getattr(st, "hands", 0) or 0),
                "tsumo": int(getattr(st, "tsumo", 0) or 0),
                "ron": int(getattr(st, "ron", 0) or 0),
                "deal_in": int(getattr(st, "deal_in", 0) or 0),
                "draws": int(getattr(st, "draws", 0) or 0),
                "riichi": int(getattr(st, "riichi", 0) or 0),
                "kita": int(getattr(st, "kita", 0) or 0),
                "calls": int(getattr(st, "calls", 0) or 0),
                "avg_han": (round(sum(st.win_han) / len(st.win_han), 3)
                            if getattr(st, "win_han", None) else 0.0),
                "avg_turns": (round(sum(st.turns) / len(st.turns), 2)
                              if getattr(st, "turns", None) else 0.0),
            })
        profiles = []
        for i, info in enumerate(infos):
            if self.noting(i):
                profiles.append(self.profiles.get(
                    info.get("uid", f"ai:{i}"),
                    name=info.get("name", ""),
                    kind=info.get("kind", "human"),
                ))
            else:
                profiles.append(PlayerProfile(uid=f"optout:{i}", elo=ELO_START))
        deltas = elo_update(profiles, list(result.scores), list(result.ranks))
        rows = []
        for i, info in enumerate(infos):
            if self.noting(i):
                apply_match(profiles[i], rank=result.ranks[i], score=result.scores[i],
                            delta=deltas[i], stats=raw_stats[i])
                self.profiles.touch()
            rows.append({
                "seat": i,
                "player": info.get("uid", ""),
                "name": info.get("name", ""),
                "kind": info.get("kind", "?"),
                "ai": info.get("ai"),
                "recorded": self.noting(i),
                "rank": result.ranks[i],
                "score": result.scores[i],
                "start_score": result.start_scores[i],
                "delta": result.deltas[i],
                "rank_point": result.rank_points[i] if result.rank_points else 0.0,
                "elo_before": round(self.elo_before.get(i, ELO_START), 2),
                "elo_after": round(profiles[i].elo, 4),
                # 原始值严格零和，四舍五入过的那个是给人看的
                "elo_delta_raw": deltas[i],
                "elo_delta": round(deltas[i], 2),
                "stats": raw_stats[i],
                "summary": (result.stats[i].summary() if i < len(result.stats) else {}),
                "profile": self.profiles.snapshot(info.get("uid", "")),
            })
        # 一条 match 记录里**每个人一行**。除了 ``players`` 数组，
        # 还给人类玩家各补一份顶层记录（``player`` = 匿名 ID）——
        # 「同一个人打了 10 局」这种查询直接按 player 过滤就行，
        # 不用先去 players 数组里翻。
        base = {
            "finished_at": time.time(),
            "duration_s": round(time.time() - self.started_at, 2),
            "hands": len(result.hands),
            "scores": list(result.scores),
            "ranks": list(result.ranks),
            "rank_points": [round(x, 2) for x in (result.rank_points or [])],
            "players": rows,
            "decisions": self.decisions,
        }
        # 写完全部记录**再**置 closed —— 顺序反了的话下面那些 per-player
        # 记录会被 ``_write`` 直接丢掉（这个坑踩过一次：ELO 零和校验
        # 拿不到任何 match 记录，0 场可查）。
        for i, row in enumerate(rows):
            if row["kind"] != "human" or not row["player"]:
                continue
            self._write("match", {
                **base,
                "seat": i,
                "player": row["player"],
                "name": row["name"],
                "kind_player": "human",
                "rank": row["rank"],
                "score": row["score"],
                "start_score": row["start_score"],
                "delta": row["delta"],
                "elo_before": row["elo_before"],
                "elo_after": row["elo_after"],
                "elo_delta_raw": row["elo_delta_raw"],
                "elo_delta": row["elo_delta"],
                "stats": row["stats"],
                "profile": row["profile"],
            })
        self._closed = True
        self.profiles.save()
        return deltas

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._write("abort", {"reason": reason, "decisions": self.decisions})

    def flush(self) -> None:
        """把画像表和写盘缓冲都刷一遍（关服 / 验证脚本用）。"""
        self.profiles.save()
        self.writer.flush()
