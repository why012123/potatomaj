"""简易 Web UI：三个人一起打三麻。

三个座位随便配::

    human      人手点（浏览器里点牌）
    external   外部接口（我 = AI 助手，用命令行出牌）
    script     脚本 AI（teacher / ml / random）

跑法::

    # 你坐 0 号位、我坐 1 号位、教师坐 2 号位
    .venv/bin/python -m aimodel.ui.server --seats human,external,teacher

    # 然后浏览器打开
    open http://127.0.0.1:8765/

轮到我（external 座）时，我用::

    .venv/bin/python -m aimodel.ui.assistant          # 看现在该我打什么
    .venv/bin/python -m aimodel.ui.assistant -p 3p    # 打 3p

设计上就一句话：**引擎在后台线程里跑，需要人/外部决策时就把线程挂住**，
HTTP 服务照常响应。每个座位只拿得到自己视角的局面（`/api/state?seat=N`），
所以不管是你还是我，都看不到别人的手牌。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import sanma as S
from ..agents import HeuristicAgent, MLAgent, RandomAgent
from ..engine import ACTION_TABLE, Decision, Game, View
from ..fastshanten import shanten, ukeire_count
from ..encode import visible_counts
from ..net import Net

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
TEMPLATES = HERE.parents[1] / "templates"
WEIGHTS_DIR = HERE.parents[0] / "weights"
WEIGHTS = WEIGHTS_DIR / "policy.npz"

#: 模型权重缓存（同一个模型被多个座位/多局用到时不用反复读盘）
_NET_CACHE: dict[str, object] = {}


#: 每种人数的默认座位（有模型就用模型，没有才退回教师）
def default_seats(players: int = 3) -> list[str]:
    from .. import variants

    # 按风格排个有代表性的顺序：标准 -> 速攻 -> 防守 -> 大牌
    order = {"normal": 0, "flash": 1, "mountain": 2, "huge": 3}
    specs = [
        sp for sp in variants.all_specs()
        if sp.players == players and sp.path.exists()
    ]
    specs.sort(key=lambda sp: order.get(sp.style, 9))
    picks = [sp.name for sp in specs]
    if not picks:
        picks = ["teacher"] * (players - 1)
    seats = ["human"] + picks
    while len(seats) < players:
        seats.append("teacher")
    return seats[:players]


#: 这些后缀的文件是训练/评测过程中的**内部产物**，不该出现在 UI 的下拉里：
#: ``_bcbase`` 是纯克隆基线（用来做「RL 有没有用」的对照），
#: ``_prev`` 是重训前的备份。它们能加载，但不是给人选的。
_INTERNAL_SUFFIX = ("_bcbase", "_prev")


def available_models() -> list[str]:
    """weights/ 目录里**给人用**的模型名（内部对照产物会被滤掉）。"""
    if not WEIGHTS_DIR.exists():
        return []
    out = []
    for f in sorted(WEIGHTS_DIR.glob("*.npz")):
        if f.stem.endswith(_INTERNAL_SUFFIX):
            continue
        out.append(f.stem)
    return out


def load_net(name: str):
    from ..net import Net

    if name in _NET_CACHE:
        return _NET_CACHE[name]
    path = WEIGHTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"没有这个模型: {path}")
    net = Net.load(path)
    _NET_CACHE[name] = net
    return net


class PacedAgent:
    """给 AI 座位加一个节拍：每手停 ``delay`` 秒，方便人盯着看。"""

    name = "paced"

    def __init__(self, inner, delay: float = 1.0) -> None:
        self.inner = inner
        self.delay = delay

    def decide(self, decision, view):
        if self.delay > 0:
            time.sleep(self.delay)
        return self.inner.decide(decision, view)

#: 决策种类 -> 中文提示
KIND_CN = {
    "discard": "打哪张",
    "tsumo": "自摸和牌",
    "ron": "荣和",
    "riichi": "立直",
    "call": "鸣牌（碰 / 杠）",
    "kita": "拔北",
    "kan": "杠",
    "kyuushu": "九种九牌（途中流局）",
}
OPTION_CN = {
    "yes": "要", "no": "不要", "pass": "过", "pon": "碰",
    "kan": "杠", "ron": "荣和", "tsumo": "自摸",
}

#: 出牌顺序提示用的座位名
SEAT_TAG = ("下家", "上家", "自己")


def tile_id(t: int) -> str:
    return S.tile_id(t)


def tile_cn(t: int) -> str:
    return S.tile_cn(t)


class RemoteAgent:
    """把决策交给外面（浏览器 / 命令行），线程挂起等答复。

    超时会自动打第一张合法牌 —— 免得有人关掉浏览器把整局卡死。
    """

    def __init__(self, name: str, timeout: float = 900.0, fallback: Any = None) -> None:
        self.name = name
        self.timeout = timeout
        #: 超时托管：宁可让教师替一手，也不要随便打一张
        self.fallback = fallback
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.pending: dict | None = None
        self.timeout_count = 0

    def decide(self, decision: Decision, view: View) -> int:
        self.event.clear()
        with self.lock:
            self.pending = {"decision": decision, "view": view, "at": time.time()}
        if not self.event.wait(self.timeout):
            self.timeout_count += 1
            with self.lock:
                pending = self.pending
                self.pending = None
            if self.fallback is not None and pending is not None:
                return self.fallback.decide(decision, view)
            return 0  # 没有托管的兜底：只能打第一张
        with self.lock:
            answer = self.pending["answer"]  # type: ignore[index]
            self.pending = None
        return int(answer)

    def submit(self, index: int) -> bool:
        with self.lock:
            if self.pending is None:
                return False
            self.pending["answer"] = index
        self.event.set()
        return True

    def current(self) -> dict | None:
        with self.lock:
            if self.pending is None:
                return None
            d: Decision = self.pending["decision"]
            return {
                "kind": d.kind,
                "options": list(d.options),
                "tile": d.tile,
                "waited": time.time() - self.pending["at"],
            }


def make_script_agent(spec: str, seed: int) -> Any:
    """座位名 -> agent。

    ``teacher`` / ``random`` 是内置的；其它名字一律当成 ``weights/<名字>.npz``
    里的模型（例如 ``maj-flash`` / ``maj-normal4``）。
    """
    spec = spec.strip().lower()
    if spec in ("teacher", "heuristic", "t"):
        return HeuristicAgent(seed)
    if spec in ("random", "r"):
        return RandomAgent(seed)
    if spec in ("ml", "net", "model"):
        spec = "policy"
    try:
        net = load_net(spec)
        return MLAgent(net, seed=seed, greedy=True)
    except FileNotFoundError:
        print(f"⚠️ 找不到模型 {spec}，这个座位退化成教师")
        return HeuristicAgent(seed)


class Session:
    """一场对局 + 给 UI 用的状态快照。"""

    def __init__(
        self,
        seats: list[str],
        seed: int,
        rounds: int | None = None,
        human_wait: float = 900.0,
        delay: float = 0.0,
    ) -> None:
        self.seats = seats
        self.seed = seed
        self.players = len(seats)
        self.rounds = rounds if rounds is not None else self.players
        self.human_wait = human_wait
        self.delay = delay
        self.rule = S.rule_for(self.players)
        #: 局间是否等玩家确认（有人类座位时默认等，start() 里定）
        self.confirm_between = False
        self.between_timeout = 3600.0
        self.next_event = threading.Event()
        self.awaiting_next = False
        self.last_hand_note = ""
        #: 回放帧（每个动作一帧，含所有手牌）
        self.frames: list[dict] = []
        self.max_frames = 20000
        self.log: list[str] = []
        self.lock = threading.Lock()
        self.game: Game | None = None
        self.finished = False
        self.result: dict | None = None
        self.remote: dict[int, RemoteAgent] = {}
        # 有人类 / 外部接口座位就默认「局间等确认」（这时 remote 还没建，直接看座位名）
        self.confirm_between = any(sp in ("human", "external") for sp in seats)
        self.agents: list[Any] = []
        self.started = time.time()

    # -- 跑对局 -------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        for i, spec in enumerate(self.seats):
            if spec == "human":
                a = RemoteAgent("你", self.human_wait, HeuristicAgent(self.seed * 3 + i))
                self.remote[i] = a
                self.agents.append(a)
            elif spec == "external":
                a = RemoteAgent(
                    "AI 助手（外部接口）", self.human_wait, HeuristicAgent(self.seed * 3 + i)
                )
                self.remote[i] = a
                self.agents.append(a)
            else:
                agent = make_script_agent(spec, self.seed * self.players + i)
                # 模型对战时慢放，每手停一下，方便看牌
                if self.delay > 0:
                    agent = PacedAgent(agent, self.delay)
                self.agents.append(agent)
        try:
            game = Game(
                self.agents, seed=self.seed, rounds=self.rounds,
                rule=self.rule, on_event=self._log, on_between=self._between,
            )
            self.game = game
            r = game.play()
            self.result = {
                "scores": r.scores,
                "ranks": r.ranks,
                "rank_points": [round(x, 1) for x in r.rank_points],
                "deltas": r.deltas,
                "hands": [
                    {
                        "kind": h.kind,
                        "winner": h.winner,
                        "winners": list(h.winners),
                        "loser": h.loser,
                        "text": h.describe(),
                        "deltas": list(h.deltas),
                        "turns": h.turns,
                    }
                    for h in r.hands
                ],
                "stats": [s.summary() for s in r.stats],
            }
            self._log("═══ 对局结束 ═══")
        except Exception as exc:  # pragma: no cover - 出问题也要让 UI 看到
            self._log(f"💥 引擎异常：{exc!r}")
            self.result = {"error": repr(exc)}
        finally:
            self.finished = True

    def _log(self, msg: str) -> None:
        with self.lock:
            self.log.append(msg)
            if len(self.log) > 500:
                del self.log[:100]
            self._capture(msg)

    # -- 回放：每一条动作都记一帧 ------------------------------------------
    def _capture(self, text: str) -> None:
        """记一帧完整局面（**所有手牌都记下来**，回放时要全亮牌）。

        客户端 500ms 轮询根本抓不全每一手（一手可能只要几十毫秒），
        所以必须在引擎这一侧逐动作记录。
        """
        g = self.game
        if g is None:
            return
        players = []
        for i, pl in enumerate(g.players):
            players.append({
                "hand": [tile_id(t) for t in range(34) for _ in range(pl.hand[t])],
                "hand_count": pl.hand_size,
                "pond": [tile_id(t) for t in pl.pond],
                "melds": [
                    {"kind": m.kind, "tile": tile_id(m.tile), "label": str(m)}
                    for m in pl.melds
                ],
                "kita": pl.kita,
                "riichi": pl.riichi,
                "double_riichi": pl.double_riichi,
                "riichi_tile": (
                    tile_id(pl.riichi_tile) if pl.riichi_tile is not None else None
                ),
                "ippatsu": pl.ippatsu,
                "is_dealer": i == g.dealer,
                "seat_wind": tile_cn(g.seat_wind(i)),
            })
        frame = {
            "i": len(self.frames),
            "text": text,
            "hand_no": len(g.hands) + 1,
            "turn": g.turn + 1,
            "wall": g.wall_left(),
            "round_wind": tile_cn(g.round_wind),
            "dealer": g.dealer,
            "dora": [tile_id(g._indicator(k)) for k in range(g.dora_slots)],
            "scores": list(g.scores),
            "sticks": g.sticks,
            "players": players,
            "agari": dict(g.last_agari) if g.last_agari else None,
        }
        if frame["agari"] and frame["agari"].get("tile", -1) >= 0:
            frame["agari"]["tile"] = tile_id(frame["agari"]["tile"])
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            del self.frames[: len(self.frames) - self.max_frames]

    # -- 快照 ---------------------------------------------------------------
    def pending_seat(self) -> int | None:
        for seat, agent in self.remote.items():
            if agent.current() is not None:
                return seat
        return None

    def hand_result_detail(self) -> dict | None:
        """最近一局的**结构化**结果：牌型、役种、番符、点数、谁放铳。

        结算页要用它把「和了什么牌、什么役、几番几符」画出来 ——
        光有一句中文日志是不够的。
        """
        g = self.game
        if g is None or not g.hands:
            return None
        h = g.hands[-1]
        base = {
            "hand_no": len(g.hands),
            "kind": h.kind,
            "turns": h.turns,
            "deltas": list(h.deltas),
            "scores": list(g.scores),
            "sticks": g.sticks,
            "notes": h.note,
            "players": len(g.scores),
            "winner": h.winner,
            "loser": h.loser,
            "winners": list(getattr(h, "winners", ()) or ()),
        }
        if h.kind in ("draw", "kyuushu", "abort"):
            base["tenpai"] = list(h.tenpai) if h.tenpai else []
            base["label"] = {
                "draw": "流局", "kyuushu": "九种九牌", "abort": "途中流局",
            }.get(h.kind, "流局")
            return base

        a = h.agari
        if a is None:
            return base
        pl = g.players[h.winner]
        hand = [tile_id(t) for t in range(34) for _ in range(pl.hand[t])]
        win_tile = None
        if g.last_agari:
            wt = g.last_agari.get("tile", -1)
            win_tile = tile_id(wt) if wt >= 0 else None
        yaku = [
            {"name": S.YAKU_CN.get(k, k), "key": k, "han": v} for k, v in a.yaku
        ]
        for k, v in a.dora_detail.items():
            if v:
                yaku.append({"name": S.YAKU_CN.get(k, k), "key": k, "han": v})
        base.update({
            "label": "自摸" if h.kind == "tsumo" else "荣和",
            "tsumo": h.kind == "tsumo",
            "hand": hand,
            "melds": [
                {"kind": m.kind, "tile": tile_id(m.tile), "label": str(m)}
                for m in pl.melds
            ],
            "win_tile": win_tile,
            "yaku": yaku,
            "han": a.han,
            "dora_han": a.dora_han,
            "total_han": a.total_han,
            "fu": a.fu,
            "yakuman": a.yakuman,
            "points": h.deltas[h.winner] if h.deltas else 0,
            "describe": a.describe(),
        })
        return base

    def _between(self, done: int) -> None:
        """一局打完、下一局开打之前。等玩家点「下一局」才继续。"""
        if not self.confirm_between or self.finished:
            return
        last = self.game.hands[-1] if self.game and self.game.hands else None
        if last is not None:
            who = "流局" if last.kind in ("draw", "kyuushu", "abort") else (
                f"P{last.winner} " + ("自摸" if last.kind == "tsumo" else "荣和")
            )
            self.last_hand_note = f"第 {done} 局结束（{who}）"
        self.awaiting_next = True
        self._log(f"⏸ 第 {done} 局结束 —— 等玩家确认下一局（点「下一局」继续）")
        self.next_event.clear()
        self.next_event.wait(timeout=self.between_timeout)
        self.awaiting_next = False
        self.last_hand_note = ""

    def confirm_next(self) -> None:
        self.next_event.set()

    def spectator_ok(self) -> bool:
        """这桌能不能开「上帝视角」（全亮牌）。

        只有**没有任何真人 / 外部接口座位**时才允许 —— 否则观战的人可以
        把别家手牌念给正在打的人听。
        """
        return not self.remote

    def snapshot(self, seat: int, reveal: bool = False) -> dict:
        g = self.game
        out: dict = {
            # 人数用 player_count；players / players_detail 都是座位明细。
            # players 这个名字历史包袱重（早先是明细数组），留着兼容老客户端，
            # 新代码请用 players_detail。
            "player_count": len(self.seats),
            "delay": self.delay,
            "allow_chi": self.rule.players == 4,
            "allow_kita": self.rule.kita_dora,
            "seats": [
                {"index": i, "type": s, "name": self._seat_name(i)}
                for i, s in enumerate(self.seats)
            ],
            "seed": self.seed,
            "finished": self.finished,
            "result": self.result,
            "you": seat,
            "log": self.log[-120:],
            "wall_left": 0,
            "turn": 0,
            "dealer": 0,
            "dora": [],
            "round_wind": "东",
            "hand_no": len(g.hands) + 1 if g else 1,
            "players": [],
            "players_detail": [],
            "pending": None,
            "hint": None,
            "awaiting_next": self.awaiting_next,
            "hand_result": self.hand_result_detail() if (
                self.awaiting_next or self.finished
            ) else None,
            "spectator_ok": self.spectator_ok(),
            "last_hand_note": self.last_hand_note,
            "confirm_between": self.confirm_between,
        }
        if g is None:
            return out
        with self.lock:
            out["wall_left"] = g.wall_left()
            out["turn"] = g.turn + 1
            out["dealer"] = g.dealer
            out["dora"] = [tile_id(g._indicator(i)) for i in range(g.dora_slots)]
            out["round_wind"] = tile_cn(g.round_wind)
            for i, pl in enumerate(g.players):
                is_you = i == seat or (reveal and self.spectator_ok())
                entry = {
                    "score": g.scores[i],
                    "pond": [tile_id(t) for t in pl.pond],
                    "melds": [
                        {"kind": m.kind, "tile": tile_id(m.tile),
                         "label": str(m)}
                        for m in pl.melds
                    ],
                    "kita": pl.kita,
                    "riichi": pl.riichi,
                    "double_riichi": pl.double_riichi,
                    # 立直宣言牌：前端横放标「立直」用。**必须后端给** ——
                    # 让前端靠轮询「看到立直那一刻」去猜，500ms 一刷新就会漏。
                    "riichi_tile": (
                        tile_id(pl.riichi_tile) if pl.riichi_tile is not None else None
                    ),
                    "ippatsu": pl.ippatsu,
                    "is_dealer": i == g.dealer,
                    "seat_wind": tile_cn(g.seat_wind(i)),
                    "hand_count": pl.hand_size,
                    "hand": (
                        [tile_id(t) for t in range(34) for _ in range(pl.hand[t])]
                        if is_you else None
                    ),
                    "red": (
                        [tile_id(t) for t in range(34) for _ in range(pl.red[t])]
                        if is_you else None
                    ),
                }
                out["players_detail"].append(entry)

        # 兼容别名：老客户端读 players（数组），新代码读 players_detail
        out["players"] = out["players_detail"]

        seat_pending = self.pending_seat()
        if seat_pending is not None:
            agent = self.remote[seat_pending]
            cur = agent.current()
            view = agent.pending["view"] if agent.pending else None
            entry = {
                "seat": seat_pending,
                "kind": cur["kind"],
                "kind_cn": KIND_CN.get(cur["kind"], cur["kind"]),
                "waited": round(cur["waited"], 1),
                "options": [],
                "yours": seat_pending == seat,
            }
            # 被鸣 / 被荣的是哪张（UI 上要标出来）
            tile = (cur.get("tile") if isinstance(cur, dict) else None)
            if tile is None and agent.pending is not None:
                dec = agent.pending.get("decision")
                tile = getattr(dec, "tile", None)
            if isinstance(tile, int):
                entry["tile"] = tile_id(tile)
                entry["tile_cn"] = tile_cn(tile)
            if cur["kind"] == "discard":
                for i, t in enumerate(cur["options"]):
                    entry["options"].append(
                        {"index": i, "label": tile_id(t), "cn": tile_cn(t), "tile": tile_id(t)}
                    )
            else:
                for i, o in enumerate(cur["options"]):
                    entry["options"].append(
                        {"index": i, "label": str(o), "cn": OPTION_CN.get(str(o), str(o))}
                    )
            # 只有本座位的决策才带上自己的手牌信息
            if seat_pending == seat and view is not None:
                entry["your_hand"] = [
                    tile_id(t) for t in range(34) for _ in range(view.hand[t])
                ]
                entry["your_melds"] = [str(m) for m in view.melds]
                entry["drawn"] = tile_id(view.drawn) if view.drawn is not None else None
            out["pending"] = entry
        elif not self.finished and self.game is not None:
            out["waiting_for"] = "脚本 AI / 其它座位"

        # 牌效提示（只给自己看）
        if g is not None and not self.finished and seat < len(self.agents):
            pl = g.players[seat]
            n_melds = len(pl.melds)
            vis = visible_counts(g.view_for(seat))
            # 摸完牌是 (4-副露)*3+2 张，没摸牌是 +1 张
            total = (4 - n_melds) * 3 + (2 if pl.drawn is not None else 1)
            if pl.hand_size == total:
                s = shanten(pl.hand, n_melds)
                out["hint"] = {
                    "shanten": s,
                    "ukeire": ukeire_count(pl.hand, vis, n_melds),
                    "menzen": pl.menzen,
                }
        return out

    def _seat_name(self, i: int) -> str:
        spec = self.seats[i]
        if spec == "human":
            return f"你（P{i}）"
        if spec == "external":
            return f"AI 助手（P{i}）"
        return f"脚本·{spec}（P{i}）"


class Handler(BaseHTTPRequestHandler):
    session: Session  # 由 serve() 注入
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # 别把访问日志刷到终端
        pass

    # -- 工具 ---------------------------------------------------------------
    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str, cache: str = "max-age=3600") -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    # -- 路由 ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            # 页面本身不缓存，改一行刷新就能看到；牌图缓存久一点无所谓
            self._file(STATIC / "index.html", "text/html; charset=utf-8", "no-store")
            return
        if u.path == "/api/state":
            seat = int(q.get("seat", ["0"])[0])
            reveal = q.get("reveal", ["0"])[0] not in ("0", "false", "")
            self._json(self.session.snapshot(seat, reveal=reveal))
            return
        if u.path == "/api/replay":
            # 元信息：有多少帧、哪些座位
            if "i" not in q and "from" not in q:
                self._json({
                    "frames": len(self.session.frames),
                    "players": self.session.players,
                    "spectator_ok": self.session.spectator_ok(),
                    "seats": [self.session._seat_name(i) for i in range(self.session.players)],
                    "finished": self.session.finished,
                    "hands": len(self.session.frames),
                })
                return
            frames = self.session.frames
            if "i" in q:
                idx = int(q["i"][0])
                if not -len(frames) <= idx < len(frames):
                    self._json({"ok": False, "error": f"帧号越界：{idx}"}, 400)
                    return
                self._json(frames[idx])
                return
            a = max(0, int(q.get("from", ["0"])[0]))
            b = min(len(frames) - 1, int(q.get("to", [str(a + 99)])[0]))
            if b < a:
                self._json({"ok": False, "error": "区间反了"}, 400)
                return
            self._json({"from": a, "to": b, "frames": frames[a : b + 1]})
            return
        if u.path == "/api/models":
            from .. import variants

            models = available_models()
            info: dict[str, dict] = {}
            for name in models:
                try:
                    spec = variants.get(name)
                    info[name] = {
                        "cn": spec.cn,              # 例如「三麻·速攻」
                        "style": variants.STYLE_CN.get(spec.style, spec.style),
                        "players": spec.players,
                        "reward": spec.reward,
                    }
                except KeyError:
                    info[name] = {"cn": "", "style": "", "players": 0, "reward": ""}
            self._json(
                {
                    "models": models,
                    "builtin": ["teacher", "random"],
                    "seats": ["human", "external"],
                    "info": info,
                    # 每种人数推荐用哪几个（按人数过滤后风格各来一个）
                    "suggest": {
                        "3": [n for n in models if info[n]["players"] == 3][:3],
                        "4": [n for n in models if info[n]["players"] == 4][:4],
                    },
                }
            )
            return
        if u.path == "/api/config":
            s = self.session
            self._json(
                {
                    "players": s.players,
                    "seats": list(s.seats),
                    "delay": s.delay,
                    "seed": s.seed,
                    "human_wait": s.human_wait,
                    "models": available_models(),
                }
            )
            return
        if u.path.startswith("/tiles/"):
            name = Path(u.path).name
            if not name.endswith(".png") or "/" in name:
                self.send_error(400)
                return
            self._file(TEMPLATES / name, "image/png")
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json({"ok": False, "error": "bad json"}, 400)
            return

        if u.path == "/api/next":
            self.session.confirm_next()
            self._json({"ok": True, "awaiting_next": self.session.awaiting_next})
            return

        if u.path == "/api/config":
            seats = [str(x) for x in (payload.get("seats") or [])]
            players = int(payload.get("players") or len(seats) or 3)
            if players not in (3, 4):
                self._json({"ok": False, "error": "人数只能是 3 或 4"}, 400)
                return
            seats = (seats + ["teacher"] * players)[:players]
            old = self.session
            new = Session(
                seats,
                int(payload.get("seed", old.seed + 1)),
                players,
                float(payload.get("human_wait", old.human_wait)),
                float(payload.get("delay", old.delay)),
            )
            if "confirm_between" in payload:
                new.confirm_between = bool(payload["confirm_between"])
            Handler.session = new
            new.start()
            self._json({
                "ok": True, "players": players, "seats": seats,
                "delay": new.delay, "seed": new.seed, "human_wait": new.human_wait,
            })
            return

        if u.path == "/api/new":
            old = self.session
            new = Session(
                old.seats, old.seed + 1, old.rounds, old.human_wait, old.delay
            )
            Handler.session = new
            new.start()
            self._json({"ok": True, "seed": new.seed})
            return

        if u.path == "/api/act":
            seat = int(payload.get("seat", -1))
            agent = self.session.remote.get(seat)
            if agent is None:
                self._json({"ok": False, "error": "该座位不是人 / 外部接口"}, 400)
                return
            index = payload.get("index")
            if index is None and payload.get("label"):
                index = self._label_to_index(agent, str(payload["label"]))
            if index is None:
                self._json({"ok": False, "error": "缺少 index 或 label"}, 400)
                return
            cur = agent.current()
            if cur is None:
                self._json({"ok": False, "error": "这个座位现在没有待决策"}, 409)
                return
            idx = int(index)
            if not 0 <= idx < len(cur["options"]):
                self._json(
                    {"ok": False,
                     "error": f"下标越界：{idx}（可选 0~{len(cur['options']) - 1}）"},
                    400,
                )
                return
            ok = agent.submit(idx)
            self._json({"ok": ok, "seat": seat, "index": idx})
            return
        self.send_error(404)

    def _label_to_index(self, agent: RemoteAgent, label: str) -> int | None:
        cur = agent.current()
        if cur is None:
            return None
        opts = cur["options"]
        if cur["kind"] == "discard":
            for i, t in enumerate(opts):
                if tile_id(t) == label or tile_cn(t) == label:
                    return i
            return None
        low = label.lower()
        for i, o in enumerate(opts):
            if str(o).lower() == low or OPTION_CN.get(str(o)) == label:
                return i
        return None


def serve(
    seats: list[str],
    port: int = 8765,
    seed: int = 0,
    rounds: int | None = None,
    host: str = "127.0.0.1",
    human_wait: float = 900.0,
    delay: float = 0.0,
) -> None:
    session = Session(seats, seed, rounds, human_wait, delay)
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    session.start()
    url = f"http://{host}:{port}/?seat={seats.index('human') if 'human' in seats else 0}"
    print("🀄 三麻 UI 已启动")
    print(f"   座位: " + "  ".join(f"P{i}={s}" for i, s in enumerate(seats)))
    print(f"   人数: {len(seats)}（{'四麻' if len(seats) == 4 else '三麻'}）"
          f"   AI 停顿 {delay:.1f}s")
    print(f"   浏览器打开: {url}")
    print("   外部座位（我）用: .venv/bin/python -m aimodel.ui.assistant")
    print(f"   等待超时 {human_wait:.0f}s（超时自动由教师托管，不会卡死）")
    print("   Ctrl-C 结束")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n再见")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    try:
        from ..variants import audit_weights
        audit_weights(verbose=True)
    except Exception as e:                          # noqa: BLE001
        print(f"（权重自检跳过：{type(e).__name__}: {e}）")
    ap = argparse.ArgumentParser(description="三麻简易 UI")
    ap.add_argument(
        "--seats", type=str, default="",
        help="座位，逗号分隔：human / external / teacher / random / 模型名"
             "（不给就自动用训练好的模型，三麻 3 个、四麻 4 个）",
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=None, help="默认按人数（三麻 3 局 / 四麻 4 局）")
    ap.add_argument("--delay", type=float, default=0.0, help="AI 每手停顿秒数（看牌用）")
    ap.add_argument(
        "--wait", type=float, default=900.0,
        help="人 / 外部座位等多久算超时（超时由教师托管），秒",
    )
    args = ap.parse_args(argv)
    seats = [s.strip().lower() for s in args.seats.split(",") if s.strip()]
    if not seats:
        seats = default_seats(3)
        print(f"（没指定座位，自动用训练好的模型：{', '.join(seats)}）")
    players = 4 if len(seats) >= 4 else 3
    while len(seats) < players:
        seats.append("teacher")
    serve(seats[:players], args.port, args.seed, args.rounds, args.host, args.wait,
          args.delay)


if __name__ == "__main__":
    main()
