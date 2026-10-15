#!/usr/bin/env python
"""验收用测试客户端：能自己打牌的「机器人真人」。

这是所有验收脚本（冒烟 / 隔离 / 防作弊 / 压测 / 重连 / 截图）的公共工具：

* 走**和浏览器完全一样**的身份流程（``/api/name`` 拿签名 cookie），
  所以「同一个人跨局跨连接」是同一个匿名 ID —— 画像验收要靠这个；
* 会自己合法地出牌（策略可选），也能被要求「只打某一手」；
* ``play_until`` 里带**卡死检测**，卡住会明确报错而不是干等。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import urllib.request  # noqa: E402

import websockets  # noqa: E402


class Bot:
    """一条 WebSocket 连接 + 一个很笨但合法的出牌策略。"""

    def __init__(
        self,
        url: str,
        name: str,
        *,
        uid: str | None = None,
        strategy: str = "first",
        token: str = "",
        password: str = "",
    ) -> None:
        self.url = url
        self.name = name
        self.uid = uid
        self.strategy = strategy
        #: 身份 cookie（``maj_id``）—— 和浏览器一样带着它，同一个 Bot
        #: 换了连接也还是同一个人（画像验收要用）
        self.auth = token
        self.password = password
        self.ws: Any = None
        self.seat = -1
        self.room = ""
        self.token = ""
        self.state: dict | None = None
        self.states: list[dict] = []
        self.errors: list[str] = []
        self.rejects: list[str] = []
        self.acts = 0
        self.actions: list[tuple[str, int | str]] = []
        self.last_raw = ""
        self.rng = __import__("random").Random(hash(name) & 0xFFFF)
        #: state.seq -> 已经在这个决策点提交过的动作种类（防重复提交）
        self._decided: dict[int, set[str]] = {}
        #: **后台读协程**把消息收进这个队列
        self._inbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._dropped = 0

    # -- 身份 ---------------------------------------------------------------
    def _http_url(self) -> str:
        u = self.url
        if u.startswith("ws://"):
            u = "http://" + u[5:]
        elif u.startswith("wss://"):
            u = "https://" + u[6:]
        return u.split("/ws")[0]

    async def login(self, name: str | None = None) -> str:
        """走一次 ``/api/name`` 拿签名 cookie —— 和浏览器完全一样的身份流程。

        传了 ``self.uid`` 就带上，服务端会沿用这个匿名 ID。**这是「同一个人
        跨连接还是同一个人」的关键**：验收 8（同一玩家打 10 局、记录按 ID
        串起来）靠的就是它。
        """
        body = json.dumps({"name": name or self.name, "uid": self.uid or ""}).encode()

        def _do() -> tuple[dict, str]:
            req = urllib.request.Request(
                self._http_url() + "/api/name", data=body,
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read()), (r.headers.get("Set-Cookie") or "")

        data, cookie = await asyncio.get_running_loop().run_in_executor(None, _do)
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("maj_id="):
                self.auth = part[len("maj_id="):]
        self.name = data.get("name") or self.name
        if data.get("uid"):
            self.uid = data["uid"]
        return self.name

    async def reconnect_as(self, uid: str, name: str | None = None) -> None:
        """换一条连接，但**还是同一个人**（沿用 uid / cookie）。"""
        self.uid = uid
        self.name = name or self.name
        if not self.auth:
            await self.login()
        await self.open()

    # -- 连接 ---------------------------------------------------------------
    async def open(self, **kw) -> None:
        if not self.auth:
            try:
                await self.login()
            except Exception:  # noqa: BLE001 - 拿不到 cookie 也能匿名连
                pass
        if self.auth:
            kw.setdefault("additional_headers", {"Cookie": f"maj_id={self.auth}"})
        self.ws = await websockets.connect(self.url, max_size=16 << 20, **kw)
        # **必须先起后台读协程**：牌桌是服务端主动推的，客户端不能等到
        # 「自己想读了」才去 recv。TCP 窗口读满之后 aiohttp 的 send_str
        # 会被流控挡住，房间的写协程就卡在 await 上 —— 症状是
        # 「服务端 seq 一直在涨、客户端却一帧都收不到」，很容易误判成
        # 服务端的广播坏了。（这个坑花了一轮才定位到。）
        self._reader = asyncio.create_task(self._read_loop(), name=f"bot-{self.name}")
        # 连上就先收一条 hello（服务端会主动推），顺便把 lobby 也吃掉。
        # 顺序很重要：**先等 hello，再发 hello**，不然自己发的那条回执
        # 会被下一句 ``wait(hello)`` 吃掉、接着等一条永远不来的消息。
        await self.wait(lambda m: m.get("t") == "hello", timeout=30)
        await self.send({"t": "hello", "name": self.name})

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self.ws.recv()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    if "t" in msg:
                        self.absorb(msg)
                    self.last_raw = raw
                    self._inbox.put_nowait(msg)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - 连接断了就结束
            return
        finally:
            try:
                self._inbox.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None

    async def send(self, obj: dict) -> None:
        assert self.ws is not None, "没连接"
        await self.ws.send(json.dumps(obj))

    async def recv(self, timeout: float = 10.0) -> dict:
        """从后台读协程的队列里取一条。

        队列满了就丢**最旧的** —— 状态快照本来就是「后一帧覆盖前一帧」，
        测试脚本永远只关心最新的那一帧；不丢的话队列会无界增长。
        """
        if self._inbox.qsize() > 4000:
            for _ in range(1000):
                try:
                    self._inbox.get_nowait()
                    self._dropped += 1
                except asyncio.QueueEmpty:
                    break
        try:
            msg = await asyncio.wait_for(self._inbox.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            raise TimeoutError(f"{self.name} 等消息超时") from None
        if msg is None:
            raise ConnectionError(f"{self.name} 连接已断开")
        return msg

    def absorb(self, msg: dict) -> None:
        t = msg.get("t")
        if t == "state":
            self.state = msg
            self.states.append(msg)
            if len(self.states) > 6000:
                del self.states[:3000]
        elif t == "joined":
            self.seat = msg["seat"]
            self.room = msg["room"]
            self.token = msg.get("token", "")
        elif t == "error":
            self.errors.append(msg.get("message", ""))
        elif t == "reject":
            self.rejects.append(msg.get("message", ""))

    async def wait(self, pred: Callable[[dict], bool], timeout: float = 20.0) -> dict:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while True:
            left = end - loop.time()
            if left <= 0:
                raise TimeoutError(
                    f"{self.name} 等不到想要的消息（最后一条："
                    f"{str(getattr(self, 'last_raw', ''))[:160]}）"
                )
            msg = await self.recv(left)
            if not isinstance(msg, dict) or "t" not in msg:
                continue        # 没有类型的消息（比如裸事件）直接跳过
            if pred(msg):
                return msg

    async def wait_reject(self, timeout: float = 10.0) -> dict:
        """等一条**新的** reject（跳过之前攒下的）。"""
        n = len(self.rejects)
        return await self.wait(
            lambda m: m.get("t") == "reject" and len(self.rejects) > n, timeout
        )

    async def pump(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while loop.time() < end:
            try:
                await self.recv(0.25)
            except (asyncio.TimeoutError, TimeoutError):
                return

    # -- 房间 ---------------------------------------------------------------
    async def create_room(
        self,
        *,
        players: int = 4,
        rounds: int = 1,
        seats: list[dict] | None = None,
        ai_delay: float = 0.0,
        name: str = "",
        autostart: bool = False,
    ) -> str:
        """建房。``seats`` 是**完整的一张桌子**（第 0 格 = 房主自己）。

        不传就等于「我坐 0 号位，其余空着」。
        """
        full = list(seats or [])
        if not full:
            full = [{"kind": "human"}] * players
        await self.send({
            "t": "room.create",
            "name": name or f"{self.name} 的牌桌",
            "players": players,
            "rounds": rounds,
            "ai_delay": ai_delay,
            "seats": full[:players],
            "autostart": autostart,
            "password": self.password,
        })
        m = await self.wait(lambda x: x["t"] == "joined")
        return m["room"]

    async def join(self, room: str, *, spectator: bool = False,
                   autostart: bool = True) -> dict:
        await self.send({
            "t": "room.join", "room": room, "seat": None, "token": "",
            "spectator": spectator, "reveal": True, "autostart": autostart,
            "password": self.password,
        })
        return await self.wait(lambda x: x["t"] == "joined")

    # -- 自己出牌 -----------------------------------------------------------
    def my_action(self) -> dict | None:
        st = self.state
        if not st or st["phase"] != "playing":
            return None
        a = st.get("actions")
        if not a or a.get("waiting") or a.get("seat") != st["viewer"]["seat"]:
            return None
        return a

    async def maybe_act(self) -> bool:
        """该我动就动一手。返回是否动了。

        **同一个决策只提交一次**：``state`` 是异步推过来的，不管 seq 就
        重复提交，服务端会正确地回一堆「现在没轮到你」—— 那是我这边刷太快。
        """
        a = self.my_action()
        if a is None:
            return False
        st = self.state
        seq = st.get("seq") if st else None
        ledger = self._decided.setdefault(seq, set())
        kind = a["kind"]
        if kind in ledger:
            return False
        opts = a.get("options") or []
        if not opts:
            return False
        ledger.add(kind)
        if len(self._decided) > 64:      # 只留最近 64 个 seq
            for old in sorted(self._decided)[:-32]:
                self._decided.pop(old, None)

        if kind == "discard":
            tiles = [o for o in opts if o.get("tile")]
            if not tiles:
                return False
            if self.strategy == "riichi":
                pick = next((o for o in tiles if o.get("riichi_ok")), None) or tiles[0]
            elif self.strategy == "random":
                pick = self.rng.choice(tiles)
            else:
                pick = tiles[0]
            await self.send({"t": "action", "kind": "discard", "tile": pick["tile"]})
            self.acts += 1
            self.actions.append(("discard", pick["tile"]))
            return True

        # 非出牌：能和就和 / 能立就立 / 能拔就拔，鸣牌看策略，其余过
        want = {"tsumo": "yes", "ron": "yes", "riichi": "yes", "kita": "yes"}
        chosen = None
        target = want.get(kind)
        if target:
            chosen = next((o for o in opts if o.get("label") == target), None)
        if chosen is None and kind in ("call", "chi", "kan"):
            if self.strategy != "pass":
                chosen = next((o for o in opts
                               if o.get("label") in ("pon", "kan", "chi")), None)
        if chosen is None:
            chosen = next((o for o in opts if o.get("label") in ("pass", "no")), None)
        if chosen is None:
            chosen = opts[-1]
        await self.send({"t": "action", "kind": kind, "index": chosen["index"]})
        self.acts += 1
        self.actions.append((kind, chosen.get("label") or chosen.get("tile") or ""))
        return True

    async def play_until(
        self,
        pred: Callable[[dict], bool],
        *,
        timeout: float = 300.0,
        ack_between: bool = True,
        stall: float = 100.0,
    ) -> bool:
        """一直打到 ``pred(state)`` 成立。超时抛 TimeoutError。

        ``stall`` 是「多久没有任何自己出的牌就算卡住」。**要留得比服务端的
        局间等待（默认 60s）长** —— 局间两个真人都在等对面点「下一局」，
        这期间本来就没有动作。
        """
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        last_progress = loop.time()
        acts_before = self.acts
        while loop.time() < end:
            st = self.state
            if st is not None and pred(st):
                return True
            if st is not None and st["phase"] == "hand_end" and ack_between:
                await self.send({"t": "next"})
            moved = await self.maybe_act()
            if moved and self.acts != acts_before:
                last_progress = loop.time()
                acts_before = self.acts
            try:
                await self.recv(0.05 if moved else 0.25)
            except (asyncio.TimeoutError, TimeoutError):
                pass
            if loop.time() - last_progress > stall:
                raise TimeoutError(
                    f"{self.name} 卡住了（{self.acts} 手没有进展，"
                    f"phase={self.state and self.state['phase']}）"
                )
        raise TimeoutError(f"{self.name} 超时（{self.acts} 手）")


async def gather_play(bots: list[Bot], pred: Callable[[dict], bool], **kw) -> list[bool]:
    """多个 bot 一起打（各自跑自己的循环）。"""
    return list(await asyncio.gather(*[b.play_until(pred, **kw) for b in bots]))
