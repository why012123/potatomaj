"""aiohttp 应用：静态资源 + 登录/列表（HTTP） + 实时推送（WebSocket）。

为什么要 aiohttp 而不是自己写 ``http.server``：牌桌要**服务端主动推**。
WebSocket 是主线，HTTP 只干三件事 —— 发静态文件、发牌面图、房间列表/身份
这类一次性查询。轮询在这里是被明确禁止的。

一条 WebSocket 连接的生命周期::

    匿名连接 ──hello──▶ 绑定 uid（签名 cookie 里的匿名 ID）
      │                     │
      │◀── lobby ──────────┘         （房间列表 / 自己所在的房间）
      │
      ├─room.join─▶ 绑定房间 R ──▶ 只推 R 的状态（**别的房间一个字都不推**）
      │                │
      │                ├─action────▶ Room.submit_action（服务端校验）
      │                ├─chat──────▶ Room.chat_msg
      │                ├─privacy───▶ Room.set_no_record（本局不记录）
      │                └─room.leave▶ 解绑，回到 lobby
      │
      └─ 断开 ▶ 位置保留 seat_grace 秒，重连凭证对得上就回原座位

隔离保证：连接上的 ``rid`` 变了就**先清空** ``last_room_payload``；
每条连接有独立的**有界发送队列 + 写协程**，一个慢客户端拖不住别人。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from aimodel import __version__ as AIMODEL_VERSION

from . import __version__ as VERSION
from . import agents as AG
from . import protocol as P
from .config import ROOT, STATIC, Config
from .recording import AsyncWriter, ProfileStore, anon_id
from .rooms import RoomManager
from .table import PHASE_FINISHED, PHASE_PLAYING, Conn, Room, RoomError

log = logging.getLogger("maj.web")

MAX_WS_MSG = 64 * 1024

#: 匿名玩家 ID 前缀。真正的 ID 就是一串随机字符 —— **绝不掺昵称 / 邮箱 /
#: 手机号**，这样它能安全地进训练数据。
ANON_PREFIX = "p_"

_TEXT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


# ---------------------------------------------------------------------------
# 身份：一个签名 cookie，够用就行
# ---------------------------------------------------------------------------

_MEM_SECRET: bytes | None = None


def _secret() -> bytes:
    """进程级密钥。持久化到 data/.secret，重启后 cookie 还能认。"""
    global _MEM_SECRET
    p = Path(os.environ.get("MAJ_SECRET_FILE", str(ROOT / "data" / ".secret")))
    try:
        if p.exists():
            return p.read_bytes()
        p.parent.mkdir(parents=True, exist_ok=True)
        s = secrets.token_bytes(32)
        p.write_bytes(s)
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
        return s
    except OSError:
        if _MEM_SECRET is None:
            _MEM_SECRET = secrets.token_bytes(32)
        return _MEM_SECRET


def sign_uid(uid: str, name: str) -> str:
    body = json.dumps({"u": uid, "n": name, "t": int(time.time())},
                      separators=(",", ":"))
    raw = base64.urlsafe_b64encode(body.encode()).rstrip(b"=")
    mac = hmac.new(_secret(), raw, hashlib.sha256).digest()[:16]
    return raw.decode() + "." + base64.urlsafe_b64encode(mac).rstrip(b"=").decode()


def verify_uid(token: str) -> dict | None:
    try:
        raw_b64, mac_b64 = token.split(".", 1)
        raw = raw_b64.encode()
        pad = "=" * (-len(mac_b64) % 4)
        want = hmac.new(_secret(), raw, hashlib.sha256).digest()[:16]
        got = base64.urlsafe_b64decode(mac_b64 + pad)
        if not hmac.compare_digest(want, got):
            return None
        body = json.loads(base64.urlsafe_b64decode(raw_b64 + "=" * (-len(raw_b64) % 4)))
        name = str(body.get("n") or "").strip()[:16]
        uid = str(body.get("u") or "").strip()
        if not uid:
            return None
        # 对外的玩家 ID 一律是匿名串
        if not uid.startswith(ANON_PREFIX):
            uid = anon_id()
        return {"uid": uid, "name": name or "玩家"}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------


class WSClient:
    """一条 WebSocket 连接。

    每条连接自带一个**有界发送队列 + 写协程**：房间广播只往队列里塞，
    真正的 ``await send_str`` 由写协程做。这样一个卡住的客户端只会把
    自己的队列塞满（然后丢最旧的），不会拖住整个房间或事件循环。
    """

    __slots__ = ("id", "ws", "uid", "name", "rid", "conn", "hello",
                 "connected_at", "addr", "closed", "outbox", "writer_task",
                 "dropped", "seen")

    OUTBOX_MAX = 600

    def __init__(self, ws: web.WebSocketResponse, uid: str, name: str, addr: str) -> None:
        self.id = secrets.token_urlsafe(8)
        self.ws = ws
        self.uid = uid
        self.name = name
        self.rid: str | None = None
        self.conn: Conn | None = None
        self.hello = False
        self.connected_at = time.time()
        self.addr = addr
        self.closed = False
        self.outbox: asyncio.Queue[str] = asyncio.Queue(maxsize=self.OUTBOX_MAX)
        self.writer_task: asyncio.Task | None = None
        self.dropped = 0
        self.seen: dict[str, Any] = {}
        setattr(ws, "_maj_outbox", self.outbox)   # 房间层靠这个拿队列

    async def writer(self) -> None:
        try:
            while True:
                text = await self.outbox.get()
                if self.closed:
                    return
                try:
                    await self.ws.send_str(text)
                except (ConnectionResetError, RuntimeError):
                    self.closed = True
                    return
        except asyncio.CancelledError:
            return

    async def push(self, payload: dict, *, quiet: bool = False) -> None:
        """入队（不等待真正发出）。"""
        if self.closed:
            return
        try:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        if self.outbox.full():
            try:
                self.outbox.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.outbox.put_nowait(text)
        except asyncio.QueueFull:
            pass
        del quiet

    def set_room(self, rid: str | None, conn: Conn | None) -> None:
        """切换 / 解绑房间。

        **这里就是隔离的实现**：一旦房间变了，先把上一条房间快照丢掉，
        绝不可能出现「A 房的数据还挂在连接上，然后被当成 B 房的推出去」。
        """
        if rid != self.rid:
            self.seen.clear()
        self.rid = rid
        self.conn = conn


# ---------------------------------------------------------------------------
# 官方 ELO 反代（公网隧道只映射 8790；8964 的页面/接口不暴露）
# ---------------------------------------------------------------------------

#: 官方 ELO / 回放服务（``aimodel.ui.arena_server``）的本机地址。
#: 环境变量 ``MAJ_ELO_UPSTREAM`` 只用于测试时指到假上游；线上不设。
ELO_UPSTREAM = os.environ.get("MAJ_ELO_UPSTREAM", "http://127.0.0.1:8964")

#: 原样转发的请求头（Host / Cookie 不转：上游是本机服务，不需要）。
_FORWARD_HEADERS = ("Content-Type", "Accept", "User-Agent")


async def _proxy_elo(request: web.Request) -> web.Response:
    """把官方 ELO 的页面 / API 请求**原样**反代到 8964。

    只加路由，不改已有的 7 个 webapp 接口 ✓
    · ``path_qs`` 含 query string -> ``?season=`` / ``?days=all`` 都能过 ✓
    · ``request.read()`` 后按原方法发出 -> POST（start/pause/favorite…）也能过 ✓
    · 不引新依赖：aiohttp 客户端本来就是 webapp 的依赖 ✓
    """
    upstream = os.environ.get("MAJ_ELO_UPSTREAM", ELO_UPSTREAM).rstrip("/")
    url = upstream + request.path_qs
    body = await request.read()
    headers = {name: request.headers[name]
               for name in _FORWARD_HEADERS if request.headers.get(name)}
    try:
        timeout = ClientTimeout(total=60.0, sock_connect=10.0)
        async with ClientSession(timeout=timeout) as session:
            async with session.request(request.method, url, data=body or None,
                                       headers=headers, allow_redirects=False) as r:
                data = await r.read()
                resp = web.Response(body=data, status=r.status)
                for name in ("Content-Type", "Cache-Control"):
                    if r.headers.get(name):
                        resp.headers[name] = r.headers[name]
                return resp
    except asyncio.TimeoutError:
        err = "上游 8964 超时"
    except Exception as exc:                             # noqa: BLE001
        err = f"上游 8964 不可达：{type(exc).__name__}: {exc}"
    if request.path.startswith("/api/"):
        return web.json_response({"ok": False, "error": err}, status=502)
    return web.Response(text=f"官方 ELO 服务暂时不可用：{err}",
                        status=502, content_type="text/plain; charset=utf-8")


def register_elo_proxy(app: web.Application) -> None:
    """注册官方 ELO 页面 / API 的反代路由（单独成函数，方便测试）。"""
    # 页面：官方 / 手动池 / 回放 —— 都是 8964 的静态页
    for path in ("/official", "/official.html", "/arena", "/arena.html",
                 "/replay", "/replay.html"):
        app.router.add_get(path, _proxy_elo)
    # API：/api/arena、/api/arena/*、/api/elo/*；GET 和 POST 都转
    app.router.add_route("*", "/api/arena", _proxy_elo)
    app.router.add_route("*", "/api/arena/", _proxy_elo)
    app.router.add_route("*", "/api/arena/{tail:.*}", _proxy_elo)
    app.router.add_route("*", "/api/elo/{tail:.*}", _proxy_elo)


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


def create_app(cfg: Config) -> web.Application:
    app = web.Application(client_max_size=256 * 1024)
    app["cfg"] = cfg
    app["clients"] = set()
    app["started_at"] = time.time()

    async def on_startup(app: web.Application) -> None:
        AG.audit_at_startup()      # 权重自检：缺头 / 载不进来的都写进日志
        writer = profiles = None
        if cfg.record:
            profiles = ProfileStore(cfg.profiles_path)
            profiles.load()
            writer = AsyncWriter(
                cfg.data_dir,
                max_bytes=cfg.record_part_bytes,
                max_total_bytes=cfg.record_total_bytes,
            )
            writer.start()
        app["writer"] = writer
        app["profiles"] = profiles
        app["rooms"] = RoomManager(cfg, asyncio.get_running_loop(),
                                   writer=writer, profiles=profiles)
        app["rooms"].start_sweeper()
        log.info("服务就绪 http://%s:%d  （AI workers=%d, 节拍=%.1fs, 超时=%.0fs, 记录=%s）",
                 cfg.host, cfg.port, cfg.ai_workers, cfg.ai_delay, cfg.human_timeout,
                 ("开 -> " + str(cfg.data_dir)) if cfg.record else "关")

    async def on_cleanup(app: web.Application) -> None:
        app["rooms"].shutdown()
        writer = app.get("writer")
        if writer is not None:
            log.info("记录统计：%s", writer.stats())
            writer.stop(timeout=8.0)
        profiles = app.get("profiles")
        if profiles is not None:
            profiles.save()
        AG.shutdown_pool()
        log.info("服务已停止")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/hello", api_hello)
    app.router.add_post("/api/name", api_name)
    app.router.add_get("/api/rooms", api_rooms)
    app.router.add_get("/api/models", api_models)
    app.router.add_get("/api/health", api_health)
    app.router.add_get("/api/profiles", api_profiles)
    app.router.add_get("/api/record", api_record)
    # 官方 ELO 页 / API：公网只映射 8790，所以在这里反代到 8964。
    # **只加路由**：上面 7 个既有接口一个都没动 ✓
    register_elo_proxy(app)
    app.router.add_get("/tiles/{name}", tile_handler)
    app.router.add_get("/", index_handler)
    app.router.add_get("/room/{rid}", index_handler)
    app.router.add_static("/app/", path=str(STATIC / "app"), show_index=False,
                          follow_symlinks=False)
    app.router.add_static("/assets/", path=str(STATIC), show_index=False,
                          follow_symlinks=False)
    app.router.add_get("/favicon.ico", favicon_handler)
    return app


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _client_uid(request: web.Request) -> dict:
    tok = request.cookies.get("maj_id", "")
    who = verify_uid(tok) if tok else None
    if who is None:
        who = {"uid": anon_id(), "name": ""}
    return who


async def index_handler(request: web.Request) -> web.StreamResponse:
    who = _client_uid(request)
    html = STATIC / "app" / "index.html"
    if not html.exists():
        return web.Response(
            status=503,
            text=("前端还没构建。\n在 webapp/client 下跑： npm install && npm run build\n"),
            content_type="text/plain",
        )
    resp = web.FileResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    resp.set_cookie("maj_id", sign_uid(who["uid"], who["name"]),
                    max_age=90 * 24 * 3600, httponly=True, samesite="Lax")
    return resp


async def favicon_handler(request: web.Request) -> web.StreamResponse:
    p = STATIC / "favicon.svg"
    if p.exists():
        return web.FileResponse(p)
    return web.Response(status=404)


async def tile_handler(request: web.Request) -> web.StreamResponse:
    """牌面图。优先 png，退回 svg（仓库里存的是 svg，省空间又清晰）。"""
    name = request.match_info["name"]
    if "/" in name or "\\" in name or ".." in name or len(name) > 24:
        raise web.HTTPBadRequest(text="bad tile name")
    stem = name.rsplit(".", 1)[0]
    tiles = request.app["cfg"].tiles_dir
    for cand in (tiles / f"{stem}.png", tiles / f"{stem}.svg"):
        if cand.exists():
            resp = web.FileResponse(cand)
            resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
            return resp
    raise web.HTTPNotFound(text=f"no such tile: {stem}")


async def api_hello(request: web.Request) -> web.Response:
    who = _client_uid(request)
    rooms: RoomManager = request.app["rooms"]
    mine = rooms.find_seat(who["uid"]) if who["name"] else None
    resp = web.json_response({
        "ok": True,
        "uid": who["uid"],
        "name": who["name"],
        "server": {"version": VERSION, "aimodel": AIMODEL_VERSION},
        "at_room": ({"id": mine[0], "seat": mine[1]} if mine else None),
        "config": {
            "ai_delay": request.app["cfg"].ai_delay,
            "human_timeout": request.app["cfg"].human_timeout,
            "allow_chat": request.app["cfg"].allow_chat,
            "max_rooms": request.app["cfg"].max_rooms,
            "record": bool(request.app["cfg"].record),
            "need_password": bool(request.app["cfg"].room_password),
        },
    })
    resp.set_cookie("maj_id", sign_uid(who["uid"], who["name"]),
                    max_age=90 * 24 * 3600, httponly=True, samesite="Lax")
    return resp


async def api_name(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise web.HTTPBadRequest(text="bad json")
    name = str(body.get("name") or "").strip()[:16]
    if not name:
        raise web.HTTPBadRequest(text="昵称不能为空")
    uid = str(body.get("uid") or "").strip()
    if not uid.startswith(ANON_PREFIX) or len(uid) > 32:
        uid = anon_id()
    resp = web.json_response({"ok": True, "uid": uid, "name": name})
    resp.set_cookie("maj_id", sign_uid(uid, name), max_age=90 * 24 * 3600,
                    httponly=True, samesite="Lax")
    return resp


async def api_rooms(request: web.Request) -> web.Response:
    rooms: RoomManager = request.app["rooms"]
    return web.json_response({
        "ok": True,
        "rooms": rooms.list_rooms(),
        "stats": rooms.stats(),
        "models": AG.available_models(),
        "ai_delay": request.app["cfg"].ai_delay,
        "record": bool(request.app["cfg"].record),
    })


async def api_models(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "models": AG.available_models()})


async def api_health(request: web.Request) -> web.Response:
    rooms: RoomManager = request.app["rooms"]
    writer = request.app.get("writer")
    return web.json_response({
        "ok": True,
        "uptime": round(time.time() - request.app["started_at"], 1),
        "clients": len(request.app["clients"]),
        "record": (writer.stats() if writer is not None else None),
        **rooms.stats(),
    })


async def api_profiles(request: web.Request) -> web.Response:
    """玩家画像表（**带匿名 ID，别对外发**）。

    公网部署时建议在 nginx 里给 ``/api/profiles`` 和 ``/api/record``
    加一层 basic auth / IP 白名单。
    """
    profiles = request.app.get("profiles")
    if profiles is None:
        return web.json_response({"ok": True, "record": False, "players": []})
    rows = profiles.all()
    rows.sort(key=lambda r: -float(r.get("elo") or 0))
    return web.json_response({"ok": True, "record": True, "players": rows})


async def api_record(request: web.Request) -> web.Response:
    writer = request.app.get("writer")
    if writer is None:
        return web.json_response({"ok": True, "record": False})
    cfg: Config = request.app["cfg"]
    # 顺手刷一次缓冲：运维 / 验收脚本查「写了没有」时不会被 gzip 缓冲骗到
    writer.flush()
    return web.json_response({
        "ok": True,
        "record": True,
        "dir": str(cfg.data_dir),
        "part_max_bytes": cfg.record_part_bytes,
        "total_max_bytes": cfg.record_total_bytes,
        "stats": writer.stats(),
    })


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


async def ws_handler(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse(heartbeat=25.0, max_msg_size=MAX_WS_MSG,
                               autoping=True, compress=False)
    await ws.prepare(request)
    who = _client_uid(request)
    addr = request.remote or "?"
    client = WSClient(ws, who["uid"], who["name"], addr)
    clients: set[WSClient] = request.app["clients"]
    clients.add(client)
    client.writer_task = asyncio.create_task(client.writer(), name=f"ws-{client.id}")
    log.info("WS 连接 %s 来自 %s（%s）", client.id, addr, who["name"] or "匿名")

    try:
        await client.push(_bootstrap(request, client))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await client.push({"t": "error", "message": "不是合法 JSON"})
                    continue
                if not isinstance(data, dict):
                    await client.push({"t": "error", "message": "消息必须是对象"})
                    continue
                await _handle(request, client, data)
            elif msg.type == WSMsgType.BINARY:
                await client.push({"t": "error", "message": "不支持二进制消息"})
            elif msg.type == WSMsgType.ERROR:
                break
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        log.exception("WS 连接 %s 异常", client.id)
    finally:
        client.closed = True
        clients.discard(client)
        await _detach(request, client)
        if client.writer_task is not None:
            client.writer_task.cancel()
        if client.dropped:
            log.info("WS %s 因客户端太慢丢弃了 %d 条推送", client.id, client.dropped)
        log.info("WS 断开 %s", client.id)
    return ws


async def _detach(request: web.Request, client: WSClient) -> None:
    room = _room_of(request, client)
    if room is not None and client.conn is not None:
        room.remove_conn(client.conn.id)
    client.set_room(None, None)


def _room_of(request: web.Request, client: WSClient) -> Room | None:
    if not client.rid:
        return None
    return request.app["rooms"].get(client.rid)


def _bootstrap(request: web.Request, client: WSClient) -> dict:
    cfg: Config = request.app["cfg"]
    return {
        "t": "hello",
        "v": P.PROTOCOL_VERSION,
        "client": client.id,
        "uid": client.uid,
        "name": client.name,
        "server": {"version": VERSION, "aimodel": AIMODEL_VERSION, "host": cfg.host},
        "config": {
            "ai_delay": cfg.ai_delay,
            "human_timeout": cfg.human_timeout,
            "allow_chat": cfg.allow_chat,
            "record": bool(cfg.record),
            "need_password": bool(cfg.room_password),
        },
        "lobby": lobby_payload(request.app["rooms"], client),
    }


def lobby_payload(rooms: RoomManager, client: WSClient) -> dict:
    return {
        "t": "lobby",
        "rooms": rooms.list_rooms(),
        "stats": rooms.stats(),
        "models": AG.available_models(),
        "record": bool(rooms.cfg.record),
    }


async def _push_lobby(request: web.Request, client: WSClient) -> None:
    await client.push(lobby_payload(request.app["rooms"], client))


async def _handle(request: web.Request, client: WSClient, data: dict) -> None:
    t = str(data.get("t") or "").strip()
    cfg: Config = request.app["cfg"]
    rooms: RoomManager = request.app["rooms"]

    if t in ("hello", "ping"):
        client.hello = True
        if t == "ping":
            await client.push({"t": "pong", "at": data.get("at")})
            return
        name = str(data.get("name") or "").strip()[:16]
        if name:
            client.name = name
            await client.push({"t": "identity", "uid": client.uid, "name": name})
        await _push_lobby(request, client)
        return

    if t == "lobby":
        await _push_lobby(request, client)
        return

    # --- 建房 ---------------------------------------------------------
    if t == "room.create":
        if not client.name:
            await client.push({"t": "error", "message": "先取个昵称再开房"})
            return
        if cfg.room_password and str(data.get("password") or "") != cfg.room_password:
            await client.push({"t": "error", "message": "进房口令不对"})
            return
        players = 4 if int(data.get("players") or 4) == 4 else 3
        raw_seats = data.get("seats") or []
        # 座位表就是**完整的一张桌子**（第 0 格 = 房主自己）。
        # 前端建「纯 AI 观战房」时会明确把第 0 格也标成 ai。
        spec: list[tuple[str, str]] = []
        for s in raw_seats[:players]:
            if isinstance(s, dict) and str(s.get("kind")) == "ai":
                spec.append(("ai", str(s.get("ai") or "teacher")))
            else:
                spec.append(("human", ""))
        while len(spec) < players:
            spec.append(("human", ""))
        host_is_ai = spec[0][0] == "ai"
        try:
            room = rooms.create(
                name=str(data.get("name") or "")[:24],
                players=players,
                host_uid=("" if host_is_ai else client.uid),
                host_name=client.name,
                rounds=int(data.get("rounds") or 0),
                ai_delay=(None if data.get("ai_delay") is None
                          else max(0.0, min(5.0, float(data["ai_delay"])))),
                seats=spec,
                spectator_reveal=bool(data.get("spectator_reveal", True)),
            )
        except RoomError as exc:
            await client.push({"t": "error", "message": str(exc)})
            return
        except (TypeError, ValueError) as exc:
            await client.push({"t": "error", "message": f"参数不对：{exc}"})
            return
        # 「纯 AI 观战房」：房主自己也不坐 —— 这样桌上一张真人牌也没有，
        # 才满足「全亮牌只在纯 AI 房生效」的规矩。房主自动变成观战者。
        await _enter_room(request, client, room, as_spectator=host_is_ai,
                          reserve_seat=not host_is_ai)
        if data.get("autostart", True):
            try:
                room.start()
            except RoomError as exc:
                await client.push({"t": "error", "message": str(exc)})
        return

    # --- 进房 ---------------------------------------------------------
    if t == "room.join":
        rid = str(data.get("room") or "").strip().upper()
        room = rooms.get(rid)
        if room is None:
            await client.push({"t": "error", "message": f"没有这个房间：{rid}"})
            return
        if not client.name:
            await client.push({"t": "error", "message": "先取个昵称"})
            return
        if cfg.room_password and str(data.get("password") or "") != cfg.room_password:
            await client.push({"t": "error", "message": "进房口令不对"})
            return
        spectator = bool(data.get("spectator"))
        want_seat = data.get("seat")
        token = str(data.get("token") or "")
        await _enter_room(request, client, room, as_spectator=spectator,
                          seat=(int(want_seat) if want_seat is not None else None),
                          token=token, reveal=bool(data.get("reveal", True)))
        if room.phase == "lobby" and room.human_count >= 1 and data.get("autostart"):
            try:
                room.start()
            except RoomError as exc:
                await client.push({"t": "error", "message": str(exc)})
        return

    # --- 房间内操作 ----------------------------------------------------
    room = _room_of(request, client)
    if room is None:
        await client.push({"t": "error", "message": "你还没进房间"})
        return

    if t in ("action", "act"):
        ok, err = room.submit_action(
            client.conn,      # type: ignore[arg-type]
            kind=str(data.get("kind") or ""),
            index=data.get("index"),
            tile=data.get("tile"),
        )
        if not ok:
            await client.push({"t": "reject", "message": err, "kind": data.get("kind")})
        return

    if t == "chat":
        ok, err = room.chat_msg(client.conn, str(data.get("text") or ""))  # type: ignore[arg-type]
        if not ok:
            await client.push({"t": "error", "message": err})
        return

    if t == "next":
        room.ack_next()
        return

    if t == "reveal":
        ok, err = room.set_reveal(client.conn, bool(data.get("on", True)))  # type: ignore[arg-type]
        await client.push({"t": "reveal", "ok": ok,
                           "on": client.conn.reveal if client.conn else False,
                           "message": err})
        if ok:
            room.broadcast_later()
        return

    if t == "privacy":
        # 玩家自己关掉「本局记录」——选了就**真的**一条都不落盘
        if client.conn is None or client.conn.seat < 0:
            await client.push({"t": "error", "message": "观战没有记录开关"})
            return
        room.set_no_record(client.conn.seat, bool(data.get("no_record")))
        room.system_msg(
            f"{client.name or '玩家'} "
            f"{'关闭' if data.get('no_record') else '开启'}了本局数据记录"
        )
        room.broadcast_later()
        return

    if t == "profile":
        # 自报段位 / 主玩模式（纯展示）
        if client.conn is not None and client.conn.seat >= 0:
            s = room.seats[client.conn.seat]
            if s is not None:
                s.declared_rank = str(data.get("rank") or "")[:16]
                s.main_mode = str(data.get("mode") or "")[:16]
        room.broadcast_later()
        return

    if t == "room.leave":
        await _detach(request, client)
        await _push_lobby(request, client)
        return

    if t == "room.start":
        try:
            room.start()
        except RoomError as exc:
            await client.push({"t": "error", "message": str(exc)})
        return

    await client.push({"t": "error", "message": f"不认识的消息类型：{t}"})


async def _enter_room(
    request: web.Request, client: WSClient, room: Room, *, as_spectator: bool,
    seat: int | None = None, token: str = "", reveal: bool = True,
    reserve_seat: bool = True,
) -> None:
    """把连接挂到房间上。

    ``reserve_seat=False`` 表示「只连进来看，不要占座位」——
    纯 AI 观战房的房主走这条路，否则他会被自动塞进 0 号位，
    房间就不再是「纯 AI」了。
    """
    # 已经在别的房间坐着了？先退出那边（一个连接只属于一个房间）
    if client.rid and client.rid != room.id:
        await _detach(request, client)

    if not reserve_seat and seat is None:
        seat = room.seat_of(client.uid, token)

    try:
        conn = room.add_conn(
            client.id, client.ws, uid=client.uid, name=client.name,
            seat=None if as_spectator else seat, token=token,
            spectator=as_spectator or not reserve_seat, reveal=reveal,
        )
    except RoomError as exc:
        await client.push({"t": "error", "message": str(exc)})
        return

    client.set_room(room.id, conn)
    seat_obj = room.seats[conn.seat] if conn.seat >= 0 else None
    tok = seat_obj.token if seat_obj is not None else ""
    await client.push({
        "t": "joined",
        "room": room.id,
        "seat": conn.seat,
        "spectator": conn.seat < 0,
        "token": tok,
        "name": (seat_obj.name if seat_obj else client.name),
        "reveal_ok": room.spectator_ok,
        "shared": f"/room/{room.id}",
    })
    if seat_obj is not None and seat_obj.kind == "human":
        room.system_msg(f"{seat_obj.name} 入座了")
    elif conn.seat < 0:
        room.system_msg(f"{client.name or '观众'} 进入观战")
    # 立刻推一帧完整局面给新连接，别等他去猜。这件事必须**同步做完**
    # （不能只 call_soon_threadsafe）：房间可能正在跑，下一帧广播要等到
    # 引擎的下一个动作才来；「刚进来什么都没有」在客户端看起来就是卡住了。
    room.push_state()


def main(argv: list[str] | None = None) -> None:
    from .config import parse_args
    from .logging_setup import setup

    cfg = parse_args(argv)
    setup(cfg.log_dir, cfg.log_level, cfg.log_max_bytes, cfg.log_backups)
    app = create_app(cfg)
    web.run_app(app, host=cfg.host, port=cfg.port, access_log=None,
                print=lambda *a, **k: None)
