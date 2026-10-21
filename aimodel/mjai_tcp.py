"""最小 mjai **TCP server 侧** —— 给 akochan 这类只做 TCP client 的 AI 用。

标准 mjai 对局服务器（``gimite/mjai``）会把事件广播给所有客户端，客户端
只在能动作时回一行 JSON。本项目原来的 ``ExternalMJaiAgent`` 只会起 stdio
子进程；akochan 只提供::

    ./system.exe mjai_client <port> setup_mjai.json

也就是它自己连到服务器上。所以这里补一个**最小的服务器**：

1. 监听 127.0.0.1:port，接受一个客户端；
2. 发 ``{"type":"hello",...}``，等客户端回 ``{"type":"join",...}``；
3. 发 ``{"type":"start_game","id":<seat>,...}``；
4. 之后把引擎事件按行推过去，等动作（只在需要决策时读）。

只做四麻、只做 akochan 需要的事件。协议细节来自 akochan 的
``main.cpp`` / ``mjai_client.cpp``（见 docs/接入外部AI.md）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any

from . import external
from . import sanma as S
from .agents import Agent, HeuristicAgent
from .engine import Game


def find_akochan() -> str | None:
    """找 akochan 可执行文件；找不到就返回 None（**不监听、不等待**）。"""
    for env in ("POTATOMAJ_EXTERNAL_AKOCHAN_BIN", "POTATOMAJ_AKOCHAN_BIN"):
        v = os.environ.get(env, "").strip()
        if v and Path(v).expanduser().exists():
            return str(Path(v).expanduser())
    for rel in ("akochan/system.exe", "akochan/system", "akochan/akochan",
                "akochan/ai_src/akochan"):
        p = external.EXTERNAL_DIR / rel
        if p.exists():
            return str(p)
    return shutil.which("akochan") or shutil.which("system.exe")


def require_akochan() -> str:
    """没装 akochan 就立刻抛，绝不进入 listen/accept。"""
    found = find_akochan()
    if not found:
        raise external.ExternalAINotInstalled(
            "akochan 没装：找不到 system.exe / akochan 可执行文件。\n"
            f"  → 安装步骤见 {external.DOC_HINT}\n"
            "  → 或设置 POTATOMAJ_EXTERNAL_AKOCHAN_BIN 指向 system.exe"
        )
    return found


class MJaiTCPServer:
    """单客户端 mjai TCP server。

    ``accept()`` 会阻塞到客户端连上并完成 ``hello/join`` 握手；之后
    ``send(event)`` / ``recv(timeout)`` 就是一行一个 JSON。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11600,
                 timeout: float = 20.0, greeting: str = "potatomaj") -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.greeting = greeting
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._q: queue.Queue = queue.Queue()
        self._closed = False
        self._connected = False
        self._lock = threading.RLock()

    # -- 生命周期 -----------------------------------------------------------
    def listen(self) -> None:
        with self._lock:
            if self._sock is not None:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            self._sock = s
            self.port = int(s.getsockname()[1])

    def accept(self, timeout: float | None = None) -> None:
        """等客户端连上；发 hello、读 join。"""
        self.listen()
        assert self._sock is not None
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        self._sock.settimeout(max(0.05, deadline - time.monotonic()))
        try:
            conn, _addr = self._sock.accept()
        except socket.timeout as exc:
            raise external.ExternalAITimeout(
                f"等 mjai TCP 客户端连接超时（{self.host}:{self.port}）"
            ) from exc
        self._conn = conn
        conn.settimeout(None)
        # 先起读线程，再发 hello —— 否则 recv() 等的是一个没人填的队列
        self._reader = threading.Thread(target=self._read_loop,
                                        name="mjai-tcp-reader", daemon=True)
        self._reader.start()
        self._send_raw({"type": "hello", "protocol": "mjai", "name": self.greeting})
        hello = self.recv(timeout=self.timeout)
        if hello is None:
            raise external.ExternalAICrashed("mjai TCP 客户端没有回 join")
        if hello.get("type") not in ("join", "none"):
            raise external.ExternalAIProtocolError(
                f"mjai TCP 握手期望 join，收到 {hello!r}"
            )
        self._connected = True

    def _read_loop(self) -> None:
        assert self._conn is not None
        try:
            for line in self._conn.makefile("r", encoding="utf-8", newline="\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    self._q.put(("bad", line[:500]))
                    continue
                if isinstance(obj, dict):
                    self._q.put(("action", obj))
                else:
                    self._q.put(("bad", str(obj)[:500]))
        except OSError:
            pass
        finally:
            self._q.put(("eof", None))

    def _send_raw(self, obj: dict) -> None:
        if self._conn is None:
            raise external.ExternalAICrashed("mjai TCP 还没连接")
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        try:
            self._conn.sendall((line + "\n").encode("utf-8"))
        except OSError as exc:
            raise external.ExternalAICrashed(f"mjai TCP 写失败：{exc}") from exc

    def send(self, event: dict) -> None:
        self._send_raw(event)

    def recv(self, timeout: float | None = None) -> dict | None:
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise external.ExternalAITimeout(
                    f"mjai TCP 客户端 {timeout:.1f}s 内没有给动作"
                )
            try:
                kind, payload = self._q.get(timeout=remain)
            except queue.Empty:
                continue
            if kind == "eof":
                raise external.ExternalAICrashed("mjai TCP 客户端断开了")
            if kind == "bad":
                raise external.ExternalAIProtocolError(
                    f"mjai TCP 客户端输出不是 JSON 对象：{payload!r}"
                )
            return payload

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for s in (self._conn, self._sock):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            self._conn = None
            self._sock = None


class ExternalTCPAgent(external.ExternalMJaiAgent):
    """把 TCP mjai client（akochan）包装成本项目的 Agent。

    复用 :class:`ExternalMJaiAgent` 的事件翻译 / 超时 / 决策映射，只把
    ``subprocess`` 换成 :class:`MJaiTCPServer`。
    """

    name = "外部AI:TCP"

    def __init__(self, spec: str = "akochan", *, seed: int = 0,
                 player_id: int = 0, players: int = 4,
                 host: str = "127.0.0.1", port: int = 11600,
                 timeout: float = 20.0, connect_timeout: float = 1.5,
                 fallback: Agent | None = None,
                 allow_missing: bool = False,
                 **kwargs: Any) -> None:
        self.host = host
        self.connect_timeout = max(0.1, float(connect_timeout))
        self._akochan_bin = None if allow_missing else require_akochan()
        self.port = int(port)
        super().__init__(spec, seed=seed, player_id=player_id, players=players,
                         timeout=timeout, fallback=fallback,
                         allow_missing=True, **kwargs)
        self.available = True
        self.available_reason = ""
        self.name = f"外部AI:{self.spec.title}（TCP {host}:{port}）"

    # -- 生命周期 -----------------------------------------------------------
    def _server(self) -> MJaiTCPServer:
        srv = getattr(self, "_tcp", None)
        if srv is None:
            srv = MJaiTCPServer(self.host, self.port, timeout=self.timeout)
            self._tcp = srv
        return srv

    def _ensure_started(self) -> None:
        if self._dead:
            return
        srv = getattr(self, "_tcp", None)
        if srv is not None and getattr(srv, "_connected", False):
            return
        if self._started:
            return
        self._start()

    def _start(self) -> None:
        if self._closed:
            raise external.ExternalAIError(f"{self.name} 已经 close()")
        srv = self._server()
        srv.listen()
        # 把端口暴露出去，方便调用方先启动 akochan 再 play
        self.port = srv.port
        # 先标记已启动：握手失败时不要在下一次 feed 里又 accept 一次（会卡死）
        self._started = True
        # 本地/CI 没有 akochan 是常态：只等 1.5s，且只等一次
        srv.accept(timeout=self.connect_timeout)
        srv.send({"type": "start_game", "id": self.player_id,
                  "names": [f"P{i}" for i in range(self.players)]})
        # akochan 对 start_game 会回一行 none；读掉，别让它变成第一个决策的脏数据
        try:
            first = srv.recv(timeout=min(10.0, self.timeout))
            if first is not None and first.get("type") not in ("none", "join"):
                self._pending_action = first
        except external.ExternalAITimeout:
            pass

    def _write_one(self, event: dict) -> None:
        event = {k: v for k, v in event.items() if not k.startswith("_")}
        self._server().send(event)

    def _write_batch(self, events: list[dict]) -> None:
        for ev in events:
            self._write_one(ev)

    def _read_action(self, timeout: float, *, allow_none: bool = False,
                     count_timeout: bool = True) -> dict | None:
        try:
            return self._server().recv(timeout=timeout)
        except external.ExternalAITimeout:
            if count_timeout:
                self.timeout_count += 1
            raise

    def close(self) -> None:
        srv = getattr(self, "_tcp", None)
        if srv is not None:
            try:
                srv.send({"type": "end_game"})
            except Exception:  # noqa: BLE001
                pass
            srv.close()
        self._closed = True
        super().close()


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="最小 mjai TCP server（四麻，给 akochan 用）")
    ap.add_argument("--port", type=int, default=11600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--player-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    ag = ExternalTCPAgent("akochan", seed=args.seed, player_id=args.player_id,
                          players=4, host=args.host, port=args.port,
                          timeout=args.timeout)
    print(f"监听 {args.host}:{ag.port}；现在启动：")
    print(f"  ./system.exe mjai_client {ag.port} setup_mjai.json")
    others = [HeuristicAgent(seed=100 + i) for i in range(4) if i != args.player_id]
    agents = []
    it = iter(others)
    for seat in range(4):
        agents.append(ag if seat == args.player_id else next(it))
    game = Game(agents, seed=args.seed, rule=S.rule_for(4), rounds=args.rounds)
    external.attach(game)
    try:
        result = game.play()
        print("结果：", result.scores, result.ranks)
    finally:
        ag.close()


if __name__ == "__main__":
    _main()
