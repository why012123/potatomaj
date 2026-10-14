"""ELO 混战池的网页界面：一个后台线程一直打，页面实时看天梯 + 图表。

跑法::

    python -m aimodel.ui.arena_server              # 默认 http://127.0.0.1:8770
    python -m aimodel.ui.arena_server --players 4 --port 8771

页面功能：按 ELO 排序的天梯、三条实时柱状图（ELO / 一位率 / 和了率）、
分数曲线、场次数量可调、开始/暂停/重置、K 值与分差上限可调。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..arena import Arena, STATE_FILE

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


class Session:
    """一个池子 + 一个一直在跑的后台线程。"""

    def __init__(self, players: int = 3, k: float = 24.0, max_gap: float = 200.0,
                 strict: bool = True, pace: float = 0.0, copies: int = 3,
                 divisor: float = 500.0) -> None:
        self.copies = copies
        self.divisor = divisor
        self.arena = Arena(players=players, k=k, max_gap=max_gap, strict=strict,
                           copies=copies, divisor=divisor)
        self.lock = threading.Lock()
        self.running = False
        self.target = 0          # 还要跑多少场（0 = 无限）
        self.done = 0
        self.pace = pace         # 每场之间停多少秒（0 = 全速）
        self.error = ""
        self.thread: threading.Thread | None = None
        self.t0 = time.time()

    # -- 控制 ---------------------------------------------------------------
    def start(self, matches: int = 100, pace: float | None = None) -> None:
        with self.lock:
            self.target = matches
            self.done = 0
            if pace is not None:
                self.pace = pace
            self.error = ""
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False

    def reset(self) -> None:
        self.pause()
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        with self.lock:
            self.arena = Arena(players=self.arena.players, k=self.arena.k,
                               max_gap=self.arena.max_gap, strict=self.arena.strict,
                               copies=self.copies, divisor=self.divisor)
            self.done = 0
            self.error = ""

    def _run(self) -> None:
        while True:
            with self.lock:
                if not self.running:
                    return
                if self.target and self.done >= self.target:
                    self.running = False
                    return
                pace = self.pace
            try:
                res = self.arena.run_match()
            except Exception as exc:                    # noqa: BLE001
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.running = False
                return
            with self.lock:
                if res is None:
                    self.error = (f"池子里已经没有分差 ≤ {self.arena.max_gap:.0f} 的组合了"
                                  f"（存活 {len(self.arena.pool())} 个）")
                    self.running = False
                    return
                self.done += 1
            if pace > 0:
                time.sleep(pace)

    # -- 快照 ---------------------------------------------------------------
    def snapshot(self, top: int = 40) -> dict:
        with self.lock:
            a = self.arena
            rows = a.table()[:top]
            return {
                "players": a.players,
                "k": a.k,
                "max_gap": a.max_gap,
                "strict": a.strict,
                "running": self.running,
                "target": self.target,
                "done_total": a.n_matches,
                "done_run": self.done,
                "pace": self.pace,
                "seed": a.seed,
                "divisor": a.divisor,
                "error": self.error,
                "alive": sum(1 for p in a.participants if not p.at_floor),
                "total": len(a.participants),
                "elapsed": round(time.time() - self.t0, 1),
                "table": rows,
                "summary": self._summary(a),
                "log": a.log[-14:],
                "copies": self.copies,
            }

    @staticmethod
    def _summary(a: Arena) -> list[dict]:
        """按模型汇总：同一个模型的几个参赛者合起来看（均分 / 最好 / 最差）。"""
        groups: dict[str, list] = {}
        for p in a.participants:
            groups.setdefault(p.source or p.key, []).append(p)
        out = []
        for src, ps in groups.items():
            rated = [x for x in ps if x.games]
            if not rated:
                continue
            elos = [x.rating for x in ps]
            hands = sum(x.hands for x in ps)
            out.append({
                "模型": src,
                "份数": len(ps),
                "均分": round(sum(elos) / len(elos), 1),
                "最好": round(max(elos), 1),
                "最差": round(min(elos), 1),
                "极差": round(max(elos) - min(elos), 1),
                "场数": sum(x.games for x in ps),
                "一位率": round(sum(x.wins for x in ps) / max(1, sum(x.games for x in ps)), 4),
                "和了率": round(sum(x.agari for x in ps) / max(1, hands), 4),
                "放铳率": round(sum(x.deal_in for x in ps) / max(1, hands), 4),
                "版本": ps[0].version, "风格": ps[0].style,
                "保底": sum(1 for x in ps if x.at_floor and x.games),
            })
        out.sort(key=lambda r: -r["均分"])
        return out

    def settings(self, payload: dict) -> None:
        with self.lock:
            a = self.arena
            if "k" in payload:
                a.k = float(payload["k"])
            if "max_gap" in payload:
                a.max_gap = float(payload["max_gap"])
            if "strict" in payload:
                a.strict = bool(payload["strict"])
            if "pace" in payload:
                self.pace = float(payload["pace"])
            if "seed" in payload:
                a.seed = int(payload["seed"])
            if "divisor" in payload:
                a.divisor = float(payload["divisor"])


class Handler(BaseHTTPRequestHandler):
    session: Session
    quiet = True

    def log_message(self, fmt: str, *args) -> None:      # noqa: A003
        if not self.quiet:
            super().log_message(fmt, *args)

    def _json(self, data, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                            # noqa: N802
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            page = STATIC / "arena.html"
            if not page.exists():
                self.send_error(404, "缺少 arena.html")
                return
            body = page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/arena":
            self._json(self.session.snapshot())
            return
        if u.path == "/api/arena/player":
            q = parse_qs(u.query)
            key = (q.get("key") or [""])[0]
            with self.session.lock:
                d = self.session.arena.player_detail(key)
            if d is None:
                self._json({"ok": False, "error": f"没有这个参赛者: {key}"}, 404)
                return
            self._json(d)
            return
        self.send_error(404)

    def do_POST(self) -> None:                           # noqa: N802
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except json.JSONDecodeError:
            self._json({"ok": False, "error": "JSON 解析失败"}, 400)
            return
        if u.path == "/api/arena/start":
            self.session.settings(payload)
            self.session.start(int(payload.get("matches", 100)),
                               float(payload["pace"]) if "pace" in payload else None)
            self._json({"ok": True})
            return
        if u.path == "/api/arena/pause":
            self.session.pause()
            self._json({"ok": True})
            return
        if u.path == "/api/arena/reset":
            self.session.reset()
            self._json({"ok": True})
            return
        if u.path == "/api/arena/settings":
            self.session.settings(payload)
            self._json({"ok": True, "settings": self.session.snapshot()})
            return
        self.send_error(404)


def serve(players: int = 3, port: int = 8770, host: str = "127.0.0.1",
          copies: int = 3, divisor: float = 500.0) -> None:
    session = Session(players=players, copies=copies, divisor=divisor)
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"🏟  ELO 混战池：http://{host}:{port}")
    print(f"     {players} 麻 · {len(session.arena.participants)} 个参战者"
          f"（每个模型 {copies} 个，各记各的分）· 初始 500 分 · 分差上限 200 · 0 分保底不出局")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收工")
        session.pause()
        session.arena.save()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ELO 混战池网页版")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--copies", type=int, default=3, help="每个模型派几个参赛者")
    ap.add_argument("--divisor", type=float, default=500.0,
                    help="点数段位分的除数（0 = 关掉点数项，只按名次算）")
    args = ap.parse_args(argv)
    serve(args.players, args.port, args.host, args.copies, args.divisor)


if __name__ == "__main__":
    main()
