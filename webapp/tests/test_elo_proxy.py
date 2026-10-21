"""8790 -> 8964 官方 ELO 反代测试（不依赖 pytest，直接跑）。

    python -m webapp.tests.test_elo_proxy

覆盖的四件事（每件错了都会让公网 /official 静默坏掉）：
  ① 页面路径原样转发（/official、/arena、/replay …）
  ② API 路径原样转发（/api/arena、/api/arena/*、/api/elo/*）
  ③ **query string 与 POST body 都保留**（?season= / ?days=all / favorite）
  ④ 上游 8964 挂了 -> 502 + 可读错误（不是白屏 / 不是 500 堆栈）

负向验证：
  换成一个「只转 request.path、不转 query」的坏代理
  -> 「query 被保留」那条必红（这里直接跑坏代理证明断言不是恒真）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aiohttp import ClientSession, web  # noqa: E402

from webapp.server.app import register_elo_proxy  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  🔴 {name}   {detail}")


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - 测试拿随机端口
    return runner, f"http://127.0.0.1:{port}"


async def _fake_upstream() -> tuple[web.AppRunner, str]:
    app = web.Application()

    async def echo(request: web.Request) -> web.Response:
        body = await request.text()
        return web.json_response({
            "path": request.path,
            "query": request.query_string,
            "method": request.method,
            "body": body,
        })

    app.router.add_route("*", "/{tail:.*}", echo)
    return await _serve(app)


async def _bad_proxy_app() -> tuple[web.AppRunner, str]:
    """负向对照：只转 path，**故意不转 query**。"""
    app = web.Application()

    async def bad(request: web.Request) -> web.Response:
        upstream = os.environ["MAJ_ELO_UPSTREAM"].rstrip("/")
        async with ClientSession() as s:
            async with s.request(request.method, upstream + request.path,
                                 data=await request.read()) as r:
                return web.Response(
                    body=await r.read(), status=r.status,
                    headers={"Content-Type": r.headers.get("Content-Type",
                                                           "application/json")})

    app.router.add_route("*", "/{tail:.*}", bad)
    return await _serve(app)


async def main() -> int:
    print("=" * 70)
    print("  8790 -> 8964 官方 ELO 反代测试")
    print("=" * 70)
    env_key = "MAJ_ELO_UPSTREAM"
    saved = os.environ.get(env_key)
    upstream_runner = proxy_runner = bad_runner = None
    try:
        upstream_runner, upstream = await _fake_upstream()
        os.environ[env_key] = upstream

        proxy_app = web.Application()
        register_elo_proxy(proxy_app)
        proxy_runner, proxy = await _serve(proxy_app)

        print("\n① 页面路径")
        async with ClientSession() as s:
            async with s.get(f"{proxy}/official?season=2026-10-20T20") as r:
                d = await r.json()
            check("GET /official -> 200", r.status == 200, str(r.status))
            check("页面 path 原样转发", d["path"] == "/official", d["path"])
            check("页面 query 原样转发", d["query"] == "season=2026-10-20T20", d["query"])
            async with s.get(f"{proxy}/replay?season=x&index=7") as r:
                d = await r.json()
            check("GET /replay 带 query", d["query"] == "season=x&index=7", d["query"])

        print("\n② API GET")
        async with ClientSession() as s:
            async with s.get(f"{proxy}/api/arena?days=3") as r:
                d = await r.json()
            check("GET /api/arena -> 200", r.status == 200, str(r.status))
            check("API path 原样转发", d["path"] == "/api/arena", d["path"])
            check("API query 原样转发", d["query"] == "days=3", d["query"])
            async with s.get(f"{proxy}/api/arena/player?key=maj-pro4v1.1%233") as r:
                d = await r.json()
            check("GET /api/arena/player 子路径", d["path"] == "/api/arena/player", d["path"])
            async with s.get(f"{proxy}/api/elo/history?season=2026-10-20T20&days=all") as r:
                d = await r.json()
            check("GET /api/elo/history 子路径 + 两个 query",
                  d["path"] == "/api/elo/history"
                  and d["query"] == "season=2026-10-20T20&days=all", str(d))

        print("\n③ POST body")
        payload = {"sid": "2026-10-20T20", "key": "k", "favorite": True}
        async with ClientSession() as s:
            async with s.post(f"{proxy}/api/elo/favorite?from=test", json=payload) as r:
                d = await r.json()
            check("POST /api/elo/favorite -> 200", r.status == 200, str(r.status))
            check("POST 方法保留", d["method"] == "POST", d["method"])
            check("POST query 保留", d["query"] == "from=test", d["query"])
            check("POST body 原样保留", json.loads(d["body"]) == payload, d["body"])
            async with s.post(f"{proxy}/api/arena/start", json={"matches": 1}) as r:
                d = await r.json()
            check("POST /api/arena/start 子路径", d["path"] == "/api/arena/start", d["path"])

        print("\n④ 上游挂了")
        os.environ[env_key] = "http://127.0.0.1:9"
        async with ClientSession() as s:
            async with s.get(f"{proxy}/api/arena") as r:
                body = await r.text()
            check("上游不可达 -> 502", r.status == 502, str(r.status))
            check("502 是 JSON 且 ok=false", '"ok": false' in body or '"ok":false' in body,
                  body[:200])

        print("\n⑤ 负向验证（故意坏掉的代理）")
        os.environ[env_key] = upstream
        bad_runner, bad = await _bad_proxy_app()
        async with ClientSession() as s:
            async with s.get(f"{bad}/api/arena?days=3") as r:
                d = await r.json()
        check("坏代理确实丢掉了 query（所以上面「query 保留」那条能红）",
              d["query"] == "", str(d["query"]))
        check("坏代理仍然保留 path（证明差异只来自 query）",
              d["path"] == "/api/arena", d["path"])
    finally:
        for runner in (upstream_runner, proxy_runner, bad_runner):
            if runner is not None:
                await runner.cleanup()
        if saved is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = saved

    print("\n" + "=" * 70)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
