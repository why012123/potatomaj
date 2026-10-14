"""一个很小的 CDP 驱动（无头 Edge/Chrome），给截图与端到端自检用。

为什么不用 Playwright/Selenium：那要装几百 MB 的运行时和浏览器内核。这里
直接用系统上已经有的 Edge + 原生 CDP（WebSocket），依赖只有 ``websockets``
和标准库 —— 部署机上不用多装任何东西。

用法::

    proc, cdp = await launch(EDGE, 9222, 1920, 1080, profile_dir)
    await cdp.call("Page.navigate", {"url": "http://127.0.0.1:8790/"})
    await cdp.wait_for("!!document.querySelector('.table')", label="牌桌")
    await cdp.shot(Path("shot.png"))
    await cdp.close(); proc.kill()
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import websockets


class TimeoutError_(TimeoutError):
    pass


class CDP:
    def __init__(self, ws: Any, proc: subprocess.Popen) -> None:
        self.ws = ws
        self.proc = proc
        self._id = 0
        self._lock = asyncio.Lock()
        self.events: list[dict] = []

    async def call(self, method: str, params: dict | None = None,
                   timeout: float = 30.0) -> dict:
        async with self._lock:
            self._id += 1
            mid = self._id
            await self.ws.send(json.dumps({"id": mid, "method": method,
                                           "params": params or {}}))
            end = time.time() + timeout
            while True:
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError_(f"CDP {method} 超时")
                raw = await asyncio.wait_for(self.ws.recv(), left)
                msg = json.loads(raw)
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method}: {msg['error']}")
                    return msg.get("result") or {}
                if "method" in msg:
                    self.events.append(msg)

    async def eval(self, js: str, timeout: float = 30.0) -> str:
        r = await self.call(
            "Runtime.evaluate",
            {"expression": js, "returnByValue": True, "awaitPromise": True},
            timeout,
        )
        if r.get("exceptionDetails"):
            raise RuntimeError(f"JS 异常：{r['exceptionDetails'].get('text')}")
        val = r.get("result", {}).get("value")
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    async def wait_for(self, js: str, *, label: str = "条件", timeout: float = 30.0,
                       interval: float = 0.25) -> str:
        end = time.time() + timeout
        last = ""
        while time.time() < end:
            try:
                last = await self.eval(js, timeout=10)
            except Exception as exc:  # noqa: BLE001
                last = f"<{exc}>"
            if last == "true":
                return last
            await asyncio.sleep(interval)
        raise TimeoutError_(f"超时：等不到条件：{label}（最后错误 {last}）")

    async def shot(self, path: Path) -> None:
        r = await self.call("Page.captureScreenshot", {"format": "png",
                                                       "captureBeyondViewport": False})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(r["data"]))

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def _http_json(url: str, timeout: float = 20.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


async def launch(
    browser: str,
    port: int,
    width: int,
    height: int,
    profile: Path | None = None,
    *,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen, CDP]:
    """起一个无头浏览器并接上 CDP。"""
    if not Path(browser).exists():
        found = _find_browser()
        if not found:
            raise SystemExit(f"找不到浏览器：{browser}")
        browser = found
    profile = profile or Path(tempfile.mkdtemp(prefix="majcdp-"))
    args = [
        browser,
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        f"--window-size={width},{height}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--disable-features=Translate,BackForwardCache",
        "about:blank",
        *(extra_args or []),
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ver = None
    for _ in range(60):
        try:
            ver = _http_json(f"http://127.0.0.1:{port}/json/version")
            break
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.5)
    if ver is None:
        proc.kill()
        raise SystemExit("无头浏览器起不来（CDP 端口没响应）")

    ws_url = ver["webSocketDebuggerUrl"]
    ws = await websockets.connect(ws_url, max_size=64 << 20)
    cdp = CDP(ws, proc)
    # 找 / 开一个页面 target，接管它
    target = None
    for _ in range(40):
        try:
            tabs = _http_json(f"http://127.0.0.1:{port}/json/list")
        except Exception:  # noqa: BLE001
            tabs = []
        for t in tabs:
            if t.get("type") == "page":
                target = t
                break
        if target:
            break
        await asyncio.sleep(0.25)
    if target is None:
        r = await cdp.call("Target.createTarget", {"url": "about:blank"})
        target = {"id": r["targetId"]}
    sess = await cdp.call("Target.attachToTarget",
                          {"targetId": target["id"], "flatten": True})
    sid = sess["sessionId"]
    # flatten 模式下要带 sessionId，包一层
    raw_call = cdp.call

    async def call(method: str, params: dict | None = None, timeout: float = 30.0):
        async with cdp._lock:  # noqa: SLF001
            cdp._id += 1  # noqa: SLF001
            mid = cdp._id
            await ws.send(json.dumps({"id": mid, "method": method,
                                      "params": params or {}, "sessionId": sid}))
            end = time.time() + timeout
            while True:
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError_(f"CDP {method} 超时")
                msg = json.loads(await asyncio.wait_for(ws.recv(), left))
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method}: {msg['error']}")
                    return msg.get("result") or {}

    cdp.call = call  # type: ignore[method-assign]
    del raw_call
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )
    await asyncio.sleep(0.2)
    return proc, cdp


def _find_browser() -> str | None:
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/usr/bin/chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
    ]
    for c in cands:
        if Path(c).exists():
            return c
    for name in ("msedge", "chrome", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return None
