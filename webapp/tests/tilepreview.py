"""把所有牌面拼成一页，用无头浏览器截图，确认 37 张都画得出来。

这是**素材自检**：换牌面素材、重新生成 SVG 之后跑一次，
比在牌桌上「感觉哪里不对」快得多。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.cdp import launch  # noqa: E402

IDS = (
    [f"{i}m" for i in range(1, 10)] + ["0m"]
    + [f"{i}p" for i in range(1, 10)] + ["0p"]
    + [f"{i}s" for i in range(1, 10)] + ["0s"]
    + [f"{i}z" for i in range(1, 8)]
    + ["back", "blank"]
)


async def main() -> int:
    import http.server
    import threading

    tiles = ROOT / "webapp" / "static" / "tiles"
    page = ROOT / "webapp" / "static" / "_tilepreview.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        f'<figure><img src="/tiles/{t}.svg" width="90"><figcaption>{t}</figcaption></figure>'
        for t in IDS
    )
    page.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "body{background:#14532d;margin:0;padding:16px;font:12px sans-serif;color:#fff}"
        "figure{display:inline-block;margin:4px;text-align:center}"
        "img{background:#f3efe9;border-radius:5px}"
        "</style></head><body>" + cards + "</body></html>",
        encoding="utf-8",
    )

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT / "webapp" / "static"), **kw)

        def log_message(self, *a):  # noqa: ARG002
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8792), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    out = ROOT.parent / "_verify" / "tiles-preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 每次都写一个新文件名：截图工具 / 附件缓存按内容哈希去重，
    # 同名文件有时候会看到上一版（这个坑踩过一次，白等了一轮）。
    stamp = time.strftime("%H%M%S")
    out = out.with_name(f"tiles-preview-{stamp}.png")
    profile = Path(tempfile.mkdtemp(prefix="majtiles-"))
    browser = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    proc, cdp = await launch(browser, 9521, 1500, 900, profile)
    try:
        await cdp.call("Page.navigate", {"url": "http://127.0.0.1:8792/_tilepreview.html"})
        await asyncio.sleep(1.0)
        # 强制重画：清掉图片缓存再加载一次
        await cdp.eval(
            "(() => {for (const i of document.images) "
            "{const s=i.src; i.src=''; i.src=s+'?v='+Date.now();} return 'ok';})()"
        )
        await asyncio.sleep(1.5)
        await cdp.shot(out)
        bad = await cdp.eval(
            "JSON.stringify([...document.images]"
            ".filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.src))"
        )
    finally:
        await cdp.close()
        proc.kill()
        srv.shutdown()
    print(f"截图：{out}")
    print("加载失败的牌：", bad)
    return 0 if bad.strip() in ("[]", "null") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
