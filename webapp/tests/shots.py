#!/usr/bin/env python
"""验收 9：**截图**（牌桌 / 结算 / 观战 / 房间列表，各 1366×768 与 1920×1080）。

用无头 Edge + 原生 CDP（见 :mod:`webapp.tests.cdp`），不依赖 Playwright。

跑法::

    .venv/Scripts/python.exe -m webapp.tests.shots
    .venv/Scripts/python.exe -m webapp.tests.shots --out E:\\WHYidea\\_verify\\shots

截图之前会**先自己打一局**（两个测试客户端），保证画面是「真的在打」而不是
空桌子；观战那张用纯 AI 房（桌上一张真人牌也没有，全亮才有意义）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.bot import Bot  # noqa: E402
from webapp.tests.cdp import launch  # noqa: E402

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
AI3 = ["maj-normalv1.2", "maj-flashv1.2", "maj-mountainv1.2"]
AI4 = ["maj-normal4v1.3", "maj-flash4v1.2", "maj-mountain4v1.2", "maj-huge4v1.2"]

JS_NAME = "(() => {const e=document.querySelector('#name');if(e){e.value=%s;" \
          "e.dispatchEvent(new Event('change',{bubbles:true}));}return 'ok';})()"


def http_base(ws_url: str) -> str:
    """``ws://host:port/ws`` -> ``http://host:port``。

    **必须换协议**：直接把 ``ws://...`` 塞给 ``Page.navigate`` 会得到
    ``ERR_UNKNOWN_URL_SCHEME``，截出来的是一张浏览器错误页
    （这个坑踩过一次 —— 十张截图全是「无法访问此页面」，白跑一轮）。
    """
    u = ws_url.strip()
    if u.startswith("wss://"):
        u = "https://" + u[6:]
    elif u.startswith("ws://"):
        u = "http://" + u[5:]
    return u.split("/ws")[0].rstrip("/")


async def shot(cdp, out: Path, tag: str, size: str) -> None:
    p = out / f"{tag}-{size}-{time.strftime('%H%M%S')}.png"
    await cdp.shot(p)
    print(f"  ✓ {p.name}")


async def main_async(args) -> int:
    url = args.url
    base = http_base(url)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("验收 9：截图")
    print("=" * 70)

    # --- 准备桌子 -------------------------------------------------------
    # **每个分辨率都要新桌子**：打完的房间不会自己重开，而且第一轮结束后
    # 陪练任务会被取消，桌子就停在那里等真人（实测：第二轮怎么等都等不到
    # 结算面板 —— 因为牌局根本没在推进）。
    seed = Bot(url, "截图准备工作")
    await seed.open()
    room_ai = await seed.create_room(
        players=4, rounds=3, ai_delay=args.ai_delay,
        seats=[{"kind": "ai", "ai": m} for m in AI4],
        autostart=True, name="AI 观战桌")
    print(f"  纯 AI 观战房 {room_ai}")

    try:
        for i, (label, w, h) in enumerate((("1366x768", 1366, 768),
                                           ("1920x1080", 1920, 1080))):
            room4 = await seed.create_room(
                players=4, rounds=2, ai_delay=args.ai_delay,
                seats=[{"kind": "human"}] +
                      [{"kind": "ai", "ai": m} for m in AI4[1:]],
                autostart=True, name=f"四麻牌桌{label}")
            room3 = await seed.create_room(
                players=3, rounds=2, ai_delay=args.ai_delay,
                seats=[{"kind": "human"}] +
                      [{"kind": "ai", "ai": m} for m in AI3[1:]],
                autostart=True, name=f"三麻牌桌{label}")
            keeper4 = Bot(url, f"四麻陪练{label}", strategy="random")
            keeper3 = Bot(url, f"三麻陪练{label}", strategy="random")
            await keeper4.open()
            await keeper3.open()
            # 陪练**当观战者**：桌子是 `seed` 开的、座位 0 留给浏览器，
            # 陪练只负责「局间按下一局」和「别家没人时也不至于停桌」。
            # （之前让它入座，结果页面轮到自己出牌就停住了。）
            await keeper4.join(room4, spectator=True, autostart=False)
            await keeper3.join(room3, spectator=True, autostart=False)
            tasks = [
                asyncio.create_task(keeper4.play_until(
                    lambda s: s["phase"] == "finished", timeout=1800, stall=600)),
                asyncio.create_task(keeper3.play_until(
                    lambda s: s["phase"] == "finished", timeout=1800, stall=600)),
            ]
            print(f"  --- {label}：四麻 {room4} / 三麻 {room3}")
        for i, (label, w, h) in enumerate((("1366x768", 1366, 768),
                                           ("1920x1080", 1920, 1080))):
            profile = Path(tempfile.mkdtemp(prefix=f"majshot-{label}-"))
            proc, cdp = await launch(EDGE, 9600 + i, w, h, profile)
            try:
                # 1) 房间列表
                await cdp.call("Page.navigate", {"url": f"{base}/"})
                try:
                    await cdp.wait_for(
                        "!!document.querySelector('.lobby, .roomlist, .rooms')",
                        label="大厅", timeout=30)
                except Exception:  # noqa: BLE001
                    pass
                await cdp.eval(JS_NAME % '"截图"')
                await asyncio.sleep(2.0)
                await shot(cdp, out, "roomlist", label)

                # 2) 观战页（纯 AI 房，全亮）
                await cdp.call("Page.navigate",
                               {"url": f"{base}/room/{room_ai}"})
                await cdp.eval(JS_NAME % '"截图"')
                await asyncio.sleep(3)
                # 勾上「全亮牌」（纯 AI 房才允许）
                await cdp.eval(
                    "(() => {const b=document.querySelector('#revealbox');"
                    "if(b && !b.checked){b.checked=true;"
                    "b.dispatchEvent(new Event('change',{bubbles:true}));}"
                    "return 'ok';})()")
                await asyncio.sleep(6)
                await shot(cdp, out, "spectate", label)

                # 3) 三麻牌桌（有真人座位）
                await cdp.call("Page.navigate",
                               {"url": f"{base}/room/{room3}"})
                await cdp.eval(JS_NAME % '"截图"')
                await asyncio.sleep(9)
                await shot(cdp, out, "table-3p", label)

                # 4) 四麻牌桌
                await cdp.call("Page.navigate",
                               {"url": f"{base}/room/{room4}"})
                await cdp.eval(JS_NAME % '"截图"')
                await asyncio.sleep(9)
                await shot(cdp, out, "table-4p", label)

                # 5) 结算页。**要抢时机**：结算面板只在 `phase == hand_end`
                #    （一局结束）和 `phase == finished`（一场结束）显示，
                #    一进入下一局就隐藏。
                #
                #    另外**别忘了自己按 Enter**：浏览器是 0 号位的真人，
                #    她不确认，整桌就停在局间最多 60 秒 —— 那样一局都推不完。
                #    `bindKeys()` 里 Enter 就是「下一局」，等价于玩家自己按。
                #
                #    踩过的两个坑：(a) 盲等 9 秒只会在下一局的开局画面上截图；
                #    (b) 用 `wait_for` 每秒查一次几乎必然错过 hand_end 那个窗口，
                #    所以这里 40ms 高频轮询。
                got = False
                for _ in range(args.result_tries):
                    vis = await cdp.eval(
                        "(() => {const o=document.querySelector('.overlay');"
                        "return (o && !o.hidden && "
                        "!!o.querySelector('.agari, .drawbox, .result'))?'1':'0';})()")
                    if vis == "1":
                        got = True
                        break
                    await asyncio.sleep(0.04)
                    if _ % 12 == 11:
                        await cdp.eval(
                            "window.dispatchEvent(new KeyboardEvent('keydown',"
                            "{key:'Enter',bubbles:true}))")
                if got:
                    await shot(cdp, out, "result", label)
                else:
                    print(f"  ! result {label}：没抢到结算面板，截当前画面")
                    await shot(cdp, out, "result", label)
            finally:
                await cdp.close()
                proc.kill()
                for t in tasks:
                    t.cancel()
                for b in (keeper3, keeper4):
                    await b.close()
    finally:
        await seed.close()

    print(f"\n✅ 截图完成 -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--out", default=str(ROOT.parent / "_verify" / "shots"))
    ap.add_argument("--ai-delay", type=float, default=1.0)
    ap.add_argument("--result-tries", type=int, default=6000,
                    help="抢结算面板的轮询次数（每次 40ms，默认 ~240 秒）")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
