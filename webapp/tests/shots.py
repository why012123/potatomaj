#!/usr/bin/env python
"""验收 9：**截图**（房间列表 / 观战 / 三麻牌桌 / 四麻牌桌 / 结算，各两种分辨率）。

用无头 Edge + 原生 CDP（见 :mod:`webapp.tests.cdp`），不依赖 Playwright。

**关键点：截图里那张桌子必须由浏览器自己开**。让别人开好房间、浏览器再进去，
它只能当观战者 —— 观战看别家手牌就是牌背，「自家手牌也全是牌背」，
看起来像 bug（实测踩过：``viewer.seat = -1 / spectator = true``）。
所以这里**走真实的大厅流程**：点「建立房间」→ 用「快速预设」选座 →
点「开桌并开始」，浏览器就是房主、就是 0 号位，手牌自然是牌面。

跑法::

    .venv/Scripts/python.exe -m webapp.tests.shots
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

JS_NAME = ("(() => {const e=document.querySelector('#nick');if(e){e.value=%s;"
           "e.dispatchEvent(new Event('change',{bubbles:true}));}return 'ok';})()")

#: 走真实大厅流程建房：预设按钮的 data-preset 是「human,ai,ai,ai|4」这种
JS_BUILD = """
(() => {
  const preset = %s;      // 例如 "human,ai,ai,ai|4"
  const name = %s;
  const rounds = %s;
  const openBtn = document.querySelector('#mk');
  if (!openBtn) return 'no-mk';
  openBtn.click();
  const el = (s) => document.querySelector(s);
  const set = (sel, val) => {
    const n = el(sel); if (!n) return false;
    n.value = val;
    n.dispatchEvent(new Event('change', {bubbles: true}));
    return true;
  };
  const b = el('#bname'); if (b) { b.value = name;
    b.dispatchEvent(new Event('change', {bubbles: true})); }
  // 预设按钮会一次性把「人数 + 每个座位」都设好，最省事
  const pre = [...document.querySelectorAll('[data-preset]')]
    .find(x => (x.dataset.preset || '') === preset);
  if (pre) pre.click();
  set('#brounds', String(rounds));
  set('#bdelay', '0.3');
  const go = el('#bgo');
  if (!go) return 'no-bgo';
  go.click();
  return 'built';
})()
"""

JS_ROOM = ("(() => {const m=/\\/room\\/([A-Za-z0-9]+)/.exec(location.pathname);"
           "return m ? m[1] : '';})()")

JS_VIEWER = ("(() => {const s=window.maj&&window.maj.state?window.maj.state():null;"
             "if(!s) return 'none';"
             "return JSON.stringify({seat:s.viewer.seat,"
             "hand:(s.seats[s.viewer.seat]||{}).hand?1:0,phase:s.phase});})()")


def http_base(ws_url: str) -> str:
    """``ws://host:port/ws`` -> ``http://host:port``。

    **必须换协议**：把 ``ws://...`` 塞给 ``Page.navigate`` 会得到
    ``ERR_UNKNOWN_URL_SCHEME``，截出来是一张浏览器错误页（踩过一次）。
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


async def got_room(cdp, timeout: float = 40.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        rid = await cdp.eval(JS_ROOM)
        if rid:
            return rid
        await asyncio.sleep(0.3)
    raise TimeoutError("建房后没有跳到 /room/…")


async def main_async(args) -> int:
    url = args.url
    base = http_base(url)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("验收 9：截图")
    print("=" * 70)

    # 纯 AI 观战房：用真实的测试客户端开（浏览器进去当观战者正合适）
    seed = Bot(url, "观战房主")
    await seed.open()
    room_ai = await seed.create_room(
        players=4, rounds=3, ai_delay=args.ai_delay,
        seats=[{"kind": "ai", "ai": m} for m in AI4],
        autostart=True, name="AI 观战桌")
    print(f"  纯 AI 观战房 {room_ai}")

    keepers: list[Bot] = []
    tasks: list[asyncio.Task] = []
    try:
        for i, (label, w, h) in enumerate((("1366x768", 1366, 768),
                                           ("1920x1080", 1920, 1080))):
            profile = Path(tempfile.mkdtemp(prefix=f"majshot-{label}-"))
            proc, cdp = await launch(EDGE, 9600 + i, w, h, profile)
            try:
                await cdp.call("Page.enable")
                await cdp.call("Runtime.enable")

                # 1) 房间列表
                await cdp.call("Page.navigate", {"url": f"{base}/"})
                await asyncio.sleep(3)
                await cdp.eval(JS_NAME % '"截图"')
                await asyncio.sleep(1.5)
                await shot(cdp, out, "roomlist", label)

                # 2) 观战页（纯 AI 房 → 全亮有意义）
                await cdp.call("Page.navigate", {"url": f"{base}/room/{room_ai}"})
                await asyncio.sleep(4)
                await cdp.eval(
                    "(() => {const b=document.querySelector('#revealbox');"
                    "if(b && !b.checked){b.checked=true;"
                    "b.dispatchEvent(new Event('change',{bubbles:true}));}"
                    "return 'ok';})()")
                await asyncio.sleep(5)
                await shot(cdp, out, "spectate", label)

                # 3) **浏览器自己开一张四麻桌**（这样它才是 0 号位、手牌是牌面）
                await cdp.call("Page.navigate", {"url": f"{base}/"})
                await asyncio.sleep(3)
                await cdp.eval(JS_NAME % f'"截图{label}"')
                await cdp.eval(JS_BUILD % ('"human,ai,ai,ai|4"',
                                           f'"四麻牌桌{label}"', "2"))
                room4 = await got_room(cdp)
                print(f"  --- {label}：浏览器开的四麻房 {room4}")
                # 陪练当观战者：只负责局间推进 + 别家手牌有变化
                k4 = Bot(url, f"陪练4-{label}", strategy="random")
                await k4.open()
                await k4.join(room4, spectator=True, autostart=False)
                keepers.append(k4)
                tasks.append(asyncio.create_task(k4.play_until(
                    lambda s: s["phase"] == "finished", timeout=3600, stall=900)))
                await asyncio.sleep(8)
                await shot(cdp, out, "table-4p", label)

                # 4) **浏览器自己开一张三麻桌**
                await cdp.call("Page.navigate", {"url": f"{base}/"})
                await asyncio.sleep(3)
                await cdp.eval(JS_NAME % f'"截图{label}"')
                await cdp.eval(JS_BUILD % ('"human,ai,ai|3"',
                                           f'"三麻牌桌{label}"', "2"))
                room3 = await got_room(cdp)
                print(f"  --- {label}：浏览器开的三麻房 {room3}")
                k3 = Bot(url, f"陪练3-{label}", strategy="random")
                await k3.open()
                await k3.join(room3, spectator=True, autostart=False)
                keepers.append(k3)
                tasks.append(asyncio.create_task(k3.play_until(
                    lambda s: s["phase"] == "finished", timeout=3600, stall=900)))
                await asyncio.sleep(8)
                await shot(cdp, out, "table-3p", label)

                # 5) 结算页：抢 hand_end / finished 那个窗口（40ms 轮询），
                #    并按 Enter 推进局间 —— 浏览器是 0 号位的真人，
                #    她不确认整桌会停在局间最多 60 秒。
                await cdp.call("Page.navigate", {"url": f"{base}/room/{room4}"})
                await asyncio.sleep(3)
                got = False
                for _ in range(args.result_tries):
                    vis = await cdp.eval(
                        "(() => {const o=document.querySelector('.overlay');"
                        "return (o && !o.hidden && "
                        "!!o.querySelector('.agari, .drawbox, .result'))"
                        "?'1':'0';})()")
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
    finally:
        for t in tasks:
            t.cancel()
        for b in keepers:
            await b.close()
        await seed.close()

    print(f"\n✅ 截图完成 -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--out", default=str(ROOT.parent / "_verify" / "shots"))
    ap.add_argument("--ai-delay", type=float, default=1.0)
    ap.add_argument("--result-tries", type=int, default=6000,
                    help="抢结算面板的轮询次数（每次 40ms）")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
