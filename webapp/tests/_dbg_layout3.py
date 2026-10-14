import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from webapp.tests.shots import launch, JS_SET_NAME
from webapp.tests.bot import Bot


async def main():
    url = "http://127.0.0.1:8790"
    ws = "ws://127.0.0.1:8790/ws"
    b = Bot(ws, "布局快照")
    await b.open()
    room = await b.create_room(players=4, rounds=1, ai_delay=0.35,
                              seats=[{"kind": "ai", "ai": m} for m in (
                                  "maj-normal4v1.2", "maj-flash4v1.2",
                                  "maj-mountain4v1.2", "maj-huge4v1.2")],
                              autostart=True)
    for w, h, tag in ((1920, 1080, "1080p"), (1366, 768, "768p")):
        profile = Path(tempfile.mkdtemp(prefix=f"majlay-{tag}-"))
        proc, cdp = await launch(
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            9410 if w == 1920 else 9411, w, h, profile)
        try:
            await cdp.call("Page.navigate", {"url": f"{url}/room/{room}"})
            await cdp.wait_for("!!document.querySelector('.seat--bottom .tile')", timeout=30)
            await cdp.eval(JS_SET_NAME % json.dumps("摄影师"))
            # 等牌河和副露都有东西
            await cdp.wait_for(
                "(() => {const s=window.maj.state(); return s && "
                "s.seats.filter(x=>x.pond.length>=3).length>=3;})()",
                timeout=90)
            await cdp.eval("document.querySelector('.overlay').hidden = true")
            await asyncio.sleep(0.5)
            out = Path(rf"E:\WHYidea\_verify\layout-{tag}.png")
            await cdp.shot(out)
            print("saved", out)
        finally:
            await cdp.close(); proc.kill()
    await b.close()

asyncio.run(main())
