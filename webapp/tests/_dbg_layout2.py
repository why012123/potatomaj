import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from webapp.tests.shots import launch, JS_SET_NAME
from webapp.tests.bot import Bot

EVAL = r"""
(() => {
  const out = {seats: {}};
  for (const dir of ['bottom','top','left','right']) {
    const seat = document.querySelector('.seat--'+dir);
    if (!seat) { out.seats[dir] = 'missing'; continue; }
    const edge = seat.querySelector('.edge');
    const er = edge.getBoundingClientRect();
    const hand = seat.querySelector('.edge__hand');
    const hr = hand.getBoundingClientRect();
    const pond = seat.querySelector('.edge__pond');
    const pr = pond.getBoundingClientRect();
    out.seats[dir] = {
      edgeRect: [er.x|0, er.y|0, er.width|0, er.height|0],
      hand: {n: hand.children.length, rect: [hr.x|0, hr.y|0, hr.width|0, hr.height|0]},
      pond: {n: pond.children.length, rect: [pr.x|0, pr.y|0, pr.width|0, pr.height|0]},
      cls: edge.className,
    };
  }
  out.viewport = [innerWidth, innerHeight];
  return JSON.stringify(out);
})()
"""


async def main():
    url = "http://127.0.0.1:8790"
    ws = "ws://127.0.0.1:8790/ws"
    b = Bot(ws, "布局调试2")
    await b.open()
    room = await b.create_room(players=4, rounds=1, ai_delay=0.3,
                              seats=[{"kind": "ai", "ai": m} for m in (
                                  "maj-normal4v1.2", "maj-flash4v1.2",
                                  "maj-mountain4v1.2", "maj-huge4v1.2")],
                              autostart=True)
    profile = Path(tempfile.mkdtemp(prefix="majlay2-"))
    proc, cdp = await launch(
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        9402, 1920, 1080, profile)
    try:
        await cdp.call("Page.navigate", {"url": f"{url}/room/{room}"})
        await cdp.wait_for("!!document.querySelector('.seat--bottom .tile')", timeout=30)
        await cdp.eval(JS_SET_NAME % json.dumps("布局调试2"))
        await asyncio.sleep(4.0)
        raw = await cdp.eval(EVAL)
        print(json.dumps(json.loads(raw), ensure_ascii=False, indent=1))
    finally:
        await cdp.close(); proc.kill()
    await b.close()

asyncio.run(main())
