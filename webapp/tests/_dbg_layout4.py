import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from webapp.tests.shots import launch
from webapp.tests.bot import Bot

EVAL = r"""
(() => {
  const out = {};
  for (const dir of ['left','right']) {
    const seat = document.querySelector('.seat--'+dir);
    const edge = seat.querySelector('.edge');
    const cs = getComputedStyle(edge);
    const er = edge.getBoundingClientRect();
    out[dir] = {
      fixx: edge.style.getPropertyValue('--fix-x'),
      fixy: edge.style.getPropertyValue('--fix-y'),
      transform: cs.transform,
      rect: [Math.round(er.x), Math.round(er.y), Math.round(er.width), Math.round(er.height)],
      w: cs.width, h: cs.height,
      children: [...edge.children].map(c => {
        const r = c.getBoundingClientRect();
        return [c.className, c.children.length,
                Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];
      }),
    };
  }
  return JSON.stringify(out);
})()
"""


async def main():
    url = "http://127.0.0.1:8790"
    ws = "ws://127.0.0.1:8790/ws"
    b = Bot(ws, "L4")
    await b.open()
    room = await b.create_room(players=4, rounds=1, ai_delay=0.3,
                              seats=[{"kind": "ai", "ai": m} for m in (
                                  "maj-normal4v1.2", "maj-flash4v1.2",
                                  "maj-mountain4v1.2", "maj-huge4v1.2")],
                              autostart=True)
    profile = Path(tempfile.mkdtemp(prefix="majl4-"))
    proc, cdp = await launch(
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        9420, 1920, 1080, profile)
    try:
        await cdp.call("Page.navigate", {"url": f"{url}/room/{room}?nopanel=1"})
        await cdp.wait_for("!!document.querySelector('.seat--right .edge__hand')", timeout=30)
        await asyncio.sleep(4)
        raw = await cdp.eval(EVAL)
        print(json.dumps(json.loads(raw), ensure_ascii=False, indent=1))
    finally:
        await cdp.close(); proc.kill()
    await b.close()

asyncio.run(main())
