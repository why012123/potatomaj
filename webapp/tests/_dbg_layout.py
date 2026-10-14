import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from webapp.tests.shots import launch, JS_SET_NAME
from webapp.tests.bot import Bot

EVAL = r"""
(() => {
  const out = {};
  const seat = document.querySelector('.seat--bottom');
  out.seatFound = !!seat;
  if (seat) {
    const r = seat.getBoundingClientRect();
    out.seatRect = [r.x|0, r.y|0, r.width|0, r.height|0];
    const edge = seat.querySelector('.edge');
    out.edge = edge ? edge.className : null;
    if (edge) {
      const er = edge.getBoundingClientRect();
      out.edgeRect = [er.x|0, er.y|0, er.width|0, er.height|0];
      const cs = getComputedStyle(edge);
      out.edgeStyle = {display: cs.display, width: cs.width, height: cs.height,
                       transform: cs.transform, visibility: cs.visibility,
                       position: cs.position};
    }
    for (const cls of ['.edge__hand', '.edge__pond', '.edge__melds']) {
      const e = seat.querySelector(cls);
      if (!e) { out[cls] = 'missing'; continue; }
      const r2 = e.getBoundingClientRect();
      out[cls] = {children: e.children.length, rect: [r2.x|0, r2.y|0, r2.width|0, r2.height|0]};
    }
    const t = seat.querySelector('.tile');
    if (t) {
      const r3 = t.getBoundingClientRect();
      const cs3 = getComputedStyle(t);
      out.tile = {rect: [r3.x|0, r3.y|0, r3.width|0, r3.height|0],
                  w: cs3.width, h: cs3.height, display: cs3.display};
      const img = t.querySelector('img');
      out.img = img ? {src: img.src.slice(-30), w: img.width, h: img.height,
                       complete: img.complete, natural: img.naturalWidth} : null;
    } else out.tile = 'none';
  }
  out.tableRect = (() => { const r = document.querySelector('.table').getBoundingClientRect();
                           return [r.x|0, r.y|0, r.width|0, r.height|0]; })();
  out.playfield = (() => { const e = document.querySelector('.playfield');
                           const r = e.getBoundingClientRect();
                           return [r.x|0, r.y|0, r.width|0, r.height|0]; })();
  return JSON.stringify(out);
})()
"""


async def main():
    url = "http://127.0.0.1:8790"
    ws = "ws://127.0.0.1:8790/ws"
    b = Bot(ws, "布局调试")
    await b.open()
    room = await b.create_room(players=4, rounds=1, ai_delay=0.4,
                              seats=[{"kind": "ai", "ai": m} for m in (
                                  "maj-normal4v1.2", "maj-flash4v1.2",
                                  "maj-mountain4v1.2", "maj-huge4v1.2")],
                              autostart=True)
    profile = Path(tempfile.mkdtemp(prefix="majlay-"))
    proc, cdp = await launch(
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        9401, 1920, 1080, profile)
    try:
        await cdp.call("Page.navigate", {"url": f"{url}/room/{room}"})
        await cdp.wait_for("!!document.querySelector('.seat--bottom .tile')", timeout=30)
        await cdp.eval(JS_SET_NAME % json.dumps("布局调试"))
        await asyncio.sleep(2.0)
        raw = await cdp.eval(EVAL)
        print(json.dumps(json.loads(raw), ensure_ascii=False, indent=1))
        await cdp.shot(Path(r"E:\WHYidea\_verify\layout-debug.png"))
    finally:
        await cdp.close(); proc.kill()
    await b.close()

asyncio.run(main())
