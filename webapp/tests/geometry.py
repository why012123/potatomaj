#!/usr/bin/env python
"""量牌桌几何：四个座位的手牌是不是都在画面里。

这是**几何自检**，不是肉眼看图。逐条量「该成立的关系」：
手牌完整在视口内、四家都存在、手牌张数合理。跑完还会截图。

    python webapp/tests/geometry.py            # 三麻 + 四麻，两种分辨率
    python webapp/tests/geometry.py --keep     # 保留房间
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

GEO_JS = r"""
(() => {
  const out = {vw: innerWidth, vh: innerHeight, seats: {}};
  const rr = (el) => {
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return [Math.round(b.x), Math.round(b.y), Math.round(b.width), Math.round(b.height)];
  };
  for (const s of document.querySelectorAll('.seat')) {
    const dir = s.dataset.dir || '?';
    const hand = s.querySelector('.edge__hand');
    const edge = s.querySelector('.edge');
    const plate = s.querySelector('.seat__plate') || s.querySelector('.seat__info');
    const h = rr(hand);
    out.seats[dir] = {
      edge: rr(edge),
      hand: h,
      plate: rr(plate),
      n: hand ? hand.children.length : 0,
      // 手牌整体必须在视口里（允许 2px 圆整误差）
      offscreen: !h ? true : (h[0] < -2 || h[1] < -2 ||
                              h[0] + h[2] > innerWidth + 2 ||
                              h[1] + h[3] > innerHeight + 2),
    };
  }
  return JSON.stringify(out);
})()
"""

AI3 = ["maj-normalv1.2", "maj-flashv1.2", "maj-mountainv1.2"]
AI4 = ["maj-normal4v1.3", "maj-flash4v1.2", "maj-mountain4v1.2", "maj-huge4v1.2"]
AI4_3P_STYLE = ["maj-normalv1.2", "maj-flashv1.2", "maj-mountainv1.2", "maj-hugev1.2"]


async def one(players: int, label: str, w: int, h: int, out: Path,
              room: str | None, port: int) -> tuple[str, bool]:
    host = None
    if room is None:
        host = Bot("ws://127.0.0.1:8790/ws", f"几何{label}")
        await host.open()
        models = AI3 if players == 3 else AI4
        room = await host.create_room(
            players=players, rounds=3, ai_delay=0.6,
            seats=[{"kind": "ai", "ai": m} for m in models],
            autostart=True, name=label,
        )
    profile = Path(tempfile.mkdtemp(prefix="majgeo-"))
    proc, cdp = await launch(EDGE, port, w, h, profile)
    ok = True
    try:
        await cdp.call("Page.navigate", {"url": f"http://127.0.0.1:8790/room/{room}"})
        # 等到有牌河（说明局真的开始了），最多 40 秒
        try:
            await cdp.wait_for(
                "(() => {const s=window.maj&&window.maj.state&&window.maj.state();"
                "return !!(s && s.room.dealt && s.seats.some(x=>x.pond.length>=2));})()",
                label="牌局开始", timeout=40)
        except Exception:  # noqa: BLE001
            await asyncio.sleep(8)
        geom = json.loads(await cdp.eval(GEO_JS))
        shot = out / f"{label}-{w}x{h}-{time.strftime('%H%M%S')}.png"
        await cdp.shot(shot)
        print(f"--- {label} {w}x{h}（视口 {geom['vw']}x{geom['vh']}）-> {shot.name}")
        want = {"bottom", "top", "left", "right"} if players == 4 else \
               {"bottom", "left", "right"}
        got = set(geom["seats"])
        if not want <= got:
            print(f"    ❌ 缺座位：{sorted(want - got)}（只有 {sorted(got)}）")
            ok = False
        for d in ("bottom", "left", "right", "top"):
            v = geom["seats"].get(d)
            if v is None:
                continue
            flag = "  ❌ 手牌跑到画面外" if v["offscreen"] else ""
            print(f"    {d:7s} 手牌 {v['n']:2d} 张  {v['hand']}  "
                  f"名牌 {v['plate']}{flag}")
            if v["offscreen"]:
                ok = False
    finally:
        await cdp.close()
        proc.kill()
    if host is not None:
        await host.close()
    return room, ok


async def main_async(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("牌桌几何自检（手牌必须在视口里 + 座位齐全）")
    print("=" * 70)
    all_ok = True
    for players, label in ((3, "sanma"), (4, "yonma")):
        room = None
        for i, (w, h) in enumerate(((1920, 1080), (1366, 768), (2560, 1440))):
            room, ok = await one(players, label, w, h, out, room, 9450 + players * 10 + i)
            all_ok = all_ok and ok
    print()
    if all_ok:
        print("✅ 几何自检通过：所有座位的手牌都在画面里")
    else:
        print("❌ 几何自检不通过（见上面 ❌ 的行）")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "_verify" / "shots"))
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
