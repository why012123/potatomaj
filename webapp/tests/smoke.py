#!/usr/bin/env python
"""协议冒烟脚本：两个真人 + 一桌 AI，跑完一局，检查关键不变量。

这是**最小闭环**的自检：能不能建房、能不能入座、能不能打完一局、
结算数据齐不齐、别家手牌在协议层存不存在。

用法::

    .venv/Scripts/python -m webapp.tests.smoke
    .venv/Scripts/python -m webapp.tests.smoke --url ws://127.0.0.1:8790/ws
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.bot import Bot  # noqa: E402


async def main_async(url: str, players: int, rounds: int) -> int:
    print("=" * 70)
    print("最小闭环冒烟")
    print("=" * 70)
    a = Bot(url, "房主", strategy="riichi")
    b = Bot(url, "小美", strategy="random")
    await a.open()
    await b.open()
    try:
        # 座位表是完整一张桌子，第 0 格留给自己；其余给 AI
        seats = [{"kind": "human"}] + [
            {"kind": "ai", "ai": "maj-normalv1.2"} for _ in range(players - 1)
        ]
        room = await a.create_room(players=players, rounds=rounds, seats=seats,
                                   ai_delay=0.05, autostart=False)
        print(f"✓ 建房成功 房间号={room} 座位={a.seat}")
        j = await b.join(room, autostart=False)
        print(f"✓ 第二人入座 座位={j['seat']}")

        await a.send({"t": "room.start"})
        st = await a.wait(lambda m: m["t"] == "state" and m["phase"] == "playing"
                          and m["room"].get("dealt"), timeout=30)
        mine = st["seats"][st["viewer"]["seat"]]
        print(f"✓ 开局 人数={st['room']['players']} 我的手牌={len(mine['hand'] or [])} 张")

        leaks = [s["seat"] for s in st["seats"]
                 if s["seat"] != st["viewer"]["seat"] and s["hand"] is not None]
        assert not leaks, f"❌ 泄漏了别家手牌：{leaks}"
        print("✓ 别家手牌不下发（hand=null）")

        await asyncio.gather(
            a.play_until(lambda s: s["phase"] == "finished", timeout=400),
            b.play_until(lambda s: s["phase"] == "finished", timeout=400),
        )
        final = a.state
        assert final is not None and final["phase"] == "finished", \
            f"没结束：{final and final['phase']}"
        res = final.get("result")
        print(f"✓ 打完：{res and res.get('label')} 收支={res and res.get('deltas')}")
        m = final.get("match") or {}
        print(f"✓ 终局点数={m.get('scores')} 顺位={m.get('ranks')}")
        ag = (res or {}).get("agari") or []
        for e in ag:
            yaku = "、".join(y["name"] for y in e["yaku"]) or "（无真役）"
            dora = "、".join(f"{y['name']}{y['han']}" for y in e["dora"])
            print(f"   座位 {e['seat']}：{e['total_han']} 番 {e['fu']} 符 "
                  f"[{yaku}] {'宝牌 ' + dora if dora else ''} 手牌={len(e['hand'])} 张")
        print(f"  房主出了 {a.acts} 手，小美出了 {b.acts} 手")
        if a.errors or b.errors:
            print(f"⚠ 服务端错误：{a.errors + b.errors}")
        print("\n✅ 最小闭环通过")
        return 0
    finally:
        await a.close()
        await b.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args.url, args.players, args.rounds))
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        return 1
    except TimeoutError as exc:
        print(f"\n❌ 超时：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
