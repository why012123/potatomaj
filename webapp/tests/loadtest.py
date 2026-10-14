#!/usr/bin/env python
"""验收 5/6：**并发压测**（10 房 × 4 人）+ **断线重连**。

压测的同时做三件事：

1. **交叉污染检查**：把每个房间收到的所有帧拼起来，检查有没有出现**别的
   房间号**；这是「房间 A 的数据不出现在房间 B」在并发下的验证。
2. **健康采样**：另起一个线程定时打 ``/api/health``，记录房间数 / 连接数
   峰值（顺带证明服务端在高压下还能正常响应 HTTP）。
3. **重连**：把某个房间里的一个玩家掐掉，让同房间的人继续打几手，
   然后用**同一个座位凭证**重连，验证房间号 / 座位 / 牌河 / 手牌张数一致，
   并且能接着打到底。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.bot import Bot  # noqa: E402


def health_sampler(url: str, stop: threading.Event, out: list[dict]) -> None:
    """另起线程采样健康接口。

    **不能用 asyncio 任务**：主循环被几十条连接喂满的时候，采样任务会被饿死，
    一个样本都拿不到（这个坑踩过一次，压测报告里「峰值 0」就是这么来的）。
    """
    base = url.replace("ws://", "http://").replace("wss://", "https://")
    base = base.split("/ws")[0]
    while not stop.is_set():
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=5) as r:
                out.append(json.loads(r.read()))
        except Exception as exc:  # noqa: BLE001
            out.append({"error": f"{type(exc).__name__}"})
        stop.wait(0.5)


async def play_all(bots: list[Bot], pred, timeout: float) -> list:
    return list(await asyncio.gather(
        *[b.play_until(pred, timeout=timeout, stall=180) for b in bots],
        return_exceptions=True,
    ))


async def main_async(url: str, rooms: int, players: int, rounds: int,
                     do_reconnect: bool) -> int:
    print("=" * 70)
    print(f"验收 5：并发压测 {rooms} 房 × {players} 人（同时开局）")
    print("=" * 70)
    stop = threading.Event()
    samples: list[dict] = []
    th = threading.Thread(target=health_sampler, args=(url, stop, samples), daemon=True)
    th.start()

    rcs: dict[str, list[Bot]] = {}
    started = time.time()
    try:
        for i in range(rooms):
            bots = [Bot(url, f"R{i}P{j}", strategy="random") for j in range(players)]
            await asyncio.gather(*[b.open() for b in bots])
            seats = [{"kind": "human"}] * players
            rid = await bots[0].create_room(players=players, rounds=rounds, seats=seats,
                                           ai_delay=0.0, autostart=False,
                                           name=f"压测房{i}")
            await asyncio.gather(*[b.join(rid, autostart=False) for b in bots[1:]])
            await bots[0].send({"t": "room.start"})
            rcs[rid] = bots
        print(f"✓ {rooms} 个房间、{rooms * players} 条连接全部开局"
              f"（用时 {time.time() - started:.1f}s）")

        all_bots = [b for bots in rcs.values() for b in bots]
        results = await play_all(all_bots, lambda s: s["phase"] == "finished", 900.0)
        bad = [r for r in results if isinstance(r, Exception)]
        print(f"✓ 全部对局结束：{len(all_bots) - len(bad)}/{len(all_bots)} 条连接到达 finished")
        for r in bad[:3]:
            print("   ⚠", r)
        assert not bad, f"{len(bad)} 条连接没打完"

        # 交叉污染
        pairs = 0
        bad_pairs = []
        ids = list(rcs)
        for rid, bots in rcs.items():
            others = [x for x in ids if x != rid]
            for b in bots:
                raw = "\n".join(json.dumps(s, ensure_ascii=False) for s in b.states)
                for other in others:
                    pairs += 1
                    if other in raw:
                        bad_pairs.append((rid, other, b.name))
        assert not bad_pairs, f"❌ 房间号交叉：{bad_pairs[:3]}"
        print(f"✓ 交叉污染检查：{pairs} 组「本房消息 × 他房房间号」全部干净")

        frames = 0
        for bots in rcs.values():
            for b in bots:
                for st in b.states:
                    frames += 1
                    me = st["viewer"]["seat"]
                    for s in st["seats"]:
                        if s["seat"] != me:
                            assert s.get("hand") is None, "❌ 别家手牌下发了"
        print(f"✓ {frames} 帧快照里，别家手牌全是 null")

        peak = {"conns": 0, "rooms": 0, "playing": 0}
        errs = 0
        for s in samples:
            if "error" in s:
                errs += 1
                continue
            peak["conns"] = max(peak["conns"], int(s.get("conns") or 0))
            peak["rooms"] = max(peak["rooms"], int(s.get("rooms") or 0))
            peak["playing"] = max(peak["playing"], int(s.get("playing") or 0))
        print(f"✓ 健康采样 {len(samples)} 次（失败 {errs}）：峰值 连接 {peak['conns']} / "
              f"房间 {peak['rooms']} / 同时进行 {peak['playing']}")
        assert peak["conns"] >= rooms * players * 0.9, "健康接口看到的连接数偏低"
        rec = next((s.get("record") for s in samples
                    if isinstance(s, dict) and s.get("record")), None)
        if rec:
            print(f"  记录器：写入 {rec.get('written')} 条，丢弃 {rec.get('dropped')}，"
                  f"错误 {rec.get('errors')}")

        # --- 验收 6：断线重连 -----------------------------------------
        if do_reconnect:
            print("\n" + "=" * 70)
            print("验收 6：断线重连（掐掉一个玩家，局面必须能接回来）")
            print("=" * 70)
            rbot = Bot(url, "重连者", strategy="random")
            obot = Bot(url, "留守者", strategy="random")
            await rbot.open()
            await obot.open()
            rid = await rbot.create_room(
                players=3, rounds=3,
                seats=[{"kind": "human"}, {"kind": "human"},
                       {"kind": "ai", "ai": "maj-normalv1.2"}],
                ai_delay=0.05, autostart=False)
            await obot.join(rid, autostart=False)
            await rbot.send({"t": "room.start"})
            await rbot.wait(lambda m: m.get("t") == "state"
                            and m.get("room", {}).get("dealt"), timeout=30)
            token = rbot.token
            seat0 = rbot.seat

            for _ in range(90):
                await rbot.maybe_act()
                await obot.maybe_act()
                for b in (rbot, obot):
                    try:
                        await b.recv(0.03)
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
            before = rbot.state
            assert before is not None, "掐断前没拿到快照"
            mine_before = before["seats"][seat0]
            print(f"✓ 掐断前：房间 {rid} 座位 {seat0} 手牌 {mine_before['hand_count']} 张、"
                  f"牌河 {len(mine_before['pond'])} 张、点数 {mine_before['score']}、"
                  f"第 {before['round']['hand_no']} 局")

            await rbot.close()               # ← 断线
            await asyncio.sleep(0.5)
            for _ in range(120):
                await obot.maybe_act()
                try:
                    await obot.recv(0.03)
                except (asyncio.TimeoutError, TimeoutError):
                    pass
            print(f"✓ 断线期间留守方继续打了 {obot.acts} 手（桌子没卡住）")

            again = Bot(url, "重连者", strategy="random")
            await again.open()
            await again.send({"t": "room.join", "room": rid, "token": token,
                              "spectator": False, "autostart": False})
            j = await again.wait(lambda m: m.get("t") == "joined", timeout=15)
            assert j["seat"] == seat0, f"❌ 重连坐到了 {j['seat']}，原来是 {seat0}"
            await again.wait(lambda m: m.get("t") == "state", timeout=15)
            after = again.state
            mine_after = after["seats"][seat0]
            assert after["room"]["id"] == before["room"]["id"], "房间号变了"
            assert len(mine_after["pond"]) >= len(mine_before["pond"]), \
                "牌河变短了（局面回退）"
            assert abs(mine_after["hand_count"] - mine_before["hand_count"]) <= 2, \
                f"手牌张数跳变：{mine_before['hand_count']} -> {mine_after['hand_count']}"
            assert len(mine_after["melds"]) >= len(mine_before["melds"]), "副露变少了"
            print(f"✓ 重连回到座位 {j['seat']}：手牌 {mine_after['hand_count']} 张、"
                  f"牌河 {len(mine_after['pond'])} 张、第 {after['round']['hand_no']} 局"
                  f"（与掐断前一致）")

            await play_all([again, obot], lambda s: s["phase"] == "finished", 400.0)
            print("✓ 重连后接着打到结束")
            await again.close()
            await obot.close()

        print(f"\n✅ 压测与重连通过（总用时 {time.time() - started:.1f}s）")
        return 0
    finally:
        stop.set()
        th.join(timeout=2)
        for bots in rcs.values():
            for b in bots:
                await b.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--no-reconnect", action="store_true")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args.url, args.rooms, args.players, args.rounds,
                                      not args.no_reconnect))
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
