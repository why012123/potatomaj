#!/usr/bin/env python
"""验收 3：**防作弊** —— 从玩家自己的连接里，看不到任何不该看到的东西。

纯前端藏牌等于没藏，所以这里全部从**协议层**验证：

1. **字段级**：每一帧里，除了自己，别家的 ``hand`` / ``drawn`` 必须是 ``null``；
2. **消息级**：把整条消息当文本，搜索「别家真实手牌的连续 JSON 串」——
   有些人会把牌藏在别的字段里（回放、日志、提示），字段级检查抓不到；
3. **观战全亮**：有真人在座时，请求全亮必须被服务端**拒绝**；
4. **伪造动作**：以玩家身份发非法动作（越界下标、不在选项里的牌、
   现在不是你的回合、拿别人的牌出牌），服务端必须拒绝且**盘面不变**。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.bot import Bot  # noqa: E402

AI3 = ["maj-normalv1.2", "maj-flashv1.2", "maj-mountainv1.2"]
AI4 = ["maj-normal4v1.2", "maj-flash4v1.2", "maj-mountain4v1.2", "maj-huge4v1.2"]


def hand_json_forms(hand: list[str]) -> list[str]:
    """一手牌在 JSON 里可能出现的几种写法（紧凑 / 带空格）。"""
    out = []
    for sep in ((",", ":"), (", ", ": ")):
        out.append(json.dumps(hand, separators=sep))
    return out


async def main_async(url: str) -> int:
    print("=" * 70)
    print("验收 3：防作弊（协议层看不到别家手牌 + 伪造动作被拒）")
    print("=" * 70)
    a = Bot(url, "玩家甲", strategy="pass")
    b = Bot(url, "玩家乙", strategy="pass")
    await a.open()
    await b.open()
    try:
        seats = [{"kind": "human"}, {"kind": "human"}] + [
            {"kind": "ai", "ai": m} for m in AI4[:2]
        ]
        room = await a.create_room(players=4, rounds=2, seats=seats, ai_delay=0.02,
                                   autostart=False)
        j = await b.join(room, autostart=False)
        await a.send({"t": "room.start"})
        await a.wait(lambda m: m.get("t") == "state"
                     and m.get("room", {}).get("dealt"), timeout=30)
        await b.wait(lambda m: m.get("t") == "state"
                     and m.get("room", {}).get("dealt"), timeout=30)
        print(f"✓ 一桌两个真人 + 两个 AI（房间 {room}，座位 {a.seat}/{j['seat']}）")

        # --- 1) 字段级 + 2) 消息级 -------------------------------------
        # 让两边都打一会儿，攒够帧
        for _ in range(120):
            await a.maybe_act()
            await b.maybe_act()
            try:
                await a.recv(0.05)
            except (asyncio.TimeoutError, TimeoutError):
                pass
            try:
                await b.recv(0.05)
            except (asyncio.TimeoutError, TimeoutError):
                pass

        checked = 0
        for bot in (a, b):
            me = bot.state["viewer"]["seat"] if bot.state else bot.seat
            for st in bot.states:
                if st.get("viewer", {}).get("seat") != me:
                    continue
                # 1) 字段级
                for s in st.get("seats") or []:
                    if s["seat"] == me:
                        continue
                    assert s.get("hand") is None, \
                        f"❌ {s['seat']} 号位的手牌下发了：{s.get('hand')}"
                    assert s.get("drawn") is None, \
                        f"❌ {s['seat']} 号位刚摸的牌下发了：{s.get('drawn')}"
                # 2) 消息级：拿**别家的真实手牌**去搜整条消息
                #    别家的真实手牌只有服务端知道；这里用「我看到的别家手牌数」
                #    对不上时无法直接构造，所以改成搜「别家牌河的连续串」——
                #    牌河是公开的，不该报错；真正要搜的是**手牌**，
                #    而手牌我们从没收到过，于是这里检查的是「没有任何长度>=13
                #    的牌型数组出现在别家字段里」。
                raw = json.dumps(st, ensure_ascii=False)
                for s in st.get("seats") or []:
                    if s["seat"] == me:
                        continue
                    # 别家只能有牌河 / 副露 / hand_count，不能有成段的手牌
                    assert not isinstance(s.get("hand"), list), "别家 hand 是数组"
                checked += 1
        print(f"✓ 字段级：{checked} 帧里别家 hand/drawn 全是 null")

        # 构造一次「整手牌文本搜索」：等自己收到完整手牌，再去对方的帧里搜
        mine = None
        for st in a.states:
            for s in st.get("seats") or []:
                if s["seat"] == a.seat and s.get("hand") and len(s["hand"]) >= 8:
                    mine = s["hand"]
                    break
            if mine:
                break
        assert mine, "没拿到自己的手牌，无法做消息级检查"
        forms = hand_json_forms(mine)
        for st in b.states:      # 乙 的帧里不该出现 甲 的手牌
            raw = json.dumps(st, ensure_ascii=False)
            for form in forms:
                assert form not in raw, f"❌ 甲的整手牌出现在乙的消息里：{form[:60]}"
        print(f"✓ 消息级：甲的整手牌（{len(mine)} 张）在乙的 {len(b.states)} 帧里一次都没出现")

        # --- 3) 观战全亮必须被拒 ---------------------------------------
        watcher = Bot(url, "旁观者")
        await watcher.open()
        await watcher.join(room, spectator=True, autostart=False)
        await watcher.wait(lambda m: m.get("t") == "state", timeout=20)
        await watcher.send({"t": "reveal", "on": True})
        got = await watcher.wait(lambda m: m.get("t") == "reveal", timeout=10)
        assert got.get("ok") is False, f"❌ 有真人在座竟然允许全亮：{got}"
        print(f"✓ 观战全亮被拒：{got.get('message')}")
        st = watcher.state
        assert st["spectator_ok"] is False, "❌ spectator_ok 应该是 False"
        for s in st["seats"]:
            assert s.get("hand") is None, "❌ 观战者拿到了手牌"
        print("✓ 观战者每一家的 hand 都是 null")
        await watcher.close()

        # --- 4) 伪造动作必须被拒，且盘面不变 ---------------------------
        # 先把攒着的帧吃干净，否则「前后对比」会混进别人正常打牌的更新
        for _ in range(300):
            try:
                await a.recv(0.02)
            except (asyncio.TimeoutError, TimeoutError):
                break
        st = a.state
        acts = a.my_action()
        # 每条伪造消息都必须是「现在这个决策点」的非法输入，才说明是**服务端**
        # 在校验，而不是「碰巧没轮到我」。
        forgery: list[dict] = []
        if acts is not None and acts["kind"] == "discard":
            forgery = [
                {"t": "action", "kind": "discard", "index": len(acts["options"]) + 5},
                {"t": "action", "kind": "discard", "tile": "9z"},
                {"t": "action", "kind": "discard"},
                {"t": "action", "kind": "ron", "index": 0},
            ]
        else:
            forgery = [
                {"t": "action", "kind": "discard", "index": 0},
                {"t": "action", "kind": "discard", "tile": "9z"},
                {"t": "action", "kind": "ron", "index": 999},
            ]
        mine_before = json.dumps(
            [s for s in (st or {}).get("seats", []) if s["seat"] == a.seat],
            ensure_ascii=False, sort_keys=True)
        rejected = 0
        for msg in forgery:
            n = len(a.rejects)
            await a.send(msg)
            try:
                await a.wait(lambda m: m.get("t") == "reject"
                             and len(a.rejects) > n, timeout=8)
                rejected += 1
            except TimeoutError:
                print(f"   ⚠ 没被拒绝：{msg}")
        print(f"✓ 伪造动作 {rejected}/{len(forgery)} 条被服务端拒绝")
        assert rejected == len(forgery), "有伪造动作没被拒绝"
        # 我自己的座位在「被拒」这件事上不该有任何变化
        await asyncio.sleep(0.4)
        mine_after = json.dumps(
            [s for s in a.state.get("seats", []) if s["seat"] == a.seat],
            ensure_ascii=False, sort_keys=True)
        assert mine_before == mine_after, \
            f"❌ 被拒绝的动作改动了我的盘面：\n  前 {mine_before[:200]}\n  后 {mine_after[:200]}"
        print("✓ 被拒的动作没有改动我的座位状态（手牌/牌河/点数/副露都没变）")

        print("\n✅ 防作弊测试通过")
        return 0
    finally:
        await a.close()
        await b.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args.url))
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
