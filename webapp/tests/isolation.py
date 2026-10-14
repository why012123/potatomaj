#!/usr/bin/env python
"""验收 2：**房间隔离** —— A 房的数据不许出现在 B 房的消息里。

做法：两个房间同时跑不同的牌局，把两边收到的**每一条原始文本**都存下来，
然后做两类检查：

1. **房间号**：A 房的房间号不许出现在 B 房的任何一条消息里（反之亦然）
2. **牌张序列**：把每条消息当**连续 JSON** 看，检查 B 房的消息里有没有出现
   A 房**独有的**牌张串（不是简单的子串搜索 —— 那样会误报，因为
   ``"1m"`` 这种短串会跨字段出现）

判定用的是「A 房某一家手里/河里的整段牌型（连着的 JSON 数组）」，
这种串出现在别的房间几乎不可能是巧合。
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

AI4 = ["maj-normal4v1.2", "maj-flash4v1.2", "maj-mountain4v1.2", "maj-huge4v1.2"]
AI3 = ["maj-normalv1.2", "maj-flashv1.2", "maj-mountainv1.2"]


async def build(url: str, who: str, players: int, ai: list[str]) -> tuple[Bot, str]:
    b = Bot(url, who, strategy="random")
    await b.open()
    seats = [{"kind": "human"}] + [{"kind": "ai", "ai": m} for m in ai[: players - 1]]
    room = await b.create_room(players=players, rounds=players, seats=seats,
                               ai_delay=0.02, autostart=True)
    return b, room


def unique_hands(states: list[dict]) -> set[str]:
    """从一堆快照里抽出「整段手牌 / 整段牌河」的 JSON 串。

    只取长度足够的（>=6 张），免得短串巧合命中。
    """
    out: set[str] = set()
    for st in states:
        for s in st.get("seats") or []:
            for key in ("hand", "pond"):
                v = s.get(key)
                if isinstance(v, list) and len(v) >= 6:
                    out.add(json.dumps(v, separators=(",", ":")))
            melds = s.get("melds")
            if melds:
                out.add(json.dumps(melds, separators=(",", ":")))
    return out


async def main_async(url: str, seconds: float) -> int:
    print("=" * 70)
    print("验收 2：房间隔离（A 房数据不许出现在 B 房消息里）")
    print("=" * 70)
    a, room_a = await build(url, "甲", 3, AI3)
    b, room_b = await build(url, "乙", 4, AI4)
    print(f"✓ 两个房间同时在跑：A={room_a}（三麻） B={room_b}（四麻）")

    async def drive(bot: Bot, secs: float) -> None:
        end = asyncio.get_running_loop().time() + secs
        while asyncio.get_running_loop().time() < end:
            await bot.maybe_act()
            try:
                await bot.recv(0.05)
            except (asyncio.TimeoutError, TimeoutError):
                pass

    try:
        await asyncio.gather(drive(a, seconds), drive(b, seconds))
    finally:
        pass

    try:
        # 只取**这个房间**的消息。Bot 自己维护的是自己那条连接。
        a_states = [s for s in a.states if s.get("room", {}).get("id") == room_a]
        b_states = [s for s in b.states if s.get("room", {}).get("id") == room_b]
        print(f"  A 房快照 {len(a_states)} 帧，B 房快照 {len(b_states)} 帧")
        assert a_states and b_states, "有一边没收到快照"

        # 1) 房间号不许串
        bad = []
        for st in a.states:
            raw = json.dumps(st, ensure_ascii=False)
            if room_b in raw:
                bad.append(("A 房里出现了 B 的房间号", st.get("seq")))
        for st in b.states:
            raw = json.dumps(st, ensure_ascii=False)
            if room_a in raw:
                bad.append(("B 房里出现了 A 的房间号", st.get("seq")))
        assert not bad, f"❌ 房间号串了：{bad[:3]}"
        print("✓ 房间号零交叉")

        # 2) A 房独有的牌张串不许出现在 B 房里
        a_hands = unique_hands(a_states)
        b_raw = "\n".join(json.dumps(s, ensure_ascii=False) for s in b.states)
        leaked = [h for h in a_hands if h in b_raw]
        assert not leaked, f"❌ 泄漏了 A 房的牌：{leaked[:2]}"
        print(f"✓ A 房的 {len(a_hands)} 段独有牌型串，一段都没出现在 B 房消息里")

        b_hands = unique_hands(b_states)
        a_raw = "\n".join(json.dumps(s, ensure_ascii=False) for s in a_states)
        leaked2 = [h for h in b_hands if h in a_raw]
        assert not leaked2, f"❌ 泄漏了 B 房的牌：{leaked2[:2]}"
        print(f"✓ 反向同理：B 房的 {len(b_hands)} 段也没进 A 房")

        # 3) 每帧里只有自己那家有手牌
        for tag, states in (("A", a_states), ("B", b_states)):
            for st in states:
                me = st["viewer"]["seat"]
                for s in st["seats"]:
                    if s["seat"] != me and s.get("hand") is not None:
                        raise AssertionError(
                            f"❌ {tag} 房第 {st['seq']} 帧里 {s['seat']} 号位的手牌下发了"
                        )
        print("✓ 每一帧里「别家手牌」都是 null")

        print("\n✅ 隔离测试通过：A 房与 B 房零交叉")
        return 0
    finally:
        await a.close()
        await b.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args.url, args.seconds))
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
