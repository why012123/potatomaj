#!/usr/bin/env python3
"""测试用的假 mjai 子进程 —— 不依赖网络、不依赖任何第三方 AI。

它故意只实现**最小**的 mjai 状态机：
- 自己摸牌 -> 打刚摸的那张（摸切）
- 别家打牌 -> ``{"type":"none"}``（不鸣牌、不荣和）
- 立直 -> 不立直
这样一局能完整跑完，而且能验证桥的几件事：

* 事件顺序（start_kyoku / tsumo / dahai / ...）
* 自己的 ``tsumogiri`` 必须和刚摸的牌一致（不一致就退出，测桥的翻译）
* 外部动作 -> ``decision.options`` 下标
* 非法动作（``FAKE_MJAI_MODE=illegal``）要能触发协议错误，而不是静默 return 0
* 超时（``sleep``）/ 崩溃（``crash``）能触发降级

环境变量：
    FAKE_MJAI_MODE = normal | illegal | sleep | crash
    FAKE_MJAI_BATCH = 1   # 一行一个 JSON 数组（Akagi 的 Mortal bot 格式）
"""

from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_MJAI_MODE", "normal")
BATCH = os.environ.get("FAKE_MJAI_BATCH") == "1"

NAMES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("ESWN PFC".replace(" ", ""))

player_id = 0
last_draw: str | None = None
hand: dict[str, int] = {}


def _remove(tile: str) -> None:
    if hand.get(tile, 0) > 0:
        hand[tile] -= 1


def _first_tile() -> str:
    for t, n in hand.items():
        if n > 0:
            return t
    return "1m"


def _illegal_tile() -> str:
    for t in NAMES:
        if hand.get(t, 0) == 0:
            return t
    return "1m"


def _own_meld(ev: dict) -> None:
    for t in ev.get("consumed") or []:
        _remove(t)
    # 被鸣的那张不在自己手里，不用减


def handle(ev: dict) -> dict | None:
    global player_id, last_draw, hand
    typ = ev.get("type")
    if ev.get("can_act") is False:
        # 真实 mjai bot（Mortal）会遵守这个字段；假 bot 也遵守，确保桥
        # 只把「最后一条」当决策触发。
        return None
    if typ == "start_game":
        player_id = int(ev.get("id", 0))
        hand = {}
        last_draw = None
        return None
    if typ == "start_kyoku":
        hand = {}
        tehais = ev.get("tehais") or []
        if player_id < len(tehais):
            for t in tehais[player_id]:
                if t != "?":
                    hand[t] = hand.get(t, 0) + 1
        last_draw = None
        return None
    if typ == "tsumo":
        if int(ev.get("actor", -1)) != player_id:
            return None
        last_draw = ev.get("pai")
        if MODE == "sleep":
            time.sleep(30)
        if MODE == "crash":
            sys.stderr.write("fake bot: intentional crash\n")
            sys.stderr.flush()
            raise SystemExit(3)
        if last_draw == "?":
            return {"type": "none"}
        if MODE == "illegal":
            # 打一张手里没有的牌 -> 适配层必须在 options 里找不到而报错
            pai = _illegal_tile()
            return {"type": "dahai", "actor": player_id, "pai": pai,
                    "tsumogiri": False}
        hand[last_draw] = hand.get(last_draw, 0) + 1
        return {"type": "dahai", "actor": player_id, "pai": last_draw,
                "tsumogiri": True}
    if typ == "dahai":
        if int(ev.get("actor", -1)) == player_id:
            if ev.get("tsumogiri") and last_draw is not None \
                    and ev.get("pai") != last_draw:
                sys.stderr.write(
                    f"fake bot: tsumogiri {ev.get('pai')} != drawn {last_draw}\n"
                )
                sys.stderr.flush()
                raise SystemExit(4)
            _remove(ev.get("pai", ""))
            last_draw = None
            return None
        return {"type": "none"}
    if typ in ("chi", "pon", "daiminkan", "ankan", "kakan"):
        if int(ev.get("actor", -1)) == player_id:
            _own_meld(ev)
            last_draw = None
            return {"type": "dahai", "actor": player_id,
                    "pai": _first_tile(), "tsumogiri": False}
        return None
    return None


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"fake bot: bad json {line[:100]}\n")
            sys.stderr.flush()
            raise SystemExit(2)
        if BATCH:
            events = data if isinstance(data, list) else [data]
            resp = None
            for ev in events:
                r = handle(ev)
                if r is not None:
                    resp = r
            if resp is None:
                resp = {"type": "none"}
            print(json.dumps(resp, ensure_ascii=False), flush=True)
        else:
            if not isinstance(data, dict):
                continue
            resp = handle(data)
            if resp is not None:
                print(json.dumps(resp, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
