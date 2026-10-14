"""给我（AI 助手）用的命令行接口。

浏览器里那局是「你 vs 我 vs 脚本 AI」，我的座位是 ``external``：
轮到我时对局会挂起等我出牌，这个脚本就是我的操作面板。

用法::

    python -m aimodel.ui.assistant              # 看现在什么情况、该我出什么
    python -m aimodel.ui.assistant -p 3p        # 打 3p
    python -m aimodel.ui.assistant -a riichi    # 立直 / 碰 / 过 …（按提示里的名字）
    python -m aimodel.ui.assistant --watch      # 一直等到轮到我，然后打印局面
    python -m aimodel.ui.assistant --board      # 只看牌桌（不等着出牌）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8765"

KIND_HINT = {
    "discard": "打哪张",
    "riichi": "要不要立直",
    "call": "要不要鸣牌",
    "ron": "要不要荣和",
    "tsumo": "要不要自摸",
    "kita": "要不要拔北",
    "kan": "要不要杠",
    "kyuushu": "要不要九种九牌流局",
}


def get(url: str, path: str) -> dict:
    try:
        with urllib.request.urlopen(url + path, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as exc:
        print(f"❌ 连不上 {url}（UI 起了吗？）：{exc}")
        sys.exit(2)


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def find_seat(state: dict, want: str) -> int:
    for s in state["seats"]:
        if s["type"] == want:
            return int(s["index"])
    return 0


def show_board(state: dict, seat: int) -> None:
    print(f"── 第 {state.get('hand_no', '?')} 局 · {state.get('turn', '?')} 巡 · "
          f"剩余 {state.get('wall_left', '?')} 张 · {state.get('round_wind', '?')}场 ──")
    print("宝牌指示: " + " ".join(state.get("dora") or []))
    for p in state["players_detail"]:
        me = p.get("hand") is not None
        tag = "★" if p["riichi"] else " "
        flags = f"{p['seat_wind']}家" + ("（亲）" if p["is_dealer"] else "")
        melds = " ".join(m["label"] for m in p["melds"]) or "—"
        line = (
            f"  P{state['players'].index(p)} {flags} {p['score']:>6d} "
            f"手牌{p['hand_count']:>3d}张 副露[{melds}] 北{p['kita']} {tag}"
        )
        print(line)
        if p["pond"]:
            print(f"      牌河: {' '.join(p['pond'])}")
        if me:
            red = set(p.get("red") or [])
            hand = " ".join(("赤" if t in red else "") + t for t in p["hand"])
            print(f"      ★我的手牌: {hand}")


def show_pending(state: dict, seat: int) -> bool:
    pend = state.get("pending")
    if not pend:
        who = state.get("waiting_for", "…")
        if state.get("finished"):
            print("🏁 这局已经结束了")
        else:
            print(f"⏳ 现在等 {who} 出牌（不是我）")
        return False
    if not pend["yours"]:
        other = next(
            (s["name"] for s in state["seats"] if s["index"] == pend["seat"]), "?"
        )
        print(f"⏳ 等 {other} 决策：{pend['kind_cn']}（已等 {pend['waited']}s）")
        return False
    print(f"🎯 轮到我：{KIND_HINT.get(pend['kind'], pend['kind_cn'])}")
    if pend.get("your_hand"):
        red = set(state["players_detail"][seat].get("red") or [])
        hand = " ".join(("赤" if t in red else "") + t for t in pend["your_hand"])
        print(f"   我的手牌: {hand}")
        if pend.get("drawn"):
            print(f"   刚摸到: {pend['drawn']}")
    for o in pend["options"]:
        extra = f"  ({o['cn']})" if o.get("cn") and o["cn"] != o["label"] else ""
        print(f"   [{o['index']}] {o['label']}{extra}")
    hint = state.get("hint")
    if hint:
        print(f"   向听 {hint['shanten']}  进张 {hint['ukeire']} 枚")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AI 助手的外部座位操作面板")
    ap.add_argument("-u", "--url", default=DEFAULT_URL, help="UI 地址")
    ap.add_argument("-p", "--play", help="打哪张牌，如 3p / 东")
    ap.add_argument("-a", "--act", help="其它动作：riichi / pon / kan / pass / yes / no …")
    ap.add_argument("--index", type=int, help="直接给选项下标")
    ap.add_argument("--board", action="store_true", help="只看牌桌")
    ap.add_argument("--watch", action="store_true", help="等到轮到我再显示")
    ap.add_argument("--interval", type=float, default=1.5, help="watch 的轮询间隔")
    args = ap.parse_args(argv)

    state = get(args.url, "/api/state?seat=0")
    seat = find_seat(state, "external")

    if args.watch:
        while True:
            state = get(args.url, f"/api/state?seat={seat}")
            pend = state.get("pending")
            if state.get("finished") or (pend and pend.get("yours")):
                break
            time.sleep(args.interval)

    state = get(args.url, f"/api/state?seat={seat}")
    if args.board or not (args.play or args.act or args.index is not None):
        show_board(state, seat)
        print()
        show_pending(state, seat)
        if not (args.play or args.act or args.index is not None):
            return

    if args.index is not None:
        payload = {"seat": seat, "index": args.index}
    elif args.play:
        payload = {"seat": seat, "label": args.play}
    else:
        payload = {"seat": seat, "label": args.act}
    res = post(args.url, "/api/act", payload)
    if res.get("ok"):
        print(f"✅ 已提交：{payload.get('label', payload.get('index'))}")
    else:
        print(f"❌ 提交失败：{res.get('error', res)}")


if __name__ == "__main__":
    main()
