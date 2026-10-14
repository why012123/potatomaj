"""自动接管 UI 里的 external 座位：轮到我时自己出牌。

给「一直监听到结束」用的：它轮询 ``/api/state``，发现待决策的是我的座位，
就用训练好的神经网络（或教师）决定，再 ``POST /api/act``。

用法::

    python -m aimodel.ui.autopilot                 # 自动找 external 座位
    python -m aimodel.ui.autopilot --policy teacher
    python -m aimodel.ui.autopilot --seat 1 --poll 0.4

它每一手都会把局面和自己的选择打在终端里，所以打完能回看整局。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from .. import sanma as S
from ..agents import HeuristicAgent, MLAgent
from ..engine import Decision, View
from ..net import Net

DEFAULT_URL = "http://127.0.0.1:8765"
WEIGHTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "weights" / "policy.npz"

_WIND = {"东": S.EAST, "南": S.SOUTH, "西": S.WEST, "北": S.NORTH}


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.loads(r.read().decode())


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def idx(tile_id: str) -> int:
    return S.parse_hand(tile_id).index(1)


def counts(ids) -> list[int]:
    out = [0] * 34
    for t in ids or []:
        for i, v in enumerate(S.parse_hand(t)):
            out[i] += v
    return out


def build_view(state: dict, seat: int, pending: dict) -> tuple[View, Decision]:
    """把 JSON 快照还原成引擎侧的 View + Decision（只含我这个座位看得到的东西）。"""
    players = state["players_detail"]
    me = players[seat]
    melds = [S.Meld(m["kind"], idx(m["tile"])) for m in me["melds"]]
    view = View(
        player=seat,
        hand=counts(me["hand"]),
        drawn=idx(pending["drawn"]) if pending.get("drawn") else None,
        melds=melds,
        hand_red=counts(me.get("red")),
        discards=[[idx(t) for t in p["pond"]] for p in players],
        melds_all=[[S.Meld(m["kind"], idx(m["tile"])) for m in p["melds"]] for p in players],
        kita=[p["kita"] for p in players],
        scores=[p["score"] for p in players],
        dora_indicators=[idx(d) for d in state["dora"]],
        round_wind=_WIND.get(str(state.get("round_wind", "东"))[0], S.EAST),
        seat_wind=_WIND.get(str(me["seat_wind"])[0], S.EAST),
        wall_left=int(state["wall_left"]),
        turn=int(state["turn"]),
        riichi=[bool(p["riichi"]) for p in players],
        ippatsu=[bool(p["ippatsu"]) for p in players],
        is_dealer=bool(me["is_dealer"]),
        last_discard=None,
        rule=S.DEFAULT_RULE,
    )
    kind = pending["kind"]
    if kind == "discard":
        options = [idx(o["tile"]) for o in pending["options"]]
    else:
        options = [str(o["label"]) for o in pending["options"]]
    tile = idx(pending["tile"]) if pending.get("tile") else None
    return view, Decision(kind, options, tile)


def describe(view: View, decision: Decision, choice: int) -> str:
    if decision.kind == "discard":
        t = int(decision.options[choice])
        return f"打 {S.tile_cn(t)}"
    return f"{decision.kind} -> {decision.options[choice]}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="自动接管 external 座位")
    ap.add_argument("-u", "--url", default=DEFAULT_URL)
    ap.add_argument("--seat", type=int, default=-1, help="座位号，默认自动找 external")
    ap.add_argument("--policy", choices=["ml", "teacher"], default="ml")
    ap.add_argument("--poll", type=float, default=0.4)
    ap.add_argument("--quiet", action="store_true", help="只在结束时打印")
    ap.add_argument(
        "--once", action="store_true",
        help="打完这一场就退出（默认：继续守着，等你开下一局）",
    )
    args = ap.parse_args(argv)

    state = get(args.url, "/api/state?seat=0")
    seat = args.seat
    if seat < 0:
        seat = next(
            (int(s["index"]) for s in state["seats"] if s["type"] == "external"), -1
        )
    if seat < 0:
        print("❌ 这桌没有 external 座位")
        sys.exit(2)

    agent = None
    if args.policy == "ml" and WEIGHTS.exists():
        agent = MLAgent(Net.load(WEIGHTS), seed=7, greedy=True)
        print(f"🤖 托管 P{seat}（神经网络：{WEIGHTS.name}）")
    else:
        agent = HeuristicAgent(7)
        print(f"🤖 托管 P{seat}（启发式教师）")

    last_seed = None
    reported: set = set()
    plays = 0
    while True:
        try:
            st = get(args.url, f"/api/state?seat={seat}")
        except urllib.error.URLError:
            print("连接断了，收工")
            return
        if st.get("seed") != last_seed:
            last_seed = st.get("seed")
            print(f"\n══ 新的一局（种子 {last_seed}） ══")
        if st.get("finished"):
            if last_seed not in reported:
                reported.add(last_seed)
                r = st.get("result") or {}
                print("\n🏁 对局结束")
                if r.get("scores"):
                    for i, s in enumerate(st["seats"]):
                        print(
                            f"   {s['name']}: {r['scores'][i]} 分"
                            f"  第 {r['ranks'][i] + 1} 位  段位分 {r['rank_points'][i]:+.1f}"
                        )
                if args.once:
                    return
                print("   （继续守着，你点「再来一局」我就接着打）")
            time.sleep(1.0)
            continue
        pend = st.get("pending")
        if pend and pend.get("yours"):
            view, decision = build_view(st, seat, pend)
            try:
                choice = agent.decide(decision, view)
            except Exception as exc:  # 兜底：出问题也别把局卡住
                print(f"   ⚠️ 决策异常 {exc!r}，打第一张")
                choice = 0
            if not (0 <= choice < len(decision.options)):
                choice = 0
            try:
                res = post(args.url, "/api/act", {"seat": seat, "index": int(choice)})
            except urllib.error.URLError as exc:
                print(f"   ⚠️ 提交失败 {exc}")
                time.sleep(1.0)
                continue
            plays += 1
            if not args.quiet:
                sh = st.get("hint") or {}
                extra = f"  (向听 {sh.get('shanten')}, 进张 {sh.get('ukeire')})" if sh else ""
                print(
                    f"  [{st['turn']}巡 剩{st['wall_left']}] {describe(view, decision, choice)}"
                    f"{extra}{'' if res.get('ok') else '  ← 提交被拒'}"
                )
            time.sleep(0.15)
            continue
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
