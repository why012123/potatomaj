"""跑一局三麻（可以自己在终端里打）。

用法::

    # 神经网络 vs 两个教师
    python -m aimodel.play

    # 自己下场打（0 号位是你）
    python -m aimodel.play --human

    # 指定三家
    python -m aimodel.play --agents teacher,random,ml

    # 只看结果，不看过程（批量自检）
    python -m aimodel.play --games 5 --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import sanma as S
from . import variants
from .agents import Agent, HeuristicAgent, HumanAgent, MLAgent, RandomAgent
from .engine import Game
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"


def make_agent(
    spec: str,
    seat: int,
    seed: int,
    net: Net | None,
    human: bool,
    human_input=input,
    style: str = "normal",
) -> Agent:
    spec = spec.strip().lower()
    if human and seat == 0:
        return HumanAgent(input_fn=human_input)
    if spec in ("teacher", "t"):
        return HeuristicAgent(seed, style=style)
    if spec in ("random", "r"):
        return RandomAgent(seed)
    # 其余一律当模型名（maj-flash / flash4 / ml ...）
    if spec in ("ml", "net", "model"):
        spec = "policy"
    try:
        name = variants.resolve(spec) if spec not in ("policy",) else "policy"
        path = variants.WEIGHTS_DIR / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return MLAgent(Net.load(path), seed=seed, greedy=True)
    except (KeyError, FileNotFoundError):
        if net is not None and spec in ("ml", "net", "policy"):
            return MLAgent(net, seed=seed, greedy=True)
        print(f"⚠️ 找不到模型 {spec}，{seat} 号位退化为教师")
        return HeuristicAgent(seed, style=style)


#: 人类玩家视角下不应该看到的信息（别人摸到什么牌）
_HIDDEN = re.compile(r"\] (摸|岭上) ")


def make_logger(show_all: bool = True, sink: list[str] | None = None):
    def log(msg: str) -> None:
        if sink is not None:
            sink.append(msg)
        if not show_all and _HIDDEN.search(msg):
            return
        print(msg)

    return log


def describe_result(r) -> str:
    lines = []
    for i, h in enumerate(r.hands, 1):
        who = "—"
        if h.kind == "tsumo":
            who = f"P{h.winner} 自摸 {h.agari.describe() if h.agari else ''}"
        elif h.kind == "ron":
            ws = "、".join(f"P{w}" for w in (h.winners or (h.winner,)))
            who = f"{ws} 荣和 {h.agari.describe() if h.agari else ''}（放铳 P{h.loser}）"
        elif h.kind == "draw":
            who = "流局 " + "".join("T" if t else "-" for t in h.tenpai)
        elif h.kind == "kyuushu":
            who = "九种九牌"
        else:
            who = h.note or "途中流局"
        lines.append(
            f"  {i:>2d}. {who}   点数变化 {h.deltas}   第 {h.turns} 巡"
        )
    return "\n".join(lines)


def print_summary(r) -> None:
    print("─" * 68)
    print("终局点数:", "  ".join(f"P{i} {s}" for i, s in enumerate(r.scores)))
    print("顺位:", "  ".join(f"P{i} {rk + 1}位" for i, rk in enumerate(r.ranks)))
    print("得失:", "  ".join(f"P{i} {d:+d}" for i, d in enumerate(r.deltas)))
    if r.rank_points:
        print(
            "段位分:",
            "  ".join(f"P{i} {rp:+.1f}" for i, rp in enumerate(r.rank_points)),
            "  （= (点数-40000)/1000 + 顺位点 + 额外段位分）",
        )
    if r.coins:
        print("铜币收支:", "  ".join(f"P{i} {c:+.0f}" for i, c in enumerate(r.coins)))
    for i in range(3):
        s = r.stats[i].summary()
        print(
            f"  P{i}: {s['局数']}局  和了 {s['和了率']:.0%}  放铳 {s['放铳率']:.0%}"
            f"  立直 {s['立直率']:.0%}  副露 {s['副露率']:.0%}  平均翻 {s['平均翻数']:.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="三麻对局（简易脚本）")
    ap.add_argument("--agents", type=str, default="maj-normal,teacher,teacher",
                    help="逗号分隔的座位：给 3 个打三麻、4 个打四麻；"
                         "可用模型名（maj-flash / flash4 …）、teacher、random")
    ap.add_argument("--style", type=str, default="auto",
                    help="教师口味：normal / mountain / flash / huge（auto=跟着第一个模型）")
    ap.add_argument("--human", action="store_true", help="0 号位由你自己打")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=1, help="打几场")
    ap.add_argument("--rounds", type=int, default=0, help="每场几局（0=按人数：三麻 3 / 四麻 4）")
    ap.add_argument("--weights", type=str, default=str(WEIGHTS / "policy.npz"))
    ap.add_argument("--quiet", action="store_true", help="只打印结果")
    ap.add_argument("--log", type=str, default="", help="把过程日志写到文件")
    args = ap.parse_args(argv)

    net = None
    if Path(args.weights).exists():
        net = Net.load(args.weights)
    elif "ml" in args.agents:
        print(f"⚠️ 找不到 {args.weights}，先跑一次 `python -m aimodel.train bc`")

    specs = [s for s in args.agents.split(",") if s.strip()]
    if len(specs) not in (3, 4):
        # 没给够就补教师：给 4 个就是四麻，否则三麻
        target = 4 if len(specs) > 3 else 3
        while len(specs) < target:
            specs.append("teacher")
        specs = specs[:target]
    n_players = len(specs)
    style = args.style if args.style != "auto" else "normal"
    if args.style == "auto" and specs[0] in variants.MODELS:
        style = variants.get(specs[0]).style

    sink: list[str] = []
    show_all = not args.human
    log = make_logger(show_all=show_all, sink=sink)

    for g in range(args.games):
        seed = args.seed + g
        print("=" * 68)
        print(f"🀄  {'四麻' if n_players == 4 else '三麻'} · 第 {g + 1} 场（种子 {seed}）  座位: "
              + "  ".join(f"P{i}={specs[i]}" for i in range(3)))
        print("=" * 68)
        agents = [
            make_agent(specs[i], i, seed * n_players + i, net, args.human, style=style)
            for i in range(n_players)
        ]
        game = Game(
            agents, seed=seed, rounds=args.rounds or n_players, on_event=log
        )
        t0 = time.perf_counter()
        r = game.play()
        dt = time.perf_counter() - t0
        if not args.quiet:
            print("\n── 本场经过 ──")
            print(describe_result(r))
        print_summary(r)
        print(f"耗时 {dt:.1f}s")

    if args.log and sink:
        Path(args.log).write_text("\n".join(sink), encoding="utf-8")
        print(f"日志已写入 {args.log}")


if __name__ == "__main__":
    main()
