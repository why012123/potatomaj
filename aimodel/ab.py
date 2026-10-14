"""两个模型直接对打（A/B 测试）。

模仿准确率不等于棋力 —— 「和教师选同一张的比例」高，可能是因为那些局面本来
就没得选。要判断新特征 / 新一轮训练到底有没有用，只能让两个版本**坐在同一桌**
上打，用同一批种子、轮换座位，再做配对检验。

用法::

    # 新旧模型 + 一个教师，跑 400 场
    python -m aimodel.ab --a weights/policy_bc3.npz --b weights/policy_bc4.npz

    # 两家都是新模型，第三家换随机
    python -m aimodel.ab --a weights/policy_rl2.npz --b weights/policy.npz --third random

输出：A、B 各自的段位分 / 顺位，以及**配对差值 + t 值**。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .agents import HeuristicAgent, MLAgent, RandomAgent
from .engine import Game
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"


def make_third(spec: str, seed: int):
    if spec == "random":
        return RandomAgent(seed)
    if spec == "teacher":
        return HeuristicAgent(seed)
    raise SystemExit(f"未知的第三方: {spec}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="两个模型对打")
    ap.add_argument("--a", default=str(WEIGHTS / "policy.npz"), help="A 的权重")
    ap.add_argument("--b", default=str(WEIGHTS / "policy_bc3.npz"), help="B 的权重")
    ap.add_argument("--third", default="teacher", help="第三家：teacher / random")
    ap.add_argument("--matches", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--sample", action="store_true", help="按概率采样（默认取 argmax）")
    args = ap.parse_args(argv)

    net_a = Net.load(args.a)
    net_b = Net.load(args.b)
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"第三家 = {args.third}   对局数 = {args.matches}\n")

    rp_a = rp_b = 0.0
    rk_a = rk_b = 0.0
    win_a = win_b = 0
    deal_a = deal_b = 0
    hands = 0
    diffs: list[float] = []
    positive = 0

    # 人数从权重维度推断：486 是三麻、560 是四麻
    n_players = 4 if net_a.dim > 500 else 3
    print(f"人数：{n_players}（{'四麻' if n_players == 4 else '三麻'}）\n")
    for m in range(args.matches):
        seed = args.seed + m
        # 座位轮换：第 m 场把整桌名单左移 m 位
        base = ["A", "B"] + ["third"] * (n_players - 2)
        rot = m % n_players
        seats = [base[(s + rot) % n_players] for s in range(n_players)]
        agents = []
        for s, tag in enumerate(seats):
            if tag == "A":
                agents.append(MLAgent(net_a, seed=seed * 3 + s, greedy=not args.sample))
            elif tag == "B":
                agents.append(MLAgent(net_b, seed=seed * 3 + s, greedy=not args.sample))
            else:
                agents.append(make_third(args.third, seed * 3 + s))
        r = Game(agents, seed=seed).play()
        ia = seats.index("A")
        ib = seats.index("B")
        rp = r.rank_points
        rp_a += rp[ia]
        rp_b += rp[ib]
        rk_a += r.ranks[ia]
        rk_b += r.ranks[ib]
        win_a += r.stats[ia].tsumo + r.stats[ia].ron
        win_b += r.stats[ib].tsumo + r.stats[ib].ron
        deal_a += r.stats[ia].deal_in
        deal_b += r.stats[ib].deal_in
        hands += len(r.hands)
        d = rp[ia] - rp[ib]
        diffs.append(d)
        positive += 1 if d > 0 else 0
        if (m + 1) % 50 == 0:
            print(f"  {m + 1}/{args.matches} …", flush=True)

    n = args.matches
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)
    se = math.sqrt(var / n)
    h = max(1, hands)
    print("\n" + "═" * 62)
    print(f"{'':<6}{'段位分':>10}{'平均顺位':>10}{'和了率':>9}{'放铳率':>9}")
    print(f"{'A':<6}{rp_a / n:>+10.2f}{rk_a / n + 1:>10.3f}{win_a / h:>9.1%}{deal_a / h:>9.1%}")
    print(f"{'B':<6}{rp_b / n:>+10.2f}{rk_b / n + 1:>10.3f}{win_b / h:>9.1%}{deal_b / h:>9.1%}")
    print("═" * 62)
    print(f"配对差值（A − B）: {mean:+.2f} ± {se:.2f}   t = {mean / se if se else 0:.2f}")
    print(f"A 单场更高的比例: {positive / n:.1%}   （50% = 完全打平）")
    if se > 0:
        t = mean / se
        if abs(t) < 1.0:
            verdict = "打平：这点差距看不出来"
        elif abs(t) < 2.0:
            verdict = "略有倾向，但还不够显著（样本再加一倍看看）"
        else:
            verdict = "显著：" + ("A 更强" if t > 0 else "B 更强")
        print(f"结论: {verdict}")


if __name__ == "__main__":
    main()
