"""两个模型直接对打（A/B 测试）—— **相同牌山配对**。

模仿准确率不等于棋力 —— 「和教师选同一张的比例」高，可能是因为那些局面本来
就没得选。要判断新特征 / 新一轮训练到底有没有用，只能让两个版本**坐在同一桌**
上打，再做配对检验。

为什么必须**同牌山**：审计实测 300 场配对检验的单模型 95% 置信区间是
**±6.7~10.5 段位分**，而要检测的效果只有 +3.65 —— 区间比效应还宽两倍。
牌运方差是主要噪声源，把两边灌**同一副牌山**能把它几乎消掉。

用法::

    # 新旧模型 + 一个教师，跑 400 组（每组两副座位）
    python -m aimodel.ab --a weights/policy_bc3.npz --b weights/policy_bc4.npz

    # 两家都是新模型，第三家换随机
    python -m aimodel.ab --a weights/policy_rl2.npz --b weights/policy.npz --third random

    # 关掉同牌山（退回老行为，用来对比方差）
    python -m aimodel.ab --a A.npz --b B.npz --no-same-wall

输出：A、B 各自的段位分 / 顺位，**配对差值 + 置信区间 + 还需多少场**。
"""

# ⚠️ **这个工具的评测设计已被证明是负优化，新评测请用 `aimodel/arena_test.py`。**
#
# 实测（1200 场三麻）：两座位同局的相关系数 r = **−0.501**（正好是零和博弈的理论值
# −0.5），于是「同局配对」的 A−B 差值 σ = **90.4**，而**独立采样只要 73.9** ✗
# 也就是说配对不是削方差，是把方差**放大 1.22 倍**。
#
# 后果：这里跑 300 场的**最小可检出效应是 14.7 段位分** —— 连 +4 都检不出来 ✗
# （项目报告里那个「RL 比克隆 +3.65 段位分」就是在这个分辨率下得出的，
#   而检查点选择偏差一项就有 +9.46，见 `aimodel/EVAL_PROTOCOL.md`）
#
# 正确做法（`arena_test.py` 已实现）：同一副牌山 + 同一个座位 + 同种子的同一批对手，
# A/B 各打一遍（duplicate），σ 降到 54~61，再配 SPRT 提前停。


from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import sanma as S
from .agents import HeuristicAgent, MLAgent, RandomAgent
from .engine import Game, build_wall
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"

#: 正态分布 95% 双侧分位（样本量够大时 t 与 z 差别可忽略；这里不依赖 scipy）
Z95 = 1.959963984540054


def make_third(spec: str, seed: int):
    if spec == "random":
        return RandomAgent(seed)
    if spec == "teacher":
        return HeuristicAgent(seed)
    # 打法风格陪练（见 aimodel/styles.py）：--third style:speed / style:defense / ...
    if spec.startswith("style:") or spec in _style_names():
        from .styles import make as _make_style

        return _make_style(spec, seed)
    raise SystemExit(f"未知的第三方: {spec}")


def _style_names() -> set[str]:
    from .styles import STYLES

    return set(STYLES)


def wall_for(seed: int, rule: S.Rule) -> list[tuple[int, int]]:
    """按 ``seed`` 生成一副确定的牌山（和 ``build_wall`` 同源，所以可复现）。"""
    import random as _random

    return build_wall(_random.Random(seed), rule)


def paired_stats(diffs: list[float]) -> dict:
    """配对差值的均值 / 标准误 / 95% 置信区间 / 显著性。"""
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "t": 0.0, "p_one_sided": 1.0}
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mean, "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "t": 0.0, "p_one_sided": 1.0}
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else 0.0
    # 单侧 p：正态近似
    p_one = 0.5 * math.erfc(t / math.sqrt(2))
    return {
        "n": n, "mean": mean, "se": se,
        "lo": mean - Z95 * se, "hi": mean + Z95 * se,
        "t": t, "p_one_sided": p_one,
    }


def matches_needed(effect: float, se_at_n: float, n_at: float,
                   power_z: float = 2.802) -> float | None:
    """要达到「单侧 p<0.05 且 80% 功效」，还需要多少场？

    ``power_z`` = z_{0.05} + z_{0.20} = 1.6449 + 0.8416 ≈ 2.4865（单侧）。
    这里用更保守的 2.802（对应双侧 0.05 + 80% 功效）以免低估。
    se 按 1/sqrt(n) 缩放：se(N) = se_at_n * sqrt(n_at / N)。
    """
    if effect <= 0 or se_at_n <= 0:
        return None
    # 需要 se(N) <= effect / power_z
    need_se = effect / power_z
    factor = (se_at_n / need_se) ** 2
    return max(0.0, n_at * factor)


def _run_one(agents, seed: int, rule: S.Rule, wall, same_wall: bool):
    g = Game(agents, seed=seed, rule=rule)
    if same_wall and wall is not None:
        g.rigged_wall = list(wall)
    return g.play()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="两个模型对打（相同牌山配对）")
    ap.add_argument("--a", default=str(WEIGHTS / "policy.npz"), help="A 的权重")
    ap.add_argument("--b", default=str(WEIGHTS / "policy_bc3.npz"), help="B 的权重")
    ap.add_argument("--third", default="teacher", help="第三家：teacher / random")
    ap.add_argument("--matches", type=int, default=2500,
                    help="牌局组数（每组默认打两副座位，所以实际对局是它的两倍）")
    ap.add_argument("--seed", type=int, default=777001,
                    help="基础种子。**多组配置请给不同的种子** —— "
                         "如果每组都用默认值，各组就是同一批牌，不独立、z 值会被高估")
    ap.add_argument("--sample", action="store_true", help="按概率采样（默认取 argmax）")
    ap.add_argument("--no-same-wall", action="store_true",
                    help="关掉同牌山注入（退回老行为，用于对比方差）")
    ap.add_argument("--swap", dest="swap", action="store_true", default=True,
                    help="每组打两副座位（A/B 互换），默认开")
    ap.add_argument("--no-swap", dest="swap", action="store_false")
    args = ap.parse_args(argv)

    net_a = Net.load(args.a)
    net_b = Net.load(args.b)
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    # 人数从权重维度推断：486 是三麻、560 是四麻
    n_players = 4 if net_a.dim > 500 else 3
    rule = S.rule_for(n_players)
    per_group = 2 if args.swap else 1
    print(f"第三家 = {args.third}   牌局组数 = {args.matches}"
          f"（每组 {per_group} 副座位 -> {args.matches * per_group} 场）")
    print(f"人数：{n_players}（{'四麻' if n_players == 4 else '三麻'}）")
    print(f"同牌山注入：{'关（老行为）' if args.no_same_wall else '开'}   "
          f"基础种子 = {args.seed}\n")

    rp_a = rp_b = 0.0
    rk_a = rk_b = 0.0
    win_a = win_b = 0
    deal_a = deal_b = 0
    hands = 0
    diffs: list[float] = []
    positive = 0
    ties = 0
    n_games = 0

    for m in range(args.matches):
        deal_seed = args.seed * 1000003 + m
        wall = wall_for(deal_seed, rule) if not args.no_same_wall else None
        for half in range(per_group):
            gseed = deal_seed * 7 + half
            base = ["A", "B"] + ["third"] * (n_players - 2)
            # 每组第二副把名单右移一位，让 A/B 都坐过不同座位
            shift = half if args.swap else 0
            seats = [base[(s - shift) % n_players] for s in range(n_players)]
            agents = []
            for s, tag in enumerate(seats):
                if tag == "A":
                    agents.append(MLAgent(net_a, seed=gseed * 3 + s, greedy=not args.sample))
                elif tag == "B":
                    agents.append(MLAgent(net_b, seed=gseed * 3 + s, greedy=not args.sample))
                else:
                    agents.append(make_third(args.third, gseed * 3 + s))
            r = _run_one(agents, gseed, rule, wall, same_wall=not args.no_same_wall)
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
            if d > 0:
                positive += 1
            elif d == 0:
                ties += 1
            n_games += 1
        if (m + 1) % 100 == 0:
            print(f"  {m + 1}/{args.matches} 组 …", flush=True)

    st = paired_stats(diffs)
    n = max(1, n_games)
    h = max(1, hands)
    print("\n" + "═" * 66)
    print(f"{'':<6}{'段位分':>10}{'平均顺位':>10}{'和了率':>9}{'放铳率':>9}")
    print(f"{'A':<6}{rp_a / n:>+10.2f}{rk_a / n + 1:>10.3f}{win_a / h:>9.1%}{deal_a / h:>9.1%}")
    print(f"{'B':<6}{rp_b / n:>+10.2f}{rk_b / n + 1:>10.3f}{win_b / h:>9.1%}{deal_b / h:>9.1%}")
    print("═" * 66)
    print(f"配对差值（A − B）: {st['mean']:+.2f}   标准误 {st['se']:.2f}")
    print(f"  95% 置信区间: [{st['lo']:+.2f}, {st['hi']:+.2f}]"
          f"   宽度 ±{Z95 * st['se']:.2f}")
    print(f"  t = {st['t']:+.2f}   单侧 p = {st['p_one_sided']:.4f}   （n = {st['n']} 场）")
    print(f"A 单场更高的比例: {positive / n:.1%}   打平 {ties / n:.1%}"
          f"   （50% = 完全打平）")

    # 结论 + 「还要多少场」
    if st["se"] > 0 and st["se"] == st["se"]:
        if st["p_one_sided"] < 0.05:
            verdict = "显著：" + ("A 更强" if st["mean"] > 0 else "B 更强")
        elif abs(st["t"]) < 1.0:
            verdict = "打平：这点差距看不出来"
        else:
            verdict = "略有倾向，但还不够显著"
        print(f"结论: {verdict}")
        target = abs(st["mean"]) if st["mean"] != 0 else 0.0
        if target > 0:
            need = matches_needed(target, st["se"], st["n"])
            if need:
                print(f"要在这点效应上达到 单侧 p<0.05 且 80% 功效，约需 "
                      f"**{need:,.0f} 场**（现在 {st['n']:,} 场）")
            else:
                print("效应量接近 0，无法用加样本量的方式「测出」差异")
        else:
            print("配对差值恰好为 0，先加大样本量再看")
    print("═" * 66)


if __name__ == "__main__":
    main()
