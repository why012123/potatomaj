"""模型强度评测 —— 低方差、可信、带功效分析。

## 为什么不能用简单的「A B 同坐一桌打 300 场」

实测（三麻，见 ``aimodel/EVAL_PROTOCOL.md``）：

* 单局段位分 σ ≈ 52；**同局配对**的 A−B 差值 σ ≈ 90（不是 √2·52=74）——
  三麻是零和博弈，两个座位的相关系数实测 **−0.501**，正好是理论值 −0.5，
  配对不但没削方差，反而把它放大成 σ√3 ✗
* 改成**同一副牌山、同一个座位、重放两遍**（duplicate），σ 降到 **54~61**
* 于是要检出 **+2 段位分**，需要 **5700~7300 场**；+4 需要 1400~1800 场
* 而项目原来用的 300 场，**能检出的最小效应是 8.8 段位分** ✗

## 这个工具做了什么

1. **duplicate 设计**：每「组」用 ``Game.rigged_wall`` 灌同一副牌山，
   A、B 各自在**同一个座位**、对**种子相同的同一批对手**各打一遍。
   两个模型看到完全一样的起手和摸牌，差异只来自决策。
2. **座位轮换**：逐组轮换被测座位，把亲家优势平掉。
3. **SPRT 提前停**：达到显著性就收工，实测比固定样本量省 50%（+2）/ 76%（+4）。
4. **功效分析**：直接告诉你「按现在实测的 σ，要多少组才能检出目标效应」，
   而不是跑完 300 场拿着一个 p=0.4 的结果说「打平」。

用法::

    # 目标：检出 +2 段位分（默认 duplicate + SPRT + 多进程）
    .venv/bin/python -m aimodel.arena_test \\
        --a aimodel/weights/maj-normalv1.1.npz \\
        --b aimodel/weights/maj-normalv1.1_bcbase.npz \\
        --target-delta 2

    # 只看功效、不真跑（先估算要多久）
    .venv/bin/python -m aimodel.arena_test --a X --b Y --plan-only --target-delta 2

    # 固定样本量、不要提前停
    .venv/bin/python -m aimodel.arena_test --a X --b Y --games 2000 --no-sprt
"""

from __future__ import annotations

import warnings
import argparse
import math
import multiprocessing as mp
import random
import statistics as st
import sys
import time
from pathlib import Path

# 允许 `python aimodel/arena_test.py` 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent, MLAgent, RandomAgent  # noqa: E402
from aimodel.engine import Game, build_wall  # noqa: E402
from aimodel.net import Net  # noqa: E402

WEIGHTS = Path(__file__).resolve().parent / "weights"

#: 从实测得到的先验 σ（duplicate 设计的 A−B 段位分差）。
#: 三麻 54~61，四麻按比例小一些；SPRT 需要一个 σ 才能起步，
#: 跑之前会先用若干组做试验估计，估不准就用这个兜底。
PRIOR_SIGMA = {3: 57.0, 4: 42.0}

_NET_A: Net | None = None
_NET_B: Net | None = None
_NET_C: Net | None = None


def _load_quiet(path):
    """加载权重，**屏蔽「旧权重缺决策头」的警告** —— 那是预期内的
    （见 net.py 的 zero-init 策略），评测时刷一屏没用。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Net.load(path)


def _load(path: str) -> Net:
    return Net.load(path)


def _init_worker(a: str, b: str, third: str | None) -> None:
    global _NET_A, _NET_B, _NET_C
    _NET_A, _NET_B = _load(a), _load(b)
    _NET_C = _load(third) if third and third.endswith(".npz") else None


def _third_agent(kind: str, seed: int):
    if kind == "random":
        return RandomAgent(seed)
    if kind == "third" and _NET_C is not None:
        return MLAgent(_NET_C, seed=seed, greedy=True)
    return HeuristicAgent(seed, style="normal")


def _run_one_side(wall, net, seat: int, third: str, opp_seed: int,
                  players: int, sample: bool):
    """一副牌山、一个座位、一套固定对手，打一局。"""
    agents = []
    for s in range(players):
        if s == seat:
            agents.append(MLAgent(net, seed=opp_seed, greedy=not sample))
        else:
            agents.append(_third_agent(third, opp_seed + 1000 + s))
    game = Game(agents, seed=0, rule=S.rule_for(players))
    game.rigged_wall = wall
    return game.play()


def _group(args_tuple):
    """一个「组」= 同一副牌山 + 同一个座位，A、B 各打一遍。"""
    m, wall_seed, seat, third, players, sample, a_path, b_path = args_tuple
    net_a = _load(a_path) if _NET_A is None else _NET_A
    net_b = _load(b_path) if _NET_B is None else _NET_B
    wall = build_wall(random.Random(wall_seed), S.rule_for(players))
    opp = wall_seed * 31 + 7          # 对手种子固定 ⇒ 两边对手完全一致
    ra = _run_one_side(wall, net_a, seat, third, opp, players, sample)
    rb = _run_one_side(wall, net_b, seat, third, opp, players, sample)
    return {
        "diff_rp": ra.rank_points[seat] - rb.rank_points[seat],
        "diff_rank": ra.ranks[seat] - rb.ranks[seat],
        "a_first": 1.0 if ra.ranks[seat] == 0 else 0.0,
        "b_first": 1.0 if rb.ranks[seat] == 0 else 0.0,
        "a_rp": ra.rank_points[seat],
        "b_rp": rb.rank_points[seat],
    }


def _batched(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


class Sprt:
    """Wald 序贯概率比检验：H0: μ=0  vs  H1: μ=δ。

    每来一组就更新对数似然比，越过边界立刻停。
    实测：真值 +2 时平均 3638 组（固定 7315 组，省 50%）；
    真值 +4 时 1767 组（省 76%）；真值 0 时 95% 正确判 H0。
    """

    def __init__(self, delta: float, sigma: float, alpha=0.05, beta=0.20):
        self.delta, self.sigma = delta, sigma
        self.hi, self.lo = _sprt_bounds(alpha, beta)
        self.llr = 0.0
        self.n = 0

    def update(self, x: float) -> str | None:
        self.llr += (self.delta * x) / self.sigma**2 - self.delta**2 / (2 * self.sigma**2)
        self.n += 1
        if self.llr >= self.hi:
            return "H1"          # 判定 A 更强
        if self.llr <= self.lo:
            return "H0"          # 判定打平
        return None


def plan(target_delta: float, sigma: float, alpha=0.05, power=0.80) -> int:
    """达到目标功效需要多少组（单侧）。"""
    z_a = st.NormalDist().inv_cdf(1 - alpha)
    z_b = st.NormalDist().inv_cdf(power)
    return math.ceil(((z_a + z_b) * sigma / abs(target_delta)) ** 2)


def mde(sigma: float, n: int, alpha=0.05, power=0.80) -> float:
    """n 组能检出的最小效应。"""
    z_a = st.NormalDist().inv_cdf(1 - alpha)
    z_b = st.NormalDist().inv_cdf(power)
    return (z_a + z_b) * sigma / math.sqrt(max(1, n))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="模型强度评测（duplicate + SPRT + 功效分析）")
    ap.add_argument("--a", default=str(WEIGHTS / "maj-normalv1.1.npz"))
    ap.add_argument("--b", default=str(WEIGHTS / "maj-normalv1.1_bcbase.npz"))
    ap.add_argument("--third", default="teacher", help="teacher / random / <权重路径>")
    ap.add_argument("--target-delta", type=float, default=2.0,
                    help="要检出的段位分差（默认 2）")
    ap.add_argument("--games", type=int, default=None,
                    help="固定组数；默认按 target-delta 算出的需求（上限 --max-games）")
    ap.add_argument("--max-games", type=int, default=20000)
    ap.add_argument("--no-sprt", action="store_true", help="关掉提前停")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--jobs", type=int, default=0, help="并行进程数（0=自动）")
    ap.add_argument("--chunk", type=int, default=64, help="每批跑多少组再判 SPRT")
    ap.add_argument("--sample", action="store_true", help="按概率采样（默认取 argmax）")
    ap.add_argument("--plan-only", action="store_true", help="只给功效和耗时估算")
    ap.add_argument("--sigma", type=float, default=None, help="已知 σ（不给就用先验）")
    args = ap.parse_args(argv)

    net_a, net_b = _load_quiet(args.a), _load_quiet(args.b)
    players = 4 if net_a.dim > 500 else 3
    sigma = args.sigma or PRIOR_SIGMA[players]
    need = plan(args.target_delta, sigma, args.alpha, args.power)
    games = args.games or min(need, args.max_games)

    print("═" * 74)
    print("  模型强度评测")
    print("═" * 74)
    print(f"  A = {Path(args.a).name}")
    print(f"  B = {Path(args.b).name}")
    print(f"  人数 {players}   第三家 {args.third}   设计 duplicate（同牌山 + 同座位重放）")
    print(f"  先验 σ = {sigma:.1f}（实测区间 54~61，三麻）")
    print(f"  目标效应 δ = {args.target_delta:+.2f} 段位分"
          f"（α={args.alpha} 单侧，功效 {args.power:.0%}）")
    print(f"  → 固定样本量方案需要 **{need} 组**；本次计划跑 {games} 组")
    if args.no_sprt:
        print("  模式：固定样本量（未开提前停）")
    else:
        print("  模式：SPRT 提前停（预计省 30~75%，取决于真实效应）")
    print("═" * 74)
    if args.plan_only:
        return

    jobs = args.jobs or max(1, (mp.cpu_count() or 4) - 1)
    print(f"  并行 {jobs} 进程\n")

    t0 = time.perf_counter()
    diffs: list[float] = []
    a_first = b_first = 0.0
    a_rp = b_rp = 0.0
    verdict = None
    sprt = Sprt(args.target_delta, sigma, args.alpha, 1 - args.power)

    work = ((m, args.seed + m * 7919, m % players, args.third, players,
             args.sample, args.a, args.b) for m in range(games))

    pool = mp.Pool(jobs, initializer=_init_worker, initargs=(args.a, args.b, args.third))
    try:
        for batch in _batched(work, args.chunk):
            for r in pool.imap(_group, batch):
                diffs.append(r["diff_rp"])
                a_first += r["a_first"]; b_first += r["b_first"]
                a_rp += r["a_rp"]; b_rp += r["b_rp"]
                if not args.no_sprt and verdict is None:
                    verdict = sprt.update(r["diff_rp"])
            n = len(diffs)
            if verdict is not None and not args.no_sprt:
                print(f"  SPRT 边界达到，第 {n} 组提前停")
                break
            if n % (args.chunk * 5) == 0 or n >= games:
                sd = st.pstdev(diffs) if n > 1 else sigma
                se = sd / math.sqrt(n)
                el = time.perf_counter() - t0
                print(f"  {n:>6}/{games} 组  Δ {st.mean(diffs):+6.2f} ± {se:4.2f}"
                      f"（σ {sd:5.1f}）  {el:5.0f}s  预计还要 {el/n*(games-n):5.0f}s",
                      flush=True)
    finally:
        pool.close(); pool.join()

    n = len(diffs)
    mean, sd = st.mean(diffs), st.pstdev(diffs) if n > 1 else sigma
    se = sd / math.sqrt(n)
    z_a = st.NormalDist().inv_cdf(1 - args.alpha)
    lo, hi = mean - z_a * se, mean + z_a * se
    p = 0.5 * math.erfc((mean / se) / math.sqrt(2)) if se else 1.0

    print("\n" + "═" * 74)
    print(f"  A 平均段位分 {a_rp/n:+7.3f}   一位率 {a_first/n:6.1%}")
    print(f"  B 平均段位分 {b_rp/n:+7.3f}   一位率 {b_first/n:6.1%}")
    print(f"  配对差值（A−B）  {mean:+.3f}   实测 σ {sd:.2f}   SE ±{se:.3f}")
    print(f"  {1-args.alpha:.0%} 单侧置信下界 {lo:+.3f}   单侧 p ≈ {p:.3f}")
    print(f"  本次 {n} 组能检出的最小效应（功效 {args.power:.0%}）= {mde(sd, n, args.alpha, args.power):.2f}")
    if verdict == "H1":
        print(f"  SPRT 判决：**A 更强**（比 H0 强出 {args.target_delta:+.1f} 以上）")
    elif verdict == "H0":
        print(f"  SPRT 判决：**打平**（差异不到 {args.target_delta:+.1f}）")
    elif se:
        print(f"  未达 SPRT 边界；按当前 σ 要检出 {args.target_delta:+.1f}"
              f" 需要 {plan(args.target_delta, sd, args.alpha, args.power)} 组")
    print(f"  总耗时 {time.perf_counter()-t0:.0f}s")
    print("═" * 74)


if __name__ == "__main__":
    main()
