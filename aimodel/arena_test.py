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

from aimodel import external  # noqa: E402
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


def _net_path(spec: str) -> Path | None:
    """把 A/B 的写法解析成权重路径；不是神经网络就返回 None。

    支持三种写法：
      * ``aimodel/weights/xxx.npz``（路径）
      * ``maj-pro4v1.1_mac_rl``（注册表里的模型名）
      * ``strong`` / ``teacher`` / ``random`` / ``style:speed``（规则 agent）
    """
    if not spec:
        return None
    p = Path(spec)
    if p.suffix == ".npz":
        return p if p.exists() else None
    try:
        from aimodel import variants

        sp = variants.get(spec)
        return sp.path if sp.path.exists() else None
    except Exception:                       # noqa: BLE001 - 不是注册模型
        return None


def _load_spec_net(spec: str) -> Net | None:
    p = _net_path(spec)
    return _load_quiet(str(p)) if p is not None else None


def _style_names() -> set[str]:
    from .styles import STYLES

    return set(STYLES)


def _make_agent(spec: str, seed: int, players: int, *,
                sample: bool = False, net: Net | None = None,
                player_id: int = 0):
    """按 spec 造一个 agent —— 神经网络 / 教师 / 随机 / strong / 风格 / 外部 AI。"""
    s = (spec or "teacher").strip()
    low = s.lower()
    ext_key = external.normalize_key(s)
    if ext_key is not None:
        return external.ExternalMJaiAgent(
            ext_key, seed=seed, player_id=player_id, players=players,
            timeout=20.0, max_restarts=2,
        )
    if low in ("teacher", "heuristic"):
        return HeuristicAgent(seed)
    if low in ("random", "r"):
        return RandomAgent(seed)
    # ``strong`` / ``strong:attack`` / ``strong3`` / ``strong4``
    if low.startswith("strong") or low in ("attack", "defense", "value"):
        from .strong import make as make_strong

        if low in ("strong3", "strong4"):
            low = "strong"
        return make_strong(low, seed)
    # 风格陪练（styles.py）
    if low.startswith("style:") or low in _style_names():
        from .styles import make as make_style

        return make_style(s, seed)
    if net is None:
        net = _load_spec_net(s)
    if net is None:
        raise SystemExit(
            f"未知的 agent spec: {spec!r}（可用：teacher / random / strong / "
            f"strong:<变体> / style:<风格> / <权重路径> / <注册模型名>）"
        )
    return MLAgent(net, seed=seed, greedy=not sample)


def _init_worker(a: str, b: str, third: str | None) -> None:
    global _NET_A, _NET_B, _NET_C
    _NET_A = None if external.is_external_spec(a) else _load_spec_net(a)
    _NET_B = None if external.is_external_spec(b) else _load_spec_net(b)
    _NET_C = (None if third and external.is_external_spec(third)
              else (_load_spec_net(third) if third else None))


def _third_agent(kind: str, seed: int, players: int, net: Net | None = None,
                 *, seat: int = 0, third_seat: int = 0):
    ext_key = external.normalize_key(kind)
    if ext_key is not None:
        # 外部 AI 只占一个座位，其它座位继续用教师：A/B 两边面对同一批对手
        if seat == third_seat:
            return external.ExternalMJaiAgent(
                ext_key, seed=seed, player_id=seat, players=players,
                timeout=20.0, max_restarts=2,
            )
        return HeuristicAgent(seed, style="normal")
    if kind == "third":
        if net is not None:
            return MLAgent(net, seed=seed, greedy=True)
        # 老行为：没给第三家权重时退回教师，而不是把 "third" 当未知 spec 报错。
        return HeuristicAgent(seed, style="normal")
    return _make_agent(kind, seed, players, sample=True, net=net,
                       player_id=seat)


def _run_one_side(wall, spec: str, net: Net | None, seat: int, third: str,
                  third_net: Net | None, opp_seed: int,
                  players: int, sample: bool, third_seat: int = 0):
    """一副牌山、一个座位、一套固定对手，打一局。"""
    agents = []
    for s in range(players):
        if s == seat:
            agents.append(_make_agent(spec, opp_seed, players,
                                      sample=sample, net=net, player_id=s))
        else:
            agents.append(_third_agent(third, opp_seed + 1000 + s, players,
                                       net=third_net, seat=s,
                                       third_seat=third_seat))
    game = Game(agents, seed=0, rule=S.rule_for(players))
    game.rigged_wall = wall
    external.attach(game)
    try:
        return game.play()
    finally:
        external.close_external(agents)


def _group(args_tuple):
    """一个「组」= 同一副牌山 + 同一个座位，A、B 各打一遍。"""
    m, wall_seed, seat, third, players, sample, a_spec, b_spec = args_tuple
    net_a = _NET_A if _NET_A is not None else _load_spec_net(a_spec)
    net_b = _NET_B if _NET_B is not None else _load_spec_net(b_spec)
    wall = build_wall(random.Random(wall_seed), S.rule_for(players))
    opp = wall_seed * 31 + 7          # 对手种子固定 ⇒ 两边对手完全一致
    third_seat = 0 if seat != 0 else 1
    ra = _run_one_side(wall, a_spec, net_a, seat, third, _NET_C, opp, players,
                       sample, third_seat)
    rb = _run_one_side(wall, b_spec, net_b, seat, third, _NET_C, opp, players,
                       sample, third_seat)
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
    ap.add_argument("--a", default=str(WEIGHTS / "maj-normalv1.1.npz"),
                    help=f"A：权重路径 / 注册模型名 / teacher / random / strong[:变体] / "
                         f"style:<风格> / 外部 AI（{external.spec_help()}）")
    ap.add_argument("--b", default=str(WEIGHTS / "maj-normalv1.1_bcbase.npz"),
                    help="B：同 --a")
    ap.add_argument("--third", default="teacher",
                    help=f"其余座位的对手：teacher / random / strong / <权重路径> / "
                         f"外部 AI（{external.spec_help()}）")
    ap.add_argument("--players", type=int, default=0, choices=[0, 3, 4],
                    help="人数；0 = 自动（有神经网络按权重维度推断，否则四麻）")
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

    # A/B 现在既可以是权重，也可以是纯规则 agent（strong / teacher / ...），
    # 还可以是外部 mjai AI（mortal / manue / ...）。
    ext_a = external.is_external_spec(args.a)
    ext_b = external.is_external_spec(args.b)
    ext_third = external.normalize_key(args.third)
    net_a = None if ext_a else _load_spec_net(args.a)
    net_b = None if ext_b else _load_spec_net(args.b)
    dims = []
    for net in (net_a, net_b):
        if net is not None:
            dims.append(4 if net.dim > 500 else 3)
    if args.players:
        players = args.players
    elif dims:
        players = dims[0]
        if len(set(dims)) > 1:
            raise SystemExit(
                f"A/B 的权重人数不一致（{dims}）—— 不能同桌，请用 --players 明确指定"
            )
    else:
        players = 4                 # 纯规则对照 / 外部 AI 默认四麻
    if (ext_a or ext_b or ext_third is not None) and players != 4:
        raise SystemExit(
            f"外部 AI 只支持四麻（mjai 的 start_kyoku 是 4 人格式）；"
            f"当前 {players} 人。见 {external.DOC_HINT}"
        )
    for key, is_ext in ((args.a, ext_a), (args.b, ext_b)):
        if is_ext:
            external.require(key)
    if ext_third is not None:
        external.require(ext_third)
    sigma = args.sigma or PRIOR_SIGMA[players]
    need = plan(args.target_delta, sigma, args.alpha, args.power)
    games = args.games or min(need, args.max_games)

    print("═" * 74)
    print("  模型强度评测")
    print("═" * 74)
    print(f"  A = {args.a if ext_a else Path(args.a).name}"
          f"    B = {args.b if ext_b else Path(args.b).name}")
    print(f"  人数 {players}   第三家 {args.third}   设计 duplicate（同牌山 + 同座位重放）")
    if ext_a or ext_b or ext_third is not None:
        print(f"  外部 AI：mjai stdio 子进程（{external.spec_help()}）")
    print(f"  先验 σ = {sigma:.1f}（实测区间 54~61，三麻；四麻按比例更小）")
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
