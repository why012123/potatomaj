"""对战评测：让不同 agent 坐同一桌，用可复现的种子跑统计。

做法
----

* 每一场用固定种子（``seed0 + 局号``），**座位轮换**（第 m 场把名单整体
  左移 ``m % 3``）—— 三麻亲家优势很大，不轮换的话「谁赢」很大程度取决于座位。
* 指标：平均顺位、平均得点变化、和了率、放铳率、立直率、副露率、平均翻数。
* 头对头比较时用**配对检验**（同一批种子、同一副牌），给出差值均值和 t 值。

用法::

    python -m aimodel.evaluate                       # 跑一套标准对比
    python -m aimodel.evaluate --matches 500         # 加大样本
    python -m aimodel.evaluate --agents ml,teacher,teacher
    python -m aimodel.evaluate --markdown            # 输出 markdown 表
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import sanma as S
from . import variants
from .agents import Agent, HeuristicAgent, MLAgent, RandomAgent
from .engine import Game
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"


# ---------------------------------------------------------------------------
# 组装 agent
# ---------------------------------------------------------------------------


def build_agent(spec: str, seed: int, net: Net | None, greedy: bool = True) -> Agent:
    """座位名 -> agent。模型名（maj-flash4 等）会自动去 weights/ 找权重。"""
    spec = spec.strip().lower()
    if spec in ("teacher", "heuristic", "t"):
        return HeuristicAgent(seed)
    if spec in ("random", "r"):
        return RandomAgent(seed)
    if spec in ("ml", "net", "model", "policy"):
        if net is None:
            raise SystemExit("需要 --weights 才能评测神经网络")
        return MLAgent(net, seed=seed, greedy=greedy)
    # 其它一律当模型名
    try:
        name = variants.resolve(spec)
    except KeyError:
        raise SystemExit(f"未知 agent / 模型: {spec}")
    path = variants.WEIGHTS_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"模型 {name} 还没训练（缺 {path}）")
    return MLAgent(Net.load(path), seed=seed, greedy=greedy)


@dataclass
class Tally:
    """一个「名单组合」的累计结果。"""

    spec: str
    matches: int = 0
    hands: int = 0
    delta_sum: float = 0.0
    rank_point_sum: float = 0.0
    rank_sum: float = 0.0
    win_sum: int = 0
    tsumo_sum: int = 0
    ron_sum: int = 0
    deal_in_sum: int = 0
    draw_sum: int = 0
    riichi_sum: int = 0
    call_sum: int = 0
    kita_sum: int = 0
    han_sum: float = 0.0
    han_n: int = 0
    score_samples: list[int] = field(default_factory=list)
    firsts: int = 0
    seconds: int = 0
    thirds: int = 0

    def add(self, delta: int, rank: int, st, rank_point: float = 0.0) -> None:
        self.matches += 1
        self.rank_point_sum += rank_point
        self.hands += st.hands
        self.delta_sum += delta
        self.rank_sum += rank
        self.win_sum += st.tsumo + st.ron
        self.tsumo_sum += st.tsumo
        self.ron_sum += st.ron
        self.deal_in_sum += st.deal_in
        self.draw_sum += st.draws
        self.riichi_sum += st.riichi
        self.call_sum += st.calls
        self.kita_sum += st.kita
        self.han_sum += sum(st.win_han)
        self.han_n += len(st.win_han)
        self.score_samples.append(delta)
        if rank == 0:
            self.firsts += 1
        elif rank == 1:
            self.seconds += 1
        else:
            self.thirds += 1

    def row(self) -> dict:
        m = max(1, self.matches)
        h = max(1, self.hands)
        return {
            "agent": self.spec,
            "对局数": self.matches,
            "平均顺位": self.rank_sum / m + 1,
            "平均段位分": self.rank_point_sum / m,
            "一位率": self.firsts / m,
            "平均得点变化": self.delta_sum / m,
            "和了率": self.win_sum / h,
            "自摸率": self.tsumo_sum / h,
            "荣和率": self.ron_sum / h,
            "放铳率": self.deal_in_sum / h,
            "流局率": self.draw_sum / h,
            "立直率": self.riichi_sum / h,
            "副露率": self.call_sum / h,
            "拔北/局": self.kita_sum / h,
            "平均翻数": (self.han_sum / self.han_n) if self.han_n else 0.0,
        }


def run_lineup(
    specs: list[str],
    matches: int,
    seed0: int,
    net: Net | None = None,
    greedy: bool = True,
    on_game=None,
) -> tuple[list[Tally], list[list[int]]]:
    """跑若干场；座位每场轮换。返回每个 spec 的汇总和每场的得分表。"""
    tallies = [Tally(spec) for spec in specs]
    per_match: list[list[int]] = []
    n = len(specs)
    for m in range(matches):
        seed = seed0 + m
        rot = m % n
        agents = []
        for seat in range(n):
            spec = specs[(seat + rot) % n]
            agents.append(build_agent(spec, seed * 3 + seat, net, greedy=greedy))
        r = Game(agents, seed=seed).play()
        deltas = r.deltas
        per_match.append(deltas)
        for seat in range(n):
            spec_i = (seat + rot) % n
            tallies[spec_i].add(
                deltas[seat], r.ranks[seat], r.stats[seat], r.rank_points[seat]
            )
        if on_game is not None:
            on_game(m, r)
    return tallies, per_match


# ---------------------------------------------------------------------------
# 头对头（配对检验）
# ---------------------------------------------------------------------------


def _players_of(*specs: str) -> int:
    """从座位名里推断打三麻还是四麻（模型名带 4 后缀就是四麻）。"""
    for sp in specs:
        low = sp.strip().lower()
        if low.endswith("4") and low not in ("ml",):
            try:
                return variants.get(low).players
            except KeyError:
                pass
    return 3


def head_to_head(
    spec_a: str, spec_b: str, matches: int, seed0: int, net: Net | None, greedy: bool = True
) -> dict:
    """A 坐 0 号位、两家 B 坐 1/2 号位，座位每场轮换后再汇总。

    返回 A 与 B 的配对差值（每场取 A 的得分变化减去 B 的平均得分变化）。
    """
    n_players = _players_of(spec_a, spec_b)
    diffs: list[float] = []
    a_delta = a_rank = a_rp = 0.0
    b_delta = b_rank = b_rp = 0.0
    n_a = n_b = 0
    for m in range(matches):
        seed = seed0 + m
        rot = m % n_players
        specs = [spec_b] * n_players
        specs[rot] = spec_a
        agents = [
            build_agent(specs[s], seed * n_players + s, net, greedy=greedy)
            for s in range(n_players)
        ]
        r = Game(agents, seed=seed).play()
        d = r.deltas
        a = rot  # A 实际坐的座位
        bs = [s for s in range(n_players) if s != a]
        rp = r.rank_points
        diffs.append(rp[a] - sum(rp[s] for s in bs) / len(bs))
        a_delta += d[a]
        a_rp += rp[a]
        a_rank += r.ranks[a]
        n_a += 1
        for s in bs:
            b_delta += d[s]
            b_rp += rp[s]
            b_rank += r.ranks[s]
            n_b += 1
    mean = sum(diffs) / max(1, len(diffs))
    if len(diffs) > 1:
        var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
        se = math.sqrt(var / len(diffs))
    else:
        se = 0.0
    return {
        "A": spec_a,
        "B": spec_b,
        "对局数": matches,
        "A平均段位分": a_rp / max(1, n_a),
        "B平均段位分": b_rp / max(1, n_b),
        "A平均得点变化": a_delta / max(1, n_a),
        "B平均得点变化": b_delta / max(1, n_b),
        "A平均顺位": a_rank / max(1, n_a) + 1,
        "B平均顺位": b_rank / max(1, n_b) + 1,
        "段位分差值均值": mean,
        "段位分差值标准误": se,
        "t值": (mean / se) if se > 0 else 0.0,
        "胜率(A段位分更高)": sum(1 for x in diffs if x > 0) / max(1, len(diffs)),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    head = ["agent", "对局数", "平均段位分", "平均顺位", "一位率", "平均得点变化",
            "和了率", "放铳率", "立直率", "副露率", "平均翻数"]
    print("  " + "".join(f"{h:>12s}" for h in head))
    for r in rows:
        vals = [
            r["agent"], str(r["对局数"]), f"{r['平均段位分']:+.1f}",
            f"{r['平均顺位']:.3f}", f"{r['一位率']:.1%}",
            f"{r['平均得点变化']:+.0f}", f"{r['和了率']:.1%}", f"{r['放铳率']:.1%}",
            f"{r['立直率']:.1%}", f"{r['副露率']:.1%}", f"{r['平均翻数']:.2f}",
        ]
        print("  " + "".join(f"{v:>12s}" for v in vals))


def markdown_table(rows: list[dict]) -> str:
    head = ["agent", "对局数", "平均段位分", "平均顺位", "一位率", "平均得点变化",
            "和了率", "放铳率", "立直率", "副露率", "平均翻数"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append(
            "| {} | {} | {:+.1f} | {:.3f} | {:.1%} | {:+.0f} | {:.1%} | {:.1%} | {:.1%} | {:.1%} | {:.2f} |".format(
                r["agent"], r["对局数"], r["平均段位分"], r["平均顺位"], r["一位率"],
                r["平均得点变化"], r["和了率"], r["放铳率"], r["立直率"],
                r["副露率"], r["平均翻数"],
            )
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="三麻 AI 评测")
    ap.add_argument("--agents", type=str, default="", help="逗号分隔的三家名单，如 ml,teacher,teacher")
    ap.add_argument("--matches", type=int, default=300, help="每种组合的对局数")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--weights", type=str, default=str(WEIGHTS / "policy.npz"))
    ap.add_argument("--sample", action="store_true", help="按概率采样出牌（默认取 argmax）")
    ap.add_argument("--markdown", action="store_true", help="输出 markdown 表格")
    ap.add_argument("--benchmark", action="store_true", default=False, help="跑一套标准对比")
    ap.add_argument("--out", type=str, default="", help="把结果写进 json")
    args = ap.parse_args(argv)

    net = None
    if Path(args.weights).exists():
        try:
            net = Net.load(args.weights)
            print(
                f"载入权重 {args.weights}"
                f"（{net.n_params()} 参数，累计更新 {net.total_steps} 次）"
            )
        except ValueError as exc:
            print(f"⚠️ {exc}")
            print("   → 这次只评测非神经网络对手")
    else:
        print(f"⚠️ 找不到权重 {args.weights}，只能评测非神经网络对手")

    greedy = not args.sample
    t0 = time.perf_counter()
    results: dict = {"lineups": [], "head_to_head": []}

    if args.agents:
        specs = [s for s in args.agents.split(",") if s]
        tallies, _ = run_lineup(
            specs, args.matches, args.seed, net, greedy=greedy,
            on_game=lambda m, r: print(f"\r  {m + 1}/{args.matches}", end="", flush=True),
        )
        print()
        rows = [t.row() for t in tallies]
        print_table(rows, "对战结果")
        results["lineups"].append({"specs": specs, "rows": rows})
    else:
        lineups = [
            ["random", "random", "random"],
            ["teacher", "teacher", "teacher"],
            ["ml", "teacher", "teacher"],
            ["ml", "random", "random"],
            ["ml", "ml", "ml"],
        ]
        if net is None:
            lineups = [lu for lu in lineups if "ml" not in lu]
        for specs in lineups:
            label = " vs ".join(specs)
            print(f"\n▶ {label}")
            tallies, _ = run_lineup(
                specs, args.matches, args.seed, net, greedy=greedy,
                on_game=lambda m, r: print(f"\r  {m + 1}/{args.matches}", end="", flush=True),
            )
            print()
            rows = [t.row() for t in tallies]
            print_table(rows, f"对战结果 · {label}")
            if args.markdown:
                print(markdown_table(rows))
            results["lineups"].append({"specs": specs, "rows": rows})

    if net is not None:
        print("\n▶ 头对头：ML vs 教师（座位轮换 + 配对检验）")
        h2h = head_to_head("ml", "teacher", args.matches, args.seed, net, greedy=greedy)
        for k, v in h2h.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        results["head_to_head"].append(h2h)
        print("\n▶ 头对头：ML vs 随机")
        h2h2 = head_to_head("ml", "random", args.matches, args.seed, net, greedy=greedy)
        for k, v in h2h2.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        results["head_to_head"].append(h2h2)

    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
