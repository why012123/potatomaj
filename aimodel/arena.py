"""ELO 混战池：让不同模型、不同版本互相打，打出一个天梯。

规则（按需求定的，都能在 UI 里调）：

* 每个参战者初始 **500 分**
* 一场 = 一局三麻 / 四麻（3 或 4 个人上桌）
* 多人大逃杀式的 ELO：桌上每两个人之间都算一次对局，名次靠前就是赢
* **低于 0 分保底为 0**，不出局（一直在池子里打）
* 匹配时**桌上分差不超过 200 分**（可关掉严格模式放宽）

为什么要多人 ELO 而不是简单「赢了加分」：一桌三个人的实力差会互相影响，
两两结算才能让「跟强者打得接近」和「虐菜赢」得到不同的分数。

跑法::

    python -m aimodel.arena --matches 200              # 命令行跑 200 场，打印天梯
    python -m aimodel.arena --players 4 --matches 100  # 四麻池
    python -m aimodel.arena --reset                    # 清空战绩重来
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import variants
from .agents import Agent, HeuristicAgent, MLAgent, RandomAgent
from . import sanma as S
from .engine import Game
from .net import Net

STATE_FILE = Path(__file__).resolve().parent / "arena_state.json"

#: 参战者的来源：版本目录里的模型 + 两个内置基线
BUILTIN = ("teacher", "random")


@dataclass
class Participant:
    """一个参战者（某个模型的某个版本，或者教师 / 随机基线）。"""

    key: str  #: 唯一标识，例如 "maj-flashv1.2#2"（同一个模型的第 2 号参赛者）
    label: str  #: 显示名
    players: int  #: 3 或 4
    kind: str  #: model / teacher / random
    source: str = ""  #: 权重来源（同一个模型派出的多个参赛者共用一份权重）
    version: str = ""  #: 模型版本（基线为空）
    style: str = ""  #: 风格（基线为空）
    rating: float = 500.0
    games: int = 0
    wins: int = 0  #: 一位次数
    last: int = 0  #: 末位次数
    score_term: float = 0.0  #: 累计「点数段位分」
    place_term: float = 0.0  #: 累计「额外段位分」
    hands: int = 0  #: 参与过的局数（算和了率的分母）
    agari: int = 0  #: 和了次数（自摸 + 荣和）
    deal_in: int = 0  #: 放铳次数
    ranks: list[int] = field(default_factory=list)
    floor_hits: int = 0  #: 有多少次被打到 0 分（保底触底次数）
    matches: list[dict] = field(default_factory=list)  #: 逐场明细（点开看战绩用）
    history: list[float] = field(default_factory=list)  #: 分数曲线

    @property
    def alive(self) -> bool:
        #: **不出局**：0 分保底，掉到 0 就停在 0，继续在池子里打
        return True

    @property
    def at_floor(self) -> bool:
        return self.rating <= 0.0

    @property
    def avg_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    @property
    def first_rate(self) -> float:
        """一位率。"""
        return self.wins / self.games if self.games else 0.0

    @property
    def agari_rate(self) -> float:
        """和了率（每局和牌的比例）。"""
        return self.agari / self.hands if self.hands else 0.0

    @property
    def deal_in_rate(self) -> float:
        return self.deal_in / self.hands if self.hands else 0.0


#: 早期权重的显示名（看名字不知道是啥，直接写清楚）
LOOSE_LABELS = {
    "policy_bc4": "bc4（纯克隆基线·5000局）",
    "policy": "policy（v1 早期·带强化学习）",
}


def loose_participants(players: int | None = None, copies: int = 3) -> list[Participant]:
    """把 ``weights/`` 里**不符合 maj-* 命名**的权重也收进来（比如 policy_bc4）。

    人数从网络维度推断：特征 486 = 三麻、560 = 四麻。
    """
    from .encode import layout_for
    from .net import Net

    out: list[Participant] = []
    for f in sorted(variants.WEIGHTS_DIR.glob("*.npz")):
        stem = f.stem
        if stem.startswith("maj-") or stem.endswith(("_bcbase", "_prev")):
            continue
        try:
            net = Net.load(f)
        except Exception:                       # noqa: BLE001
            continue
        n = 3 if net.dim < 500 else 4
        if players and n != players:
            continue
        if net.dim != layout_for(n)[1]:
            continue
        label = LOOSE_LABELS.get(stem, stem)
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"{stem}#{i}" if copies > 1 else stem,
                label=f"{label}{suffix}", source=stem, players=n, kind="model",
                version="早期", style="—",
            ))
    return out


def collect(
    players: int | None = None,
    include: list[str] | None = None,
    copies: int = 3,
) -> list[Participant]:
    """把所有可用的模型（含各版本）+ 内置基线收进来。

    **每个模型版本派出 ``copies`` 个独立参赛者**，各记各的分。
    它们用的是同一份权重，但座位、牌运、对手组合都不一样 ——
    分开记才能看出「同一个模型自己的方差有多大」，而不是被平均掉。
    """
    out: list[Participant] = []
    seen: set[str] = set()

    def add(spec: variants.ModelSpec) -> None:
        if not spec.path.exists() or (players and spec.players != players):
            return
        if spec.name in seen:
            return
        seen.add(spec.name)
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"{spec.name}#{i}" if copies > 1 else spec.name,
                label=f"{spec.name}{suffix}",
                source=spec.name,
                players=spec.players, kind="model",
                version=spec.version, style=variants.STYLE_CN.get(spec.style, spec.style),
            ))

    # ★ 注册表里的（当前版本）+ 盘上所有带版本号的权重 ✓
    #   ⚠️ 这两段被我删过一次 ✗ —— 用 s[:start]+s[end:] 做拼接时，
    #      把「_want 那段的开头」到「loose 那段的开头」之间的东西全删了 ✓
    #      症状是**池子里只剩教师和随机**，一个模型都没有 ✗
    #      （靠 `--models` 试跑时看到「2 个参战者」才发现 ✓ —— 改完必须试跑 ✓）
    for spec in variants.all_specs():          # 当前版本
        add(spec)
    for f in sorted(variants.WEIGHTS_DIR.glob("maj-*v*.npz")):
        if f.stem.endswith(("_bcbase", "_prev")):
            continue
        try:
            add(variants.get(f.stem))
        except KeyError:
            continue

    # 不在命名规则里的早期权重（policy_bc4 / policy 这些）也拉进来当参照
    for p in loose_participants(players, copies):
        if p.source in seen or (players and p.players != players):
            continue
        seen.add(p.source)
        out.extend(p if isinstance(p, list) else [p])
    if include:
        keys = {k.lower() for k in include}
        out = [p for p in out if p.source.lower() in keys]
    for n in (3, 4):
        if players and n != players:
            continue
        for i in range(1, max(1, copies) + 1):
            suffix = f" #{i}" if copies > 1 else ""
            out.append(Participant(
                key=f"teacher{n}#{i}", label=f"教师（{n}麻基线）{suffix}",
                source=f"teacher{n}", players=n, kind="teacher",
            ))
            out.append(Participant(
                key=f"random{n}#{i}", label=f"随机（{n}麻基线）{suffix}",
                source=f"random{n}", players=n, kind="random",
            ))
    return out


class Arena:
    """一张天梯榜 + 匹配 + ELO 结算。"""

    def __init__(
        self,
        players: int = 3,
        k: float = 24.0,
        max_gap: float = 200.0,
        strict: bool = True,
        seed: int = 0,
        participants: list[Participant] | None = None,
        copies: int = 3,
        divisor: float = 500.0,
        models: list[str] | None = None,
    ) -> None:
        self.copies = copies
        #: 点数段位分的除数：``(最终点数 - 初始点数) / divisor``。
        #: 越小点数项越强势（500 时一局能到 ±90，会把名次分完全压过去）；
        #: 设 0 就是关掉点数项，只按名次算。
        self.divisor = divisor
        self.players = players
        self.k = k
        self.max_gap = max_gap
        self.strict = strict
        self.seed = seed
        self.participants = (
            participants if participants is not None
            else collect(players, include=models, copies=copies)
        )
        self.n_matches = 0
        self.log: list[str] = []
        self._nets: dict[str, Net] = {}
        self._rng = random.Random(seed)

    # -- 持久化 -------------------------------------------------------------
    def save(self, path: Path = STATE_FILE) -> None:
        data = {
            "players": self.players, "k": self.k, "max_gap": self.max_gap,
            "n_matches": self.n_matches, "seed": self.seed, "copies": self.copies,
            "divisor": self.divisor,
            "ts": time.time(),
            "participants": [asdict(p) for p in self.participants],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "Arena | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        parts = [Participant(**p) for p in d.get("participants", [])]
        a = cls(players=d.get("players", 3), k=d.get("k", 24.0),
                max_gap=d.get("max_gap", 200.0), participants=parts,
                copies=d.get("copies", 3), seed=d.get("seed", 0),
                divisor=d.get("divisor", 500.0))
        a.n_matches = d.get("n_matches", 0)
        return a

    # -- 匹配 ---------------------------------------------------------------
    def pool(self) -> list[Participant]:
        return sorted([p for p in self.participants if p.alive], key=lambda p: -p.rating)

    def pick(self) -> list[Participant] | None:
        """挑一桌分差不超过 ``max_gap`` 的人；严格模式下挑不出来就返回 None。"""
        alive = sorted(self.pool(), key=lambda p: p.rating)
        n = self.players
        if len(alive) < n:
            return None
        # 在所有「分差 ≤ 上限」的组合里选。**同分差时要优先照顾上场少的**，
        # 否则一堆人都是 500 分、分差都是 0，排序又稳定，排在列表后面的
        # 参赛者会永远抢不到上场机会（真实踩过的坑：bc4 进池后 25 场没上过桌）。
        cands: list[tuple[float, int, list[Participant]]] = []
        for i in range(len(alive) - n + 1):
            group = alive[i : i + n]
            spread = group[-1].rating - group[0].rating
            if spread <= self.max_gap:
                cands.append((spread, sum(p.games for p in group), group))
        if cands:
            min_spread = min(c[0] for c in cands)
            near = [c for c in cands if c[0] <= min_spread + 1e-6]
            fewest = min(c[1] for c in near)
            pool = [c[2] for c in near if c[1] == fewest]
            return self._rng.choice(pool) if len(pool) > 1 else pool[0]
        if self.strict:
            return None
        # 放宽：拿分差最小的那一组，但别总是同样几个人
        i = min(range(len(alive) - n + 1), key=lambda i: alive[i + n - 1].rating - alive[i].rating)
        group = alive[i : i + n]
        self._rng.shuffle(group)
        return group

    # -- 出牌的人 -----------------------------------------------------------
    def _agent(self, p: Participant, seed: int) -> Agent:
        if p.kind == "teacher":
            return HeuristicAgent(seed)
        if p.kind == "random":
            return RandomAgent(seed)
        src = p.source or p.key          # 同一个模型派出的多个参赛者共用权重
        net = self._nets.get(src)
        if net is None:
            net = Net.load(variants.WEIGHTS_DIR / f"{src}.npz")
            self._nets[src] = net
        return MLAgent(net, seed=seed, greedy=True)

    # -- 打一场 -------------------------------------------------------------
    def seed_of(self, index: int) -> int:
        """第 ``index`` 场的随机种子。**只跟种子基数和场次序号有关** ——
        同一套参战者、同一个顺序，重跑就得到一模一样的牌。"""
        return self.seed + index * 7919

    def run_match(self) -> dict | None:
        group = self.pick()
        if group is None:
            return None
        seed = self.seed_of(self.n_matches)
        agents = [self._agent(p, seed * self.players + i) for i, p in enumerate(group)]
        r = Game(agents, seed=seed, rounds=self.players).play()
        ranks = list(r.ranks)
        start_score = S.rule_for(self.players).start_score
        self._settle(group, ranks, r.scores, r, start_score)
        self.n_matches += 1
        text = "  ".join(
            f"{p.label} {r.scores[i]}({ranks[i] + 1}位"
            f" {(r.scores[i] - start_score) / self.divisor if self.divisor else 0:+.0f})"
            for i, p in enumerate(group)
        )
        self.log.append(f"[{self.n_matches}] 🌱{seed}  {text}")
        del self.log[:-200]
        return {
            "group": [p.key for p in group], "ranks": ranks, "scores": list(r.scores),
            "seed": seed,
        }

    def _settle(self, group: list[Participant], ranks: list[int], scores: list[int],
                r=None, start_score: int = 35000) -> None:
        """一局结束给每个人加分 = **额外段位分 + 点数段位分**。

        * **额外段位分**：多人大逃杀 ELO，桌上两两结算，名次靠前算赢
        * **点数段位分**：``(最终点数 - 初始点数) / 500``，可正可负 ——
          这样「同样第 2 名，赢 2 万分」和「小输 3 千」不该是同一个分
        """
        n = len(group)
        delta = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = 1.0 if ranks[i] < ranks[j] else (0.0 if ranks[i] > ranks[j] else 0.5)
                e = 1.0 / (1.0 + 10 ** ((group[j].rating - group[i].rating) / 400.0))
                delta[i] += (s - e)
        for i, p in enumerate(group):
            before = p.rating
            place_term = self.k / (n - 1) * delta[i]          # 额外段位分
            score_term = (
                (scores[i] - start_score) / self.divisor if self.divisor else 0.0
            )
            raw = p.rating + place_term + score_term
            if raw <= 0.0:                       # 0 分保底，不出局
                p.floor_hits += 1
                raw = 0.0
            p.rating = raw
            p.score_term += score_term
            p.place_term += place_term
            p.games += 1
            if r is not None:                      # 和了率 / 放铳率
                st = r.stats[i]
                p.hands += st.hands
                p.agari += st.tsumo + st.ron
                p.deal_in += st.deal_in
            p.ranks.append(ranks[i])
            if ranks[i] == 0:
                p.wins += 1
            if ranks[i] == n - 1:
                p.last += 1
            p.history.append(round(p.rating, 1))
            del p.history[:-300]
            p.matches.append({
                "n": self.n_matches + 1,
                "seed": self.seed_of(self.n_matches),
                "rank": ranks[i] + 1,
                "score": scores[i],
                "pts": round(score_term, 1),
                "place": round(place_term, 1),
                "delta": round(place_term + score_term, 1),
                "rating": round(p.rating, 1),
                "opponents": [q.label for k, q in enumerate(group) if k != i],
            })
            del p.matches[:-200]
            if raw == 0.0 and before > 0:
                self.log.append(f"⬇️ {p.label} 触底 0 分（保底，不出局）")

    # -- 批量跑 -------------------------------------------------------------
    def run(self, matches: int, verbose: bool = True, on_match=None) -> int:
        done = 0
        for _ in range(matches):
            res = self.run_match()
            if res is None:
                if verbose:
                    print("⚠️ 池子里已经没有分差 ≤ "
                          f"{self.max_gap:.0f} 的组合了（活着 {len(self.pool())} 个）")
                break
            done += 1
            if on_match:
                on_match(res)
            if verbose and done % 20 == 0:
                print(f"  跑了 {done} 场…")
        return done

    # -- 报表 ---------------------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        for p in sorted(self.participants, key=lambda x: -x.rating):
            rows.append({
                "名": p.label, "模型": p.source or p.key, "人数": p.players,
                "版本": p.version, "风格": p.style,
                "分数": round(p.rating, 1), "场数": p.games, "一位": p.wins,
                "末位": p.last, "平均顺位": round(p.avg_rank + 1, 2) if p.ranks else 0,
                "一位率": round(p.first_rate, 4), "和了率": round(p.agari_rate, 4),
                "放铳率": round(p.deal_in_rate, 4), "局数": p.hands,
                "点数分": round(p.score_term, 1), "名次分": round(p.place_term, 1),
                "触底": p.floor_hits,
                "状态": "0分保底" if (p.at_floor and p.games) else "存活",
                "曲线": p.history[-120:],
            })
        return rows

    def player_detail(self, key: str) -> dict | None:
        """某个参赛者的完整战绩（点开看历史用）。"""
        p = next((x for x in self.participants if x.key == key or x.label == key), None)
        if p is None:
            return None
        return {
            "key": p.key, "label": p.label, "模型": p.source or p.key,
            "版本": p.version, "风格": p.style, "人数": p.players,
            "分数": round(p.rating, 1), "场数": p.games,
            "一位": p.wins, "末位": p.last,
            "一位率": round(p.first_rate, 4), "和了率": round(p.agari_rate, 4),
            "放铳率": round(p.deal_in_rate, 4),
            "平均顺位": round(p.avg_rank + 1, 2) if p.ranks else 0,
            "名次分": round(p.place_term, 1), "点数分": round(p.score_term, 1),
            "局数": p.hands, "触底": p.floor_hits,
            "状态": "0分保底" if (p.at_floor and p.games) else "存活",
            "曲线": p.history,
            "对局": list(reversed(p.matches)),     # 新的在前
        }

    def print_table(self) -> None:
        print(f"\n{'名次':<5}{'参战者':<22}{'版本':<7}{'风格':<6}{'分数':>8}"
              f"{'场数':>6}{'一位':>6}{'末位':>6}{'均位':>7}  状态")
        for i, r in enumerate(self.table(), 1):
            print(f"{i:<5}{r['名']:<22}{r['版本']:<7}{r['风格']:<6}{r['分数']:>8}"
                  f"{r['场数']:>6}{r['一位']:>6}{r['末位']:>6}{r['平均顺位']:>7}  {r['状态']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ELO 混战池")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--k", type=float, default=24.0, help="ELO 的 K 值（越大分数动得越快）")
    ap.add_argument("--max-gap", type=float, default=200.0, help="一桌允许的最大分差")
    ap.add_argument("--loose", action="store_true", help="分差不够就放宽（默认严格）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子基数：第 n 场的种子 = seed + n×7919（同一种子重跑得到同样的牌）")
    ap.add_argument("--divisor", type=float, default=500.0,
                    help="点数段位分的除数（越小点数项越强；0 = 关掉点数项）")
    ap.add_argument("--models", nargs="*", default=None,
                    help="只让这几个模型参赛（默认全部）✓ 直接接到 collect 的 "
                         "include 上。**要写带版本号的正名**（如 maj-normal4v1.2），"
                         "因为 include 是精确匹配 source ✓。"
                         "例：--models maj-normal4v1.2 maj-pro4v1.1")
    ap.add_argument("--copies", type=int, default=3,
                    help="每个模型版本派几个参赛者（默认 3，各记各的分）")
    ap.add_argument("--reset", action="store_true",
                    help="清空战绩，全部回到 500 分（0 分保底，不出局）")
    ap.add_argument("--save", action="store_true", help="跑完把状态存下来")
    args = ap.parse_args(argv)

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("已清空天梯")

    arena = None if args.reset else Arena.load()
    if arena is None or arena.players != args.players:
        arena = Arena(players=args.players, k=args.k, max_gap=args.max_gap,
                      strict=not args.loose, seed=args.seed, copies=args.copies,
                      divisor=args.divisor, models=args.models)
        print(f"新建 {args.players} 麻池：{len(arena.participants)} 个参战者"
              f"（每个模型 {args.copies} 个），"
              f"初始 500 分，分差上限 {args.max_gap:.0f}，0 分保底（不出局）")
    arena.k, arena.max_gap, arena.strict = args.k, args.max_gap, not args.loose

    t0 = time.perf_counter()
    done = arena.run(args.matches)
    dt = time.perf_counter() - t0
    arena.print_table()
    print(f"\n跑了 {done} 场，用时 {dt:.0f}s（{dt / max(1, done):.1f}s/场）")
    alive = len(arena.pool())
    print(f"存活 {alive} / {len(arena.participants)}")
    if args.save:
        arena.save()
        print(f"已保存到 {STATE_FILE}")


if __name__ == "__main__":
    main()
