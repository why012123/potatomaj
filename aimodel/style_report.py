"""风格对照报告：证明四个模型真的打出了四种脾气。

四个模型的规则完全一样，差别只在「得分条件」。如果训练有效，那么：

* ``mountain``（重罚放铳）→ **放铳率最低**，但和牌也少、流局多
* ``flash``（按和牌次数给钱）→ **和了率最高**、平均番数最低
* ``huge``（按番数给钱）→ **平均番数最高**
* ``normal``（只看段位分）→ 各项都在中间

做法：每个模型各占满一桌打 N 场（三家/四家都是它自己），统计风格指标；
再让四个模型**同桌混战**，看它们互相对上是什么结果。

用法::

    python -m aimodel.style_report            # 三麻
    python -m aimodel.style_report --players 4
    python -m aimodel.style_report --matches 200 --markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import rewards, variants
from . import sanma as S
from .agents import HeuristicAgent, MLAgent, RandomAgent
from .engine import Game
from .net import Net

WEIGHTS = variants.WEIGHTS_DIR


def load(spec: variants.ModelSpec):
    return Net.load(spec.path)


def profile(spec: variants.ModelSpec, matches: int, seed0: int, teacher_style: str) -> dict:
    """这个模型自己占满一桌打若干场，看它的「脾气」。"""
    net = load(spec)
    n = spec.players
    tot = {
        "hands": 0, "win": 0, "tsumo": 0, "ron": 0, "deal_in": 0, "draw": 0,
        "riichi": 0, "calls": 0, "kita": 0, "han": 0, "han_n": 0,
        "win_pts": 0.0, "deal_pts": 0.0, "rp": 0.0, "rank": 0.0,
    }
    for m in range(matches):
        game_seed = seed0 * 7919 + m
        agents = [MLAgent(net, seed=game_seed * n + i, greedy=True) for i in range(n)]
        r = Game(agents, seed=game_seed).play()
        for p in range(n):
            st = r.stats[p]
            tot["hands"] += st.hands
            tot["win"] += st.tsumo + st.ron
            tot["tsumo"] += st.tsumo
            tot["ron"] += st.ron
            tot["deal_in"] += st.deal_in
            tot["draw"] += st.draws
            tot["riichi"] += st.riichi
            tot["calls"] += st.calls
            tot["kita"] += st.kita
            tot["han"] += sum(st.win_han)
            tot["han_n"] += len(st.win_han)
            tot["win_pts"] += sum(st.win_score)
            tot["deal_pts"] += sum(st.deal_in_score)
            tot["rp"] += r.rank_points[p]
            tot["rank"] += r.ranks[p]
    h = max(1, tot["hands"])
    per = max(1, matches * n)
    return {
        "模型": spec.name,
        "风格": variants.STYLE_CN.get(spec.style, spec.style),
        "段位分": tot["rp"] / per,
        "平均顺位": tot["rank"] / per + 1,
        "和了率": tot["win"] / h,
        "自摸率": tot["tsumo"] / h,
        "放铳率": tot["deal_in"] / h,
        "流局率": tot["draw"] / h,
        "立直率": tot["riichi"] / h,
        "副露/局": tot["calls"] / h,
        "拔北/局": tot["kita"] / h,
        "平均番数": (tot["han"] / tot["han_n"]) if tot["han_n"] else 0.0,
        "和了收入": (tot["win_pts"] / max(1, tot["win"])) if tot["win"] else 0.0,
        "放铳支出": (tot["deal_pts"] / max(1, tot["deal_in"])) if tot["deal_in"] else 0.0,
    }


def mixed(specs: list[variants.ModelSpec], matches: int, seed0: int) -> list[dict]:
    """同桌混战：每次从这些模型里挑「一桌人」上场，座位轮换。

    注意三麻只有 3 个座位 —— 4 个三麻模型不能同时上桌（那会变成四麻），
    所以每场轮着把其中一个晾在一边。
    """
    n = specs[0].players
    nets = {s.name: load(s) for s in specs}
    tot: dict[str, dict] = {
        s.name: dict(hands=0, win=0, deal_in=0, han=0, han_n=0, rp=0.0, rank=0.0, plays=0)
        for s in specs
    }
    for m in range(matches):
        game_seed = seed0 * 104729 + m
        roster = [specs[(m + i) % len(specs)] for i in range(n)]
        agents = [
            MLAgent(nets[sp.name], seed=game_seed * n + seat, greedy=True)
            for seat, sp in enumerate(roster)
        ]
        r = Game(agents, seed=game_seed, rule=S.rule_for(n)).play()
        for seat, sp in enumerate(roster):
            t = tot[sp.name]
            st = r.stats[seat]
            t["hands"] += st.hands
            t["win"] += st.tsumo + st.ron
            t["deal_in"] += st.deal_in
            t["han"] += sum(st.win_han)
            t["han_n"] += len(st.win_han)
            t["rp"] += r.rank_points[seat]
            t["rank"] += r.ranks[seat]
            t["plays"] += 1
    out = []
    for spec in specs:
        t = tot[spec.name]
        h = max(1, t["hands"])
        plays = max(1, t["plays"])
        out.append({
            "模型": spec.name,
            "风格": variants.STYLE_CN.get(spec.style, spec.style),
            "上场": t["plays"],
            "段位分": t["rp"] / plays,
            "平均顺位": t["rank"] / plays + 1,
            "和了率": t["win"] / h,
            "放铳率": t["deal_in"] / h,
            "平均番数": (t["han"] / t["han_n"]) if t["han_n"] else 0.0,
        })
    return out


def table(rows: list[dict], cols: list[str], title: str) -> str:
    head = f"{'模型':<14}{'风格':<6}" + "".join(f"{c:>10}" for c in cols)
    out = [f"\n{title}", "  " + head]
    for r in rows:
        line = f"  {r['模型']:<14}{r['风格']:<6}"
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                line += f"{v:>10.2f}" if abs(v) < 100 else f"{v:>10.0f}"
            else:
                line += f"{v:>10}"
        out.append(line)
    return "\n".join(out)


def markdown(rows: list[dict], cols: list[str], title: str) -> str:
    out = [f"\n**{title}**\n", "| 模型 | 风格 | " + " | ".join(cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for r in rows:
        cells = [f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        out.append(f"| {r['模型']} | {r['风格']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="四个模型的风格对照")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--matches", type=int, default=120, help="每个模型自己打几场")
    ap.add_argument("--mixed", type=int, default=160, help="混战几场")
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    specs = [s for s in variants.all_specs() if s.players == args.players and s.path.exists()]
    if not specs:
        print(f"⚠️ 还没有 {args.players} 麻的模型权重，先训练："
              f"python -m aimodel.train_all --models "
              f"{','.join(s.name for s in variants.all_specs() if s.players == args.players)}")
        return

    style_cols = ["和了率", "放铳率", "流局率", "立直率", "副露/局", "平均番数", "和了收入", "放铳支出"]
    rows = []
    for spec in specs:
        print(f"▶ {spec.name} 自己打 {args.matches} 场 …", flush=True)
        rows.append(profile(spec, args.matches, args.seed, spec.style))

    print(table(rows, style_cols, f"【{args.players} 麻】各模型的脾气（自己占满一桌）"))
    if args.markdown:
        print(markdown(rows, style_cols, f"{args.players} 麻：各模型风格"))

    if len(specs) >= 2:
        print(f"\n▶ 混战 {args.mixed} 场 …", flush=True)
        mrows = mixed(specs, args.mixed, args.seed)
        mcols = ["上场", "段位分", "平均顺位", "和了率", "放铳率", "平均番数"]
        print(table(mrows, mcols, f"【{args.players} 麻】同桌混战（座位轮换）"))
        if args.markdown:
            print(markdown(mrows, mcols, f"{args.players} 麻：混战结果"))

    # 简单自检：四种风格是否真的分开了
    print("\n风格是否真的不同？")
    by = {r["风格"]: r for r in rows}
    checks = [
        ("防守", "放铳率", "min"), ("速攻", "和了率", "max"), ("大牌", "平均番数", "max"),
    ]
    for style, col, kind in checks:
        if style not in by:
            continue
        vals = {r["风格"]: r[col] for r in rows}
        best = min(vals, key=vals.get) if kind == "min" else max(vals, key=vals.get)
        ok = "✅" if best == style else "❌"
        print(f"  {ok} {col} 的 {kind}: 期望「{style}」，实际是「{best}」"
              f"（" + "  ".join(f"{k} {v:.2f}" for k, v in vals.items()) + "）")


if __name__ == "__main__":
    main()
