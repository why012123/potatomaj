#!/usr/bin/env python
"""导出/统计决策级对局记录。

记录是按天分文件的 gzip JSONL（``data/games/games-YYYY-MM-DD.jsonl.gz``）。
这个工具把它们读成人能看的东西。

用法::

    python tools/dump_games.py --list                       # 有哪些分片
    python tools/dump_games.py --stats                      # 全局统计
    python tools/dump_games.py --since 2026-10-01 --out x.jsonl
    python tools/dump_games.py --kind decision --game <uid> --out one.jsonl
    python tools/dump_games.py --player p_ab12cd34 --stats
    python tools/dump_games.py --show-profiles              # 玩家画像表
    python tools/dump_games.py --since 2026-10-01 --pretty  # 人看的（不压缩）

导出的是**未压缩** JSONL（一行一条），直接能喂给训练脚本。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jsonlgz import iter_jsonl_gz  # noqa: E402  （容忍截断的 gzip 读取器）

DEFAULT_DIR = ROOT / "data" / "games"
PROFILE_PATH = ROOT / "data" / "profiles.json"


def parse_day(text: str) -> date | None:
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"看不懂的日期：{text}（要 YYYY-MM-DD）")


def iter_files(games_dir: Path, since: date | None, until: date | None):
    for f in sorted(games_dir.glob("games-*.jsonl.gz")):
        stem = f.name[len("games-") : -len(".jsonl.gz")]
        day_part = stem.split(".")[0]
        try:
            d = datetime.strptime(day_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if since and d < since:
            continue
        if until and d > until:
            continue
        yield f, d


def iter_records(games_dir: Path, since: date | None, until: date | None,
                 kind: str | None = None, game: str | None = None,
                 player: str | None = None):
    for f, _d in iter_files(games_dir, since, until):
        for rec in iter_jsonl_gz(f):
            if kind and rec.get("kind") != kind:
                continue
            if game and rec.get("game_uid") != game:
                continue
            if player and rec.get("player") != player \
                    and (rec.get("profile") or {}).get("uid") != player:
                continue
            yield rec, f


def cmd_list(games_dir: Path) -> int:
    files = list(iter_files(games_dir, None, None))
    if not files:
        print(f"（{games_dir} 下还没有记录）")
        return 0
    print(f"记录目录：{games_dir}")
    print(f"{'文件':40s} {'大小':>10s} {'行数':>10s}")
    total = 0
    for f, _d in files:
        n = 0
        for _rec in iter_jsonl_gz(f):
            n += 1
        total += n
        print(f"{f.name:40s} {f.stat().st_size / 1e6:9.2f}M {n:10d}")
    print(f"{'合计':40s} {'':>10s} {total:10d} 行")
    return 0


def cmd_stats(games_dir: Path, since: date | None, until: date | None,
              player: str | None) -> int:
    kinds: Counter[str] = Counter()
    games: set[str] = set()
    players: Counter[str] = Counter()
    names: dict[str, str] = {}
    decisions = 0
    elapsed: list[int] = []
    sources: Counter[str] = Counter()
    act_kinds: Counter[str] = Counter()
    ai_models: Counter[str] = Counter()
    hands = 0
    results: Counter[str] = Counter()
    ai_vs_ai = 0
    human_mixed = 0
    for rec, _f in iter_records(games_dir, since, until, player=player):
        k = rec.get("kind")
        kinds[k] += 1
        if rec.get("game_uid"):
            games.add(rec["game_uid"])
        if k == "decision":
            decisions += 1
            if rec.get("elapsed_ms") is not None:
                elapsed.append(int(rec["elapsed_ms"]))
            sources[rec.get("source") or "?"] += 1
            act_kinds[rec.get("decision") or "?"] += 1
            if rec.get("ai"):
                ai_models[rec["ai"]] += 1
            p = rec.get("player") or ""
            if p:
                players[p] += 1
                names[p] = rec.get("name") or names.get(p, "")
        elif k == "hand":
            hands += 1
            results[rec.get("result_kind") or "?"] += 1
        elif k == "game_start":
            seats = rec.get("seats") or []
            kinds_h = {s.get("kind") for s in seats}
            if kinds_h == {"ai"}:
                ai_vs_ai += 1
            elif "human" in kinds_h and "ai" in kinds_h:
                human_mixed += 1
    print(f"记录目录：{games_dir}")
    print(f"  对局数     {len(games)}（纯 AI {ai_vs_ai}，人机混合 {human_mixed}）")
    print(f"  局数       {hands}")
    print(f"  决策数     {decisions}")
    print(f"  记录类型   {dict(kinds)}")
    if results:
        print(f"  局结果     {dict(results)}")
    if act_kinds:
        print(f"  决策种类   {dict(act_kinds)}")
    if sources:
        print(f"  动作来源   {dict(sources)}   ← teacher_fallback / no_head 是降级")
    if ai_models:
        print("  AI 模型    " + ", ".join(f"{k}={v}"
                                          for k, v in ai_models.most_common(6)))
    if elapsed:
        elapsed.sort()
        n = len(elapsed)
        print(f"  思考耗时   中位 {elapsed[n // 2]} ms，"
              f"P90 {elapsed[int(n * 0.9)]} ms，最长 {elapsed[-1]} ms")
    if players:
        print(f"  玩家数     {len(players)}（匿名 ID）")
        for p, n in players.most_common(5):
            print(f"      {p}  {names.get(p, ''):10s} {n} 条决策")
    return 0


def cmd_export(games_dir: Path, since: date | None, until: date | None, out: Path,
               kind: str | None, game: str | None, player: str | None,
               pretty: bool) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec, _f in iter_records(games_dir, since, until, kind, game, player):
            if pretty:
                fh.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
            else:
                fh.write(json.dumps(rec, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            n += 1
    print(f"✓ 导出 {n} 条 -> {out}（{out.stat().st_size / 1e6:.2f} MB）")
    if kind == "decision" and n:
        print("  每行是一条决策：view 是局面快照，options 是引擎给的完整合法动作，"
              "choice_index 是实际选择，elapsed_ms 是思考耗时")
    return 0


def cmd_profiles() -> int:
    if not PROFILE_PATH.exists():
        print(f"（{PROFILE_PATH} 还不存在）")
        return 0
    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    players = data.get("players") or {}
    rows = sorted(players.values(), key=lambda r: -float(r.get("elo") or 0))
    print(f"玩家画像表：{PROFILE_PATH}（{len(rows)} 人）")
    print(f"{'匿名 ID':12s} {'名字':10s} {'类型':5s} {'局数':>5s} {'一位率':>7s} "
          f"{'均顺':>6s} {'和了率':>7s} {'放铳率':>7s} {'ELO':>7s}")
    for r in rows[:40]:
        print(f"{str(r.get('uid', '')):12s} {str(r.get('name', ''))[:10]:10s} "
              f"{str(r.get('kind', ''))[:5]:5s} {r.get('games', 0):5d} "
              f"{float(r.get('first_rate') or 0):7.3f} "
              f"{float(r.get('avg_rank') or 0):6.3f} "
              f"{float(r.get('agari_rate') or 0):7.3f} "
              f"{float(r.get('deal_in_rate') or 0):7.3f} "
              f"{float(r.get('elo') or 0):7.1f}")
    print("\n注：ELO 和 AI 用完全同一套公式（成对比较、零和、同一个 K），"
          "所以人和模型的分数在同一量纲上。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出 / 统计决策级对局记录")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="记录目录")
    ap.add_argument("--since", default="", help="起始日期 YYYY-MM-DD")
    ap.add_argument("--until", default="", help="结束日期 YYYY-MM-DD")
    ap.add_argument("--out", default="", help="导出到这个文件")
    ap.add_argument("--kind", default="", help="只看某类记录（decision/hand/match/…）")
    ap.add_argument("--game", default="", help="只看某一局（game_uid）")
    ap.add_argument("--player", default="", help="只看某个匿名玩家 ID")
    ap.add_argument("--list", action="store_true", help="列出分片")
    ap.add_argument("--stats", action="store_true", help="统计")
    ap.add_argument("--show-profiles", action="store_true", help="打印玩家画像表")
    ap.add_argument("--pretty", action="store_true", help="导出成人看的缩进 JSON")
    args = ap.parse_args(argv)

    games_dir = Path(args.dir)
    since, until = parse_day(args.since), parse_day(args.until)
    if until and not since:
        since = until - timedelta(days=1)
    if args.list:
        return cmd_list(games_dir)
    if args.show_profiles:
        return cmd_profiles()
    if args.out:
        return cmd_export(games_dir, since, until, Path(args.out),
                          args.kind or None, args.game or None,
                          args.player or None, args.pretty)
    return cmd_stats(games_dir, since, until, args.player or None)


if __name__ == "__main__":
    raise SystemExit(main())
