#!/usr/bin/env python
"""把官方天梯的**历史各期按「每期独立」口径重算**（回填脚本）。

## 为什么要它

官方天梯原来把 ELO **跨期累积**：每期收尾落盘的 ``arena.json`` 里
``rating`` 一路 946 -> 1209 -> 1462 -> 1680，``games`` 也累积到 48000 ✗
—— 于是页面上的「该期最终 ELO / 冠军均分」**不是该期的独立成绩** ✗
各期之间不可比 ✗（用户 2026-10-21 实测指出）。

而 ``matches.jsonl`` **每期只存该期的对局**（一行一局，含 ``ranks`` / ``scores``）✓
-> 重算**不需要重放牌局** ✓ 直接按名次跑 ELO 更新就行 ✓

## 口径（一条都不许自己发明 ✗）

* 起始分：:data:`aimodel.arena.INITIAL_RATING`（就是 ``Participant.rating`` 的默认值）
* K 值 / 点数除数：取**该期自己** ``arena.json`` 里的 ``k`` / ``divisor``
  （第 4 期确实是 ``k=30`` / ``divisor=1000``，后面几期是 ``24`` / ``500`` ✓
   用别的期的值就是换了一种口径 ✗）
* 算法：:meth:`aimodel.arena.Arena.replay_ranks` -> :meth:`Arena._settle`
  —— **和实时跑局 ``run_match()`` 同一条路径** ✓ 没有第二份公式 ✓

## 输出（**新目录**，绝不覆盖生产文件 ✗）

    _data/elo/reseason/<期号>/arena.json     <- 该期独立的分数状态（Arena.save 格式）
    _data/elo/reseason/<期号>/scores.json    <- 该期的最终 ELO 表（同 scores/<期>.json 的形状）
    _data/elo/reseason/<期号>/meta.json      <- 该期元数据（冠军 / 场次 / 新口径标记）
    _data/elo/reseason/report.json           <- 逐期 + 可比性指标（机器可读）
    _data/elo/reseason/report.md             <- 同上，人看的

跑法::

    python tools/elo_reseason.py --src <生产 _data/elo 的副本> --out <输出根>
    python tools/elo_reseason.py --dry-run          # 只算不落盘

⚠️ ``--src`` 默认是 ``<仓库根>/_data/elo``；不在那个目录时显式传 ✗
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimodel import elo_store as E                      # noqa: E402
from aimodel.arena import INITIAL_RATING, Arena, Participant  # noqa: E402


# ---------------------------------------------------------------------------
# 读盘
# ---------------------------------------------------------------------------


def read_matches(d: Path) -> list[dict]:
    """读该期的 ``matches.jsonl``（4000 行级别，直接进内存）。"""
    p = d / "matches.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def model_of_key(key: str) -> str:
    """参战者 key -> 模型名（去掉 ``#N`` 实例后缀）。"""
    return key.split("#", 1)[0]


def synth_participant(key: str) -> Participant:
    """没有名单可借时，从 key 硬造一个参战者（**兜底**，正常不该走到）。"""
    src = model_of_key(key)
    low = src.lower()
    if low.startswith("teacher"):
        kind = "teacher"
    elif low.startswith("random"):
        kind = "random"
    elif low.startswith("strong"):
        kind = "strong"
        src = "strong"
    else:
        kind = "model"
    return Participant(key=key, label=key, players=4, kind=kind, source=src)


def build_roster(season_dir: Path, sids_dirs: list[Path],
                 observed: set[str]) -> tuple[list[Participant], str]:
    """该期的参战者名单。

    优先用**该期自己的** ``arena.json``；没有（早期几期没落盘）就借**别的期**的
    名单 —— 前提是那份名单能覆盖该期出现过的所有 key ✓
    （实测 7 期的 key 集合完全一致，只是前 3 期没存 arena.json ✓）

    返回 ``(participants, 来源说明)``。
    """
    own = read_json(season_dir / "arena.json")
    if own and own.get("participants"):
        return ([Participant(**p) for p in own["participants"]],
                f"{season_dir.name}/arena.json")

    # 借名单：优先能覆盖 observed 的，且实例数最接近的
    for d in sids_dirs:
        if d == season_dir:
            continue
        o = read_json(d / "arena.json")
        if not o or not o.get("participants"):
            continue
        keys = [p["key"] for p in o["participants"]]
        if observed <= set(keys):
            return ([Participant(**p) for p in o["participants"]],
                    f"{d.name}/arena.json（借名单）")

    # 连一份能覆盖的都没有 -> 按出现顺序硬造（并**大声说**）
    print(f"  ⚠️ {season_dir.name}: 找不到能覆盖全部 key 的名单，按 key 硬造参战者")
    return ([synth_participant(k) for k in sorted(observed)], "按 key 硬造")


def season_params(season_dir: Path, records: list[dict]) -> dict:
    """该期的 ``players`` / ``k`` / ``divisor`` / ``seed``（**取该期自己的值** ✓）。"""
    own = read_json(season_dir / "arena.json") or {}
    meta = read_json(season_dir / "meta.json") or {}
    g0 = len((records[0].get("group") if records else []) or [])
    players = int(own.get("players") or g0 or 4)
    return {
        "players": players,
        # Arena 的默认值就是 24.0 / 500.0 —— 早期没落盘的期按默认 ✓
        "k": float(own.get("k", 24.0)),
        "divisor": float(own.get("divisor", 500.0)),
        "seed": int(meta.get("seed_base") or own.get("seed") or 0),
        "target": int(meta.get("target") or 0),
        "started_at": float(meta.get("started_at") or 0.0),
        "finished_at": float(meta.get("finished_at") or 0.0),
        "note": str(meta.get("note") or ""),
        "pruned": bool(meta.get("pruned")),
        "favorite": bool(meta.get("favorite")),
    }


def zero_or(v, default):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return n or default


def old_champion(src: Path, sid: str) -> tuple[str, float]:
    """旧口径下的冠军（**该期 meta.json 里记的**；没有就退回 scores/<期>.json）。"""
    meta = read_json(src / "seasons" / sid / "meta.json") or {}
    if meta.get("champion"):
        return str(meta["champion"]), float(meta.get("champion_avg") or 0.0)
    sc = read_json(src / "scores" / f"{sid}.json") or {}
    rows = sc.get("table") or []
    if rows:
        m, avg, _inst, _isc = E.champion_by_model(rows)
        return m, avg
    return "", 0.0


# ---------------------------------------------------------------------------
# 重算一期
# ---------------------------------------------------------------------------


def reseason_one(src: Path, season_dir: Path, all_dirs: list[Path]) -> dict:
    """重算一期，返回该期的结果 dict（含可比性指标）。"""
    sid = season_dir.name
    records = read_matches(season_dir)
    observed = {str(k) for r in records for k in (r.get("group") or [])}
    participants, roster_src = build_roster(season_dir, all_dirs, observed)
    prm = season_params(season_dir, records)
    players = prm["players"]

    arena = Arena(players=players, k=prm["k"], divisor=prm["divisor"],
                  max_gap=200.0, strict=True, seed=prm["seed"],
                  participants=participants)
    # ⚠️ 名单是从该期 arena.json 里读的 —— 那里的 rating/games 是**累积值** ✗
    #    重算的第一步必须是「回到初始分」✓ 否则量出来的「起始分」根本不是起点 ✓
    arena.new_season()

    # ── ① 起始分：重算前必须是**全部相同**的初始分 ──
    start_ratings = [p.rating for p in arena.participants]
    start_ok = bool(start_ratings) and all(
        abs(r - INITIAL_RATING) < 1e-9 for r in start_ratings)

    # ── ② 对手强度：结算前抓一桌当时的分数（只用于指标，不影响结果 ✓）──
    opp_sum: dict[str, float] = {}
    opp_n: dict[str, int] = {}

    def _hook(group, ranks, scores) -> None:
        for i, p in enumerate(group):
            others = [q.rating for j, q in enumerate(group) if j != i]
            if not others:
                continue
            opp_sum[p.key] = opp_sum.get(p.key, 0.0) + sum(others) / len(others)
            opp_n[p.key] = opp_n.get(p.key, 0) + 1

    done = arena.replay_ranks(records, fresh=False, before_settle=_hook)
    # 每期独立：这一期自己的 Arena 里 done_total 就是本期局数（不是服务级累计）
    arena.done_total = arena.n_matches

    table = arena.table()
    champ, champ_avg, champ_inst, champ_inst_score = E.champion_by_model(table)

    # ── 回归锚点：和「当年真实跑出来」的那份 scores 表逐实例对比 ──
    # 如果该期当年**本来就是从初始分起算**的（例：08 那期刚好 arena 归零过），
    # 那么重算必须**一个数都不差** ✓ 差一点就说明重算口径和实时路径不一致 ✗
    # 累积起算的那几期差得很大是**预期**（那正是要修的东西 ✓）
    old_table = (read_json(src / "scores" / f"{sid}.json") or {}).get("table") or []
    old_by_name = {str(r.get("名")): float(r.get("分数") or 0.0) for r in old_table}
    diffs = [abs(float(r["分数"]) - old_by_name[str(r["名"])])
             for r in table if str(r["名"]) in old_by_name]
    anchor = {
        "compared": len(diffs),
        "max_diff": round(max(diffs), 3) if diffs else None,
        "exact": bool(diffs) and max(diffs) < 1e-6,
    }

    # ── ③ 可比性指标 ──
    games = [int(p.games) for p in arena.participants]
    played = [g for g in games if g > 0]
    mean_g = (sum(games) / len(games)) if games else 0.0
    med_g = statistics.median(games) if games else 0.0
    sd_g = statistics.pstdev(games) if len(games) > 1 else 0.0
    cv = (sd_g / mean_g) if mean_g else 0.0
    lo, hi = mean_g * 0.8, mean_g * 1.2
    within = [p.key for p in arena.participants if lo <= p.games <= hi]
    opp_means = {k: opp_sum[k] / opp_n[k] for k in opp_sum if opp_n.get(k)}
    opp_vals = list(opp_means.values())

    def _spread(vals: list[float]) -> dict:
        if not vals:
            return {"min": 0.0, "median": 0.0, "max": 0.0}
        return {"min": round(min(vals), 1),
                "median": round(statistics.median(vals), 1),
                "max": round(max(vals), 1)}

    om = _spread(opp_vals)
    old_m, old_avg = old_champion(src, sid)
    old_arena = read_json(season_dir / "arena.json") or {}
    old_nm = int(old_arena.get("n_matches") or 0)
    old_games_total = sum(int(p.get("games") or 0)
                          for p in (old_arena.get("participants") or []))

    result = {
        "sid": sid,
        "roster_from": roster_src,
        "players": players,
        "k": prm["k"],
        "divisor": prm["divisor"],
        "matches": len(records),
        "replayed": done,
        # ① 起始分
        "start_rating": {"value": INITIAL_RATING,
                         "min": round(min(start_ratings), 3) if start_ratings else 0.0,
                         "max": round(max(start_ratings), 3) if start_ratings else 0.0,
                         "all_same": start_ok},
        # ② 对手强度（越高说明那批对手越强）
        "opponent_rating": om,
        # ③ 场次分布
        "games": {
            "n": len(games),
            "total": sum(games),
            "per_seat_expected": len(records) * players,
            "mean": round(mean_g, 1),
            "median": round(med_g, 1),
            "min": min(games) if games else 0,
            "max": max(games) if games else 0,
            "stdev": round(sd_g, 1),
            "cv": round(cv, 4),
            "unplayed": sum(1 for g in games if g == 0),
            "played_mean": round(sum(played) / len(played), 1) if played else 0.0,
            "band_lo": round(lo, 1),
            "band_hi": round(hi, 1),
            "within_band": len(within),
            "within_band_pct": round(100.0 * len(within) / len(games), 1) if games else 0.0,
            "worst": sorted(({"key": p.key, "games": p.games}
                             for p in arena.participants),
                            key=lambda x: x["games"])[:5],
        },
        # ④ 冠军
        "champion": {"model": champ, "avg": champ_avg,
                     "inst": champ_inst, "inst_score": champ_inst_score},
        "old_champion": {"model": old_m, "avg": old_avg},
        "anchor": anchor,
        "old": {"n_matches": old_nm, "games_total": old_games_total},
        "new": {"n_matches": arena.n_matches, "games_total": sum(games),
                "done_total": arena.done_total},
        "meta": prm,
        "_arena": arena,
        "_table": table,
    }
    return result


# ---------------------------------------------------------------------------
# 落盘（**新目录**）
# ---------------------------------------------------------------------------


def write_outputs(res: dict, out_root: Path) -> None:
    sid = res["sid"]
    d = out_root / sid
    d.mkdir(parents=True, exist_ok=True)
    arena: Arena = res["_arena"]
    table = res["_table"]

    # 1) 分数状态（Arena.save 的格式，能被 Arena.load 读回来 ✓）
    arena.save(d / "arena.json")

    # 2) 该期最终 ELO 表（**和实时收尾同一个写法** E.save_scores ✓ 只是换了目录）
    (d / "scores.json").write_text(json.dumps(
        {"sid": sid, "at": time.time(), "table": table,
         "summary": {"players": res["players"], "matches": res["matches"],
                     "reseason": True}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    # 3) 元数据（Season 的形状，能被 elo_store 认 ✓）
    prm = res["meta"]
    s = E.Season(
        sid=sid, players=res["players"], copies=3,
        seed_base=prm["seed"], started_at=prm["started_at"],
        finished_at=prm["finished_at"] or time.time(),
        target=prm["target"] or res["matches"],
        done=res["matches"],                       # <- **本期**局数（不是累计 ✗）
        champion=res["champion"]["model"], champion_key=res["champion"]["model"],
        champion_avg=res["champion"]["avg"],
        champion_inst=res["champion"]["inst"],
        champion_inst_score=res["champion"]["inst_score"],
        pruned=False, favorite=prm["favorite"],
        note=((prm["note"] + "；") if prm["note"] else "")
             + f"按每期独立口径重算（起始 {INITIAL_RATING:.0f} 分，"
               f"k={res['k']:g}，divisor={res['divisor']:g}）",
    )
    (d / "meta.json").write_text(
        json.dumps(s.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def public_view(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# 硬校验（测试也直接用这个函数，不许另写一份判据 ✗）
# ---------------------------------------------------------------------------


def audit_season(res: dict, *, expected_matches: int | None = None) -> list[str]:
    """把一期的结果按「每期独立」的硬指标检一遍，返回问题列表（空 = 通过）。

    查的事：
      ① 起始分是否**全部相同**（每期从同一个初始分起算 ✓ 跨期可比的地基）
      ② ``games`` 合计是否 == 对局数 × 人数（口径有没有串）
      ③ 是不是每一局都真的结算了（``replayed == matches``）
      ④ ``n_matches``（本期序号）是否 == 该期对局数（不是累计 ✗）

    ``expected_matches``：该期**计划**跑多少局（官方档位 4000）。给了就额外查
    「本期局数有没有**超**」——

    ⚠️ 这一条是「某期被多喂了上一期的对局」唯一能抓到的信号 ✗
       ``games`` 合计和 ``n_matches`` 自己跟自己永远对得上（多一局就两边都多）✓
       只有跟**计划**比才看得出「这期多打了」✗（用户要求的负向验证就是这条 ✓）
    """
    bad: list[str] = []
    sid = res.get("sid", "?")
    sr = res.get("start_rating") or {}
    g = res.get("games") or {}
    if not sr.get("all_same"):
        bad.append(f"{sid}: 起始分不全是 {INITIAL_RATING:g}"
                   f"（min={sr.get('min')} max={sr.get('max')}）")
    if g.get("total") != g.get("per_seat_expected"):
        bad.append(f"{sid}: games 合计 {g.get('total')} != "
                   f"对局×人数 {g.get('per_seat_expected')}")
    if res.get("replayed") != res.get("matches"):
        bad.append(f"{sid}: 只结算了 {res.get('replayed')}/{res.get('matches')} 局")
    if (res.get("new") or {}).get("n_matches") != res.get("matches"):
        bad.append(f"{sid}: n_matches {(res.get('new') or {}).get('n_matches')} != "
                   f"该期对局数 {res.get('matches')}")
    if expected_matches is not None:
        n = int(res.get("matches") or 0)
        if n > expected_matches:
            bad.append(f"{sid}: 该期对局 **{n} 局，超了计划 {expected_matches} 局**"
                       f"（多喂了 {(n - expected_matches) * int(res.get('players') or 4)} 个座位"
                       f" -> games 合计 {g.get('total')} 超了 "
                       f"{expected_matches * int(res.get('players') or 4)}）")
        elif n < expected_matches:
            bad.append(f"{sid}: 该期只跑了 {n} 局 < 计划 {expected_matches} 局")
    return bad


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def fmt_report(results: list[dict], src: Path, out_root: Path) -> str:
    L: list[str] = []
    A = L.append
    A(f"# 官方天梯「每期独立」重算报告\n")
    A(f"- 数据源：`{src}`")
    A(f"- 输出：`{out_root}`")
    A(f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- 初始分：**{INITIAL_RATING:.0f}**（`aimodel.arena.INITIAL_RATING`）")
    A("")
    A("## 1. 起点 / 场次口径\n")
    A("| 期号 | 对局数 | 起始分 min/max | 全部相同 | 新 games 合计 | 应 = 对局数×人数 | 旧 games 合计 | 旧 n_matches |")
    A("|---|---|---|---|---|---|---|---|")
    for r in results:
        sr = r["start_rating"]
        g = r["games"]
        A(f"| {r['sid']} | {r['matches']} | {sr['min']:g} / {sr['max']:g} | "
          f"{'✅' if sr['all_same'] else '🔴'} | {g['total']} | "
          f"{g['per_seat_expected']} | {r['old']['games_total'] or '—'} | "
          f"{r['old']['n_matches'] or '—'} |")
    A("")
    A("## 2. 可比性：上场次数是否均匀（**关键指标**）\n")
    A("| 期号 | 参战者数 | 均值 | 中位 | min | max | 标准差 | 变异系数 | 均值±20% 内占比 | 0 场上场者 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        g = r["games"]
        A(f"| {r['sid']} | {g['n']} | {g['mean']} | {g['median']} | {g['min']} | "
          f"{g['max']} | {g['stdev']} | {g['cv']:.2f} | "
          f"**{g['within_band_pct']}%** | {g['unplayed']} |")
    A("")
    A("## 3. 对手强度是否接近\n")
    A("| 期号 | 对手均分 min | 中位 | max |")
    A("|---|---|---|---|")
    for r in results:
        o = r["opponent_rating"]
        A(f"| {r['sid']} | {o['min']} | {o['median']} | {o['max']} |")
    A("")
    A("## 4. 冠军\n")
    A("| 期号 | 旧冠军(分) | 新冠军(分) | 冠军变化 | 新单实例第一(分) |")
    A("|---|---|---|---|---|")
    for r in results:
        oc, nc = r["old_champion"], r["champion"]
        o = f"{oc['model']}({oc['avg']})" if oc["model"] else "—"
        n = f"{nc['model']}({nc['avg']})"
        same = "✅ 不变" if oc["model"] == nc["model"] else "🔁 变了"
        A(f"| {r['sid']} | {o} | {n} | {same} | "
          f"{nc['inst']}({nc['inst_score']}) |")
    A("")
    A("## 5. 回归锚点：重算 vs 当年真实跑出来的表\n")
    A("> 某期当年若**本来就是从初始分起算**，重算必须一个数都不差 → `max_diff = 0.0` ✓")
    A("> 累积起算的那几期 `max_diff` 很大是**预期**（那正是要修的东西）。\n")
    A("| 期号 | 对比实例数 | 最大分差 | 判定 |")
    A("|---|---|---|---|")
    for r in results:
        a = r["anchor"]
        md = "—" if a["max_diff"] is None else f"{a['max_diff']}"
        verdict = "✅ 完全一致（该期本来就从初始分起算）" if a["exact"] else "（口径变了，预期）"
        A(f"| {r['sid']} | {a['compared']} | {md} | {verdict} |")
    A("")
    return "\n".join(L)


def _n_keys(r: dict) -> int:
    return int(r["games"].get("n") or 0)


def print_report(results: list[dict]) -> None:
    print()
    print("=" * 118)
    print("  ① 起点 / 场次口径")
    print("=" * 118)
    print(f"{'期号':<24}{'对局':>6}{'起始min':>9}{'起始max':>9}{'全同':>6}"
          f"{'新games合计':>12}{'=对局×人':>10}{'旧games':>9}{'旧n_m':>8}")
    for r in results:
        sr, g = r["start_rating"], r["games"]
        print(f"{r['sid']:<24}{r['matches']:>6}{sr['min']:>9.1f}{sr['max']:>9.1f}"
              f"{('✅' if sr['all_same'] else '🔴'):>6}{g['total']:>12}"
              f"{g['per_seat_expected']:>10}{r['old']['games_total']:>9}"
              f"{r['old']['n_matches']:>8}")
    print()
    print("=" * 118)
    print("  ② 可比性关键指标：上场次数分布（均值±20% 内占比越高越可比）")
    print("=" * 118)
    print(f"{'期号':<24}{'均值':>8}{'中位':>8}{'min':>6}{'max':>6}{'标准差':>9}"
          f"{'变异':>7}{'±20%内':>9}{'0场':>5}   最少的几个")
    for r in results:
        g = r["games"]
        worst = " ".join(f"{w['key'].split('#')[0][:10]}:{w['games']}"
                         for w in g["worst"][:3])
        print(f"{r['sid']:<24}{g['mean']:>8.1f}{g['median']:>8.1f}{g['min']:>6}"
              f"{g['max']:>6}{g['stdev']:>9.1f}{g['cv']:>7.2f}"
              f"{g['within_band_pct']:>8.1f}%{g['unplayed']:>5}   {worst}")
    print()
    print("=" * 118)
    print("  ③ 对手强度（结算时对手的当时分数，越接近说明同对手池）")
    print("=" * 118)
    print(f"{'期号':<24}{'min':>10}{'中位':>10}{'max':>10}")
    for r in results:
        o = r["opponent_rating"]
        print(f"{r['sid']:<24}{o['min']:>10.1f}{o['median']:>10.1f}{o['max']:>10.1f}")
    print()
    print("=" * 118)
    print("  ④ 冠军（旧 = 该期 meta.json 里记的；新 = 按每期独立重算）")
    print("=" * 118)
    print(f"{'期号':<24}{'旧冠军':<34}{'新冠军':<34}{'变化':>8}{'新单实例第一':>22}")
    for r in results:
        oc, nc = r["old_champion"], r["champion"]
        o = f"{oc['model']}({oc['avg']})" if oc["model"] else "—"
        n = f"{nc['model']}({nc['avg']})"
        same = "不变" if oc["model"] == nc["model"] else "变了"
        print(f"{r['sid']:<24}{o:<34}{n:<34}{same:>8}"
              f"{nc['inst'] + '(' + str(nc['inst_score']) + ')':>22}")
    print()
    print("=" * 118)
    print("  ⑤ 回归锚点：重算 vs 当年真实跑出来的表（当年就从初始分起算的期，必须 0 分差）")
    print("=" * 118)
    print(f"{'期号':<24}{'对比实例':>9}{'最大分差':>12}   判定")
    for r in results:
        a = r["anchor"]
        md = "—" if a["max_diff"] is None else f"{a['max_diff']:g}"
        verdict = "✅ 完全一致" if a["exact"] else "（口径变了，预期 ✗）"
        print(f"{r['sid']:<24}{a['compared']:>9}{md:>12}   {verdict}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把官方天梯历史各期按「每期独立」口径重算（只写新目录，不动生产文件）")
    ap.add_argument("--src", default=str(E.DATA_DIR),
                    help=f"生产 ELO 数据根（默认 {E.DATA_DIR}）")
    ap.add_argument("--out", default="",
                    help="输出根（默认 <src>/reseason）")
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="只算这几期（默认全部）")
    ap.add_argument("--dry-run", action="store_true", help="只算不落盘")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"🔴 数据源不存在：{src}")
        return 2
    seasons_dir = src / "seasons"
    if not seasons_dir.exists():
        print(f"🔴 没有 seasons/ 目录：{seasons_dir}")
        return 2
    out_root = Path(args.out) if args.out else src / "reseason"
    if out_root.resolve() == seasons_dir.resolve():
        print("🔴 输出目录不能是 seasons/ 本身（会覆盖生产文件 ✗）")
        return 2

    dirs = [d for d in sorted(seasons_dir.iterdir()) if d.is_dir()]
    if args.seasons:
        want = set(args.seasons)
        dirs = [d for d in dirs if d.name in want]
    if not dirs:
        print("🔴 没有可算的期")
        return 2

    print(f"数据源 {src}")
    print(f"输出   {'（dry-run，不落盘）' if args.dry_run else out_root}")
    print(f"共 {len(dirs)} 期：{', '.join(d.name for d in dirs)}")
    print()

    results: list[dict] = []
    for d in dirs:
        r = reseason_one(src, d, dirs)
        results.append(r)
        n = r["games"]
        print(f"  · {r['sid']:<24} 对局 {r['matches']:>5}  "
              f"起始 {r['start_rating']['min']:g}/{r['start_rating']['max']:g}  "
              f"games 合计 {r['new']['games_total']:>6} "
              f"(应 {n['per_seat_expected']:>6})  "
              f"±20% 内 {n['within_band_pct']:>5.1f}%  "
              f"冠军 {r['champion']['model']}")

    print_report(results)

    payload = {"generated_at": time.time(), "src": str(src),
               "out": str(out_root), "initial_rating": INITIAL_RATING,
               "seasons": [public_view(r) for r in results]}

    # 硬校验：任何一条不满足就**非 0 退出**（别让坏数据静默通过 ✗）
    bad: list[str] = []
    for r in results:
        bad.extend(audit_season(r, expected_matches=r["matches"]))

    if args.dry_run:
        print("（dry-run：没落盘）")
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        for r in results:
            write_outputs(r, out_root)
        (out_root / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_root / "report.md").write_text(fmt_report(results, src, out_root),
                                           encoding="utf-8")
        print(f"已写 {len(results)} 期到 {out_root}")
        print(json.dumps({k: payload[k] for k in ("generated_at",)}), "")

    print()
    if bad:
        print("🔴 校验未通过：")
        for b in bad:
            print(f"   · {b}")
        return 1
    print(f"✅ 全部 {len(results)} 期通过硬校验"
          f"（起始分全同 / games 合计 = 对局×人数 / n_matches = 对局数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
