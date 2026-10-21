"""官方天梯「**每期独立**」的回归测试（不依赖 pytest，直接跑）。

    python -m aimodel.ui.test_season_independent

## 为什么要有这个测试

天梯存在的唯一意义是**横向公平地比较各模型水平** ✓
只要有一处口径串了，页面上的「该期最终 ELO」就**不能和别人的分放一起比** ✗
而串法都是**静默**的：不报错、不崩、数字看着还挺正常 ✗

覆盖的事：
  ① **两条路径一致**：同一个 ``matches.jsonl``，内存里重算 == ``tools/elo_reseason.py``
     落盘的 ``scores.json``（两套入口必须一个数都不差 ✓）
  ② **每期重置**：新一期 rating 回到初始分、``Participant.games`` 归零，
     而 ``arena.done_total``（服务级累计进度）**继续保留** ✓ 两个计数器不混 ✓
  ③ **续跑要读回该期自己的分**：同一期接着跑不能从 500 重来（另一头也不能松 ✗）
  ④ 负向：故意把**上一期的对局**多喂一条进某期
     -> 「该期 games 超了 4000」必须能被检出来 ✓
  ⑤ 负向：把 ``resume_latest()`` 指向**上一期**
     -> 不允许把上一期的 rating 带进新一期 ✓
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from aimodel import elo_store as E                      # noqa: E402
from aimodel.arena import INITIAL_RATING, Arena, Participant  # noqa: E402
import aimodel.ui.arena_server as SRV                   # noqa: E402
import elo_reseason as R                                # noqa: E402


PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  🔴 {name}   {detail}")


def use_tmp() -> Path:
    """把 elo_store 的三个目录指到一个临时目录（**绝不碰真数据** ✓）。"""
    tmp = Path(tempfile.mkdtemp(prefix="elo_season_test_"))
    E.DATA_DIR = tmp
    E.SEASONS_DIR = tmp / "seasons"
    E.SCORES_DIR = tmp / "scores"
    return tmp


def roster() -> list[Participant]:
    """一桌 4 个人的名单（key/provider 都造得小一点，跑得快）。"""
    return [Participant(key=f"{n}#1", label=f"{n} #1", players=4, kind="model",
                       source=n, version="v9", style="标准")
            for n in ("A", "B", "C", "D")]


def make_records(n: int, seed: int = 20261021) -> list[dict]:
    """造 ``n`` 条**确定性**的对局记录（ranks 是排列、scores 是四麻点数）。"""
    rng = random.Random(seed)
    keys = [p.key for p in roster()]
    out = []
    for i in range(n):
        ranks = [0, 1, 2, 3]
        rng.shuffle(ranks)
        gain = [30000, 25000, 20000, 15000]
        scores = [gain[ranks[j]] + rng.randrange(-3000, 3000) for j in range(4)]
        out.append({"seed": 1000 + i, "group": keys, "ranks": ranks,
                    "scores": scores})
    return out


def write_season(sid: str, records: list[dict], *, k: float = 24.0,
                 divisor: float = 500.0, target: int = 0,
                 with_arena: bool = True) -> E.Season:
    s = E.open_season(sid, players=4, copies=1, target=target)
    with (s.dir / "matches.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    s.done = len(records)
    E.save_season(s)
    if with_arena:
        a = Arena(players=4, k=k, divisor=divisor, participants=roster())
        a.save(s.dir / "arena.json")
    return s


def main() -> int:
    print("=" * 78)
    print("  「每期独立」回归测试")
    print("=" * 78)
    tmp = use_tmp()

    # =====================================================================
    print("\n① 两条路径一致：内存重算 == elo_reseason.py 落盘")
    # =====================================================================
    sid = "2026-10-21T08"
    recs = make_records(60)
    write_season(sid, recs)

    # 路径 A：直接调 Arena.replay_ranks（和实时跑局共用 _settle ✓）
    a = Arena(players=4, participants=roster())
    a.replay_ranks(recs)
    table_a = {str(r["名"]): float(r["分数"]) for r in a.table()}

    # 路径 B：跑真脚本（另起进程，走 --src/--out 那条真实回填路径）
    out_root = tmp / "reseason"
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "elo_reseason.py"),
                        "--src", str(tmp), "--out", str(out_root),
                        "--seasons", sid],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout)
    check("elo_reseason.py 正常退出", p.returncode == 0, f"rc={p.returncode}")
    scored = out_root / sid / "scores.json"
    check("脚本落了 scores.json", scored.exists(), str(scored))
    check("脚本落了 arena.json / meta.json",
          (out_root / sid / "arena.json").exists()
          and (out_root / sid / "meta.json").exists(), str(out_root / sid))
    check("脚本**没有**覆盖生产 seasons/<期>/ 里的任何文件",
          not (E.SEASONS_DIR / sid / "scores.json").exists(), "")

    table_b = {}
    if scored.exists():
        d = json.loads(scored.read_text(encoding="utf-8"))
        table_b = {str(r["名"]): float(r["分数"]) for r in (d.get("table") or [])}
    check("两边实例数一致", len(table_a) == len(table_b) == 4,
          f"{len(table_a)} vs {len(table_b)}")
    worst = max((abs(table_a[k] - table_b[k]) for k in table_a if k in table_b),
                default=999.0)
    check("**每个实例的分数逐个相同**（两条路径同一份算法 ✓）", worst == 0.0,
          f"最大分差 {worst}")
    check("①b 重算时起点确实是初始分（new_season 在 replay 里被调用）",
          all(p_.games == len(recs) for p_ in a.participants), "")

    # =====================================================================
    print("\n② 每期重置：rating / games 归零，done_total 照旧只增不减")
    # =====================================================================
    b = Arena(players=4, participants=roster())
    for p_ in b.participants:                       # 假装已经打了一期
        p_.rating = 1680.7
        p_.games = 889
        p_.wins = 300
        p_.ranks = [0, 1, 2]
        p_.history = [900.0, 1200.0]
        p_.matches = [{"n": 1}]
        p_.floor_hits = 2
    b.n_matches = 4000
    b.done_total = 4000
    b.new_season()
    check("新一期：所有人 rating 回到初始分",
          all(abs(p_.rating - INITIAL_RATING) < 1e-9 for p_ in b.participants),
          str([p_.rating for p_ in b.participants]))
    check("新一期：Participant.games（本期场数）归零",
          all(p_.games == 0 for p_ in b.participants), "")
    check("新一期：曲线 / 名次 / 触底 也归零",
          all(not p_.history and not p_.ranks and not p_.matches
              and p_.floor_hits == 0 for p_ in b.participants), "")
    check("新一期：n_matches（本期序号）归零", b.n_matches == 0, str(b.n_matches))
    check("**done_total（服务级累计进度）不动**", b.done_total == 4000,
          str(b.done_total))

    # 重算/续跑重建**不该**把 done_total 算成两倍
    b.done_total = 4000
    b.settle_ranks(b.participants, [0, 1, 2, 3], [30000, 25000, 20000, 15000])
    check("settle_ranks 涨 n_matches 但**不涨** done_total",
          b.n_matches == 1 and b.done_total == 4000,
          f"n={b.n_matches} done_total={b.done_total}")

    # =====================================================================
    print("\n③ 续跑：同一期必须读回**该期自己**的分（不能从 500 重来）")
    # =====================================================================
    sid2 = "2026-10-21T11"
    recs2 = make_records(20, seed=7)
    s2 = write_season(sid2, recs2)
    a2 = Arena(players=4, k=24.0, divisor=500.0, participants=roster())
    a2.replay_ranks(recs2)
    a2.save(s2.dir / "arena.json")                  # 该期跑完的分数状态
    want = [round(p_.rating, 6) for p_ in a2.participants]

    srv = _session_with(roster())
    srv.season = E.load_season(sid2)
    srv.arena.seed = 123
    srv._enter_season(sid2, len(recs2), 123)
    got = [round(p_.rating, 6) for p_ in srv.arena.participants]
    check("续跑读回了该期的分数状态", got == want, f"{got} != {want}")
    check("续跑的 n_matches 接上了盘上局数", srv.arena.n_matches == len(recs2),
          str(srv.arena.n_matches))

    # 另一条续跑路径：**没有** arena.json 时用 matches.jsonl 按名次精确重算
    (s2.dir / "arena.json").unlink()
    srv2 = _session_with(roster())
    srv2.season = E.load_season(sid2)
    srv2.arena.seed = 123
    srv2._enter_season(sid2, len(recs2), 123)
    got2 = [round(p_.rating, 6) for p_ in srv2.arena.participants]
    check("没有快照时用 matches.jsonl 重算，结果**一样**", got2 == want,
          f"{got2} != {want}")

    # =====================================================================
    print("\n④ 负向：某期被多喂一条**上一期的对局** -> 必须检出「超了 4000」")
    # =====================================================================
    target = 4000
    prev_sid, cur_sid = "2026-10-21T14", "2026-10-21T17"
    prev_recs = make_records(5, seed=99)
    write_season(prev_sid, prev_recs)
    cur_recs = make_records(target, seed=100)
    write_season(cur_sid, cur_recs)

    # 先跑一次**干净**的，确认判据不会恒真（正控 ✓）
    clean = R.reseason_one(E.DATA_DIR, E.SEASONS_DIR / cur_sid,
                           [E.SEASONS_DIR / prev_sid, E.SEASONS_DIR / cur_sid])
    clean_bad = R.audit_season(clean, expected_matches=target)
    check("正控：干净的 4000 局一期，audit 无告警", clean_bad == [], str(clean_bad))
    check("正控：该期 games 合计 = 4000×4 = 16000",
          clean["games"]["total"] == 16000, str(clean["games"]["total"]))

    # 现在把上一期的第 1 局**混进当期**（真实事故的样子：续跑接错了文件）
    with (E.SEASONS_DIR / cur_sid / "matches.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(prev_recs[0], ensure_ascii=False,
                            separators=(",", ":")) + "\n")
    dirty = R.reseason_one(E.DATA_DIR, E.SEASONS_DIR / cur_sid,
                           [E.SEASONS_DIR / prev_sid, E.SEASONS_DIR / cur_sid])
    dirty_bad = R.audit_season(dirty, expected_matches=target)
    print("       检出：")
    for m in dirty_bad:
        print(f"         · {m}")
    check("负向：**检出该期局数超了计划 4000**",
          any("超了计划" in m for m in dirty_bad), str(dirty_bad))
    check("负向：报出的实际局数是 4001",
          dirty["matches"] == target + 1, str(dirty["matches"]))
    check("负向：games 合计涨到 16004（自己跟自己仍然自洽，所以必须跟计划比 ✓）",
          dirty["games"]["total"] == (target + 1) * 4, str(dirty["games"]["total"]))
    check("反证：这条负向**只靠「自洽」判据是抓不到的**"
          "（无 expected_matches 时 audit 不报超局）",
          R.audit_season(dirty) == [], str(R.audit_season(dirty)))
    # 还原，别影响后面的断言
    (E.SEASONS_DIR / cur_sid / "matches.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                  for r in cur_recs) + "\n", encoding="utf-8")

    # =====================================================================
    print("\n⑤ 负向：resume_latest 指向**上一期** -> 不得污染新一期")
    # =====================================================================
    old_sid = "2026-10-22T08"        # 「上一期」
    a3 = Arena(players=4, participants=roster())
    for p_ in a3.participants:       # 上一期跑完时的样子：分数很高
        p_.rating = 1500.0
        p_.games = 640
        p_.history = [1500.0]
    a3.n_matches = 4000
    a3.done_total = 8000
    s3 = write_season(old_sid, make_records(4, seed=5), with_arena=False)
    a3.save(s3.dir / "arena.json")

    srv.lock = __import__("threading").Lock()
    srv.season = None
    srv.resume_latest()             # 把「上一期」读回来（这就是污染源）
    check("前提：resume_latest 确实读回了上一期的 1500 分",
          abs(srv.arena.participants[0].rating - 1500.0) < 1e-9,
          str(srv.arena.participants[0].rating))

    new_sid = "2026-10-22T11"       # 「新一期」
    srv.season = E.open_season(new_sid, players=4, copies=1, target=2)
    srv._enter_season(new_sid, 0, 424242)     # have=0 -> 新一期
    rat = [p_.rating for p_ in srv.arena.participants]
    gm = [p_.games for p_ in srv.arena.participants]
    check("**新一期没有被上一期的 rating 污染**（全部回到初始分）",
          all(abs(r - INITIAL_RATING) < 1e-9 for r in rat), str(rat))
    check("**新一期没有被上一期的场数污染**（全部 0）", all(g == 0 for g in gm), str(gm))
    check("新一期没有继承上一期的 done_total=8000 之外的东西（n_matches 归零）",
          srv.arena.n_matches == 0, str(srv.arena.n_matches))
    check("done_total 仍在（服务级累计只增不减）", srv.arena.done_total == 8000,
          str(srv.arena.done_total))
    check("反证：不调 _enter_season 时 rating 还是 1500（说明这条断言不是恒真）",
          abs(a3.participants[0].rating - 1500.0) < 1e-9,
          str(a3.participants[0].rating))

    # =====================================================================
    print("\n⑥ 每期独立：两期各自从初始分起算，互不累积")
    # =====================================================================
    sa, sb = "2026-10-23T08", "2026-10-23T11"
    write_season(sa, make_records(30, seed=11))
    write_season(sb, make_records(30, seed=12))
    ra = R.reseason_one(E.DATA_DIR, E.SEASONS_DIR / sa,
                        [E.SEASONS_DIR / sa, E.SEASONS_DIR / sb])
    rb = R.reseason_one(E.DATA_DIR, E.SEASONS_DIR / sb,
                        [E.SEASONS_DIR / sa, E.SEASONS_DIR / sb])
    check("两期的起始分都是初始分",
          ra["start_rating"]["all_same"] and rb["start_rating"]["all_same"]
          and abs(ra["start_rating"]["min"] - INITIAL_RATING) < 1e-9
          and abs(rb["start_rating"]["min"] - INITIAL_RATING) < 1e-9,
          f'{ra["start_rating"]} / {rb["start_rating"]}')
    # 同样的对局内容、同样的起点 -> 同样的结果（跨期不再互相影响 ✓）
    write_season("2026-10-23T14", make_records(30, seed=11))
    rc = R.reseason_one(E.DATA_DIR, E.SEASONS_DIR / "2026-10-23T14",
                        [E.SEASONS_DIR / "2026-10-23T14"])
    check("`A 期`与`用同样对局的新一期`分数完全一致（期与期之间不残留状态 ✓）",
          ra["_table"] and [r["分数"] for r in ra["_table"]]
          == [r["分数"] for r in rc["_table"]],
          f'{[r["分数"] for r in ra["_table"]]} vs {[r["分数"] for r in rc["_table"]]}')

    shutil.rmtree(tmp, ignore_errors=True)

    # =====================================================================
    print("\n⑦ 前瞻：当前 pick() 下 4000 局的场次分布（可比性的地基）")
    # =====================================================================
    # 为什么要单独模拟：历史那 7 期的 matches.jsonl 是**旧 pick()** 抽出来的，
    # 重算只改分数口径、改不了「当时谁上过桌」✗
    # -> 那 7 期在「同场次量级」这条上必然不达标；
    #    新 pick() 的效果只能这样**只模拟抽人 + 结算**（不打牌，1 秒级）来验 ✓
    fair = _sim_draws(lambda a, rng: a.pick())
    print(f"        新 pick()：均值 {fair['mean']:.1f}  min {fair['min']}  "
          f"max {fair['max']}  CV {fair['cv']:.4f}  ±20% 内 {fair['pct']:.1f}%")
    check("新 pick()：4000 局后每人场次**全部**落在均值 ±20% 内",
          fair["pct"] == 100.0, str(fair))
    check("新 pick()：变异系数 < 0.05（不是「大致均匀」，是真均匀）",
          fair["cv"] < 0.05, str(fair["cv"]))
    # 负向对照：换成「永远取分最高的 4 个」这种抽法 -> 上面那条必红
    unfair = _sim_draws(
        lambda a, rng: sorted(a.participants, key=lambda p: -p.rating)[:4])
    print(f"        对照（永远取分最高的 4 个）：均值 {unfair['mean']:.1f}  "
          f"min {unfair['min']}  max {unfair['max']}  ±20% 内 {unfair['pct']:.1f}%")
    check("负向对照：不公平抽法在**同一个判据**下不通过（说明这条判据不恒真 ✓）",
          unfair["pct"] < 100.0 and unfair["cv"] > 0.05, str(unfair))

    print("\n" + "=" * 78)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 78)
    return 0 if FAIL == 0 else 1


def _many(n_models: int = 18, copies: int = 3) -> list[Participant]:
    """官方天梯的规模：18 个模型 × 3 份 = 54 个参战者。"""
    return [Participant(key=f"m{m:02d}#{c}", label=f"M{m:02d} #{c}", players=4,
                        kind="model", source=f"m{m:02d}")
            for m in range(n_models) for c in range(1, copies + 1)]


def _sim_draws(choose, *, matches: int = 4000, seed: int = 7) -> dict:
    """**只模拟抽人 + ELO 结算**（不打牌），返回每人上场次数的分布。

    ``choose(arena, rng)`` 返回这一桌的 4 个人（``arena.pick()`` 或任何抽法）。
    名次随机（所有人一样强）—— 这样量到的就是**纯抽人公平性** ✓
    """
    import statistics

    rng = random.Random(seed)
    a = Arena(players=4, participants=_many(), seed=seed)
    a.new_season()
    start = 25000                                  # 四麻起始点数
    for _ in range(matches):
        group = choose(a, rng)
        if not group:
            break
        ranks = [0, 1, 2, 3]
        rng.shuffle(ranks)
        # zero-sum 点数（和真四麻一样总和恒定），名次越高点数越多
        scores = [start + int((1.5 - ranks[j]) * 8000)
                  + rng.randrange(-3000, 3000) for j in range(4)]
        drift = sum(scores) - 4 * start
        scores = [s - drift // 4 for s in scores]
        a.settle_ranks(group, ranks, scores)
    g = [p.games for p in a.participants]
    mean = sum(g) / len(g)
    lo, hi = mean * 0.8, mean * 1.2
    return {"mean": mean, "min": min(g), "max": max(g),
            "cv": (statistics.pstdev(g) / mean) if mean else 0.0,
            "pct": 100.0 * sum(1 for x in g if lo <= x <= hi) / len(g),
            "matches": a.n_matches}


def _session_with(parts: list[Participant]) -> SRV.Session:
    """造一个 ``Session`` 但**不真的加载权重**（把 ``Arena`` 换成小名单 ✓）。

    为什么要这样：``Session.__init__`` 会走 ``arena.collect()`` 去扫全部权重，
    测试里没必要（而且慢）。只换**构造**那一下，``Arena.load`` 仍是真货 ✓
    """
    import threading

    real = SRV.Arena

    def fake(**kw):                                  # noqa: ANN003
        return real(players=4, participants=parts)

    SRV.Arena = fake                                 # type: ignore[assignment]
    try:
        s = SRV.Session(players=4)
    finally:
        SRV.Arena = real                             # type: ignore[assignment]
    s.lock = threading.Lock()
    return s


if __name__ == "__main__":
    sys.exit(main())
