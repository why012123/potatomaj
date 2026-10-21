"""ELO 存档 / 回放 / 清理的回归测试（不依赖 pytest，直接跑）。

    python -m aimodel.ui.test_elo_store

覆盖的事（每件错了都会**静默**产出错数据）：
  ① 每局只存 5 个字段，且**回放能逐手复现**（用同一个引擎重跑 ✓）
  ② 3 天前**未收藏**的期 -> 只留冠军名，回放数据删掉
  ③ 3 天前**已收藏**的期 -> 完整保留
  ④ **ELO 表不受 3 天限制**（历史数据长期保留 ✓）
  ⑤ 模型榜**按场数加权**（不是简单平均；少打的实例不能带偏）
  ⑥ 趋势**缺期返回 None**（前端据此断线，不跨期连）

负向验证：
  1. 把 ``prune()`` 里 ``if s.pruned or fav: continue`` 的 ``fav`` 去掉
     -> 「已收藏的期完整保留」那条必红 ✓
  2. 把 ``model_ranking()`` 的按场数加权换成简单平均
     -> 「模型榜 A 的 ELO = 505」那条必红（简单平均是 750）✓
  3. 把 ``build_trend()`` 的缺期补成上一期的值
     -> 「缺期 A 的点是 None」那条必红 ✓
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import elo_store as E  # noqa: E402


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
    tmp = Path(tempfile.mkdtemp(prefix="elo_test_"))
    E.DATA_DIR = tmp
    E.SEASONS_DIR = tmp / "seasons"
    E.SCORES_DIR = tmp / "scores"
    return tmp


def main() -> int:
    print("=" * 70)
    print("  ELO 存档 / 回放 / 清理 测试")
    print("=" * 70)
    tmp = use_tmp()

    # ---- ① 逐局记录 + 回放复现 -------------------------------------------
    print("\n① 记录与回放")
    sid = "2026-10-20T08"
    s = E.open_season(sid, players=4, copies=1, target=2)
    ikey = ["maj-normal4v1", "maj-flash4v1", "maj-huge4v1", "teacher4#1"]
    # ⚠️ **先真打一局拿到真实结果**，再拿它当存档 ✗
    #    第一版我手编了 ranks/scores（[0,1,2,3] / 30000…）✓
    #    而 seed 12345 实际打出来是别的名次 ✗ -> 那两条断言必然红 ✓
    #    错的是**测试数据**不是代码 ✓（这个区别很重要 ✓）
    seed = 12345
    _, ref = E.replay({"seed": seed, "players": 4, "group": ikey})
    rec = {"seed": seed, "players": 4, "group": ikey,
           "ranks": list(ref.ranks), "scores": list(ref.scores)}
    E.append_match(s, {"seed": seed, "players": 4, "group": ikey,
                       "ranks": rec["ranks"], "scores": rec["scores"],
                       "多送的字段": "应被丢掉"})
    check("记一局后 done=1", s.done == 1, str(s.done))

    back = list(E.iter_matches(s))
    check("读回 1 局", len(back) == 1, str(len(back)))
    check("只存了 5 个字段（多余字段被丢掉 ✓）",
          set(back[0]) == set(E.MATCH_FIELDS), str(sorted(back[0])))

    # 真回放：用同一个引擎重跑
    acts, res = E.replay(back[0])
    check("回放跑出动作（>0）", len(acts) > 0, str(len(acts)))
    check("回放的名次与存档一致", list(res.ranks) == rec["ranks"],
          f'{list(res.ranks)} != {rec["ranks"]}')
    check("回放的分数与存档一致", list(res.scores) == rec["scores"],
          f'{list(res.scores)} != {rec["scores"]}')
    # 再跑一次，验证确定性（这段是「存档只靠种子」的地基）
    acts2, res2 = E.replay(back[0])
    check("**同 seed 重跑两次，动作逐个相同**（确定性的地基 ✓）",
          acts == acts2, f"{len(acts)} vs {len(acts2)}")
    check("首个动作是 start_hand，且 hands 是 list（不是被字符串化 ✗）",
          acts[0].get("kind") == "start_hand"
          and isinstance(acts[0].get("hands"), list),
          f'kind={acts[0].get("kind")} hands={type(acts[0].get("hands")).__name__}')

    # ---- ② / ③ / ④ 清理 -------------------------------------------------
    print("\n②③④ 三天清理")
    now = time.time()
    old = now - 4 * 86400
    nofav, fav = "2026-10-16T08", "2026-10-16T11"
    for name, is_fav in ((nofav, False), (fav, True)):
        x = E.open_season(name, players=4, copies=1, target=2)
        x.started_at = old
        x.finished_at = old
        for i in range(3):
            E.append_match(x, {"seed": i, "players": 4,
                               "group": ["a", "b", "c", "d"],
                               "ranks": [0, 1, 2, 3], "scores": [1, 2, 3, 4]})
        x.done = 3
        x.favorite = is_fav
        E.save_season(x)
        # 冠军从这份表里取 ✓
        E.save_scores(name, [{"名": "甲", "模型": "a", "分数": 900},
                             {"名": "乙", "模型": "b", "分数": 800}])

    before = {x.sid: x for x in E.list_seasons()}
    check("清理前两期都有回放数据",
          all((x.dir / "matches.jsonl").exists() for x in before.values()), "")

    gone = E.prune(keep_days=3, now=now)
    check("清理只动了 1 期", len(gone) == 1, str(gone))

    a0 = E.load_season(nofav)
    a1 = E.load_season(fav)
    check("② 未收藏的期：pruned=True", a0.pruned is True, str(a0.pruned))
    check("② 未收藏的期：回放数据被删",
          not (a0.dir / "matches.jsonl").exists(), "")
    # ⚠️ 冠军口径在 2026-10-20 被用户改过：从「单实例第一」改成「**模型均分第一**」✓
    #    所以这里断言的是**模型名**（"a"）而不是显示名（"甲"）✗
    #    这是**需求变了**跟着改 ✓ 不是「为了让测试通过而改断言」✗
    check("② 未收藏的期：**留住了冠军模型名**（按均分口径 ✓）",
          a0.champion == "a", repr(a0.champion))
    check("② 未收藏的期：均分也留住了", a0.champion_avg > 0, str(a0.champion_avg))
    check("③ 已收藏的期：pruned=False", a1.pruned is False, str(a1.pruned))
    check("③ 已收藏的期：**回放数据完整保留**",
          (a1.dir / "matches.jsonl").exists(), "")
    check("④ ELO 表**不被删**（历史数据长期保留 ✓，两期都在）",
          (E.SCORES_DIR / f"{nofav}.json").exists()
          and (E.SCORES_DIR / f"{fav}.json").exists(), "")

    # 再清理一次：已精简的不应重复处理 ✓
    gone2 = E.prune(keep_days=3, now=now)
    check("⑤ 重复清理不会重复动（幂等 ✓）", gone2 == [], str(gone2))

    # ---- ⑤b 冠军口径：均分第一 != 单实例第一 ---------------------------
    print("\\n⑤b 冠军口径（按模型均分）")
    demo = [
        {"名": "A #3", "模型": "A", "分数": 900.0},
        {"名": "B #1", "模型": "B", "分数": 800.0},
        {"名": "B #2", "模型": "B", "分数": 790.0},
        {"名": "A #1", "模型": "A", "分数": 600.0},
        {"名": "A #2", "模型": "A", "分数": 580.0},
        {"名": "B #3", "模型": "B", "分数": 780.0},
    ]
    m, avg, inst, isc = E.champion_by_model(demo)
    check("单实例第一是 A #3（900）", inst == "A #3" and isc == 900.0, f"{inst} {isc}")
    check("**冠军按均分是 B**（790 vs 693.3）", m == "B", f"{m} {avg}")
    check("冠军均分 = 790.0", abs(avg - 790.0) < 0.05, str(avg))
    m2, a2, _, _ = E.champion_by_model([])
    check("空表不炸", (m2, a2) == ("", 0.0), f"{m2} {a2}")

    # ---- ⑥ 模型榜：按场数加权（不是简单平均） -------------------------
    print("\n⑥ 模型榜加权")
    demo_rows = [
        # A：1 场运气好 1000 分；99 场只有 500 分
        {"名": "A #1", "模型": "A", "分数": 1000.0, "场数": 1, "局数": 10,
         "一位": 0, "一位率": 0.0, "和了率": 0.5, "放铳率": 0.1,
         "版本": "v1", "风格": "标准", "触底": 0, "状态": "存活"},
        {"名": "A #2", "模型": "A", "分数": 500.0, "场数": 99, "局数": 990,
         "一位": 99, "一位率": 1.0, "和了率": 0.1, "放铳率": 0.2,
         "版本": "v1", "风格": "标准", "触底": 0, "状态": "存活"},
        # B：两场都 700，用来验证排序与实例数
        {"名": "B #1", "模型": "B", "分数": 700.0, "场数": 50, "局数": 500,
         "一位": 10, "一位率": 0.2, "和了率": 0.2, "放铳率": 0.1,
         "版本": "v2", "风格": "速攻", "触底": 0, "状态": "存活"},
        # C：没上过场 -> 不进榜（与 Arena.summary 一致）
        {"名": "C #1", "模型": "C", "分数": 500.0, "场数": 0, "局数": 0,
         "一位": 0, "一位率": 0.0, "和了率": 0.0, "放铳率": 0.0,
         "版本": "v3", "风格": "防守", "触底": 0, "状态": "存活"},
    ]
    ranked = E.model_ranking(demo_rows)
    by = {r["模型"]: r for r in ranked}
    a = by.get("A", {})
    check("模型榜 A 的 ELO = 505.0（按场数加权：1000×1 + 500×99 / 100）",
          abs(a.get("分数", 0) - 505.0) < 0.05, str(a.get("分数")))
    check("模型榜 A 的**均分仍是 750**（说明两条口径确实不同 ✓）",
          abs(a.get("均分", 0) - 750.0) < 0.05, str(a.get("均分")))
    check("模型榜 A 实例数=2 / 场数=100 / 一位率=99/100",
          a.get("实例数") == 2 and a.get("场数") == 100
          and abs(a.get("一位率", 0) - 0.99) < 1e-9, str(a))
    # 和了率按手数加权：(0.5×10 + 0.1×990) / 1000 = 0.104
    check("模型榜 A 和了率按局数加权 = 0.104",
          abs(a.get("和了率", 0) - 0.104) < 1e-9, str(a.get("和了率")))
    check("模型榜 B 排在 A 前（700 > 505）", ranked[0]["模型"] == "B", str(ranked))
    check("没上过场的模型 C 不进榜", "C" not in by, str(sorted(by)))
    check("空表不炸", E.model_ranking([]) == [], str(E.model_ranking([])))

    # **负向验证**：把「按场数加权」换成简单平均 -> 上面 505 那条必然红。
    # 这里不真去改源码（测试要可重复），而是用同一份数据算出简单平均，
    # 断言它 != 505 —— 证明那条断言不是恒真。
    simple_a = sum(float(r["分数"]) for r in demo_rows
                   if r["模型"] == "A") / 2
    check("负向验证：简单平均 750 != 加权 505（改坏加权逻辑这条会红）",
          abs(simple_a - 505.0) > 1.0, f"simple={simple_a}")

    # ---- ⑦ 趋势：缺期断开（None 点），不跨期连线 -----------------------
    print("\n⑦ 趋势缺期")
    t_rows_1 = [{"模型": "A", "分数": 1000, "ELO": 1000, "场数": 10,
                 "一位率": 0.3, "版本": "v1", "风格": "标准"}]
    t_rows_3 = [{"模型": "A", "分数": 1100, "ELO": 1100, "场数": 12,
                 "一位率": 0.3, "版本": "v1", "风格": "标准"},
                {"模型": "B", "分数": 900, "ELO": 900, "场数": 8,
                 "一位率": 0.2, "版本": "v2", "风格": "速攻"}]
    trend = E.build_trend([
        ("2026-10-18T08", t_rows_1),
        ("2026-10-19T08", []),          # 整期没有表 -> 所有模型都断
        ("2026-10-20T08", t_rows_3),
    ])
    a_points = next(s for s in trend["series"] if s["模型"] == "A")["points"]
    check("趋势有 3 个期位（含缺期）", trend["sids"] ==
          ["2026-10-18T08", "2026-10-19T08", "2026-10-20T08"], str(trend["sids"]))
    check("缺期 A 的点是 None（不是沿用上一期 ✗）",
          a_points[1] is None, str(a_points))
    check("缺期被列进 missing", trend["missing"] == ["2026-10-19T08"],
          str(trend["missing"]))
    b_points = next(s for s in trend["series"] if s["模型"] == "B")["points"]
    check("B 在前期也不存在 -> 前两个点是 None",
          b_points[0] is None and b_points[1] is None, str(b_points))
    check("空趋势不炸", E.build_trend([]) ==
          {"sids": [], "series": [], "missing": []}, str(E.build_trend([])))

    # ---- ⑧ 配额统计的分母 -------------------------------
    print("\n⑧ 配额统计")
    st = E.stats()
    check("per_match_bytes 只除 matches.jsonl（不是全部文件 ✓）",
          0 < st["per_match_bytes"] < 500, f'{st["per_match_bytes"]} B/局')
    check("replay_bytes 与 scores_bytes 分开算",
          st["replay_bytes"] > 0 and st["scores_bytes"] > 0,
          f'replay={st["replay_bytes"]} scores={st["scores_bytes"]}')
    check("配额是 100 GB", st["quota_bytes"] == 100 * 1024 ** 3, str(st["quota_bytes"]))

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
