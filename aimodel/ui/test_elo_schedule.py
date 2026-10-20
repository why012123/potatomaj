"""官方 ELO 档位调度的测试（不依赖 pytest，直接跑）。

    python -m aimodel.ui.test_elo_schedule

覆盖的核心不变式（每条都是「错了会静默跑歪」的那种）：
  ① 期号用**档位小时**，不是「现在几点」
  ② **同一档只跑一次**（靠落盘的期号判重，不是内存集合）
  ③ **正在跑就不叠加**（busy=True 一律不跑）
  ④ **错过太久不补**（超过 grace_min）
  ⑤ **还没到的不跑**
  ⑥ 跨天：昨天的档位绝不会被当成今天的

负向验证（改坏 -> 必须变红）：
  把 :func:`due_slot` 里的 ``if sid in existing: continue`` 删掉
  -> 「同一档只跑一次」那条必红 ✓
"""

from __future__ import annotations

import sys
import time

from . import elo_schedule as S


def _at(day: str, hh: int, mm: int) -> float:
    """构造一个本地时间戳。``day`` 形如 ``2026-10-20``。"""
    y, m, d = (int(x) for x in day.split("-"))
    return time.mktime((y, m, d, hh, mm, 0, 0, 0, -1))


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


def main() -> int:
    print("=" * 70)
    print("  官方 ELO 档位调度测试")
    print("=" * 70)

    # ---- ① 期号用档位小时，不是 now 的小时 -------------------------------
    print("\n① 期号口径")
    ts = _at("2026-10-20", 8, 0)
    check("8:00 那一档的期号是 …T08",
          S.season_id_for(ts, 8) == "2026-10-20T08",
          S.season_id_for(ts, 8))
    # 关键：9:30 补跑 8 点那档时，期号必须是 T08 而不是 T09 ✗
    late = _at("2026-10-20", 9, 30)
    check("9:30 补跑 8 点档，期号仍是 …T08（不是 T09）",
          S.season_id_for(late, 8) == "2026-10-20T08",
          S.season_id_for(late, 8))

    # ---- ② 同一档只跑一次 ------------------------------------------------
    print("\n② 同一档只跑一次")
    d = S.due_slot(_at("2026-10-20", 8, 5), existing=set())
    check("8:05 且该档没跑过 -> 该跑 8 点档",
          d is not None and d[1] == "2026-10-20T08", str(d))
    d = S.due_slot(_at("2026-10-20", 8, 5), existing={"2026-10-20T08"})
    check("8:05 但该档已存在 -> 不跑",
          d is None, str(d))

    # ---- ③ 正在跑就不叠加 ------------------------------------------------
    print("\n③ 不叠加")
    d = S.due_slot(_at("2026-10-20", 11, 2), existing=set(), busy=True)
    check("busy=True 时一律返回 None",
          d is None, str(d))

    # ---- ④ 错过太久不补 --------------------------------------------------
    print("\n④ 错过不补")
    d = S.due_slot(_at("2026-10-20", 8, 0) + 10 * 60, existing=set())
    check("8:10（10 分钟内）-> 补跑", d is not None, str(d))
    d = S.due_slot(_at("2026-10-20", 10, 40), existing=set())
    check("10:40（8 点档过去 160 分钟）-> 不补 8 点档",
          d is None, str(d))

    # ---- ⑤ 还没到的不跑 --------------------------------------------------
    print("\n⑤ 未到不跑")
    d = S.due_slot(_at("2026-10-20", 7, 30), existing=set())
    check("7:30 -> 不跑（8 点还没到）", d is None, str(d))

    # ---- ⑥ 跨天 ----------------------------------------------------------
    print("\n⑥ 跨天")
    d = S.due_slot(_at("2026-10-21", 8, 5), existing={"2026-10-20T08"})
    check("第二天 8:05，昨天跑过不影响今天 -> 该跑",
          d is not None and d[1] == "2026-10-21T08", str(d))

    # ---- ⑦ 多档同时过期时，取最早的未跑档 -------------------------------
    print("\n⑦ 多个档位待补时的顺序")
    # 14:05：8/11 都没跑（都超出 grace 了）-> 14 点该跑
    d = S.due_slot(_at("2026-10-20", 14, 5), existing=set())
    check("14:05 且前面全过期 -> 跑 14 点档",
          d is not None and d[1] == "2026-10-20T14", str(d))
    # 11:05：8 点跑过、11 点没跑 -> 跑 11
    d = S.due_slot(_at("2026-10-20", 11, 5), existing={"2026-10-20T08"})
    check("11:05 且 8 点跑过 -> 跑 11 点档",
          d is not None and d[1] == "2026-10-20T11", str(d))

    # ---- ⑧ today_slots 状态机 -------------------------------------------
    print("\n⑧ 时间轴状态")
    rows = S.today_slots(_at("2026-10-20", 15, 0), existing={"2026-10-20T08"})
    st = {r.slot: r.state for r in rows}
    check("15:00：8 点 done", st.get(8) == "done", str(st))
    check("15:00：11 点 missed（过期且没跑）", st.get(11) == "missed", str(st))
    check("15:00：14 点 due（1 小时内）", st.get(14) == "due", str(st))
    check("15:00：17 点 upcoming", st.get(17) == "upcoming", str(st))
    check("五个档位都在", len(rows) == 5, str(len(rows)))

    # ---- ⑨ next_slot_at --------------------------------------------------
    print("\n⑨ 下次时刻")
    nxt = S.next_slot_at(_at("2026-10-20", 12, 0))
    check("12:00 -> 下次是 14:00",
          time.localtime(nxt).tm_hour == 14 and time.localtime(nxt).tm_mday == 20,
          time.strftime("%m-%d %H:%M", time.localtime(nxt)))
    nxt = S.next_slot_at(_at("2026-10-20", 21, 0))
    check("21:00 -> 下次是明天 8:00",
          time.localtime(nxt).tm_hour == 8 and time.localtime(nxt).tm_mday == 21,
          time.strftime("%m-%d %H:%M", time.localtime(nxt)))

    print("\n" + "=" * 70)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
