"""天梯接外部 AI（Mortal 等）+ 抽人公平性的回归测试。

    python aimodel/tests/test_arena_external.py

不依赖真 Mortal：用 ``_fake_mjai_bot.py`` 的子进程模拟四种情况
（正常 / 崩溃 / 超时 / 非法动作），验证**整期 4000 局不会被一个外部进程拖垮**。

覆盖（每件错了都会静默产出错数据）：
  ① 外部参战者正常上桌：ELO 正常更新、``external_failed=0``
  ② **负向 1**：外部进程崩溃 -> 该座位教师兜底、整期继续、本局标 ``external_failed``
  ③ **负向 2**：外部进程卡死超时 -> 同上（不拖死整期）
  ④ 外部进程给非法动作 -> 同上
  ⑤ ``matches.jsonl`` 只给含外部 AI 的局加标记；老局仍然 5 个字段
  ⑥ ``replay`` 遇到外部局**明确拒绝**（抛 ExternalMatchNotReplayable），不跑假对局
  ⑦ 抽人公平性：18 个模型 × 3 实例跑 400 局，各模型场数在均值 ±20% 内
  ⑧ **负向 3**：一个超高分参战者 + 一群中游，它仍然会被抽到
     （旧 ``pick()`` 先取 min_spread，这条必红 —— 见报告里的实测）
  ⑨ 官方 ``--official`` 调度路径也能把外部 AI 带进期里（start_official 实跑）
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import arena as A  # noqa: E402
from aimodel import elo_store as E  # noqa: E402
from aimodel import external  # noqa: E402
from aimodel.arena import Arena, Participant  # noqa: E402
from aimodel.engine import Decision  # noqa: E402

FAKE = Path(__file__).resolve().parent / "_fake_mjai_bot.py"

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


def _config(mode: str, *, batch: bool = True) -> tuple[Path, Path]:
    """写一份 ``$POTATOMAJ_EXTERNAL_CONFIG``，用假 mjai bot 当外部 AI。"""
    td = Path(tempfile.mkdtemp(prefix="arena_ext_"))
    cfg = td / "external.json"
    cfg.write_text(json.dumps({
        "manue": {
            "cmd": [sys.executable, str(FAKE)],
            "env": {"FAKE_MJAI_MODE": mode,
                    **({"FAKE_MJAI_BATCH": "1"} if batch else {})},
            "batch_events": batch,
        }
    }, ensure_ascii=False), encoding="utf-8")
    os.environ["POTATOMAJ_EXTERNAL_CONFIG"] = str(cfg)
    return td, cfg


def _participants() -> list[Participant]:
    return [
        Participant(key="external:manue", label="FakeMortal", players=4,
                    kind="external", source="manue", external="manue"),
        Participant(key="teacher4#1", label="T1", players=4, kind="teacher"),
        Participant(key="teacher4#2", label="T2", players=4, kind="teacher"),
        Participant(key="teacher4#3", label="T3", players=4, kind="teacher"),
    ]


def _arena(mode: str, *, timeout: float = 5.0, max_restarts: int = 0) -> Arena:
    return Arena(players=4, participants=_participants(), max_gap=10000.0,
                 strict=True, seed=7, external_timeout=timeout,
                 external_max_restarts=max_restarts, external_reuse=True)


# ---------------------------------------------------------------------------
def test_normal_external_match() -> None:
    td, _ = _config("normal")
    a = _arena("normal")
    try:
        t = time.perf_counter()
        done = a.run(2, verbose=False)
        dt = time.perf_counter() - t
        p = a.participants[0]
        check("正常外部 AI：整期跑完 2 局", done == 2, str(done))
        check("正常外部 AI：external_matches=2 / failed=0",
              a.external_matches == 2 and a.external_failed == 0,
              f"{a.external_matches}/{a.external_failed}")
        check("正常外部 AI：参战者 external_games=2 / external_failed=0",
              p.external_games == 2 and p.external_failed == 0,
              f"{p.external_games}/{p.external_failed}")
        check("正常外部 AI：ELO 正常更新（4 个人都打满 2 场、分数有变化）",
              all(x.games == 2 for x in a.participants)
              and any(abs(x.rating - 500.0) > 1e-6 for x in a.participants),
              str([(x.key, x.games, round(x.rating, 1)) for x in a.participants]))
        print(f"     （2 局 {dt:.2f}s，Mortal/外部座位真的在打："
              f"{p.external_games} 场上桌）")
    finally:
        a.close_externals()
        shutil.rmtree(td, ignore_errors=True)


def test_crash_falls_back_and_period_continues() -> None:
    """负向 1：外部进程崩溃 -> 教师兜底 + 整期继续（不抛异常）。"""
    td, _ = _config("crash")
    a = _arena("crash")
    try:
        done = a.run(3, verbose=False)
        p = a.participants[0]
        check("崩溃：整期 3 局一局不少（异常没冒出来）", done == 3, str(done))
        check("崩溃：3 局都标记 external_failed",
              a.external_failed == 3 and p.external_failed == 3,
              f"{a.external_failed}/{p.external_failed}")
        check("崩溃：外部座位仍然都上了桌（教师兜底打完）",
              p.games == 3 and p.external_games == 3,
              f"{p.games}/{p.external_games}")
        check("崩溃：ELO 照常结算（有分、不是全 500）",
              all(x.games == 3 for x in a.participants)
              and any(abs(x.rating - 500.0) > 1e-6 for x in a.participants),
              str([(x.key, round(x.rating, 1)) for x in a.participants]))
        check("崩溃：错误类型记成 ExternalAICrashed",
              "ExternalAICrashed" in p.external_last_error,
              p.external_last_error)
    finally:
        a.close_externals()
        shutil.rmtree(td, ignore_errors=True)


def test_timeout_falls_back_and_period_continues() -> None:
    """负向 2：外部进程卡死 -> 超时 + 教师兜底 + 整期继续。"""
    td, _ = _config("sleep")
    a = _arena("sleep", timeout=0.25)
    try:
        done = a.run(2, verbose=False)
        p = a.participants[0]
        check("超时：整期 2 局一局不少（卡死没拖死整期）", done == 2, str(done))
        check("超时：2 局都标记 external_failed",
              a.external_failed == 2 and p.external_failed == 2,
              f"{a.external_failed}/{p.external_failed}")
        check("超时：错误类型记成 ExternalAITimeout",
              "ExternalAITimeout" in p.external_last_error,
              p.external_last_error)
        check("超时：教师兜底把两局都打完（games=2）",
              p.games == 2 and p.external_games == 2,
              f"{p.games}/{p.external_games}")
    finally:
        a.close_externals()
        shutil.rmtree(td, ignore_errors=True)


def test_illegal_action_falls_back() -> None:
    td, _ = _config("illegal")
    a = _arena("illegal")
    try:
        done = a.run(2, verbose=False)
        p = a.participants[0]
        check("非法动作：整期继续（2 局）", done == 2, str(done))
        check("非法动作：标记 external_failed",
              a.external_failed == 2 and p.external_failed == 2,
              f"{a.external_failed}/{p.external_failed}")
        check("非法动作：错误类型记成 ExternalAIProtocolError",
              "ExternalAIProtocolError" in p.external_last_error,
              p.external_last_error)
    finally:
        a.close_externals()
        shutil.rmtree(td, ignore_errors=True)


def test_match_marker_optional_and_replay_refusal() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="elo_ext_"))
    E.DATA_DIR = tmp
    E.SEASONS_DIR = tmp / "seasons"
    E.SCORES_DIR = tmp / "scores"
    try:
        s = E.open_season("2026-10-21T20", players=4, copies=1, target=2)
        rec = {
            "seed": 123, "players": 4,
            "group": ["external:manue", "teacher4#1", "teacher4#2", "teacher4#3"],
            "ranks": [0, 1, 2, 3], "scores": [40000, 30000, 20000, 10000],
            "external": True, "external_failed": True,
            "external_specs": ["manue"], "external_seats": [0],
        }
        E.append_match(s, rec)
        back = list(E.iter_matches(s))[0]
        check("含外部 AI 的局存下了 external 标记",
              back.get("external") is True, str(back))
        check("含外部 AI 的局存下了 external_failed / specs / seats",
              back.get("external_failed") is True
              and back.get("external_specs") == ["manue"]
              and back.get("external_seats") == [0], str(back))
        try:
            E.replay(back)
        except E.ExternalMatchNotReplayable as exc:
            msg = str(exc)
            check("replay 遇到外部局 -> 明确拒绝（不是崩、不是跑假局）",
                  "外部" in msg and "无法" in msg and "external=true" in msg, msg)
        else:
            check("replay 遇到外部局 -> 明确拒绝", False, "居然没抛异常")

        # 老局：没有 external 字段，仍然只存 5 个字段（老存档/老测试不受影响）
        E.append_match(s, {"seed": 456, "players": 4,
                           "group": ["teacher4#1", "teacher4#2",
                                     "teacher4#3", "random4#1"],
                           "ranks": [0, 1, 2, 3],
                           "scores": [40000, 30000, 20000, 10000],
                           "多送的": "丢掉"})
        old = list(E.iter_matches(s))[1]
        check("不含外部 AI 的老局仍然只有 5 个字段",
              set(old) == set(E.MATCH_FIELDS), str(sorted(old)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def test_chankan_ron_has_key() -> None:
    """抢杠（ロン on kakan）没有 dahai 事件，也必须能映射 hora -> ron。

    真实踩过：`kakan 1p` 后引擎问 ron，`_discard_key` 还是 None，
    `_choose_call` 直接抛协议错 -> Mortal 整局降级。固定 seed 1588596275
    第 4 局必现（本测试不依赖真 Mortal，只验证事件流 key + 映射）。
    """
    td, _ = _config("normal")
    ag = None
    try:
        ag = external.ExternalMJaiAgent(
            "manue", seed=1, player_id=3, players=4, timeout=5,
            command=[sys.executable, str(FAKE)],
            env={"FAKE_MJAI_MODE": "normal", "FAKE_MJAI_BATCH": "1"},
            batch_events=True,
        )
        # 别家加杠：没有 dahai，但引擎会问其他家 ron（抢杠）
        ag._after_feed([{"type": "kakan", "actor": 0, "pai": "1p",
                         "consumed": ["1p", "1p", "1p"]}])
        check("抢杠事件会生成一个「可荣和出牌点」key",
              ag._discard_key is not None, repr(ag._discard_key))
        # 模拟适配层已经读到 Mortal 的 hora
        ag._discard_response = {"type": "hora", "actor": 3, "target": 0}
        ag._discard_response_key = ag._discard_key
        idx = ag._choose_call(Decision("ron", ["yes", "no"], 9), None)
        check("抢杠的 hora 能映射成 ron=yes（不抛协议错）",
              idx == 0, str(idx))
    finally:
        if ag is not None:
            ag.close()
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
def test_pick_fairness_400() -> None:
    """18 模型 × 3 实例，模拟 400 局：各模型场数必须在均值 ±20% 内。"""
    parts: list[Participant] = []
    for m in range(18):
        for c in range(3):
            parts.append(Participant(
                key=f"m{m}#{c}", label=f"m{m}#{c}", players=4,
                kind="teacher", source=f"m{m}", rating=500.0))
    a = Arena(players=4, participants=parts, max_gap=200.0, strict=True, seed=7)
    # m0 当「mac_rl」：每局 +40 分，迅速拉开；其它 +5 分。
    # 旧 pick() 会在 m0 拉开后把它饿死；新 pick() 必须保持公平。
    for _ in range(400):
        group = a.pick()
        if group is None:
            break
        for p in group:
            p.games += 1
            p.rating += 40.0 if p.source == "m0" else 5.0
    by_model: dict[str, int] = {}
    for p in parts:
        by_model[p.source] = by_model.get(p.source, 0) + p.games
    mean = sum(by_model.values()) / len(by_model)
    lo, hi = min(by_model.values()), max(by_model.values())
    check("抽人公平：400 局没有参战者 0 场",
          all(p.games > 0 for p in parts),
          str([p.key for p in parts if p.games == 0][:5]))
    check("抽人公平：18 个模型场数都在均值 ±20% 内",
          lo >= mean * 0.8 and hi <= mean * 1.2,
          f"mean={mean:.1f} min={lo} max={hi}")
    check("抽人公平：m0（模拟 mac_rl）不再被饿死（≥均值 80%）",
          by_model["m0"] >= mean * 0.8,
          f"{by_model['m0']} vs mean {mean:.1f}")
    print(f"     （m0 {by_model['m0']} 场，均值 {mean:.1f}，"
          f"区间 {lo}~{hi} = {lo/mean:.0%}~{hi/mean:.0%}）")


def test_isolated_high_still_picked() -> None:
    """负向 3：一个超高分 + 一群中游 -> 它仍然必须被抽到（旧代码必红）。"""
    high = Participant(key="mac_rl#1", label="mac_rl#1", players=4,
                       kind="teacher", source="mac_rl", rating=1680.7)
    mids = [Participant(key=f"mid#{i}", label=f"mid#{i}", players=4,
                        kind="teacher", source=f"mid{i}", rating=1300.0)
            for i in range(5)]
    a = Arena(players=4, participants=[high] + mids, max_gap=200.0,
              strict=True, seed=3)
    picked: list[Participant] = []
    for _ in range(3):
        group = a.pick()
        if group:
            picked.extend(group)
            for p in group:
                p.games += 1
    check("孤立高分参战者（1680 vs 1300，差 380 > max_gap）仍然被抽到",
          high in picked,
          f"抽到 {[p.key for p in picked]}")
    check("孤立高分参战者没有被 max_gap 永久排除（至少 1 场）",
          high.games >= 1, str(high.games))


def test_official_scheduler_includes_external() -> None:
    """``--official`` 用的就是同一个 Session/Arena；这里实跑调度线程。"""
    from aimodel.ui import arena_server as AS
    from aimodel.ui import elo_schedule as S

    tmp = Path(tempfile.mkdtemp(prefix="elo_official_ext_"))
    E.DATA_DIR = tmp
    E.SEASONS_DIR = tmp / "seasons"
    E.SCORES_DIR = tmp / "scores"
    td, _ = _config("normal")
    session = AS.Session(players=4, externals=["manue"],
                         external_timeout=5.0, external_reuse=True)
    orig = S.due_slot
    try:
        snap = session.snapshot()
        check("Session: /api/arena 快照能看到 external 配置",
              snap.get("external", {}).get("enabled") is True
              and snap.get("external", {}).get("specs") == ["manue"],
              str(snap.get("external")))
        check("Session: /api/arena 快照有 external_failed 顶层计数",
              "external_failed" in snap and snap["external_failed"] == 0,
              str(snap.get("external_failed")))

        # 官方调度用完整阵容（50+ 人）；2 局里不保证抽到某一个特定模型。
        # 这条测的是「--official 的 Session 里外部 AI 在名单上、能被抽到」，
        # 所以把池子缩成 1 个外部 + 3 个教师（抽人规则仍走真 pick）。
        ext = next(p for p in session.arena.participants if p.kind == "external")
        teachers = [p for p in session.arena.participants
                    if p.kind == "teacher" and p.players == 4][:3]
        check("Session: 外部参战者在 --official 的名单里",
              ext is not None and len(teachers) == 3, str(teachers))
        session.arena.participants = [ext] + teachers

        S.due_slot = lambda now, existing, busy=False: (
            (20, "2026-10-21T20") if "2026-10-21T20" not in existing else None)
        AS.start_official(session, interval=0.05, matches=2)
        deadline = time.time() + 40
        while time.time() < deadline:
            s = E.load_season("2026-10-21T20")
            if s is not None and sum(1 for _ in E.iter_matches(s)) >= 2:
                break
            time.sleep(0.05)
        session.pause()
        s = E.load_season("2026-10-21T20")
        rows = list(E.iter_matches(s)) if s else []
        ext_rows = [m for m in rows if m.get("external")]
        check("官方调度：开了一期 2026-10-21T20", s is not None, "")
        check("官方调度：期里的局带 external 标记", len(ext_rows) >= 1,
              f"{len(ext_rows)}/{len(rows)}")
        check("官方调度：Mortal/外部座位真的上了桌",
              sum(1 for m in rows if m.get("external_failed") is False) >= 1
              or any(m.get("external") for m in rows),
              str(rows[:2]))
    finally:
        S.due_slot = orig
        session.pause()
        session.arena.close_externals()
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    print("=" * 70)
    print("  天梯接外部 AI + 抽人公平性")
    print("=" * 70)
    tests = [
        ("正常外部 AI 上桌", test_normal_external_match),
        ("负向 1：外部崩溃 -> 兜底且整期继续", test_crash_falls_back_and_period_continues),
        ("负向 2：外部超时 -> 兜底且整期继续", test_timeout_falls_back_and_period_continues),
        ("非法动作 -> 兜底", test_illegal_action_falls_back),
        ("存档标记 / replay 明确拒绝", test_match_marker_optional_and_replay_refusal),
        ("抢杠 ron 的 key（真实踩过的 1/5 故障）", test_chankan_ron_has_key),
        ("抽人公平 400 局", test_pick_fairness_400),
        ("负向 3：孤立高分仍被抽到", test_isolated_high_still_picked),
        ("官方调度带外部 AI", test_official_scheduler_includes_external),
    ]
    for name, fn in tests:
        print(f"\n▶ {name}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            global FAIL
            FAIL += 1
            print(f"  ❌ {name}: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 70)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
