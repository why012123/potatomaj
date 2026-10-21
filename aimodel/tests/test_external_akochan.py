"""akochan 外部引擎座位（``aimodel/external_akochan.py``）的回归测试。

**不需要装 akochan**：绝大部分测试用「假的子进程」验证每一根线；
装了的话可以把 ``AKOCHAN_E2E=1`` 打开，额外跑一条真进程的端到端对比。

覆盖的点：

1. 牌名 / 下标（akochan 的 38 下标，赤五单算）往返
2. ``moves`` -> 本引擎选项下标的映射（8 种决策）
3. **往返一致性**：``View`` -> MJAI -> 用独立解析器还原 -> 跟 View 逐项相等
   （手牌 / 牌河 / 副露 / 宝牌 / 点数 / 立直 / 亲家 / 最后一条是决策点）
4. **负向**：akochan 没有 ``moves`` -> 走兜底（不崩、不卡）
5. **负向**：超时 / 进程不存在 / 输出是垃圾 -> 全部走兜底
6. 临时文件：并发不撞名、成功后一定删掉
7. 缓存：同一决策点只查一次
8. 叫牌后那张打牌：用 fuuro 选择里一起给的 ``dahai``（不再查询）
9. 三麻：直接拒绝（akochan 的回放按 4 家写死）
10. 注册表：``variants`` 里是「外部引擎」，可用、但不进天梯
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import external_akochan as A  # noqa: E402
from aimodel import mjai  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from aimodel import variants  # noqa: E402
from aimodel.agents import Agent, HeuristicAgent  # noqa: E402
from aimodel.engine import Decision, Game, View  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

class _Spy(Agent):
    """把每次决策的 (decision, view) 记下来（View 是快照，之后随便变）。"""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.recs: list[tuple[Decision, View, int]] = []

    def decide(self, decision: Decision, view: View) -> int:
        idx = int(self.inner.decide(decision, view))
        self.recs.append((decision, view, idx))
        return idx


def _records(seed: int, *, players: int = 4, hands: int = 2):
    spy = _Spy(HeuristicAgent(seed * 3 + 1))
    others = [HeuristicAgent(seed + 2 + i) for i in range(players - 1)]
    game = Game([spy] + others, seed=seed, rule=S.rule_for(players),
                rounds=1, max_hands=hands)
    game.play()
    return spy.recs


def _ak(name: str) -> int:
    tile, red = mjai.tile_from_mjai(str(name))
    return A._ak_index(tile, red)


def _akname(tile: int) -> str:
    return A._tile_name(int(tile))


def _replay(events: list[dict]) -> dict:
    """独立的 MJAI 解析器（专门用来验证序列化结果）。

    语义照 akochan 的 ``go_next_state`` 写 —— 特别是 ``kakan`` 只从手牌里
    扣 ``pai`` **一张**（另外两张在更早的 ``pon`` 里扣过）。
    """
    st = {
        "hands": [[0] * 38 for _ in range(4)],
        "ponds": [[] for _ in range(4)],
        "melds": [[] for _ in range(4)],
        "scores": [25000] * 4,
        "riichi": [False] * 4,
        "dora": [],
        "oya": 0,
        "kyoku": 1,
        "bakaze": "E",
        "kyotaku": 0,
        "initial": None,
    }
    for e in events:
        t = e.get("type")
        if t == "start_kyoku":
            st["hands"] = [[0] * 38 for _ in range(4)]
            st["initial"] = []
            for pid, hand in enumerate(e.get("tehais") or []):
                names = list(hand)[:13]
                st["initial"].append(names)
                for name in names:
                    st["hands"][pid][_ak(name)] += 1
            st["ponds"] = [[], [], [], []]
            st["melds"] = [[], [], [], []]
            st["scores"] = [int(x) for x in (e.get("scores") or [25000] * 4)]
            st["oya"] = int(e.get("oya", 0))
            st["kyoku"] = int(e.get("kyoku", 1))
            st["bakaze"] = e.get("bakaze")
            st["dora"] = [e.get("dora_marker")]
            st["riichi"] = [False] * 4
            st["kyotaku"] = int(e.get("kyotaku", 0))
        elif t == "dora":
            st["dora"].append(e.get("dora_marker"))
        elif t == "tsumo":
            st["hands"][int(e["actor"])][_ak(e["pai"])] += 1
        elif t == "dahai":
            actor = int(e["actor"])
            st["hands"][actor][_ak(e["pai"])] -= 1
            st["ponds"][actor].append(str(e["pai"]))
        elif t in ("pon", "chi", "daiminkan", "ankan", "kakan"):
            actor = int(e["actor"])
            if t == "kakan":
                st["hands"][actor][_ak(e["pai"])] -= 1
                for i, m in enumerate(st["melds"][actor]):
                    if (m[0] == "pon"
                            and A._ak_base(_ak(m[1])) == A._ak_base(_ak(e["pai"]))):
                        st["melds"][actor][i] = ("kakan", e["pai"],
                                                 e.get("target", -1))
                        break
            else:
                for name in e.get("consumed") or []:
                    st["hands"][actor][_ak(name)] -= 1
                st["melds"][actor].append((t, e.get("pai"), e.get("target", -1)))
        elif t == "reach":
            st["riichi"][int(e["actor"])] = True
        elif t == "reach_accepted":
            actor = int(e["actor"])
            st["scores"][actor] -= 1000
            st["kyotaku"] += 1
    return st


def _compare(view: View, events: list[dict]) -> list:
    """``View`` 和事件流还原出来的状态逐项比。返回不一致项（空 = 一致）。"""
    st = _replay(events)
    seat = int(view.player)
    bad: list = []
    # 1) 自己的手牌（38 下标，赤五单算）
    want = A._ak_counts(view.hand, view.hand_red)
    got = st["hands"][seat]
    if want != got:
        bad.append(("hand", [(A._ak_str(i), want[i], got[i])
                            for i in range(38) if want[i] != got[i]]))
    # 2) 起手 13 张
    if not st["initial"] or len(st["initial"][seat]) != 13:
        bad.append(("initial13", st["initial"][seat] if st["initial"] else None))
    # 3) 牌河：事件里的 kawa = View 的牌河 + 被叫走的牌（多重集），
    #    且 View 的牌河顺序必须是事件 kawa 的子序列
    for q in range(4):
        ev_kawa = list(st["ponds"][q])
        called: list[str] = []
        for pid in range(4):
            if pid == q or pid >= len(view.melds_all):
                continue
            for m in view.melds_all[pid]:
                if not A._is_call(m):
                    continue
                if int(getattr(m, "from_player", -1)) != q:
                    continue
                called_t = int(getattr(m, "called", -1))
                if str(m.kind) == "chi" and called_t >= 0:
                    called.append(_akname(called_t))
                else:
                    called.append(_akname(int(m.tile)))
        want_ms = ([_akname(int(x)) for x in view.discards[q]] + called)
        if sorted(ev_kawa) != sorted(want_ms):
            bad.append((f"pond{q}", sorted(ev_kawa), sorted(want_ms)))
        it = iter(ev_kawa)
        pond_order = [_akname(int(x)) for x in view.discards[q]]
        if not all(any(x == y for y in it) for x in pond_order):
            bad.append((f"pond_order{q}", ev_kawa, pond_order))
    # 4) 副露
    for q in range(4):
        want_m = []
        for m in (view.melds_all[q] if q < len(view.melds_all) else []):
            kind = str(m.kind)
            tile = (int(m.called)
                    if (kind == "chi" and int(getattr(m, "called", -1)) >= 0)
                    else int(m.tile))
            target = -1 if kind == "ankan" else int(getattr(m, "from_player", -1))
            want_m.append((A._CALL_TYPE[kind], _akname(tile), target))
        got_m = [(kind, pai, -1 if kind == "ankan" else int(target))
                 for (kind, pai, target) in st["melds"][q]]
        if want_m != got_m:
            bad.append((f"melds{q}", got_m, want_m))
    # 5) 宝牌指示牌
    want_d = sorted(_akname(int(x)) for x in view.dora_indicators)
    if sorted(st["dora"]) != want_d:
        bad.append(("dora", st["dora"], want_d))
    # 6) 点数（reach_accepted 已经把 +1000 扣回去了）
    if list(st["scores"]) != [int(x) for x in view.scores]:
        bad.append(("scores", st["scores"], list(view.scores)))
    # 7) 立直
    if list(st["riichi"]) != [bool(x) for x in view.riichi]:
        bad.append(("riichi", st["riichi"], list(view.riichi)))
    # 8) 亲家
    dealer = (seat - (int(view.seat_wind) - A._EAST)) % 4
    if st["oya"] != dealer:
        bad.append(("oya", st["oya"], dealer))
    # 9) 最后一条必须是 akochan 认的决策点
    last = events[-1] if events else {}
    lt, la = last.get("type"), last.get("actor")
    ok_last = ((lt == "tsumo" and la == seat)
               or (lt == "dahai" and la != seat)
               or (lt in ("kakan", "daiminkan") and la != seat))
    if not ok_last:
        bad.append(("last_event", last))
    return bad


def _fake_engine(*, moves: list | None = None, raw: str | None = None,
                 rc: int = 0, timeout: bool = False,
                 paths: list | None = None,
                 delay: float = 0.0) -> A.AkochanEngine:
    """注入 ``runner`` 的假引擎（不起真进程）。"""

    def runner(args, timeout_s):
        if paths is not None:
            paths.append(args[2])
        if delay:
            time.sleep(delay)
        if timeout:
            raise subprocess.TimeoutExpired(args, timeout_s)
        if raw is not None:
            return rc, raw, ""
        review = [{"moves": moves, "review": {"pt_exp_total": 0.0}}] if moves else []
        return rc, json.dumps(review), ""

    return A.AkochanEngine(runner=runner, timeout=1.0)


def _first_view(seed: int = 20261021) -> tuple[Decision, View]:
    for dec, view, _idx in _records(seed, hands=1):
        if dec.kind == "discard" and view.drawn is not None and sum(view.hand) >= 13:
            return dec, view
    raise AssertionError("没找到可用的 View")


def _find_decision(kind: str, seeds=(11, 23, 37, 51, 67, 83)):
    for seed in seeds:
        for dec, view, _idx in _records(seed, hands=2):
            if dec.kind == kind:
                return dec, view
    raise AssertionError(f"没找到 {kind} 决策")


# ---------------------------------------------------------------------------
# 1. 牌名 / 下标
# ---------------------------------------------------------------------------

def test_tile_index_roundtrip():
    for tile in range(34):
        for red in (False, True):
            if red and not (tile < 27 and tile % 9 == 4):
                continue
            ak = A._ak_index(tile, red)
            assert A._ak_base(ak) == tile, (tile, red, ak, A._ak_base(ak))
            assert A._ak_red(ak) is red, (tile, red, ak)
            name = A._tile_name(tile, red)
            assert mjai.tile_from_mjai(name) == (tile, red), (tile, red, name)
            assert A._ak_str(ak) == name
            assert name != "0m" and name != "0p" and name != "0s"
    # 赤五必须是标准 MJAI 写法（akochan 的 hai_str_to_int 只认这个）
    assert A._tile_name(4, True) == "5mr"
    assert A._tile_name(13, True) == "5pr"
    assert A._tile_name(22, True) == "5sr"
    assert A._ak_index(13, True) == 20 and A._ak_base(20) == 13
    assert A._ak_index(22, True) == 30 and A._ak_base(30) == 22
    print("  ✅ 34 种牌 + 3 张赤五的 38 下标 / 标准 MJAI 写法往返一致")


# ---------------------------------------------------------------------------
# 2. moves -> 选项下标
# ---------------------------------------------------------------------------

def test_move_to_index():
    D = Decision
    cases = [
        (D("discard", [0, 4, 27]), [{"type": "dahai", "actor": 0, "pai": "5m"}], 1),
        # 赤五写成 5mr 也要落到「5m」这个选项上
        (D("discard", [0, 4, 27]), [{"type": "dahai", "actor": 0, "pai": "5mr"}], 1),
        (D("discard", [0, 1]), [{"type": "dahai", "actor": 0, "pai": "9p"}], None),
        (D("discard", [0, 1]), [], None),
        (D("riichi", ["yes", "no"]),
         [{"type": "reach", "actor": 0}, {"type": "dahai", "pai": "1m"}], 0),
        (D("riichi", ["yes", "no"]), [{"type": "dahai", "pai": "1m"}], 1),
        (D("tsumo", ["yes", "no"]), [{"type": "hora", "actor": 0}], 0),
        (D("tsumo", ["yes", "no"]), [{"type": "dahai", "pai": "1m"}], 1),
        (D("ron", ["yes", "no"]), [{"type": "hora", "actor": 0, "target": 1}], 0),
        (D("ron", ["yes", "no"]), [{"type": "none", "actor": 0}], 1),
        (D("kyuushu", ["yes", "no"]), [{"type": "kyushukyuhai", "actor": 0}], 0),
        (D("call", ["pass", "pon"]),
         [{"type": "pon", "actor": 0}, {"type": "dahai", "pai": "1m"}], 1),
        (D("call", ["pass", "pon", "kan"]),
         [{"type": "daiminkan", "actor": 0, "pai": "1m"}], 2),
        (D("call", ["pass", "pon"]), [{"type": "none", "actor": 0}], 0),
        (D("chi", ["pass", "chi"]),
         [{"type": "chi", "actor": 0}, {"type": "dahai", "pai": "1m"}], 1),
        (D("chi", ["pass", "chi"]), [{"type": "none", "actor": 0}], 0),
        (D("kan", ["pass", "kan"], tile=4), [{"type": "ankan", "actor": 0, "pai": "5m"}], 1),
        (D("kan", ["pass", "kan"], tile=4), [{"type": "kakan", "actor": 0, "pai": "5m"}], 1),
        # 它想杠的不是引擎问的那张 -> 不能乱杠
        (D("kan", ["pass", "kan"], tile=5), [{"type": "ankan", "actor": 0, "pai": "5m"}], 0),
        (D("kan", ["pass", "kan"], tile=4), None, None),
    ]
    for decision, moves, want in cases:
        got = A.move_to_index(decision, moves)
        assert got == want, (decision, moves, got, want)
    print(f"  ✅ {len(cases)} 条 moves -> 选项下标映射都对")


# ---------------------------------------------------------------------------
# 3. 往返一致性（核心）
# ---------------------------------------------------------------------------

def test_serialize_roundtrip():
    n_points = 0
    n_calls = n_kans = n_riichi = n_red = 0
    for seed in (11, 23, 37, 51):
        tracker = A._HandTracker()
        for dec, view, _idx in _records(seed, hands=2):
            if dec.kind not in A.SUPPORTED_KINDS:
                continue
            # 叫牌之后那张打牌：座位用上一次 fuuro 选择里一起给的 dahai，
            # 不走序列化（akochan 不接受「最后一条是自己的副露」）
            if dec.kind == "discard" and view.drawn is None:
                tracker.observe(view, view.player)
                continue
            events = A.build_events(view, view.player, tracker,
                                    kind=dec.kind, tile=dec.tile)
            bad = _compare(view, events)
            assert not bad, f"seed={seed} kind={dec.kind}: 往返不一致 {bad}"
            n_points += 1
            n_calls += 1 if view.melds else 0
            n_kans += 1 if any(m.kind in ("ankan", "kakan", "minkan")
                               for m in view.melds) else 0
            n_riichi += 1 if any(view.riichi) else 0
            n_red += 1 if any(view.hand_red) else 0
            # 四家的起手牌必须是**一次真实发牌**：52 张里每种牌不超过 4 张、
            # 赤五每色不超过 1 张。别家的手牌内容没人读，但牌数不守恒时
            # akochan 会静默返回空 review（实测 4 张 5pr 就会）。
            used = [0] * 38
            for hand in events[1]["tehais"]:
                assert len(hand) == 13, hand
                for name in hand:
                    used[_ak(name)] += 1
            for t in range(34):
                n = used[A._ak_index(t)]
                if t < 27 and t % 9 == 4:
                    n += used[A._ak_index(t, True)]
                assert n <= 4, f"seed={seed}: 起手牌里 {_akname(t)} 一共 {n} 张（>4）"
                if t < 27 and t % 9 == 4:
                    assert used[A._ak_index(t, True)] <= 1, \
                        f"seed={seed}: 赤五 {_akname(t)} 超过 1 张"
    assert n_points >= 60, n_points
    print(f"  ✅ {n_points} 个决策点往返一致（含副露 {n_calls} / 杠 {n_kans} / "
          f"立直 {n_riichi} / 带赤五 {n_red}）")


def test_three_player_rejected():
    dec, view = None, None
    for d, v, _i in _records(101, players=3, hands=1):
        if v.rule.players == 3:
            dec, view = d, v
            break
    assert view is not None
    try:
        A.build_events(view, view.player, A._HandTracker(), kind=dec.kind)
    except A.UnsupportedView:
        pass
    else:
        raise AssertionError("三麻应该抛 UnsupportedView")
    # 座位也必须兜底（不能把三麻喂给 akochan）
    agent = A.ExternalAkochanAgent(engine=_fake_engine(moves=[{"type": "none"}]))
    idx = agent.decide(dec, view)
    assert 0 <= idx < len(dec.options)
    assert agent.fallbacks == 1, agent.fallbacks
    print("  ✅ 三麻直接被拒绝（抛 UnsupportedView / 座位走兜底）")


# ---------------------------------------------------------------------------
# 4/5. 负向验证
# ---------------------------------------------------------------------------

def test_negative_no_moves():
    """akochan 跑通了但 review 里**没有 moves** -> 必须兜底，不许崩。"""
    _dec, view = _first_view()
    decision = Decision("discard", [t for t in range(34) if view.hand[t] > 0])
    engine = _fake_engine(moves=None)          # review = []
    agent = A.ExternalAkochanAgent(engine=engine)
    idx = int(agent.decide(decision, view))
    assert 0 <= idx < len(decision.options), idx
    assert engine.stats["no_moves"] == 1, engine.stats
    assert engine.stats["ok"] == 0
    assert agent.fallbacks == 1, agent.fallbacks
    assert "moves" in agent._last_error or "可用" in agent._last_error
    # 教师也给出同样的答案（兜底用的就是教师）
    teacher = HeuristicAgent(0).decide(decision, view)
    assert idx == int(teacher)
    print(f"  ✅ 没有 moves -> 兜底成教师（index={idx}），没崩")


def test_negative_timeout():
    """超时 -> 必须兜底（而且临时文件要删）。"""
    _dec, view = _first_view()
    decision = Decision("discard", [t for t in range(34) if view.hand[t] > 0])
    paths: list[str] = []
    engine = _fake_engine(timeout=True, paths=paths)
    agent = A.ExternalAkochanAgent(engine=engine)
    idx = int(agent.decide(decision, view))
    assert 0 <= idx < len(decision.options)
    assert engine.stats["timeouts"] == 1, engine.stats
    assert agent.fallbacks == 1
    assert agent.summary()["fallbacks"] == 1
    assert paths and not Path(paths[0]).exists(), paths
    print("  ✅ 超时 -> 兜底（临时文件已删），没崩")


def test_negative_missing_engine():
    """目录不存在 / 没装 -> available=False -> 兜底。"""
    missing = Path(tempfile.gettempdir()) / "definitely-no-akochan-here"
    ok, why = A.available(missing)
    assert not ok and "akochan" in why, (ok, why)
    engine = A.AkochanEngine(root=missing)
    assert not engine.available
    _dec, view = _first_view()
    decision = Decision("discard", [t for t in range(34) if view.hand[t] > 0])
    agent = A.ExternalAkochanAgent(engine=engine)
    idx = int(agent.decide(decision, view))
    assert 0 <= idx < len(decision.options)
    assert agent.fallbacks == 1
    assert engine.stats["queries"] == 0
    print("  ✅ 进程不存在 -> 兜底（一次查询都没发），没崩")


def test_negative_garbage_output():
    """rc=0 但输出是垃圾 / 退出码非 0 -> 兜底。"""
    _dec, view = _first_view()
    decision = Decision("discard", [t for t in range(34) if view.hand[t] > 0])
    for kw in ({"raw": "boom\nnot json\n"}, {"raw": "", "rc": 3}):
        engine = _fake_engine(**kw)
        agent = A.ExternalAkochanAgent(engine=engine)
        idx = int(agent.decide(decision, view))
        assert 0 <= idx < len(decision.options)
        assert engine.stats["errors"] == 1, engine.stats
        assert agent.fallbacks == 1
    print("  ✅ 垃圾输出 / 非 0 退出码 -> 兜底")


# ---------------------------------------------------------------------------
# 6. 临时文件
# ---------------------------------------------------------------------------

def test_temp_files_cleaned_and_unique():
    """并发查询的文件名必须唯一，而且成功/失败都要删掉。"""
    paths: list[str] = []
    engine = _fake_engine(moves=[{"type": "none", "actor": 0}],
                          paths=paths, delay=0.02)
    events = [A._probe_events() for _ in range(2)]
    out: list[list] = []

    def work(e):
        out.append(engine.query(e, 0))

    threads = [threading.Thread(target=work, args=(e,)) for e in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(paths) == 2
    assert len(set(paths)) == 2, paths             # uuid 不撞名
    for p in paths:
        assert not Path(p).exists(), p             # 全删了
    assert engine.stats["temp_files"] == 2
    assert all(o == [{"type": "none", "actor": 0}] for o in out)
    print("  ✅ 并发 2 次查询：文件名唯一、临时文件全部清理")


# ---------------------------------------------------------------------------
# 7/8. 缓存 / 叫牌后打牌
# ---------------------------------------------------------------------------

def test_cache_and_post_call_discard():
    # (a) 同一个决策点被问多次（立直 -> 打牌）：只查一次
    _dec, view = _first_view()
    opts = [t for t in range(34) if view.hand[t] > 0]
    first = opts[0]
    engine = _fake_engine(moves=[
        {"type": "dahai", "actor": view.player, "pai": _akname(first)},
    ])
    agent = A.ExternalAkochanAgent(engine=engine)
    i1 = agent.decide(Decision("riichi", ["yes", "no"]), view)
    i2 = agent.decide(Decision("discard", opts), view)
    assert engine.stats["queries"] == 1, engine.stats
    assert agent.cached == 1, agent.cached
    assert i1 == 1 and i2 == 0, (i1, i2)

    # (b) 碰之后那张打牌：直接用 fuuro 选择里一起给的 dahai，不再查询
    call_dec, call_view = _find_decision("call")
    hand_opts = [t for t in range(34) if call_view.hand[t] > 0]
    assert hand_opts, "call 决策的手牌是空的？"
    want = hand_opts[0]
    engine2 = _fake_engine(moves=[
        {"type": "pon", "actor": call_view.player},
        {"type": "dahai", "actor": call_view.player, "pai": _akname(want)},
    ])
    agent2 = A.ExternalAkochanAgent(engine=engine2)
    i3 = agent2.decide(call_dec, call_view)
    assert i3 == int(call_dec.options.index("pon")), (i3, call_dec.options)
    q = engine2.stats["queries"]
    i4 = agent2.decide(Decision("discard", hand_opts), call_view)
    assert i4 == 0, (i4, want)
    assert engine2.stats["queries"] == q, "碰后打牌不该再查一次"
    print("  ✅ 同一决策点只查一次；碰后打牌用缓存的 dahai")


# ---------------------------------------------------------------------------
# 9. 注册表 / 平台列表
# ---------------------------------------------------------------------------

def test_registry_external():
    spec = variants.get("akochan")
    assert spec.external is True, spec
    assert spec.players == 4 and spec.name == "akochan"
    assert variants.resolve("AKOCHAN") == "akochan"
    assert variants.resolve("akochan4") == "akochan"
    assert variants.is_external("akochan") and not variants.is_external("teacher")
    # **不进天梯**：all_specs / same_players 里不能有它
    assert "akochan" not in [s.name for s in variants.all_specs()]
    assert "akochan" not in [s.name for s in variants.same_players(4)]
    # audit_weights 不能把它报成「注册了但没权重」
    report = variants.audit_weights(verbose=False)
    assert "akochan" not in report.get("missing_files", []), report.get("missing_files")
    # 平台列表：external=True，usable 跟真探测一致
    from webapp.server import agents as AG
    rows = {r["name"]: r for r in AG.available_models(refresh=True)}
    assert "akochan" in rows, list(rows)[:5]
    row = rows["akochan"]
    assert row.get("external") is True, row
    assert row["players"] == 4
    ok, why = A.available()
    assert row["usable"] == bool(ok), (row, why)
    if not ok:
        print(f"  ⚠️ 本机没装 akochan（{why}）—— 列表里如实标成不可用")
    else:
        assert row["usable"] is True, row
    print(f"  ✅ 注册表：external=True / 不进天梯 / audit 不报缺失 / "
          f"平台 usable={row['usable']}")


# ---------------------------------------------------------------------------
# 10. 真进程端到端（可选，AKOCHAN_E2E=1）
# ---------------------------------------------------------------------------

def test_e2e_real_engine():
    if os.environ.get("AKOCHAN_E2E") != "1":
        print("  ⏭  跳过真进程端到端（设 AKOCHAN_E2E=1 才跑）")
        return
    ok, why = A.available()
    if not ok:
        print(f"  ⏭  没装 akochan：{why}")
        return
    engine = A.AkochanEngine(timeout=10.0)
    assert engine.probe(live=True)[0]
    tracker = A._HandTracker()
    n = 0
    for seed in (11,):
        for dec, view, _idx in _records(seed, hands=1):
            if dec.kind not in ("discard", "call", "chi", "ron"):
                continue
            if dec.kind == "discard" and view.drawn is None:
                continue
            events = A.build_events(view, view.player, tracker,
                                    kind=dec.kind, tile=dec.tile)
            moves = engine.query(events, view.player)
            assert moves, f"真引擎没给出 moves（{dec.kind}）"
            idx = A.move_to_index(dec, moves)
            assert idx is not None and 0 <= idx < len(dec.options), (dec, moves)
            n += 1
            if n >= 3:
                break
        if n >= 3:
            break
    assert n >= 1
    print(f"  ✅ 真 system.exe 端到端 {n} 个决策点：都给出了能对上选项的 moves")


# ---------------------------------------------------------------------------
def main() -> int:
    print("【akochan 外部引擎座位】")
    tests = [
        test_tile_index_roundtrip,
        test_move_to_index,
        test_serialize_roundtrip,
        test_three_player_rejected,
        test_negative_no_moves,
        test_negative_timeout,
        test_negative_missing_engine,
        test_negative_garbage_output,
        test_temp_files_cleaned_and_unique,
        test_cache_and_post_call_discard,
        test_registry_external,
        test_e2e_real_engine,
    ]
    failed: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  ❌ {fn.__name__}: {type(exc).__name__}: {exc}")
            if os.environ.get("AKOCHAN_TEST_DEBUG") == "1":
                import traceback

                traceback.print_exc()
    print()
    if failed:
        print(f"❌ {len(failed)} 项失败：{', '.join(failed)}")
        return 1
    print(f"✅ akochan 外部引擎 {len(tests)} 项全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
