"""四麻的「吃」必须真的被学到（P0-2 回归）。

审计证据：
* ``encode.py:29`` 的 ``HEAD_SIZE`` 里**没有 chi**
* ``net.py:30`` 的 ``KINDS`` 里**也没有 chi**
* 但 ``engine.py:41`` 的 ``ACTION_TABLE`` 有 ``"chi": ("pass", "chi")``，
  且 ``engine.py:877`` 真的会向 agent 抛 chi 决策
* 结果：``train.py:65`` 的 ``if kind in HEAD_SIZE`` 判定失败
  -> **Tracker 里 0 条 chi 记录**（实测四麻 12.6% 的决策丢失）

跑法：``.venv/bin/python aimodel/tests/test_chi_4p.py``
"""

from __future__ import annotations

import collections
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import (  # noqa: E402
    STYLES, HeuristicAgent, MLAgent, TeacherStyle,
)
from aimodel.encode import HEAD_SIZE, action_index, action_mask, layout_for  # noqa: E402
from aimodel.engine import ACTION_TABLE, Decision, Game, View  # noqa: E402
from aimodel.net import KINDS, Net  # noqa: E402
from aimodel.train import RecordingAgent  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


R4 = S.rule_for(4)

#: 牌型下标
T1M, T9M = 0, 8
T1P, T9P = 9, 17
T1S, T9S = 18, 26
HONORS = [27, 28, 29, 30, 31, 32, 33]

#: 13 张、指定向听的手牌（向听值由 fastshanten 实测，别手改）
#: call_shanten: normal=3 / mountain=1 / flash=5 / huge=1
#: 吃不吃 = 向听 <= call_shanten
HANDS = {
    2: "3444568m667p89s1z",   # 2 向听：normal 吃，mountain/huge 不吃
    3: "457789m134p1269s",    # 3 向听：normal 吃，mountain/huge 不吃
    4: "344567m15p368s16z",   # 4 向听：只有 flash 吃
    5: "149m138p1369s356z",   # 5 向听：只有 flash 吃
}

#: 每个风格在每个向听下「该不该吃」
EXPECT = {
    "normal":   {2: True,  3: True,  4: False, 5: False},
    "mountain": {2: False, 3: False, 4: False, 5: False},
    "flash":    {2: True,  3: True,  4: True,  5: True},
    "huge":     {2: False, 3: False, 4: False, 5: False},
}


def chi_view(hand_str: str, player: int = 1) -> View:
    """按 ``hand_str`` 造一个「吃不吃」局面（四麻，自己是 P1）。"""
    h = list(S.parse_hand(hand_str))
    return View(
        player=player,
        hand=h,
        drawn=None,
        melds=[],
        hand_red=[0] * 34,
        discards=[[] for _ in range(4)],
        melds_all=[[] for _ in range(4)],
        kita=[0] * 4,
        scores=[R4.start_score] * 4,
        dora_indicators=[T1S],
        round_wind=S.EAST,
        seat_wind=S.SOUTH,
        wall_left=60,
        turn=4,
        riichi=[False] * 4,
        ippatsu=[False] * 4,
        is_dealer=False,
        last_discard=None,
        rule=R4,
    )


def main() -> int:
    print("== 三个动作表必须完全一致（不能再漏一个）==")
    check("chi" in HEAD_SIZE, "encode.HEAD_SIZE 必须有 chi")
    check(HEAD_SIZE.get("chi") == 2, f"chi 头应是 2（pass/chi），得到 {HEAD_SIZE.get('chi')}")
    check("chi" in KINDS, "net.KINDS 必须有 chi")
    # 这三张表必须一一对应 —— 老 bug 就是 HEAD_SIZE/KINDS 少一张
    for k, table in ACTION_TABLE.items():
        if k == "discard":
            continue  # discard 是 34 张动态掩码，不跟 ACTION_TABLE 对齐
        check(k in HEAD_SIZE, f"ACTION_TABLE 有 {k}，HEAD_SIZE 却没有")
        check(k in KINDS, f"ACTION_TABLE 有 {k}，net.KINDS 却没有")
        if k in HEAD_SIZE:
            check(HEAD_SIZE[k] == len(table),
                  f"{k}: HEAD_SIZE={HEAD_SIZE[k]} 但 ACTION_TABLE 有 {len(table)} 项")
    check(set(KINDS) == set(HEAD_SIZE), f"KINDS 与 HEAD_SIZE 键集必须相同：{set(KINDS) ^ set(HEAD_SIZE)}")
    print(f"  ✅ ACTION_TABLE / HEAD_SIZE / KINDS 三者一致（{len(ACTION_TABLE)} 种）")

    print("\n== 30 场四麻：Tracker 里 chi 的记录数 > 0，且约占 12% ==")
    tracker_kinds = collections.Counter()
    asked = collections.Counter()

    class Spy(RecordingAgent):
        """RecordingAgent + 统计引擎到底问了哪些决策。"""

        def decide(self, decision, view):
            asked[decision.kind] += 1
            return super().decide(decision, view)

    GAMES = 30
    for g in range(GAMES):
        tr = _SharedTracker(tracker_kinds)
        agents = [Spy(HeuristicAgent(g * 4 + i), tr) for i in range(4)]
        Game(agents, seed=g, rule=R4).play()

    total_asked = sum(asked.values())
    chi_asked = asked["chi"]
    chi_frac = chi_asked / total_asked
    chi_rec = tracker_kinds["chi"]
    print(f"  引擎问了 {total_asked} 次决策，其中 chi {chi_asked} 次（{chi_frac:.1%}）")
    print(f"  Tracker 里 chi 记录 {chi_rec} 条")
    check(chi_rec > 0, f"Tracker 里必须有 chi 记录，得到 {chi_rec}（老 bug 是 0）")
    # 每条 chi 决策都应被记录：记录数 == 引擎问的次数
    check(chi_rec == chi_asked,
          f"chi 应逐条记录：问了 {chi_asked} 次却只记了 {chi_rec} 条")
    check(0.08 < chi_frac < 0.20,
          f"chi 应约占 12%（审计实测 12.6%），得到 {chi_frac:.1%}")
    print(f"  ✅ chi {chi_asked} 次全部入库，占 {chi_frac:.1%}")

    print("\n== 四个风格的吃牌规则不能完全相同 ==")
    # 逐格核对：向听 <= call_shanten 就该吃
    table = {}
    for name in STYLES:
        ag = HeuristicAgent(0, style=name)
        table[name] = {
            sh: ag._want_chi(chi_view(hs)) for sh, hs in HANDS.items()
        }
    for name in STYLES:
        print(f"  {name:<9} call_shanten={STYLES[name].call_shanten}  {table[name]}")
        check(table[name] == EXPECT[name],
              f"{name} 的吃牌规则应为 {EXPECT[name]}，得到 {table[name]}")
    # 「风格生效」的可检验含义：四个风格不能塌成同一个答案，而且
    #   * normal(3) vs mountain(1) 必须不同（mountain 更不爱叫）
    #   * flash(5)  vs normal(3)   必须不同（flash 更爱叫）
    # 注意：mountain 与 huge 的 call_shanten **都是 1**，所以它俩在
    # 吃不吃这一档上本来就相同（差别在打点/进张权重），不算 bug。
    check(len({tuple(r.values()) for r in table.values()}) > 1,
          f"四个风格的吃牌规则塌成了同一个答案：{table}")
    check(table["normal"] != table["mountain"],
          f"normal 与 mountain 必须不同（3 vs 1）：{table['normal']} / {table['mountain']}")
    check(table["flash"] != table["normal"],
          f"flash 与 normal 必须不同（5 vs 3）：{table['flash']} / {table['normal']}")
    check(any(table["flash"][sh] and not table["mountain"][sh] for sh in HANDS),
          "至少要有一个向听档位 flash 吃而 mountain 不吃")
    check(STYLES["mountain"].call_shanten == STYLES["huge"].call_shanten,
          "前提变了：mountain/huge 的 call_shanten 不再相同，测试要跟着改")
    print("  ✅ 四风格逐档核对通过（normal≠mountain≠flash；mountain=huge 是设计）")

    print("\n== MLAgent 的兜底教师必须跟着风格走 ==")
    # 老代码恒为 HeuristicAgent(seed)（normal）-> flash/huge 的吃牌规则
    # 和 normal 一模一样。这里直接看兜底教师拿到了什么 style。
    net = Net(dim=layout_for(4)[1], hidden=(8,), seed=0)
    for name in STYLES:
        ag = MLAgent(net, seed=0, style=name)
        check(getattr(ag.fallback, "style", None) == STYLES[name],
              f"style={name} 的 MLAgent 兜底教师应是 {name}，得到 "
              f"{getattr(getattr(ag.fallback, 'style', None), 'name', None)}")
    # 端到端：网络没有 chi 头 -> 走兜底 -> 兜底必须按风格回答
    old_net = Net(dim=layout_for(4)[1], hidden=(8,), seed=0)
    old_net.kinds = [k for k in KINDS if k != "chi"]
    # 选项顺序是 ACTION_TABLE["chi"] = ("pass", "chi")：index 1 = 吃。
    # 老代码 HeuristicAgent.decide 把它写反了（`0 if _want_chi else 1`），
    # 于是教师「想吃的时候过」—— 这个测试同时钉住这个反向 bug。
    opts = list(ACTION_TABLE["chi"])
    check(opts == ["pass", "chi"], f"chi 选项顺序应是 (pass, chi)，得到 {opts}")
    d = Decision("chi", opts, tile=T1S)
    for sh, hs in HANDS.items():
        v = chi_view(hs)
        per_style = {
            name: opts[MLAgent(old_net, seed=0, style=name).decide(d, v)]
            for name in STYLES
        }
        want = {name: ("chi" if EXPECT[name][sh] else "pass") for name in STYLES}
        print(f"  {sh} 向听 {hs:<22} 兜底={per_style}")
        check(per_style == want, f"{sh} 向听下兜底应为 {want}，得到 {per_style}")
    # 教师本体（不经网络）也必须是同一套语义
    check(opts[HeuristicAgent(0, style="normal").decide(d, chi_view(HANDS[2]))] == "chi",
          "normal 教师 2 向听应该吃（老代码会答「过」）")
    check(opts[HeuristicAgent(0, style="mountain").decide(d, chi_view(HANDS[2]))] == "pass",
          "mountain 教师 2 向听不该吃")
    print("  ✅ 兜底教师按风格分岔（flash 全吃 / mountain·huge 全不吃）")

    print("\n== chi 的掩码 / 下标映射 ==")
    m = action_mask(d)
    check(len(m) == 2, f"chi 掩码应是 2 维，得到 {len(m)}")
    check(m[0] == 1.0 and m[1] == 1.0, "pass 与 chi 都该合法")
    check(action_index(d, 0) == 0, "options[0]=pass -> 头下标 0")
    check(action_index(d, 1) == 1, "options[1]=chi -> 头下标 1")
    m1 = action_mask(Decision("chi", ["chi"], tile=T1P))
    check(m1[0] == 0.0 and m1[1] == 1.0, "只给 chi 时掩码应只开第 1 位")
    # 外部引擎可能把「每一种吃法」作为独立选项传进来（"chi:10:12"）。
    # 动作空间仍是二值，看冒号前的动作名。老代码在这里会把 head=1 静默变成 0（过）。
    dm = Decision("chi", ["pass", "chi:10:12", "chi:21:22"], tile=T1P)
    mm = action_mask(dm)
    check(mm.tolist() == [1.0, 1.0], f"逐吃法选项的掩码应是 [1,1]，得到 {mm.tolist()}")
    check(action_index(dm, 1) == 1 and action_index(dm, 2) == 1,
          "逐吃法选项都应映射到 chi 头（下标 1）")
    probe = MLAgent(Net(dim=layout_for(4)[1], hidden=(8,), seed=0), seed=0)
    check(probe._head_to_option(dm, 1) == 1,
          "head=1 必须映射到第一个吃法选项，不能静默变成 0（pass）")
    print("  ✅ chi 掩码 2 维 / 下标映射正确（含逐吃法写法）")

    print("\n== 旧权重（没有 chi 头）必须被安全处理，且不能静默 ==")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "old.npz"
        net.save(p)
        # 手工把 chi 头从存档里抹掉，模拟「加 chi 之前的旧权重」
        import json

        import numpy as np

        data = dict(np.load(p, allow_pickle=False))
        meta = json.loads(bytes(data["_meta"]).decode())
        meta["kinds"] = [k for k in meta["kinds"] if k != "chi"]
        data["_meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        for name in list(data):
            if name.startswith("Wp_chi") or name.startswith("bp_chi"):
                del data[name]
        np.savez_compressed(p, **data)
        # ⚠️ 契约在 2026-10-15 变过一次，这里说明为什么：
        #
        # 原契约是「缺头 → 直接抛 ValueError」。它的意图是对的（不能静默接受
        # 随机初始化的头），但**代价是加一个头就让全部 44 个存量权重加载失败** ✗
        # —— 天梯 / Java 牌桌 / 网页平台一起瘫，实测踩过一次。
        #
        # 新契约（三条一起满足，意图不变、代价没了）：
        #   ① 加载成功（不再抛错）
        #   ② **必须有 UserWarning**（不是静默）
        #   ③ 缺的头**零初始化**并记进 `untrained_heads`，
        #      MLAgent 见到它走启发式兜底 → 老模型行为与加头前一致
        import warnings as _w

        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            net2 = Net.load(p)
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        check(bool(msgs), "加载缺头的旧权重必须发 UserWarning，不能静默")
        check(any("chi" in m for m in msgs), f"警告必须点明 chi，实际：{msgs[:1]}")
        check(any("吃" in m or "重新训练" in m for m in msgs),
              f"警告必须说清原因/怎么办，实际：{msgs[:1]}")
        check("chi" in net2.untrained_heads, "缺的头必须记进 untrained_heads")
        z = net2.Wp["chi"]
        check(float(abs(z).max()) == 0.0, f"缺的头必须零初始化，实际最大 {abs(z).max()}")
        check("chi" not in net2.kinds,
              "缺的头不应进 net.kinds —— MLAgent 靠这个走启发式兜底，行为才和老版本一致")
        print(f"  ✅ 正确处理：{msgs[0][:110]}")

    print("\n== l2 必须存档并恢复（P0-4，与加头同批改动）==")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l2.npz"
        n1 = Net(dim=layout_for(4)[1], hidden=(8,), seed=0, lr=3e-4, l2=1e-3)
        n1.save(p)
        n2 = Net.load(p)
        check(abs(n2.l2 - 1e-3) < 1e-12, f"l2 应恢复为 1e-3，得到 {n2.l2}")
        check(abs(n2.lr - 3e-4) < 1e-12, f"lr 应恢复为 3e-4，得到 {n2.lr}")
        print(f"  ✅ l2 {1e-3} / lr {3e-4} 往返一致")

    if FAILED:
        print(f"\n❌ {len(FAILED)} 条失败")
        return 1
    print("\n✅ 全部通过")
    return 0


class _SharedTracker:
    """把 Tracker 的记录转发到一个全局计数器（统计用，不做真训练）。"""

    def __init__(self, counter):
        self.counter = counter
        self._n = 0

    def record(self, kind, *a, **k):
        self.counter[kind] += 1
        self._n += 1

    def __len__(self):
        return self._n


_ = TeacherStyle  # 类型标注用


if __name__ == "__main__":
    raise SystemExit(main())
