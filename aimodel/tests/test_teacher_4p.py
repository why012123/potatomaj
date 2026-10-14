"""四麻教师不能对第 4 家失明（P0-1 回归）。

审计证据：``agents.py:177`` / ``agents.py:192`` 把对手范围写死成 ``range(3)``。
三麻是 3 家所以正确，**四麻有 4 家，第 4 家（P3）被完全忽略**。
实测 649 次「场上有立直」的四麻决策里 38.7% 漏看立直家、
同一张 5m 危险度 0.08（漏看）vs 0.62（正确）。

这个文件把「教师必须按 ``view.rule.players`` 取对手」钉死。

跑法：``.venv/bin/python aimodel/tests/test_teacher_4p.py``
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent, HumanAgent, make_agents  # noqa: E402
from aimodel.encode import tile_danger  # noqa: E402
from aimodel.engine import View  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


R3 = S.DEFAULT_RULE
R4 = S.rule_for(4)

#: 牌型下标：0-8 万 / 9-17 筒 / 18-26 索 / 27-33 字
T5M = 4        # 5m —— 中张，危险度最高那一档（d = 0.62）
T5P = 9 + 4    # 5p
HONOR_E = 27   # 东（字牌，默认手牌里有 1 张）

#: 默认手牌：teacher 的 turn=5、无副露时 d_open 基线恰好是 0.30
DEFAULT_HAND = {T5M: 2, T5P: 2, HONOR_E: 1}


def make_view(
    players: int,
    player: int = 0,
    *,
    hand: dict[int, int] | None = None,
    riichi: dict[int, bool] | None = None,
    discards: dict[int, list[int]] | None = None,
    melds_all: dict[int, list] | None = None,
    turn: int = 5,
    scores: list[int] | None = None,
) -> View:
    """造一个 ``players`` 家的局面，只填这个测试关心的字段。"""
    rule = S.rule_for(players)
    h = [0] * 34
    for t, n in (DEFAULT_HAND if hand is None else hand).items():
        h[t] = n
    return View(
        player=player,
        hand=h,
        drawn=None,
        melds=[],
        hand_red=[0] * 34,
        discards=[list((discards or {}).get(q, [])) for q in range(players)],
        melds_all=[list((melds_all or {}).get(q, [])) for q in range(players)],
        kita=[0] * players,
        scores=scores or [rule.start_score] * players,
        dora_indicators=[0],
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=40,
        turn=turn,
        riichi=[bool((riichi or {}).get(q, False)) for q in range(players)],
        ippatsu=[False] * players,
        is_dealer=(player == 0),
        last_discard=None,
        rule=rule,
    )


def main() -> int:
    print("== 对手范围按实际人数取（不再写死 3）==")
    t = HeuristicAgent(0)
    check(t._opp_riichi(make_view(4, riichi={3: True})) == [3],
          f"四麻 P3 立直应为 [3]，得到 {t._opp_riichi(make_view(4, riichi={3: True}))}")
    check(t._opp_riichi(make_view(4, riichi={1: True, 2: True})) == [1, 2],
          "四麻 P1/P2 立直应为 [1, 2]")
    check(t._opp_riichi(make_view(4, riichi={0: True})) == [],
          "自己立直不算对手")
    check(t._opp_riichi(make_view(4, riichi={1: True, 2: True, 3: True})) == [1, 2, 3],
          "三家都立直应为 [1, 2, 3]")
    # 三麻行为不能变
    check(t._opp_riichi(make_view(3, riichi={1: True, 2: True})) == [1, 2],
          "三麻仍应为 [1, 2]")
    check(t._opp_riichi(make_view(3)) == [], "三麻没人立直应为空")
    # 没带 rule 的手写 View 也要能推出来
    v_norule = make_view(4, riichi={3: True})
    v_norule.rule = None
    check(t._opp_riichi(v_norule) == [3], "view.rule 缺失时应能从 scores 推出 4 家")
    print("  ✅ 四麻 [3] / [1,2] / [1,2,3] / 三麻 [1,2] / rule 缺失兜底")

    print("\n== 危险度确实把第 4 家算进去了 ==")
    # 隔离 d_open 基线：手牌/牌河/副露全空 -> d_open = 0.08（early turn 基线）
    # 手牌非空只影响调用方，不影响 danger，所以这里两个 view 的手牌都为空
    v4_p3 = make_view(4, riichi={3: True}, hand={})
    v4_no = make_view(4, hand={})
    d_p3 = tile_danger(v4_p3, T5M)
    d_no = tile_danger(v4_no, T5M)
    check(abs(d_no - 0.08) < 1e-9, f"无人立直时 5m 应只有 d_open 基线 0.08，得到 {d_no:.3f}")
    check(abs(d_p3 - 0.62) < 1e-9,
          f"P3 立直时 5m 应是纯立直项 0.62，得到 {d_p3:.3f} —— 第 4 家被漏掉了")
    check(d_p3 > d_no + 0.2, "P3 立直与无人立直必须拉开差距")
    # 显式传 opp_riichi=[3]（教师走的就是显式那条路）必须与自动推导一致
    check(abs(tile_danger(v4_p3, T5M, opp_riichi=[3]) - d_p3) < 1e-9,
          "显式 opp_riichi=[3] 与自动推导应一致")
    # P1 立直在三麻/四麻都早就算对了 —— 防止「修 P3 修坏 P1」
    check(abs(tile_danger(make_view(4, riichi={1: True}, hand={}), T5M) - 0.62) < 1e-9,
          "四麻 P1 立直也应是 0.62")
    # P3 打过 5m -> 对 P3 是现物，立直那一项归零，只剩 d_open 基线
    d_gen = tile_danger(make_view(4, riichi={3: True}, discards={3: [T5M]}, hand={}), T5M)
    check(abs(d_gen - 0.08) < 1e-9,
          f"P3 打过 5m 后立直项应归零（只剩 0.08 基线），得到 {d_gen:.3f}"
          f" —— 说明 P3 的牌河真的参与了计算")
    # 字牌在只剩 1 张时更安全
    check(tile_danger(v4_p3, HONOR_E) < d_p3, "字牌应比中张安全")
    print(f"  ✅ 5m: 无人立直 {d_no:.3f} → P3 立直 {d_p3:.3f}"
          f"；P3 现物 {d_gen:.3f}；东 {tile_danger(v4_p3, HONOR_E):.3f}")

    print("\n== 教师决策真的变了：P3 立直会触发「有压力」==")
    v_fold = make_view(4, hand={0: 1, 9: 1, 18: 1, 27: 1}, riichi={3: True})
    ag = HeuristicAgent(0)
    check(ag._pressure(v_fold, ag._opp_riichi(v_fold)) == 1.0,
          "P3 立直时 _pressure 必须是 1.0（修之前看不见 P3 -> 0.0）")
    check(ag._pressure(make_view(4), ag._opp_riichi(make_view(4))) < 1.0,
          "没人立直时 _pressure 不该是 1.0")
    # 第 4 家的副露也要算进 _pressure
    v_open3 = make_view(4, melds_all={3: [1, 2]})
    check(ag._pressure(v_open3, ag._opp_riichi(v_open3)) == 0.7,
          "P3 有两组副露时 _pressure 应为 0.7（修之前看不见 P3 -> 0.0）")
    print("  ✅ P3 立直 -> 压力 1.0；P3 两组副露 -> 压力 0.7")

    print("\n== _discard 会因 P3 立直而改选（端到端）==")
    # 同一手牌、同一巡：唯一区别是 P3 有没有立直，教师的出牌应当不同
    from aimodel.engine import Decision  # noqa: E402

    hand = {T5M: 1, T5P: 1, 18: 1, HONOR_E: 1}  # 全是孤张，向听很高
    opts = sorted(hand)
    pick_no = ag._discard(Decision("discard", list(opts)), make_view(4, hand=hand))
    pick_ri = ag._discard(
        Decision("discard", list(opts)), make_view(4, hand=hand, riichi={3: True})
    )
    face_no = opts[pick_no]
    face_ri = opts[pick_ri]
    print(f"  P3 未立直打 {S.tile_cn(face_no)} / P3 立直打 {S.tile_cn(face_ri)}")
    # 立直后必须变成「安全优先」：选中的牌危险度不高于立直前的选择
    d_sel_ri = tile_danger(make_view(4, hand=hand, riichi={3: True}), face_ri)
    d_sel_no = tile_danger(make_view(4, hand=hand, riichi={3: True}), face_no)
    check(d_sel_ri <= d_sel_no,
          f"P3 立直后教师应改打更安全的牌（{d_sel_ri:.3f} !<= {d_sel_no:.3f}）")

    print("\n== show_table / 打印点数按实际人数 ==")
    buf = io.StringIO()
    ha = HumanAgent(input_fn=lambda _: "0",
                    print_fn=lambda *a, **k: print(*a, file=buf, **k))
    ha.show_table(make_view(4, riichi={3: True}))
    out = buf.getvalue()
    check("P3" in out, f"四麻牌桌应打印 P3 的点数/牌河，实际：\n{out}")
    check("对家" in out, f"四麻座位名应有「对家」，实际：\n{out}")
    check(out.count("牌河") == 4, f"四麻应打印 4 条牌河，实际 {out.count('牌河')} 条")
    # 座位名要相对自己算：P3 从 P1 看是「对家」
    buf2 = io.StringIO()
    HumanAgent(print_fn=lambda *a, **k: print(*a, file=buf2, **k)).show_table(
        make_view(4, player=1)
    )
    check("对家" in buf2.getvalue(), "从 P1 看的「对家」应是 P3")
    print("  ✅ 四麻 4 条牌河 + 「对家」 + P3 点数")

    print("\n== make_agents 按人数造 ==")
    check(len(make_agents("teacher", seed=0, players=3)) == 3, "三麻应 3 个 agent")
    check(len(make_agents("teacher", seed=0, players=4)) == 4, "四麻应 4 个 agent")
    check(len(make_agents("random", seed=0, players=4)) == 4, "四麻 random 应 4 个")
    print("  ✅ 三麻 3 个 / 四麻 4 个")

    if FAILED:
        print(f"\n❌ {len(FAILED)} 条失败")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
