"""四麻的测试：牌张、役种、点数、引擎（含吃）。

四麻不是「三麻多一个人」—— 牌从 108 变 136、多了**吃**、万子终于是完整的、
北家有了自风、不聴罰符从 2000 变 3000。这些都要一条条验。

跑法：``.venv/bin/python aimodel/tests/test_4p.py``
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import engine as E  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


R4 = S.rule_for(4)
R3 = S.DEFAULT_RULE


def ev(hand: str, win: str, rule=R4, melds=(), tsumo=False, sw=S.EAST, rw=S.EAST):
    n = sum(S.parse_hand(hand)) + 3 * len(melds)
    if n != 14:
        raise AssertionError(f"用例写错了：{hand} + {len(melds)} 副露 = {n} 张")
    ctx = S.WinContext(
        win_tile=S.parse_hand(win).index(1), is_tsumo=tsumo, rule=rule,
        round_wind=rw, seat_wind=sw,
    )
    return S.evaluate_hand(S.parse_hand(hand), list(melds), ctx)


print("== 牌张 ==")
check(len(S.tile_types(4)) == 34 and len(S.tile_types(4)) * 4 == 136, "四麻应为 34 种 / 136 张")
check(len(S.tile_types(3)) * 4 == 108, "三麻仍应是 108 张")
check(R4.dead_wall == 14, f"四麻王牌应为 14 张，得到 {R4.dead_wall}")
check(R4.noten_penalty == 3000, f"四麻不聴罰符应为 3000，得到 {R4.noten_penalty}")
check(R4.start_score == 25000 and R4.return_score == 30000, "四麻 25000 起 / 30000 返")
check(not R4.kita_dora and R4.kita_dora if False else not R4.kita_dora, "四麻没有拔北")
check(R4.uma == (15, 5, -5, -15), f"四麻顺位点应为 +15/+5/-5/-15，得到 {R4.uma}")
check(S.aka_types(4) == (4, 13, 22), f"四麻赤 5 应是 5m/5p/5s，得到 {S.aka_types(4)}")
check(S.aka_types(3) == (13, 22), "三麻赤 5 只有 5p/5s")
check(S.dora_from_index(0, R3) == 8, "三麻 1m 指示牌 -> 宝牌 9m")
check(S.dora_from_index(0, R4) == 1, "四麻 1m 指示牌 -> 宝牌 2m（正常循环）")
print("  ✅ 136 张 / 王牌 14 / 罰符 3000 / 赤 5 三张 / 宝牌循环")

print("\n== 四麻才有的役 ==")
a = ev("123m123p123s456p88s", "3s")
check(a is not None and "sanshoku" in [k for k, _ in a.yaku], "四麻应能三色同顺")
check(a is not None and a.han == 2, f"门清三色同顺应 2 翻，得到 {a.han if a else None}")
# 副露降番
a = ev("123p123s456p88s", "3s", melds=(S.Meld("chi", S.parse_hand("1m").index(1), 3),))
check(a is not None and a.han == 1, f"副露三色同顺应降为 1 翻，得到 {a.han if a else None}")
# 三麻凑不出三色同顺
a3 = ev("123p123s456p88s", "3s", rule=R3, melds=(S.Meld("chi", 9, 3),))
check(a3 is None or "sanshoku" not in [k for k, _ in a3.yaku], "三麻不可能有三色同顺")
# 四麻的一气通贯可以在万子
a = ev("123m456m789m123p11s", "3p")
check(a is not None and "ittsuu" in [k for k, _ in a.yaku], "四麻万子应能一气通贯")
print("  ✅ 三色同顺（含副露降番）/ 万子一气通贯")

print("\n== 自风与北家 ==")
a = ev("444z234m567m567p88s", "4z", sw=S.NORTH)
check(a is not None and ("yakuhai_seat" in [k for k, _ in a.yaku]), "四麻北家的北刻子应是自风")
a = ev("444z234m567m567p88s", "4z", sw=S.EAST)
check(a is None, "四麻东家拿北刻子应当无役（客风）")
a = ev("444z234p567p567s88s", "4z", rule=R3, sw=S.SOUTH)
check(a is None, "三麻北刻子永远无役")
print("  ✅ 北家自风 / 东家客风 / 三麻北不算役")

print("\n== 四麻点数 ==")
ag = S.Agari(3, 30, [("tanyao", 3)], 0, 0, {})
p = S.settle(ag, is_dealer=False, is_tsumo=True, winner=0, loser=None, rule=R4, dealer_seat=1)
check(p.deltas[0] == 4000, f"四麻子家 3翻30符自摸应得 4000，得到 {p.deltas[0]}")
check(sorted(p.deltas[1:]) == [-2000, -1000, -1000], f"四麻自摸应 亲2000/子1000×2，得到 {p.deltas}")
p = S.settle(ag, is_dealer=True, is_tsumo=True, winner=0, loser=None, rule=R4, dealer_seat=0)
check(p.deltas[0] == 6000, f"四麻亲家 3翻30符自摸应得 6000，得到 {p.deltas[0]}")
p = S.settle(ag, is_dealer=False, is_tsumo=False, winner=0, loser=3, rule=R4)
check(p.deltas[0] == 3900 and p.deltas[3] == -3900, "四麻子家荣和 3翻30符 = 3900")
p = S.settle(ag, is_dealer=True, is_tsumo=False, winner=0, loser=3, rule=R4)
check(p.deltas[0] == 5800, f"四麻亲家荣和 3翻30符 = 5800，得到 {p.deltas[0]}")
# 三麻还是老样子：少一家分摊
p3 = S.settle(ag, is_dealer=False, is_tsumo=True, winner=0, loser=None, rule=R3, dealer_seat=1)
check(p3.deltas[0] == 3000, f"三麻子家 3翻30符自摸应得 3000（ツモ損），得到 {p3.deltas[0]}")
print("  ✅ 自摸分摊 / 荣和 / 三麻四麻差额")

print("\n== 引擎：四麻能跑、有吃、张数守恒 ==")
KINDS: collections.Counter = collections.Counter()


class Checking(E.Game):
    def play_hand(self, dealer: int):
        res = super().play_hand(dealer)
        cnt = [0] * 34
        for pl in self.players:
            for i, c in enumerate(pl.hand):
                cnt[i] += c
            for m in pl.melds:
                for t in m.tiles:   # 吃是顺子，三张不同的牌
                    cnt[t] += 1
            for t in pl.pond:
                cnt[t] += 1
        for t, _ in self.live[self._pos : self._tail]:
            cnt[t] += 1
        for t, _ in self.dead:
            cnt[t] += 1
        check(sum(cnt) == 136, f"四麻牌数守恒：账上 {sum(cnt)} 张（应 136）")
        for i, c in enumerate(cnt):
            check(c <= 4, f"{S.tile_cn(i)} 出现 {c} 张")
        return res


chi_count = 0
for seed in range(20):
    g = Checking([HeuristicAgent(seed * 4 + i, style="flash") for i in range(4)], seed=seed, rule=R4)
    r = g.play()
    for h in r.hands:
        KINDS[h.kind] += 1
    for h in r.hands:
        pass
    for st in r.stats:
        chi_count += st.calls
    check(sum(r.scores) + 1000 * g.sticks == 100000, f"四麻点数守恒：{sum(r.scores)}")
print(f"  ✅ 四麻 20 场跑通，结果分布 {dict(KINDS)}")

# 单独确认「吃」真的会发生，而且只有四麻有
found_chi = False
for seed in range(40):
    log: list[str] = []
    g = E.Game([HeuristicAgent(seed * 4 + i, style="flash") for i in range(4)],
               seed=seed, rule=R4, on_event=log.append)
    g.play()
    if any("吃" in x for x in log):
        found_chi = True
        break
check(found_chi, "四麻 40 场里一次吃都没出现（不太对劲）")
log3: list[str] = []
E.Game([HeuristicAgent(i, style="flash") for i in range(3)], seed=3, on_event=log3.append).play()
check(not any("吃" in x for x in log3), "三麻不该出现吃")
print("  ✅ 四麻会出现吃，三麻不会")

print("\n== 吃的手牌判定 ==")
melds = [S.Meld("chi", S.parse_hand("2m").index(1), 3)]
a = ev("567m567p555z88s", "8s", melds=melds)
check(a is not None and "yakuhai_haku" in [k for k, _ in a.yaku], "吃 234m + 白白白 应是役牌·白")
# ⚠️ 这里原先是 **40** ✗ —— 那个值是「吃的顺子也被按刻子加符」算出来的 ✓
#    2026-10-21 修掉 `_compute_fu` 不判 kind 的 bug 之后，正确答案是 **30** ✓
#    手算：底 20 + 白(555z 暗刻,三元) 8 + 单骑(88s 雀头) 2 + 吃(顺子) 0 = **30**
#    -> 这个断言**曾经把 bug 锁死** ✓ 所以修 bug 必须先改测试 ✓
check(a is not None and a.fu == 30, f"符数应为 30，得到 {a.fu if a else None}")
print("  ✅ 吃的顺子能正确参与和牌判定与符计算")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED[:15]:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
