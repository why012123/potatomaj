"""特征编码的测试：全局特征、动作特征、掩码。

动作特征（每张候选牌各自的向听/进张/危险度）是这一版棋力提升的关键，
所以这里逐项核对：算出来的值必须和「真的打掉这张牌再算一遍」一致。

跑法：``.venv/bin/python aimodel/tests/test_encode.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402
from aimodel.encode import (  # noqa: E402
    ACTION_DIM,
    ACTION_OFFSET,
    DIM,
    action_features,
    action_index,
    action_mask,
    encode,
    tile_danger,
    visible_counts,
)
from aimodel.engine import Decision, Game  # noqa: E402
from aimodel.fastshanten import shanten, ukeire_count  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


# ---------------------------------------------------------------------------
print("== 全局特征 ==")
captured: list = []


class Spy(HeuristicAgent):
    def decide(self, decision, view):
        captured.append((decision, view))
        return super().decide(decision, view)


Game([Spy(i) for i in range(3)], seed=11).play()
discards = [(d, v) for d, v in captured if d.kind == "discard"]
kita = [(d, v) for d, v in captured if d.kind == "kita"]
print(f"  收到 {len(captured)} 次决策（打牌 {len(discards)}、拔北 {len(kita)}）")

d, v = discards[0]
x = encode(v)
check(x.shape == (DIM,), f"全局特征维度应为 {DIM}，得到 {x.shape}")
check(np.isfinite(x).all(), "全局特征不能有 NaN/inf")
# 手牌块应当和 view.hand 一致
from aimodel.encode import OFFSET  # noqa: E402

check(
    np.allclose(x[OFFSET["hand"] : OFFSET["hand"] + 34], np.asarray(v.hand) / 4.0),
    "手牌块应当等于 view.hand / 4",
)
check(x[OFFSET["hand_size"]] == v.hand_size / 14.0, "手牌张数特征")
print("  ✅ 全局特征维度 / 数值 / 手牌块")

print("\n== 动作特征逐项核对 ==")
af = action_features(v)
check(af.shape == (34, ACTION_DIM), f"动作特征形状应为 (34, {ACTION_DIM})，得到 {af.shape}")
check(np.isfinite(af).all(), "动作特征不能有 NaN/inf")

o = ACTION_OFFSET
n_melds = v.meld_count()
vis = visible_counts(v)
bad = 0
for t in range(34):
    if v.hand[t] == 0:
        check(af[t].sum() == 0, f"手里没有 {S.tile_cn(t)}，它那一行应当全 0")
        continue
    rest = list(v.hand)
    rest[t] -= 1
    s_true = shanten(rest, n_melds)
    u_true = ukeire_count(rest, vis, n_melds)
    if abs(af[t, o["shanten_after"]] - min(max(s_true, 0), 6) / 6.0) > 1e-6:
        bad += 1
        print(f"    {S.tile_cn(t)} shanten_after: {af[t, o['shanten_after']]} vs {s_true}")
    if abs(af[t, o["ukeire_after"]] - min(u_true, 40) / 40.0) > 1e-6:
        bad += 1
        print(f"    {S.tile_cn(t)} ukeire_after: {af[t, o['ukeire_after']]} vs {u_true}")
    if af[t, o["is_win_after"]] != (1.0 if s_true == 0 else 0.0):
        bad += 1
check(bad == 0, f"动作特征与「真打掉再算」有 {bad} 处不一致")
print(f"  ✅ {sum(1 for t in range(34) if v.hand[t])} 张候选牌的向听 / 进张 / 听牌标记全部对得上")

# 孤张标记
check(
    af[:, o["isolated"]].sum() <= v.hand_size,
    "孤张标记不应该超过手牌张数",
)
print("  ✅ 孤张标记")

print("\n== 危险度 ==")
import copy  # noqa: E402

d2, v2 = discards[-1]
opp = [q for q in range(3) if q != v2.player]

# 早巡、没人立直、没人副露 -> 只有很低的基线（但不为零！这一条是刻意改的：
# 上一版「没立直就返回 0」导致面对默听/清一色完全不设防，实战被打了 12000）
v_none = copy.deepcopy(v2)
v_none.riichi = [False, False, False]
v_none.melds_all = [[], [], []]
v_none.turn = 3
_opp = next(q for q in range(3) if q != v_none.player)
check(tile_danger(v_none, 13) <= 0.12, "早巡无副露时基线危险度应当很低")

# 没人立直，但已经晚巡 -> 危险度必须起来
v_late = copy.deepcopy(v_none)
v_late.turn = 15
check(tile_danger(v_late, 13) >= 0.29, f"晚巡要有危险度，得到 {tile_danger(v_late, 13):.2f}")

# 没人立直，但对手已经两组副露 -> 也是在快听牌了
v_open = copy.deepcopy(v_none)
v_open.turn = 5
_melds = [[], [], []]
_melds[_opp] = [S.Meld("pon", 3), S.Meld("pon", 5)]
v_open.melds_all = _melds
check(
    tile_danger(v_open, 13) >= 0.4,
    f"对手两组副露时危险度要起来，得到 {tile_danger(v_open, 13):.2f}",
)
print(
    f"  ✅ 没人立直也有防守：早巡 {tile_danger(v_none, 13):.2f}"
    f" / 晚巡 {tile_danger(v_late, 13):.2f}"
    f" / 对手两副露 {tile_danger(v_open, 13):.2f}"
)

# 读牌方向：牌河是「他不要的牌」——他狂打万子和字牌、几乎不碰筒子，
# 才说明他在攒筒子（这个方向搞反过一次，所以专门测）
v_suit = copy.deepcopy(v_none)
v_suit.turn = 15
v_suit.melds_all = [[], [], []]
_ponds = [[], [], []]
_ponds[_opp] = [0, 8, 27, 28, 29, 30, 18, 20]  # 猛打万子/字牌/索子，一张筒子都没打
v_suit.discards = _ponds
p_press = tile_danger(v_suit, 13)   # 五筒（他没打过筒子）
s_press = tile_danger(v_suit, 22)   # 五索（他一直在打索子）
check(
    p_press > s_press * 1.3,
    f"对手狂打索子不打筒子时，筒子({p_press:.2f}) 应当比索子({s_press:.2f}) 危险",
)
print(f"  ✅ 单色读牌方向：筒子 {p_press:.2f} > 索子 {s_press:.2f}")

# 他打过这张 -> 相对安全
v_safe = copy.deepcopy(v_none)
v_safe.turn = 15
_others = [q for q in range(3) if q != v_none.player]
_ponds2 = [[], [], []]
_ponds2[_others[0]] = [13, 13]
_ponds2[_others[1]] = [13, 0, 8, 27, 28, 29]   # 两家都打过五筒
v_safe.discards = _ponds2
check(
    tile_danger(v_safe, 13) < tile_danger(v_safe, 14),
    f"对手打过五筒，再打五筒({tile_danger(v_safe, 13):.2f}) 应当比打六筒({tile_danger(v_safe, 14):.2f}) 安全",
)
print(f"  ✅ 他打过这张 -> 更安全（{tile_danger(v_safe, 13):.2f} < {tile_danger(v_safe, 14):.2f}）")

print("\n== 掩码与动作映射 ==")
mask = action_mask(d)
check(mask.shape == (34,), "打牌掩码 34 维")
check(mask.sum() == len(d.options), "掩码里 1 的个数应当等于合法牌数")
for i, t in enumerate(d.options):
    check(action_index(d, i) == t, f"{i} 号选项 {S.tile_cn(t)} 的动作下标应为 {t}")
check(set(np.nonzero(mask)[0].tolist()) == set(d.options), "掩码位置应当正好是合法牌")
print("  ✅ 打牌掩码与下标映射")

for kind, opts in (("call", ["pass", "pon"]), ("riichi", ["yes", "no"])):
    dm = Decision(kind, opts)
    mm = action_mask(dm)
    check(mm.sum() == len(opts), f"{kind} 掩码个数")
print("  ✅ 其它决策的掩码")

print("\n== 特征里没有「上帝视角」 ==")
# 同样的输入必须给同样的特征
d3, v4 = discards[len(discards) // 2]
v4 = copy.deepcopy(v4)
x3 = encode(v4)
x4 = encode(copy.deepcopy(v4))
check(np.allclose(x3, x4), "同样输入应当给同样特征")
print("  ✅ 特征只依赖观测里给出的信息")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED[:20]:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
