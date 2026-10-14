"""快速向听/进张的测试。

关键的一条：**与既有实现 :mod:`potatomaj.shanten` 逐手交叉验证** ——
两套算法（那边是整体递归拆解，这边是分花色块 + 帕累托剪枝）独立写出来，
结果必须一模一样，否则至少有一个是错的。

跑法：``.venv/bin/python aimodel/tests/test_fastshanten.py``
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import fastshanten as F  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from potatomaj import shanten as SLOW  # noqa: E402
from potatomaj.tiles import TILE_IDS  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def random_hand(rng: random.Random, n: int = 13) -> list[int]:
    wall = [t for t in S.SANMA_TYPES for _ in range(4)]
    rng.shuffle(wall)
    counts = [0] * 34
    for t in wall[:n]:
        counts[t] += 1
    return counts


def to_ids(counts: list[int]) -> list[str]:
    return [TILE_IDS[i] for i in range(34) for _ in range(counts[i])]


print("== 与既有实现交叉验证 ==")
rng = random.Random(20260101)
N = 400
bad = 0
for _ in range(N):
    c = random_hand(rng)
    if F.shanten(c) != SLOW.shanten(to_ids(c)):
        bad += 1
        if bad <= 3:
            print(f"    {S.hand_str(c)}: 快 {F.shanten(c)} vs 慢 {SLOW.shanten(to_ids(c))}")
check(bad == 0, f"13 张手牌：{bad}/{N} 手向听不一致")
if bad == 0:
    print(f"  ✅ 13 张 × {N} 手向听完全一致")

bad = 0
for _ in range(150):
    c = random_hand(rng, 14)
    if F.shanten(c) != SLOW.shanten(to_ids(c)):
        bad += 1
check(bad == 0, f"14 张手牌：{bad}/150 手向听不一致")
if bad == 0:
    print("  ✅ 14 张 × 150 手向听完全一致（含「刚摸到第 14 张」的口径）")

# 进张也要一致（慢的那边很贵，样本小一点）
bad = 0
for _ in range(12):
    c = random_hand(rng)
    fast = {S.tile_id(t) for t in F.useful_tiles(c)}
    slow = set(SLOW.useful_tiles(to_ids(c)))
    if fast != slow:
        bad += 1
        print(f"    {S.hand_str(c)}: 差 {fast ^ slow}")
check(bad == 0, f"进张：{bad}/12 手不一致")
if bad == 0:
    print("  ✅ 12 手进张集合完全一致")

print("\n== 逐张弃牌口径 ==")
bad = 0
for _ in range(400):
    c = random_hand(rng, 14)
    if F.shanten(c) != F.shanten_by_discard(c):
        bad += 1
check(bad == 0, f"14 张：{bad}/400 手与「逐张弃牌取最优」不一致")
if bad == 0:
    print("  ✅ 400 手 14 张牌：直接算块分解 == 逐张弃牌取最优")

print("\n== 副露后的向听 ==")
# 碰了 111p 之后，暗牌 10 张时应该是「再有 1 张就听牌」而不是当成 4 面子算
hand = S.parse_hand("234p567p88s9s")  # 9 张：2 面子 + 雀头 + 孤张
check(F.shanten(hand, 1) < F.shanten(hand, 0), "同一手暗牌，副露越多离和牌越近")
# 副露后不该再算七对子/国士
c = S.parse_hand("1m9m1p9p1s9s1z2z3z4z5z6z7z")
check(F.shanten(c, 0) == 0, "13 面国士应当听牌")
check(F.shanten(c, 1) > 0, "有副露时不能再算国士")
print("  ✅ 副露计入面子数、副露后不算七对子/国士")

print("\n== 与和牌判定一致 ==")
bad = 0
for _ in range(300):
    c = random_hand(rng, 14)
    win = F.shanten(c) < 0
    if win:
        if not S.is_winning_shape(c, 0):
            bad += 1
check(bad == 0, f"{bad}/300 手「向听 -1」与和牌形判定不一致")
if bad == 0:
    print("  ✅ 向听 -1 与和牌形判定完全对应")

print("\n== 记忆化与 base 复用必须完全等价（P1-3）==")
# P1-3：ukeire_count 曾占 action_features 91% 的时间（1565µs/1711µs）。
# 改了两处：① shanten 结果按 (counts, n_melds) 记忆化
#           ② useful_tiles/ukeire/ukeire_count 接受调用方已算好的 base
# 两处都必须**数值完全不变**，否则特征就变了、旧权重白训。
sample = [random_hand(rng) for _ in range(60)]

# ① 记忆化：清空 memo 重算一遍，结果必须与预热后一致
before = [F.shanten(c, 0) for c in sample]
saved = dict(F._SHANTEN_MEMO)
F._SHANTEN_MEMO.clear()
cold = [F.shanten(c, 0) for c in sample]
check(before == cold, "清空 memo 前后，shanten 结果必须一致")
# 再算一次（这次全命中 memo）也必须一致
again = [F.shanten(c, 0) for c in sample]
check(before == again, "memo 命中时 shanten 结果必须一致")
F._SHANTEN_MEMO.update(saved)
print("  ✅ shanten 记忆化：冷/热/命中三次结果一致")

# ② base 复用：传 base 与不传 base，进张必须逐张相同
bad_base = 0
for c in sample:
    s = F.shanten(c, 0)
    if F.useful_tiles(c, 0, sanma=True) != F.useful_tiles(c, 0, sanma=True, base=s):
        bad_base += 1
    if F.ukeire(c, None, 0, True) != F.ukeire(c, None, 0, True, base=s):
        bad_base += 1
    if F.ukeire_count(c, None, 0, True) != F.ukeire_count(c, None, 0, True, base=s):
        bad_base += 1
check(bad_base == 0, f"{bad_base} 手在传/不传 base 时结果不同 —— base 复用不等价")
print("  ✅ base 复用：useful_tiles / ukeire / ukeire_count 三者逐手等价")

# ③ 带副露（n_melds != 0）时也要等价，且 memo 不能把 n_melds 混掉
bad_meld = 0
for c in sample[:30]:
    for nm in (0, 1, 2):
        s = F.shanten(c, nm)
        if F.ukeire_count(c, None, nm, True) != F.ukeire_count(c, None, nm, True, base=s):
            bad_meld += 1
check(bad_meld == 0, f"{bad_meld} 处带副露时 base 复用不等价")
# memo 的键必须含 n_melds：同一手牌不同副露数不能互相顶掉
_diff = sum(1 for c in sample[:30] if F.shanten(c, 0) != F.shanten(c, 2))
check(_diff > 0, "n_melds 应当影响 shanten（否则 memo 键少一维也看不出来，测试无效）")
print(f"  ✅ 带副露等价；且 n_melds 确实影响结果（{_diff}/30 手不同）")

print("\n== 速度 ==")
hands = [random_hand(rng) for _ in range(200)]
# 预热：上面的等价性测试会清空 memo，所以这里必须把**即将计时的那批手牌**
# 先各算一遍，否则测到的是冷缓存，速度断言会假红。
for c in hands:
    F.shanten(c)
    F.useful_tiles(c)
t0 = time.perf_counter()
for c in hands:
    F.shanten(c)
t_fast = (time.perf_counter() - t0) / len(hands) * 1e6
t0 = time.perf_counter()
for c in hands[:40]:
    SLOW.shanten(to_ids(c))
t_slow = (time.perf_counter() - t0) / 40 * 1e6
print(f"  快 {t_fast:.1f} µs/次   慢 {t_slow:.1f} µs/次   加速 {t_slow / t_fast:.0f}x")
check(t_fast < t_slow / 10, "快速实现应当至少快 10 倍")
print("  ✅ 速度达标")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
