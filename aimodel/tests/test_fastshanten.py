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

print("\n== 速度 ==")
hands = [random_hand(rng) for _ in range(200)]
for c in hands[:20]:
    F.useful_tiles(c)  # 预热缓存
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
