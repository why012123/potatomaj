"""交叉验证：`aimodel.fastshanten` 的快路径必须和本文件的慢实现**完全一致**。

为什么值得单独写一个测试：每张牌打掉的收益（`discard_options`）原来要 1.3 秒，
换成快实现之后是 1.6 毫秒 —— **快了 800 倍**。提速本身没有意义，结果一样才有意义。
这个测试就是拿随机手牌把两条路逐字段比一遍，任何一处不一致都算失败。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from potatomaj import shanten as S  # noqa: E402

ALL34 = list(S._BASE_IDS)
#: 三麻：万子只有 1m/9m
SANMA_POOL = [t for t in ALL34 if not (t.endswith("m") and t not in ("1m", "9m"))]


def random_hand(rng: random.Random, n: int, pool: list[str]) -> list[str]:
    bag = [t for t in pool for _ in range(4)]
    rng.shuffle(bag)
    return bag[:n]


def with_slow(fn, *a, **kw):
    """把快路径关掉再跑一遍。"""
    real = S._fast
    S._fast = lambda: None
    try:
        return fn(*a, **kw)
    finally:
        S._fast = real


def main() -> int:
    if S._fast() is None:
        print("⚠️ 没找到 aimodel.fastshanten —— 快路径没启用，这个测试没有意义")
        return 1

    rng = random.Random(20261013)
    bad = 0

    # 1) 进张集合
    n = 0
    for _ in range(120):
        hand = random_hand(rng, 13, SANMA_POOL)
        for melds in (0, 1, 2):
            cut = hand[: len(hand) - 3 * melds]
            if len(cut) + 3 * melds != 13:
                continue
            a = S.useful_tiles(cut, sanma=True, melds=melds)
            b = with_slow(S.useful_tiles, cut, sanma=True, melds=melds)
            n += 1
            if a != b:
                bad += 1
                print(f"❌ 进张不一致 melds={melds}: {cut}\n   快 {a}\n   慢 {b}")
                if bad > 3:
                    return 1
    print(f"✓ 进张集合 {n} 组一致")

    # 2) 每张牌打掉的收益（这是这次提速的主角）
    n = 0
    for _ in range(60):
        hand = random_hand(rng, 14, SANMA_POOL)
        seen = {t: rng.randint(0, 3) for t in rng.sample(ALL34, 8)}
        a = S.discard_options(hand, sanma=True, seen=seen)
        b = with_slow(S.discard_options, hand, sanma=True, seen=seen)
        n += 1
        if a != b:
            bad += 1
            print(f"❌ 收益不一致: {' '.join(hand)}")
            for x, y in zip(a, b):
                if x != y:
                    print(f"   快 {x}\n   慢 {y}")
            if bad > 3:
                return 1
    print(f"✓ 每张牌的收益 {n} 手一致（含已见张数）")

    # 3) 四麻也要能用（网络用不上时就会走这条路）
    n = 0
    for _ in range(40):
        hand = random_hand(rng, 14, ALL34)
        a = S.discard_options(hand, sanma=False)
        b = with_slow(S.discard_options, hand, sanma=False)
        n += 1
        if a != b:
            bad += 1
            print(f"❌ 四麻收益不一致: {' '.join(hand)}")
            if bad > 3:
                return 1
    print(f"✓ 四麻（34 种牌）{n} 手一致")

    print()
    print("✅ 快慢两条路完全一致" if bad == 0 else f"❌ {bad} 处不一致")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
