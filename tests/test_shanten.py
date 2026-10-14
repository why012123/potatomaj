"""向听数 / 有效牌的测试。

跑法：``.venv/bin/python tests/test_shanten.py``
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from potatomaj.shanten import (  # noqa: E402
    analyze, describe, melds_from_hand_size, shanten, useful_tiles,
)
from potatomaj.tiles import mpsz_to_tiles, sort_tiles  # noqa: E402

# (牌型, 期望向听, 说明) —— 期望值都手工推过
CASES: list[tuple[str, int, str]] = [
    ("123m456m789m123p55s", -1, "4面子+雀头，已和"),
    ("111m999m111p999p11s", -1, "对对和，已和"),
    ("19m19p19s1234567z7z", -1, "国士无双十三面，已和"),
    ("123m123p123s789s99p", -1, "和牌"),
    ("123m456m789m123p5s", 0, "听 5s"),
    ("123m123p123s789s9p", 0, "听 9p"),
    ("123m456m789m123p45s", 0, "14 张：打 4s 即听牌"),
    ("1m1m2p2p3p3p4p4p5p5p6p6p7p", 0, "七对子听 7p"),
    ("1m9m1p9p1s9s1z2z3z4z5z6z7z", 0, "国士听牌"),
    ("123m456m789m124p5s", 1, "差一张成面子"),
    ("1m4m7m1p4p7p1s4s7s1z4z6z7z", 6, "全是孤张"),
]


def test_cases() -> int:
    failed = 0
    for hand, want, why in CASES:
        tiles = mpsz_to_tiles(hand)
        got = shanten(tiles)
        if got != want:
            failed += 1
            print(f"✗ {hand:32s} 向听={got} 期望={want}  ({why})")
        else:
            print(f"✓ {hand:32s} 向听={got:2d}  ({why})")
    return failed


def test_properties() -> int:
    """随机手牌的物理约束：摸一张牌不会让向听变差，且最多只降 1。"""
    rng = random.Random(20261013)
    pool = [t for t in mpsz_to_tiles("19m123456789p123456789s1234567z")]
    failed = 0
    for _ in range(60):
        hand: list[str] = []
        while len(hand) < 13:
            t = rng.choice(pool)
            if hand.count(t) < 4:
                hand.append(t)
        base = shanten(hand)
        drawn = rng.choice([t for t in set(pool) if hand.count(t) < 4])
        after = shanten(hand + [drawn])
        if after > base:
            failed += 1
            print(f"✗ 摸牌后向听变差: {sort_tiles(hand)} 摸 {drawn}: {base} -> {after}")
        if base - after > 1:
            failed += 1
            print(f"✗ 一张牌降了不止一向听: {sort_tiles(hand)} 摸 {drawn}: {base} -> {after}")
    if not failed:
        print("✓ 随机一致性（60 组：摸牌不劣化、且最多降一向听）")
    return failed


def test_useful() -> int:
    failed = 0
    tiles = mpsz_to_tiles("123m123p123s789s9p")
    ups = set(useful_tiles(tiles, sanma=True))
    if ups != {"9p"}:
        failed += 1
        print(f"✗ 听牌有效牌应为 9p，实际 {sort_tiles(ups)}")
    else:
        print("✓ 听牌有效牌 = 9p")

    # 三麻不该给出 2m~8m
    tiles2 = mpsz_to_tiles("1m9m123p456p789p11s")
    ups2 = set(useful_tiles(tiles2, sanma=True))
    bad = [t for t in ups2 if t.endswith("m") and t[0] not in "019"]
    if bad:
        failed += 1
        print(f"✗ 三麻不应出现 2m~8m: {sort_tiles(bad)}")
    else:
        print(f"✓ 三麻有效牌不含 2m~8m：{sort_tiles(ups2)}")
    return failed


def test_melds() -> int:
    """副露（碰/吃/杠）：副露数从手牌张数反推，向听要正确抵扣。"""
    failed = 0

    # 张数 -> 副露数
    for slots, want in ((14, 0), (13, 0), (11, 1), (10, 1), (8, 2), (7, 2), (5, 3), (4, 3), (2, 4), (1, 4)):
        got = melds_from_hand_size(slots)
        if got != want:
            failed += 1
            print(f"✗ 手牌 {slots} 张应推出 {want} 副露，实际 {got}")
    if not failed:
        print("✓ 副露数反推正确（14/13->0, 11/10->1, 8/7->2, 5/4->3, 2/1->4）")

    # 向听：手牌张数 + 3*副露 必须等于 13 或 14
    cases = [
        (["4m","5m","6m","7m","8m","9m","1p","2p","3p","5s","5s"], 1, -1, "3面子+副露+雀头=和牌"),
        (["4m","5m","6m","7m","8m","9m","1p","2p","3p","5s"], 1, 0, "单骑听牌"),
        (["4m","5m","6m","7m","8m","9m","1p","2p","3p","5s","6s"], 1, 0, "两面听 4s/7s"),
        (["1m","2m","3m","5p","5p","7s","9s"], 2, 0, "3面子+雀头+嵌张搭子 -> 听牌"),
        (["1m","3m","5m","7m","9m","1p","3p"], 2, 2, "3组嵌张搭子 + 2副露"),
    ]
    for tiles, m, want, why in cases:
        got = shanten(tiles, melds=m)
        if got != want:
            failed += 1
            print(f"✗ {len(tiles)}张+{m}副露 应 {want} 向听，实际 {got}（{why}）")
        else:
            print(f"✓ {len(tiles)}张+{m}副露 -> {got} 向听（{why}）")

    # analyze 也要能接受副露
    r = analyze(["4m","5m","6m","7m","8m","9m","1p","2p","3p","5s","5s"], melds=1)
    if "和牌" not in r["text"]:
        failed += 1
        print(f"✗ 和牌形应报「已经和牌」，实际：{r['text']}")
    else:
        print(f"✓ analyze 识别和牌：{r['text']}")
    return failed


if __name__ == "__main__":
    bad = test_cases() + test_properties() + test_useful() + test_melds()
    print()
    print("✅ 向听测试全部通过" if bad == 0 else f"❌ 有 {bad} 项失败")
    raise SystemExit(1 if bad else 0)
