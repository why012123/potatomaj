"""快慢两条路的**边界**一致性 —— 专挑容易出问题、随机测试又碰不到的形状。

``test_fastshanten.py`` 已经用随机手牌比对了两条路，但它的牌池是 ``_BASE_IDS``，
里面是 ``5s`` **不是赤五 ``0s``** —— 也就是说赤五从来没被测过。
而真实对局里赤五很常见（用户那把真值手里就有一张 ``0s``）。

这里补的就是这类「随机测不到、但真会出现」的形状：

* 赤五（``0s`` / ``0p``）—— 它在 ``aimodel`` 那边是另一种编码，映射错了会静默偏
* 极端手牌：清一色、国士、已经和了、只剩几张（副露很多）
* 空手牌和一张牌（不该崩，也不该给出荒唐结果）
* 四张同样的牌全在手里
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from potatomaj import shanten as S  # noqa: E402


def with_slow(fn, *a, **kw):
    real = S._fast
    S._fast = lambda: None
    try:
        return fn(*a, **kw)
    finally:
        S._fast = real


CASES: list[tuple[str, list[str], bool, int]] = [
    # 名字, 手牌, 是否三麻, 副露数
    ("赤五在手（真实那把的形状）",
     ["1p", "2p", "2p", "3p", "6p", "7p", "2s", "0s", "6s", "5s"], True, 1),
    ("两张赤五（0s + 0p）",
     ["0s", "0p", "1m", "9m", "2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "4s", "5z"], True, 0),
    ("清一色（全是筒）",
     ["1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "9p", "5p", "5p", "5p"], True, 0),
    ("国士无双十三面",
     ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z", "1m"], True, 0),
    ("已经和了（四面子一雀头）",
     ["1m", "2m", "3m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "5z", "5z"], False, 0),
    ("四张一样的全在手里",
     ["5z", "5z", "5z", "5z", "1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "9s"], False, 0),
    ("副露很多，只剩 4 张",
     ["1m", "2m", "3m", "5z"], False, 3),
    ("只剩 1 张（三麻拔北三次）",
     ["5z"], True, 3),
]


def compare(name: str, hand: list[str], sanma: bool, melds: int) -> list[str]:
    """比对两条路的全部字段。返回不一致的描述（空表示一致）。"""
    problems: list[str] = []
    try:
        a = S.discard_options(hand, sanma=sanma, melds=melds)
        b = with_slow(S.discard_options, hand, sanma=sanma, melds=melds)
    except Exception as e:                                  # noqa: BLE001
        return [f"{name}: 抛异常 {type(e).__name__}: {e}"]
    if a != b:
        problems.append(f"{name}: 收益不一致")
        for x, y in zip(a, b):
            if x != y:
                problems.append(f"    快 {x}")
                problems.append(f"    慢 {y}")
    try:
        ua = S.useful_tiles(hand, sanma=sanma, melds=melds)
        ub = with_slow(S.useful_tiles, hand, sanma=sanma, melds=melds)
        if ua != ub:
            problems.append(f"{name}: 进张不一致 快 {ua} 慢 {ub}")
    except Exception as e:                                  # noqa: BLE001
        problems.append(f"{name}: useful_tiles 抛异常 {type(e).__name__}: {e}")
    # 结果本身也得说得通：向听不该是负数或大得离谱
    try:
        # 注意 shanten() 不吃 sanma 参数（三麻只是牌池更小，不影响向听算法）
        sh = S.shanten(hand, melds=melds)
        if sh < -1 or sh > 8:
            problems.append(f"{name}: 向听算出 {sh}，不合理")
    except Exception as e:                                  # noqa: BLE001
        problems.append(f"{name}: shanten 抛异常 {type(e).__name__}: {e}")
    return problems


def red_five_equivalence() -> list[str]:
    """赤五必须和普通五**完全等价** —— 它只是张五，颜色不影响牌型。

    为什么单独测：快慢一致只说明两条路「一样」，不说明「对」—— 两条路可能一起错。
    赤五在 aimodel 那边的编码是另一个槽位，映射错了会静默偏一点点，
    外面看不出来。而把 0s 换成 5s 之后一切应该逐字段相同，这是个硬判据。
    """
    problems: list[str] = []
    for red, plain in (("0s", "5s"), ("0p", "5p")):
        for hand in (
            ["1m", "9m", red, "1p", "2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "4s", "5z"],
            [red, red, "1m", "9m", "2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "5z", "5z"],
        ):
            mirrored = [plain if t == red else t for t in hand]
            sh_red = S.shanten(hand)
            sh_plain = S.shanten(mirrored)
            if sh_red != sh_plain:
                problems.append(f"{red} 与 {plain} 向听不同：{sh_red} vs {sh_plain}（{' '.join(hand)}）")
            u_red = S.useful_tiles(hand, sanma=True)
            u_plain = [plain if t == red else t for t in S.useful_tiles(mirrored, sanma=True)]
            if u_red != u_plain:
                problems.append(f"{red} 与 {plain} 进张不同：{u_red} vs {u_plain}")
            d_red = S.discard_options(hand, sanma=True)
            d_plain = S.discard_options(mirrored, sanma=True)
            if len(d_red) != len(d_plain):
                problems.append(f"{red} 与 {plain} 候选数不同：{len(d_red)} vs {len(d_plain)}")
            else:
                for x, y in zip(d_red, d_plain):
                    if x["ukeire"] != y["ukeire"] or x["shanten"] != y["shanten"]:
                        problems.append(
                            f"{red}/{plain} 收益不同：{x['tile']} {x['ukeire']}/{x['shanten']}"
                            f" vs {y['tile']} {y['ukeire']}/{y['shanten']}")
                        break
    return problems


def main() -> int:
    if S._fast() is None:
        print("⚠️ 没找到 aimodel.fastshanten —— 快路径没启用，这个测试没有意义")
        return 1

    bad: list[str] = []
    for name, hand, sanma, melds in CASES:
        problems = compare(name, hand, sanma, melds)
        if problems:
            bad += problems
            print(f"❌ {name}")
            for p in problems:
                print("   " + p)
        else:
            sh = S.shanten(hand, melds=melds)
            print(f"✓ {name:32} {len(hand):>2} 张　{sh} 向听")

    # 赤五的实质检查（不只是「两条路一致」，而是「赤五就是五」）
    red_bad = red_five_equivalence()
    if red_bad:
        bad += red_bad
        print("❌ 赤五等价性")
        for p in red_bad:
            print("   " + p)
    else:
        print("✓ 赤五与普通五完全等价（0s=5s、0p=5p，逐字段相同）")

    print()
    if bad:
        print(f"❌ {len(bad)} 处问题")
        return 1
    print(f"✅ 边界形状 {len(CASES)} 组快慢一致，且赤五等价性成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
