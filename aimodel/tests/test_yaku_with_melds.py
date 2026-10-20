"""回归：**有「吃」副露时，顺子不能被当成刻子**。

## 这个 bug 的来历（2026-10-21，用户截图报出来的）

平台和了面板显示：`maj-mountain4 荣和 4 番 40 符`，役 = **対対和(2)** · 役牌·白 · 宝牌×1，
而那手牌里明明有 **三万四万五万**（顺子）—— 对々和结构上不可能成立 ✗

根因（`sanma.py::_score_normal_one`）::

    seqs  = [t for k, t in shape_melds if k == "seq"]   # ← 只数「暗牌」，漏了副露 ✗
    n_seq = len(seqs)
    n_tri = len(shapes) - n_seq                          # ← shapes 含副露 ✗

`n_seq` 少算一个 -> `n_tri` 多算一个 -> **任何有「吃」的手都会被判成「四组全是刻子」**
-> 白送 `toitoi`(2 番) ✗

而同一个 `seqs` 还被**三色同顺 / 一气通贯 / 混老头**分支用 ✓
所以这一处错会让「有吃的手」同时出现：**多算対々和 + 少算三色/一气** ✗

修法：`seqs` 改从 `shapes`（暗牌 + 副露）里数 ✓

## 本文件盯的三件事

1. **有吃的手不许出现 toitoi** ✓（用户报的那个）
2. **有吃的手仍要认得三色同顺 / 一气通贯** ✓（同一个 bug 的另一面）
3. **真的全刻子手必须保留 toitoi** ✓（别修过头 ✗）
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402


def tile_of(t: str) -> int:
    """牌名 -> 牌型下标。

    ⚠️ 不要写 ``parse_hand(t)[0]`` ✗ —— 那拿到的是「牌型 0 的张数」✓ 不是牌型编号 ✗
    （我自己踩过一次：于是每手牌都被判成无役）
    """
    return S.parse_hand(t).index(1)


def mk(kind: str, t: str) -> S.Meld:
    return S.Meld(kind, tile_of(t), 0)


def score(concealed: str, win: str, melds=(), seat=S.SOUTH, rule=None):
    rule = rule or S.rule_for(4)
    ctx = S.WinContext(
        win_tile=tile_of(win), is_tsumo=False,
        round_wind=S.EAST, seat_wind=seat, rule=rule,
    )
    return S.evaluate_hand(S.parse_hand(concealed), list(melds), ctx)


def yaku_keys(a) -> set[str]:
    return {k for k, _ in a.yaku} if a else set()


def main() -> int:
    failed: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'✅' if cond else '🔴'} {label}")
        if not cond:
            failed.append(label)

    print("─" * 74)
    print("① 用户截图那手：碰白 + 碰發 + **吃** 345m + 111p + 55s，荣 1p")
    print("─" * 74)
    a = score("1p1p1p5s5s", "1p",
              [mk("pon", "5z"), mk("pon", "6z"), mk("chi", "3m")])
    keys = yaku_keys(a)
    print(f"    实际: {'None（无役）' if a is None else f'{a.han} 番 {a.fu} 符 {sorted(keys)}'}")
    check(a is not None, "应当和牌（役牌 白/發 有役）")
    check("toitoi" not in keys, "**不许**出现対々和（手上有顺子）")
    check("yakuhai_haku" in keys, "认得役牌·白")
    check("yakuhai_hatsu" in keys, "认得役牌·發")

    print()
    print("─" * 74)
    print("② 同一个 bug 的另一面：**有吃**时仍须认得三色同顺 / 一气通贯")
    print("─" * 74)
    # 三色同顺：吃 234m + 吃 234p + 暗 234s + 11z
    a = score("2s3s4s1p1p1p1z1z", "4s",          # 暗牌 8 张（2*3+2）
              [mk("chi", "2m"), mk("chi", "2p")])
    keys = yaku_keys(a)
    print(f"    三色 实际: {'None' if a is None else f'{a.han} 番 {sorted(keys)}'}")
    check("sanshoku" in keys, "有吃也要认得三色同顺")
    check("toitoi" not in keys, "三色手不许出现対々和")

    # 一气通贯：吃 123p + 吃 789p + 暗 456p + 11z
    a = score("4p5p6p1z1z1z2z2z", "6p",          # 暗牌 8 张
              [mk("chi", "1p"), mk("chi", "7p")])
    keys = yaku_keys(a)
    print(f"    一气 实际: {'None' if a is None else f'{a.han} 番 {sorted(keys)}'}")
    check("ittsuu" in keys, "有吃也要认得一气通贯")
    check("toitoi" not in keys, "一气手不许出现対々和")

    print()
    print("─" * 74)
    print("③ 别修过头：真正的全刻子手必须保留対々和")
    print("─" * 74)
    a = score("1z1z1z1p1p1p5s5s", "1p",
              [mk("pon", "5z"), mk("pon", "6z")])
    keys = yaku_keys(a)
    print(f"    实际: {'None' if a is None else f'{a.han} 番 {a.fu} 符 {sorted(keys)}'}")
    check("toitoi" in keys, "全刻子手必须给対々和")
    check(a is not None and a.han >= 5, f"番数应当 >= 5（实际 {None if a is None else a.han}）")

    print()
    print("─" * 74)
    print("④ 门清全顺子仍要认得平和；有副露则不该给平和")
    print("─" * 74)
    a = score("123m456m789m23p55s4p", "4p")   # 14 张含和了牌 4p，门清两面 -> 平和
    check("pinfu" in yaku_keys(a), "门清两面顺子应当是平和")
    a = score("2p3p4p6s7s8s", "4p", [mk("pon", "1z")])  # 有碰 -> 无平和
    check("pinfu" not in yaku_keys(a), "有副露时不该给平和")

    print()
    print("─" * 74)
    print("⑤ 符：**吃（顺子）不加符** —— 和「没吃」的对照组符数必须相同")
    print("─" * 74)
    # 底：20 + 幺九暗刻(111s) 8 + 自風雀头(南) 2 + 两面 0 = 30 符
    def fu_of(melds, wait="ryanmen"):
        ctx = S.WinContext(win_tile=tile_of("2z"), is_tsumo=False,
                           round_wind=S.EAST, seat_wind=S.SOUTH, rule=S.rule_for(4))
        return S._compute_fu(menzen=False, tsumo=False, pinfu=False, melds=list(melds),
                             shape_melds=[("tri", tile_of("1s"))], pair=tile_of("2z"),
                             wait_type=wait, ctx=ctx)

    f0 = fu_of([])
    f1 = fu_of([mk("chi", "2s")])
    f2 = fu_of([mk("chi", "1s")])       # 含幺九的顺子：**仍然是顺子**，也不加符
    print(f"    无吃 {f0} 符 / 吃234s {f1} 符 / 吃123s {f2} 符")
    check(f0 == 30, f"对照组应当是 30 符（实际 {f0}）")
    check(f1 == f0, f"吃中张顺子不该改变符数（{f1} vs {f0}）")
    check(f2 == f0, f"含幺九的顺子也不该加符（{f2} vs {f0}）")

    # 真刻子必须加符：碰一个三元牌 -> +4
    f3 = fu_of([mk("pon", "5z")])
    check(f3 > f0, f"碰三元牌应当加符（{f3} vs {f0}）")

    print()
    if failed:
        print(f"🔴 {len(failed)} 项未通过：")
        for f in failed:
            print("   -", f)
        return 1
    print("✅ 有副露时的顺子/刻子判定全部正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
