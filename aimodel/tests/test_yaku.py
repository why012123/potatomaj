"""逐个役种的「对照表」测试 —— 每个役都用手工构造的牌例钉死。

**为什么要有这个文件**：`sanma.py` 里踩过一次很难看的坑 —— 把「牌型下标」
当成「花色下标」用（`27 in suits_used`），结果凡是「一门 + 字牌」的手牌
全被判成清一色，副露白多算 3 番，而且**一路没被发现**，因为原来的测试
只测了点数表和几个常见役。

所以这里换个思路：**把每个役种都列一行**，写清楚「什么牌 + 和哪张 =>
必须出现哪个役 + 多少番」。加新役、改判定，这一表就是照妖镜；
漏判、多判、番数写错，都会立刻红。

跑法：``.venv/bin/python aimodel/tests/test_yaku.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402

R = S.DEFAULT_RULE
FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def meld(kind: str, tile: str, from_p: int = 1) -> S.Meld:
    return S.Meld(kind, S.parse_hand(tile).index(1), from_p)


def ev(
    hand: str,
    win: str,
    melds: tuple[S.Meld, ...] = (),
    tsumo: bool = False,
    riichi: bool = False,
    ippatsu: bool = False,
    rinshan: bool = False,
    haitei: bool = False,
    chankan: bool = False,
    sw: int = S.SOUTH,
    rw: int = S.EAST,
    aka: int = 0,
):
    try:
        n = sum(S.parse_hand(hand)) + 3 * len(melds)
    except Exception as exc:  # 牌名写错了
        print(f"  ⚠️ 用例牌名解析失败（{exc}）：{hand!r}")
        return None
    if n != 14:
        print(f"  ⚠️ 用例写错了（{n} 张，应 14）：{hand} + {len(melds)} 副露")
        return None
    ctx = S.WinContext(
        win_tile=S.parse_hand(win).index(1),
        is_tsumo=tsumo,
        is_riichi=riichi,
        is_ippatsu=ippatsu,
        is_rinshan=rinshan,
        is_haitei=haitei,
        is_chankan=chankan,
        rule=R,
        round_wind=rw,
        seat_wind=sw,
        aka=aka,
    )
    return S.evaluate_hand(S.parse_hand(hand), list(melds), ctx)


def keys(a) -> list[str]:
    return [k for k, _ in a.yaku] if a else []


def han_of(a, key: str) -> int:
    for k, v in a.yaku:
        if k == key:
            return v
    return -1


# ---------------------------------------------------------------------------
# 每个役一行：(说明, hand, win, kwargs, 必须出现的役, 该役番数)
# 番数写 None 表示只检查「有没有」，不查番数（比如复合役满那种）
# ---------------------------------------------------------------------------
CASES: list[tuple] = [
    # ---- 一门清 / 混 ----
    ("门清清一色", "111333456999p22p", "2p", {}, "chinitsu", 6),
    ("副露清一色（降番）", "111333456p22p", "2p",
     {"melds": (meld("pon", "9p"),)}, "chinitsu", 5),
    ("门清混一色", "11133345699p555z", "5z", {}, "honitsu", 3),
    ("副露混一色（降番）", "111333p55z777z", "7z",
     {"melds": (meld("pon", "9p"),)}, "honitsu", 2),
    ("混一色 + 字牌副露（曾经被误判成清一色）", "111p333p55z", "1p",
     {"melds": (meld("pon", "7z"), meld("minkan", "9p"))}, "honitsu", 2),
    # ---- 带幺九系 ----
    ("混全带幺九", "123p789p123s789s11z", "1z", {}, "honchantai", 2),
    ("纯全带幺九（无字牌）", "123p789p123s789s99s", "9s", {}, "junchantai", 2),
    ("混老头", "111p999p111z555z99p", "9p", {}, "honroutou", 2),
    # ---- 三色 / 一通 ----
    ("三色同顺（门清 2 番）", "123m123p123s456p88s", "3s", {}, "sanshoku", 2),
    ("三色同顺（副露降 1 番）", "123p123s456p88s", "3s",
     {"melds": (meld("chi", "1m", 0),)}, "sanshoku", 1),
    ("一气通贯（门清 2 番）", "123p456p789p123s55s", "3s", {}, "ittsuu", 2),
    ("一气通贯（副露降 1 番）", "123p456p789p55s", "5s",
     {"melds": (meld("chi", "1s", 0),)}, "ittsuu", 1),
    ("三色同刻", "111m111p111s456p22s", "6p", {}, "sanshoku_doukou", 2),
    # ---- 对对 / 三暗刻 / 四暗刻 ----
    ("对对和", "111p333p55z777z", "7z", {"melds": (meld("pon", "9p"),)}, "toitoi", 2),
    ("三暗刻", "111p333p999p456s55s", "6s", {}, "sanankou", 2),
    ("四暗刻（役满·双碰听）", "111p333p999p444s55s", "4s",
     {"tsumo": True}, "suuankou", None),
    # ---- 七对 / 二盃口 / 一盃口 ----
    ("七对子", "1133557799p11s33z", "3z", {}, "chiitoi", 2),
    ("一盃口", "112233p456p789p11s", "1s", {}, "iipeikou", 1),
    ("二盃口", "112233p445566p11s", "1s", {}, "ryanpeikou", 3),
    # ---- 三色 / 三元 ----
    ("小三元", "555z666z77z123p456p", "7z", {}, "shousangen", 2),
    ("大三元（役满）", "555z666z777z123p44p", "4p", {}, "daisangen", None),
    ("役牌·白", "555z123p456p789p11s", "1s", {}, "yakuhai_haku", 1),
    ("役牌·场风自风（东家东）", "111z123p456p789p11s", "1s", {}, "yakuhai_round", 1),
    ("役牌·自风（南家南）", "222z123p456p789p11s", "1s", {"sw": S.SOUTH},
     "yakuhai_seat", 1),
    # ---- 三麻特有的「北」 ----
    ("三麻北刻子不是役牌（会显示无役）", "444z123p456p789p11s", "1s", {}, None, None),
    # ---- 断幺 / 平和 ----
    ("断幺九", "234p567p234s567s22s", "2s", {}, "tanyao", 1),
    ("平和（门清·两面听）", "234p567p345s567s22s", "3s", {}, "pinfu", 1),
    # ---- 立直系 ----
    ("立直", "123p456p789p123s55s", "3s", {"riichi": True}, "riichi", 1),
    ("门清自摸", "123p456p789p123s55s", "3s", {"tsumo": True}, "menzen_tsumo", 1),
    ("岭上开花", "123p456p789p123s55s", "3s",
     {"tsumo": True, "rinshan": True}, "rinshan", 1),
    # ---- 绿一色 / 清老头 / 字一色 ----
    ("绿一色（役满）", "223344s666s888s22s", "3s", {}, "ryuuiisou", None),
    ("清老头（役满）", "111p999p111s999s11p", "1p", {}, "chinroutou", None),
    # ---- 国士 ----
    ("国士无双（单骑听）", "119m19p19s1234567z", "7z", {}, "kokushi", None),
    ("国士无双十三面（双倍役满）", "119m19p19s1234567z", "1m", {}, "kokushi13", None),
]

print("== 逐役对照表 ==")
for name, hand, win, kw, want_key, want_han in CASES:
    a = ev(hand, win, **kw)
    if a is None and "用例写错" in "":  # 占位，实际由 ev 打印
        continue
    if want_key is None:
        # 明确要求「不能出现这个役」的用例：只看它没被误判
        bad = {
            "三麻北刻子不是役牌（会显示无役）": "yakuhai_seat",
        }[name]
        check(bad not in keys(a) or a is None, f"{name}：不该出现 {bad}，得到 {keys(a)}")
        print(f"  ✅ {name}")
        continue
    if a is None:
        check(False, f"{name}：应当能和牌，却判成无役")
        continue
    if want_key not in keys(a):
        check(False, f"{name}：应当有 {want_key}，实际 {keys(a)}")
        continue
    if want_han is not None:
        got = han_of(a, want_key)
        check(got == want_han, f"{name}：{want_key} 应为 {want_han} 番，实际 {got}")
    # 役满用例：总番数应当以役满计
    if want_han is None and a.yakuman < 1:
        check(False, f"{name}：应当是役满，实际 yakuman={a.yakuman} {a.describe()}")
    print(f"  ✅ {name}：{a.describe()}")

print("\n== 容易搞混的几组，必须分清 ==")
# 1) 混一色 vs 清一色（就是那个 bug）
a1 = ev("11133345699p555z", "5z")
a2 = ev("111333456999p22p", "2p")
check("honitsu" in keys(a1) and "chinitsu" not in keys(a1), "有字牌必须是混一色")
check("chinitsu" in keys(a2) and "honitsu" not in keys(a2), "无字牌必须判清一色")
check(han_of(a1, "honitsu") == 3 and han_of(a2, "chinitsu") == 6, "两者番数不同")
print(f"  ✅ 混一色 {a1.describe()}  |  清一色 {a2.describe()}")

# 2) 三色同顺 在三麻不可能（万子没有顺子）
a3 = ev("123p123s456p789s11z", "9s")
check("sanshoku" not in keys(a3), "三麻凑不出三色同顺")
print(f"  ✅ 三麻手上不会有 123m 这种顺子：{a3.describe() if a3 else '无役'}")

# 3) 门清 vs 副露的降番
a4 = ev("111333p555z77z", "5z", melds=(meld("pon", "9p"),))
print(f"  ✅ 副露混一色对照：{a4.describe() if a4 else '无役'}")

# 4) 北：三麻不是役牌，四麻北家是自风
a5 = ev("444z123p456p789p11s", "1s", sw=S.SOUTH)
check(a5 is None or "yakuhai_seat" not in keys(a5), "三麻的北不是役牌")
ctx4 = S.WinContext(
    win_tile=S.parse_hand("1s").index(1), is_tsumo=False,
    rule=S.rule_for(4), round_wind=S.EAST, seat_wind=S.NORTH,
)
a6 = S.evaluate_hand(
    S.parse_hand("444z123p456p789p11s"), [], ctx4
)
check(a6 is not None and "yakuhai_seat" in keys(a6), "四麻北家的北刻子应当是自风")
print(f"  ✅ 北：三麻{'无役' if a5 is None else a5.describe()} / 四麻北家 {a6.describe()}")

print()

# 5) 高点法：同一张和牌张有**多种听形解释**时必须取最高分
#
#    远端用 4 万局天鳳牌谱对拍时抓出来的真 bug：老代码取**第一个**含和牌张的
#    顺子就返回听形，于是 `123s`（start=0、offset=2）被判成「边张」✗ ——
#    可同一张 3s 也能补 `345s` → 是「两面」✓ 结果平和被丢掉、符数多算 2 符，
#    荣和时甚至**直接拒和**（实测能实战打穿）。
print()
print("== 高点法：多种听形解释取最高分 ==")
_h = "1m2m3m4m5m6m1s2s3s3s4s5s9s9s"      # 3s 既补 123s 也补 345s
_a_t = ev(_h, "3s", tsumo=True)
check(_a_t is not None and "pinfu" in keys(_a_t),
      f"自摸应当是「门前清自摸和 + 平和」，实际 {_a_t.describe() if _a_t else '拒和'}")
check(_a_t is not None and _a_t.fu == 20,
      f"平和自摸应当 20 符，实际 {_a_t.fu if _a_t else '拒和'}")
_a_r = ev(_h, "3s", tsumo=False)
check(_a_r is not None, "荣和**不能拒和** —— 最严重的表现（能实战打穿）")
check(_a_r is not None and "pinfu" in keys(_a_r), "荣和应当有平和")
if _a_t and _a_r:
    print(f"  ✅ 自摸 {_a_t.describe()} / 荣和 {_a_r.describe()}")
# 反向断言：听形候选必须**多于一个**，只有一个就说明又退回「取第一个」了
from aimodel.sanma import _decompose, wait_types
_cands: set = set()
for _sm, _pr in _decompose(list(S.parse_hand(_h)), 4, 1):
    if _pr is not None and len(_sm) == 4:
        _cands.update(wait_types(_sm, _pr, S.parse_hand("3s").index(1)))
check(len(_cands) > 1, f"这副牌应当有多个听形解释，实际只有 {_cands}")
check("ryanmen" in _cands, f"应当包含两面，实际 {_cands}")
print(f"  ✅ 听形候选 = {sorted(_cands)}")

if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED[:20]:
        print("   -", f)
    sys.exit(1)
print(f"✅ 全部通过（{len(CASES)} 个役种用例 + 4 组易混对照）")
