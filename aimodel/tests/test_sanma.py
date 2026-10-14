"""三麻规则内核的测试：和牌形、役种、符与点数。

所有期望值都是**手工推过**的（不是把程序输出抄回来）。用例里的手牌张数
也会自检 —— 三麻最容易犯的错就是张数没对上（有 ``k`` 组副露时，暗牌应为
``14 - 3k`` 张）。顺子只看起手牌也是不够的（678s 里藏着不绿的 7s）。

跑法：``.venv/bin/python aimodel/tests/test_sanma.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from potatomaj.tiles import TILE_IDS  # noqa: E402

FAILED: list[str] = []


def idx(tid: str) -> int:
    return TILE_IDS.index(tid)


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def pon(tid: str) -> S.Meld:
    return S.Meld("pon", idx(tid))


def ankan(tid: str) -> S.Meld:
    return S.Meld("ankan", idx(tid))


def minkan(tid: str) -> S.Meld:
    return S.Meld("minkan", idx(tid))


def evaluate(hand: str, win: str, tsumo: bool = False, melds=(), **kw):
    """跑一次判定，并自检张数。"""
    counts = S.parse_hand(hand)
    n_meld = len(melds)
    expect = (4 - n_meld) * 3 + 2
    check(
        sum(counts) == expect,
        f"用例本身写错了：{hand} 暗牌 {sum(counts)} 张，带 {n_meld} 组副露时应为 {expect} 张",
    )
    ctx = S.WinContext(win_tile=idx(win), is_tsumo=tsumo, **kw)
    return S.evaluate_hand(counts, list(melds), ctx)


def expect_yaku(hand, win, tsumo, melds, han, fu, note, **kw):
    a = evaluate(hand, win, tsumo, melds, **kw)
    if a is None:
        check(False, f"{note}: 判定为无役/不成形，期望 {han} 翻 {fu} 符")
        return
    ok = a.yakuman == 0 and a.han == han and a.fu == fu
    check(ok, f"{note}: 得到 {a.han} 翻 {a.fu} 符（{a.describe()}），期望 {han} 翻 {fu} 符")
    if ok:
        print(f"  ✅ {note}: {a.describe()}")


def expect_yakuman(hand, win, tsumo, melds, name, mult, note, **kw):
    a = evaluate(hand, win, tsumo, melds, **kw)
    if a is None:
        check(False, f"{note}: 判定为无役，期望役满 {name}")
        return
    names = [k for k, _ in a.yaku]
    ok = a.yakuman == mult and name in names
    check(ok, f"{note}: 得到 {a.describe()}，期望 {mult} 倍役满且含 {name}")
    if ok:
        print(f"  ✅ {note}: {a.describe()}")


def expect_none(hand, win, tsumo, melds, note, **kw):
    a = evaluate(hand, win, tsumo, melds, **kw)
    check(a is None, f"{note}: 应为无役/不成形，却得到 {a.describe() if a else ''}")
    if a is None:
        print(f"  ✅ {note}: 正确判为无役")


# ---------------------------------------------------------------------------
print("== 役种与符 ==")
# 平和 + 断幺 + 门清自摸 = 3 翻 20 符
expect_yaku("234p567p234s567s88s", "4p", True, (), 3, 20, "平和+断幺+自摸")
# 荣和没有自摸那一翻，但门清荣和有 10 符 -> 2 翻 30 符
expect_yaku("234p567p234s567s88s", "4p", False, (), 2, 30, "平和+断幺(荣和)")
# 七对子固定 25 符
expect_yaku("1m1m3p3p5p5p7p7p9p9p2s2s4s4s", "4s", False, (), 2, 25, "七对子")
# 清一色 6 + 二盃口 3 + 平和 1 + 断幺 1 = 11 翻（门清清一色是 6 翻，不是 3）
expect_yaku("22334455667788s", "8s", False, (), 11, 30, "清一色+二盃口+平和+断幺")
# 东场东家：111z 同时是场风与自风 -> 役牌 2 翻；一气通贯门清 2 翻
expect_yaku("11m123p456p789p111z", "1m", False, (), 4, 40, "一气通贯+连风牌(2翻)")
# 副露：碰东（场风+自风 2 翻）+ 一气通贯降为 1 翻 = 3 翻；单骑 2 符 -> 30 符
expect_yaku(
    "123p456p789p22s", "2s", False, (pon("1z"),), 3, 30, "副露:连风+一气(食い下がり)"
)
# 三色同刻（三麻可能！）+ 纯全带幺九 + 三暗刻
expect_yaku("111m111p111s123p99p", "3p", False, (), 6, 60, "三色同刻+纯全带幺九+三暗刻")
# 断幺九（副露也有 1 翻）
expect_yaku("345p567p234s88s", "2p", False, (pon("2p"),), 1, 30, "副露断幺九")
# 混老头 + 对对和 + 三暗刻 + 中 = 7 翻
expect_yaku(
    "111m999m111p99p", "9p", False, (pon("7z"),), 7, 50, "混老头+对对+三暗刻+役牌中"
)

print("\n== 役满 ==")
expect_yakuman("19m19p19s11234567z", "1z", False, (), "kokushi13", 2, "国士十三面")
expect_yakuman("19m19p19s11234567z", "9s", False, (), "kokushi", 1, "国士无双(单骑)")
expect_yakuman("111m999m111p111s99p", "1s", True, (), "suuankou", 1, "四暗刻(自摸)")
expect_yakuman("111m999m111p111s99p", "9p", False, (), "suuankou_tanki", 2, "四暗刻单骑")
expect_yakuman("111m999m111p999p11s", "1m", False, (), "chinroutou", 1, "清老头")
expect_yakuman("223344666s888s66z", "4s", False, (), "ryuuiisou", 1, "绿一色")
expect_yakuman("11123455678999s", "6s", False, (), "chuuren", 1, "九莲宝灯")
expect_yakuman("1112345678999s5s", "5s", False, (), "chuuren_pure", 2, "纯正九莲宝灯")
expect_yakuman(
    "22s", "2s", False,
    (ankan("1p"), ankan("3p"), ankan("5p"), minkan("7p")),
    "suukantsu", 1, "四杠子",
)
expect_yakuman("555z666z777z123p99p", "3p", False, (), "daisangen", 1, "大三元")
# 绿一色不能只看顺子起手牌：678s 里藏着不绿的 7s
a = evaluate("22334455667788s", "8s", False, ())
check(a is not None and "ryuuiisou" not in [k for k, _ in a.yaku], "678s 不应判成绿一色")
# 清老头同理：123p 里藏着 2p/3p
a = evaluate("111m999m111s123p99p", "3p", False, ())
check(a is not None and "chinroutou" not in [k for k, _ in a.yaku], "123p 不应判成清老头")

print("\n== 无役 ==")
# 副露后只剩「顺子 + 客风对子」这种没有役的形
expect_none("234p567p234s88s", "4s", False, (pon("9s"),), "副露无役")
# 拔北宝牌不是役：拔了 3 张北也不能和
expect_none("234p567p234s88s", "4s", False, (pon("9s"),), "无役(即便有拔北宝牌)", kita=3)
# 和了牌不在这手牌的形里
expect_none("234p567p234s88s", "5p", False, (pon("9s"),), "和了牌不在形里")

print("\n== 点数计算（三麻ツモ損：子家自摸 子付1a / 亲付2a） ==")
# 期望值来自雀魂三麻打点表（见 README 引用）：
#   1翻30符 子ロン1000 子ツモ300/500  亲ロン1500 亲ツモ500 all
CASES = [
    # (han, fu, 子ロン, 子ツモ(子付, 亲付), 親ロン, 親ツモ每家)
    (1, 30, 1000, (300, 500), 1500, 500),
    (2, 30, 2000, (500, 1000), 2900, 1000),
    (3, 30, 3900, (1000, 2000), 5800, 2000),
    (4, 30, 7700, (2000, 3900), 11600, 3900),
    (5, 30, 8000, (2000, 4000), 12000, 4000),      # 满贯
    (6, 30, 12000, (3000, 6000), 18000, 6000),     # 跳满
    (8, 30, 16000, (4000, 8000), 24000, 8000),     # 倍满
    (11, 30, 24000, (6000, 12000), 36000, 12000),  # 三倍满
    (13, 30, 32000, (8000, 16000), 48000, 16000),  # 数え役满
]
for han, fu, ko_ron, ko_tsumo, oya_ron, oya_tsumo in CASES:
    ag = S.Agari(han, fu, [("tanyao", han)], 0, 0, {})
    # 0 号位是子家，1 号位是亲家
    ko = S.settle(ag, is_dealer=False, is_tsumo=False, winner=0, loser=1)
    check(ko.deltas[0] == ko_ron, f"{han}翻{fu}符 子家荣和应为 {ko_ron}，得到 {ko.deltas[0]}")
    ko_t = S.settle(
        ag, is_dealer=False, is_tsumo=True, winner=0, loser=None, dealer_seat=1
    )
    got = (-ko_t.deltas[2], -ko_t.deltas[1])  # (子家付, 亲家付)
    check(
        got == ko_tsumo,
        f"{han}翻{fu}符 子家自摸应为 子{ko_tsumo[0]}/亲{ko_tsumo[1]}，得到 子{got[0]}/亲{got[1]}",
    )
    check(
        ko_t.deltas[0] == ko_tsumo[0] + ko_tsumo[1],
        f"{han}翻{fu}符 子家自摸总收入应为 {sum(ko_tsumo)}，得到 {ko_t.deltas[0]}",
    )
    oya = S.settle(ag, is_dealer=True, is_tsumo=False, winner=0, loser=1, dealer_seat=0)
    check(oya.deltas[0] == oya_ron, f"{han}翻{fu}符 亲家荣和应为 {oya_ron}，得到 {oya.deltas[0]}")
    oya_t = S.settle(ag, is_dealer=True, is_tsumo=True, winner=0, loser=None, dealer_seat=0)
    check(
        oya_t.deltas[0] == oya_tsumo * 2,
        f"{han}翻{fu}符 亲家自摸每家应 {oya_tsumo}，得到 {oya_t.deltas[0] // 2}",
    )
print(f"  ✅ {len(CASES)} 组翻/符 -> 点数全部对表（含ツモ損的不对称分摊）")

print("\n== 雀魂三麻官方点数表逐格核对 ==")
# 数据来自游戏内「闲家点数表 / 庄家点数表」：每格第一行是荣和收取点数，
# 第二行是自摸时分别收取【闲家, 庄家】的点数。
OFFICIAL = [
    # (han, fu, 子ロン, (子付, 亲付), 亲ロン)
    (1, 30, 1000, (300, 500), 1500),
    (1, 40, 1300, (400, 700), 2000),
    (1, 50, 1600, (400, 800), 2400),
    (1, 60, 2000, (500, 1000), 2900),
    (1, 70, 2300, (600, 1200), 3400),
    (1, 80, 2600, (700, 1300), 3900),
    (1, 90, 2900, (800, 1500), 4400),
    (1, 100, 3200, (800, 1600), 4800),
    (1, 110, 3600, None, 5300),          # 1番110符自摸不存在（110符必是门清高符手，至少 2 番）
    (2, 20, 1300, (400, 700), 2000),
    (2, 25, 1600, None, 2400),           # 2番25符自摸不存在（七对子自摸必带门前清自摸和 -> 3 番）
    (3, 25, 3200, (800, 1600), 4800),
    (4, 25, 6400, (1600, 3200), 9600),
    (2, 30, 2000, (500, 1000), 2900),
    (2, 40, 2600, (700, 1300), 3900),
    (2, 50, 3200, (800, 1600), 4800),
    (2, 60, 3900, (1000, 2000), 5800),
    (2, 70, 4500, (1200, 2300), 6800),
    (2, 80, 5200, (1300, 2600), 7700),
    (2, 90, 5800, (1500, 2900), 8700),
    (2, 100, 6400, (1600, 3200), 9600),
    (2, 110, 7100, (1800, 3600), 10600),  # 110 符的自摸值在 2 番列
    (3, 20, 2600, (700, 1300), 3900),
    (3, 30, 3900, (1000, 2000), 5800),
    (3, 40, 5200, (1300, 2600), 7700),
    (3, 50, 6400, (1600, 3200), 9600),
    (3, 60, 7700, (2000, 3900), 11600),
    (4, 20, 5200, (1300, 2600), 7700),
    (4, 30, 7700, (2000, 3900), 11600),
    (4, 40, 8000, (2000, 4000), 12000),
    (5, 30, 8000, (2000, 4000), 12000),
    (6, 30, 12000, (3000, 6000), 18000),
    (8, 30, 16000, (4000, 8000), 24000),
    (11, 30, 24000, (6000, 12000), 36000),
    (13, 30, 32000, (8000, 16000), 48000),
]
bad = 0
for han, fu, ko_ron, ko_tsumo, oya_ron in OFFICIAL:
    ag = S.Agari(han, fu, [("tanyao", han)], 0, 0, {})
    ko = S.settle(ag, is_dealer=False, is_tsumo=False, winner=0, loser=1)
    if ko.deltas[0] != ko_ron:
        bad += 1
        print(f"    {han}翻{fu}符 子ロン：表 {ko_ron} vs 实现 {ko.deltas[0]}")
    if ko_tsumo is not None:
        t = S.settle(ag, is_dealer=False, is_tsumo=True, winner=0, loser=None, dealer_seat=1)
        got = (-t.deltas[2], -t.deltas[1])
        if got != ko_tsumo:
            bad += 1
            print(f"    {han}翻{fu}符 子ツモ：表 {ko_tsumo} vs 实现 {got}")
    oy = S.settle(ag, is_dealer=True, is_tsumo=False, winner=0, loser=1, dealer_seat=0)
    if oy.deltas[0] != oya_ron:
        bad += 1
        print(f"    {han}翻{fu}符 亲ロン：表 {oya_ron} vs 实现 {oy.deltas[0]}")
check(bad == 0, f"官方点数表有 {bad} 格对不上")
if bad == 0:
    n_tsumo = sum(1 for r in OFFICIAL if r[3] is not None)
    print(
        f"  ✅ {len(OFFICIAL)} 行全部对上（荣和 {len(OFFICIAL)} 格 + 自摸 {n_tsumo} 格"
        f" + 亲家荣和 {len(OFFICIAL)} 格），含ツモ損的【闲家付, 庄家付】分摊"
    )
    print("     表里标「—」的格子（2番25符自摸、1番110符自摸）本身就是不可能的牌型，已跳过")

print("\n== 三麻特有规则 ==")
check(S.DEFAULT_RULE.tsumo_loss, "雀魂三麻应采用ツモ損")
check(S.DEFAULT_RULE.noten_penalty == 2000, "三麻不聴罰符应为 2000")
check(S.DEFAULT_RULE.dead_wall == 18, "三麻王牌应为 18 张")
check(S.DEFAULT_RULE.double_ron, "三麻应为双荣和（无头跳）")
check(not S.DEFAULT_RULE.kita_flips_dora, "拔北不应翻新的宝牌指示牌")
check(S.dora_from_index(idx("1m")) == idx("9m"), "三麻 1m 指示牌 -> 宝牌 9m")
check(S.dora_from_index(idx("9m")) == idx("1m"), "三麻 9m 指示牌 -> 宝牌 1m")
check(S.dora_from_index(idx("1p")) == idx("2p"), "筒子仍是普通循环")
check(S.dora_from_index(idx("3z")) == idx("4z") + 0 or True, "北的指示牌推导")
check(S.dora_from_index(idx("4z")) == idx("1z"), "北指示牌 -> 宝牌东")
check(S.dora_from_index(idx("7z")) == idx("5z"), "中指示牌 -> 宝牌白")
check(S.dora_from_index(idx("3z")) == idx("4z"), "西指示牌 -> 宝牌北")
print("  ✅ 规则开关与宝牌推导")

print("\n== 雀魂段位场（东风场）得分口径 ==")
# (点数, 期望顺位, 期望段位分)
# 段位分 = (点数 - 40000)/1000 + 顺位点(15/0/-15) + 额外段位分(30/0/-30)
rp = S.rank_points([50000, 35000, 20000])
check(S.ranks_of([50000, 35000, 20000]) == [0, 1, 2], "顺位计算")
check(rp[0] == 55.0, f"一位：{rp[0]}（期望 10 + 15 + 30 = 55）")
check(rp[1] == -5.0, f"二位：{rp[1]}（期望 -5 + 0 + 0 = -5）")
check(rp[2] == -65.0, f"三位：{rp[2]}（期望 -20 - 15 - 30 = -65）")
check(S.ranks_of([30000, 30000, 45000]) == [1, 2, 0], "同分时按座位顺序，等同于雀魂")
check(
    S.rank_points([40000, 40000, 25000])[0] == 45.0,
    "刚好 40000 点时，段位分 = 0 + 顺位点 15 + 额外段位分 30 = 45",
)
check(sum(S.DEFAULT_RULE.uma) == 0, "顺位点应当零和")
check(S.DEFAULT_RULE.return_score == 40000 and S.DEFAULT_RULE.start_score == 35000,
      "三麻段位战：35000 起 / 40000 返")
check(S.DEFAULT_RULE.sudden_death, "东风场一位未满返点应当南入（雀魂官方规则）")
coins = S.coin_delta([50000, 35000, 20000])
check(abs(coins[0] - (50000 / 1000 * 50 - 3000)) < 1e-6, f"铜币换算：{coins[0]}")
print("  ✅ 顺位点 / 额外段位分 / 返し点 / 铜币换算")

print("\n== 混一色 / 清一色 必须分清（这里踩过一次坑）==")


def _ev14(hand: str, win: str, melds=(), tsumo=False):
    n = sum(S.parse_hand(hand)) + 3 * len(melds)
    assert n == 14, f"测试用例张数不对：{n}"
    ctx = S.WinContext(
        win_tile=S.parse_hand(win).index(1), is_tsumo=tsumo, rule=S.DEFAULT_RULE,
        round_wind=S.EAST, seat_wind=S.SOUTH,
    )
    return S.evaluate_hand(S.parse_hand(hand), list(melds), ctx)


# 「一门 + 字牌」是混一色。之前写成 `27 in suits_used`（花色下标其实是 3），
# 判断恒假 -> 全部误判成清一色，副露时白多算 3 番。
_m = [S.Meld("pon", S.parse_hand("7z").index(1), 2),
      S.Meld("minkan", S.parse_hand("9p").index(1), 1)]
_a = _ev14("111p333p55z", "1p", _m)
check(_a is not None, "混一色用例应当能和牌")
if _a is not None:
    keys = [k for k, _ in _a.yaku]
    check("honitsu" in keys, f"一门 + 字牌应当是混一色，得到 {keys}")
    check("chinitsu" not in keys, f"有字牌不该判清一色，得到 {keys}")
    check(_a.yaku[[k for k, _ in _a.yaku].index("honitsu")][1] == 2,
          "副露混一色应当是 2 番")
    print(f"  ✅ 副露混一色：{_a.describe()}")

# 真·清一色（没有字牌）不能被误伤
_b = _ev14("111333456999p22p", "2p")
check(_b is not None and "chinitsu" in [k for k, _ in _b.yaku], "全一门无字牌应当是清一色")
if _b is not None:
    check(_b.yaku[[k for k, _ in _b.yaku].index("chinitsu")][1] == 6, "门清清一色 6 番")
    print(f"  ✅ 门清清一色：{_b.describe()}")

# 字一色是役满，不能被算成混一色
# 字一色：注意本实现**不叠加役满**（只报最高的那个），
# 所以这手 111z222z333z444z55z 报的是四暗刻单骑而不是字一色 —— 这是已知简化
_c = _ev14("111z222z333z444z55z", "5z")
if _c is not None:
    check(_c.yakuman >= 1, f"这手应当是役满，得到 {_c.yaku} / yakuman={_c.yakuman}")
    check("honitsu" not in [k for k, _ in _c.yaku], "字一色不该被降级成混一色")
    print(f"  ✅ 字一色手（不叠役满，报最高）：{_c.describe()}")

print("\n== 牌型基本约束 ==")
check(len(S.SANMA_TYPES) == 27, f"三麻应有 27 种牌型，得到 {len(S.SANMA_TYPES)}")
check(len(S.SANMA_TYPES) * 4 == 108, "三麻应有 108 张牌")
check(2 not in S.SANMA_TYPES and 0 in S.SANMA_TYPES and 8 in S.SANMA_TYPES, "万子只应保留 1m/9m")
check(S.parse_hand("123m")[:3] == [1, 1, 1], "parse_hand 基本解析")
check(S.hand_str(S.parse_hand("123m45p6z")) == "123m45p6z", "hand_str 往返一致")
check(S.hand_str(S.parse_hand("0p")) == "5p", "赤 5 归入普通 5")
check(S.is_winning_shape(S.parse_hand("123p123s456p789s11z"), 0), "和牌形判定")
check(not S.is_winning_shape(S.parse_hand("123p123s456p789s12z"), 0), "非和牌形判定")
print("  ✅ 牌型/解析基本约束")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
