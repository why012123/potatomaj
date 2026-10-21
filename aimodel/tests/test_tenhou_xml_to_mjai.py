# -*- coding: utf-8 -*-
"""tools/tenhou_xml_to_mjai 的回归测试：天凤 mjlog **XML** -> **标准 MJAI**（P0: dan/rate）。

这个测试**不需要** 8,219 个真牌谱，也不需要 riichienv：
    · 主用例是**手工搭的 2 局 XML**（起手/摸打/吃/碰/暗槓/立直/和了/流局 全走一遍），
      断言输出是**标准 MJAI**（`kyoutaku` / `dora_marker` / 实际点数 / `end_kyoku` …）
      并且转换器自己的 6 项自检**一个都不报**；
    · 负向验证：在同一条 XML 上**故意改坏**（打出手上没有的牌 / 起手 12 张 /
      和了点数对不上 / 跨局分数接不上 / 立直宣言牌被荣），要求检查器**必须报出来**。
      检查器一旦失灵，这些行会变 ❌、进程返回 1 —— 这就是「测试会红」的证明。

跑了真牌谱的机器上（`_data/tenhou_scc` 存在）再加一段 20 局的抽样回归。
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tenhou_xml_to_mjai import (  # noqa: E402
    _dan_rate_stats, convert_game, decode_call, game_id_of, load_xml, tile_name,
)

FAILS = []


def check(name, got, want):
    if got == want:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}\n       得 {got!r}\n       期望 {want!r}")
        FAILS.append(name)


# --------------------------------------------------------------- 测试牌谱
# 两局：東1（吃 / 碰 / 暗槓 / 立直 / 荣和）+ 東2（流局）。
# 每一张牌都手工对过：四家起手各 13 张、摸的牌不与任何人的手牌重复、
# 每种牌最多 4 张、分数跨局接得上（下面 `good` 用例就靠这些数）。

HANDS1 = [
    "120,121,122,123,36,0,4,8,12,20,24,28,32",   # 0
    "40,44,72,76,80,84,92,93,96,100,104,108,112",  # 1
    "90,91,52,56,60,124,128,132,16,74,105,1,5",   # 2
    "68,69,37,41,45,49,53,57,61,65,113,117,89",   # 3
]
HANDS2 = [
    "0,1,2,3,4,5,6,7,8,9,10,11,12",
    "13,14,15,16,17,18,19,20,21,22,23,24,25",
    "26,27,28,29,30,31,32,33,34,35,36,37,38",
    "39,40,41,42,43,44,45,46,47,48,49,50,51",
]

UN = ('<UN n0="%E3%83%86%E3%82%B9%E3%83%88" n1="b" n2="c" n3="d" '
      'dan="16,17,18,20" rate="2100.50,2200.25,2300.75,2400.00" sx="M,M,F,C"/>')

#: 東1 的动作流（注释见文件头）
ACTIONS1 = (
    '<T30/><D36/>'
    '<N who="1" m="21511"/><E72/>'
    '<V62/><F62/>'
    '<W70/><G89/>'
    '<N who="2" m="33801"/><F5/>'
    '<W118/><G117/>'
    '<T66/><N who="0" m="30720"/><DORA hai="67"/><T71/><D71/>'
    '<U94/><E76/>'
    '<V63/><REACH who="2" step="1"/><F63/>'
    '<REACH who="2" ten="250,250,240,250" step="2"/>'
    '<W119/><G113/>'
    '<T3/><D8/>'
    '<U95/><E80/>'
    '<V55/><F55/>'
    '<AGARI ba="0,1" hai="0,4,8,12,16,20,24,28" machi="55" ten="30,1000,0" '
    'yaku="1,1" doraHai="129,67" doraHaiUra="53" who="0" fromWho="2" '
    'sc="250,20,250,0,240,-10,250,0"/>'
)
ACTIONS2 = (
    '<T52/><D0/><U53/><E13/><V54/><F26/><W55/><G39/>'
    '<RYUUKYOKU type="nm" sc="270,-15,250,15,230,-15,250,15"/>'
)

INIT1 = ('<INIT seed="0,0,0,1,4,129" ten="250,250,250,250" oya="0" '
         f'hai0="{HANDS1[0]}" hai1="{HANDS1[1]}" hai2="{HANDS1[2]}" hai3="{HANDS1[3]}"/>')
INIT2 = ('<INIT seed="1,0,0,1,4,99" ten="270,250,230,250" oya="1" '
         f'hai0="{HANDS2[0]}" hai1="{HANDS2[1]}" hai2="{HANDS2[2]}" hai3="{HANDS2[3]}"/>')


def build(init1=INIT1, actions1=ACTIONS1, init2=INIT2, actions2=ACTIONS2,
          un=UN) -> str:
    return (f'<mjloggm ver="2.3"><SHUFFLE seed="x"/>{un}<TAIKYOKU oya="0"/>'
            f'{init1}{actions1}{init2}{actions2}</mjloggm>')


def evs(events, typ):
    return [e for e in events if e["type"] == typ]


# ---------------------------------------------------------------- 主用例


def test_good():
    print("▶ 标准 MJAI 形状 / 事件语义（手工 2 局牌谱）")
    events, issues, info = convert_game(build(), game_id="2026-test-log")

    # ① 转换器自检必须全绿（座位 / 手牌 / 分数 / 宝牌 / 立直棒 全对得上）
    check("自检问题数（应为 0）", issues, [])

    # ② start_game：dan / rate / 名字 / id 是 P0
    sg = events[0]
    check("start_game 是第一个事件", sg["type"], "start_game")
    check("id 是字符串（riichienv 要求）", sg["id"], "2026-test-log")
    check("game_id 原样保留", sg["game_id"], "2026-test-log")
    check("名字 URL 解码", sg["names"], ["テスト", "b", "c", "d"])
    check("dan 四个都有", sg["dans"], [16, 17, 18, 20])
    check("rate 四个都有", sg["rates"], [2100.5, 2200.25, 2300.75, 2400.0])
    check("sx", sg["sx"], ["M", "M", "F", "C"])
    check("最后一个事件是 end_game", events[-1]["type"], "end_game")

    # ③ start_kyoku：**标准** MJAI 字段（不是我们旧的 riichi_sticks/dora_markers）
    sk = evs(events, "start_kyoku")
    check("start_kyoku 2 个", len(sk), 2)
    k = sk[0]
    check("bakaze", k["bakaze"], "E")
    check("kyoku", k["kyoku"], 1)
    check("honba", k["honba"], 0)
    check("kyoutaku（立直棒根数）", k["kyoutaku"], 0)
    check("没有 riichi_sticks（旧口径）", "riichi_sticks" in k, False)
    check("dora_marker（标量）", k["dora_marker"], "F")     # 129 -> 32 -> F
    check("没有 dora_markers（旧口径）", "dora_markers" in k, False)
    check("oya", k["oya"], 0)
    check("scores 用实际点数（250 -> 25000）", k["scores"], [25000] * 4)
    check("四家起手各 13 张", [len(h) for h in k["tehais"]], [13, 13, 13, 13])
    check("起手牌名（赤五 16 -> 5mr）", k["tehais"][2][8], "5mr")

    # ④ 摸打：座位是**绝对座位**（T/D=0, U/E=1, V/F=2, W/G=3）
    ts = evs(events, "tsumo")
    dh = evs(events, "dahai")
    check("第一摸是 seat0 摸 8m", (ts[0]["actor"], ts[0]["pai"]), (0, "8m"))
    check("第一打是 seat0 打 1p（手切）", (dh[0]["actor"], dh[0]["pai"], dh[0]["tsumogiri"]),
          (0, "1p", False))
    check("第二打是 seat1 吃完打的 1s（不是摸切）",
          (dh[1]["actor"], dh[1]["pai"], dh[1]["tsumogiri"]), (1, "1s", False))
    # seat1 这一巡是「吃」，没有摸牌 -> 下一摸必须是 seat2 的 V62，而不是被吃插进来的
    check("吃之后下一摸落在 seat2（绝对座位）", (ts[1]["actor"], ts[1]["pai"]), (2, "7p"))
    check("再下一摸落在 seat3", (ts[2]["actor"], ts[2]["pai"]), (3, "9p"))
    # seat1 后面才摸到牌：U94 = 94//4 = 23 = 6s
    s1 = [e for e in ts if e["actor"] == 1][0]
    check("seat1 的摸牌落在 seat1", (s1["actor"], s1["pai"]), (1, "6s"))
    # 摸切判定：seat2 摸 62（7p）打 62 -> tsumogiri
    tg = [e for e in dh if e["actor"] == 2 and e["pai"] == "7p"][0]
    check("摸切判定 tsumogiri=True", tg["tsumogiri"], True)
    check("手切判定 tsumogiri=False",
          [e for e in dh if e["actor"] == 0 and e["pai"] == "1p"][0]["tsumogiri"], False)

    # ⑤ 吃 / 碰 / 暗槓
    chi = evs(events, "chi")
    check("chi 1 个", len(chi), 1)
    check("chi: actor/target/pai", (chi[0]["actor"], chi[0]["target"], chi[0]["pai"]),
          (1, 0, "1p"))
    check("chi: consumed 是手里的两张", chi[0]["consumed"], ["2p", "3p"])
    pon = evs(events, "pon")
    check("pon 1 个", len(pon), 1)
    check("pon: actor/target/pai", (pon[0]["actor"], pon[0]["target"], pon[0]["pai"]),
          (2, 3, "5s"))
    check("pon: consumed 两张 5s", pon[0]["consumed"], ["5s", "5s"])
    ankan = evs(events, "ankan")
    check("ankan 1 个", len(ankan), 1)
    check("ankan: actor + 四张 N", (ankan[0]["actor"], ankan[0]["consumed"]),
          (0, ["N", "N", "N", "N"]))

    # ⑥ 宝牌：start_kyoku 里带初始指示牌，槓宝牌单独一条 dora
    dora = evs(events, "dora")
    check("槓宝牌事件 1 条", len(dora), 1)
    check("槓宝牌是 8p（67）", dora[0]["dora_marker"], "8p")
    hora = evs(events, "hora")[0]
    check("hora 记的指示牌 == 初始 + 槓", hora["dora_markers"], ["F", "8p"])

    # ⑦ 立直：reach 在打牌前、reach_accepted 在后
    idx_reach = events.index(evs(events, "reach")[0])
    check("reach 的下一条就是宣言牌（dahai）", events[idx_reach + 1]["type"], "dahai")
    check("宣言牌是 seat2 的 7p",
          (events[idx_reach + 1]["actor"], events[idx_reach + 1]["pai"]), (2, "7p"))
    check("宣言牌后紧跟 reach_accepted", events[idx_reach + 2]["type"], "reach_accepted")
    check("reach_accepted 只有成立的那次", len(evs(events, "reach_accepted")), 1)

    # ⑧ 和了：deltas 与 <AGARI sc>/ten 对得上
    check("hora: actor/target/tsumo", (hora["actor"], hora["target"], hora["tsumo"]),
          (0, 2, False))
    check("hora: deltas = sc 的 variation ×100", hora["deltas"], [2000, 0, -1000, 0])
    check("hora: points/fu/han", (hora["points"], hora["fu"], hora["han"]), (1000, 30, 1))
    check("hora: 赢家 deltas == 点数+棒", hora["deltas"][0], 1000 + 1000)
    check("hora: pai 是宣言牌", hora["pai"], "5p")

    # ⑨ 流局
    ryu = evs(events, "ryukyoku")[0]
    check("ryukyoku reason", ryu["reason"], "exhaustive_draw")
    check("ryukyoku deltas 合计 0", sum(ryu["deltas"]), 0)
    check("ryukyoku deltas", ryu["deltas"], [-1500, 1500, -1500, 1500])

    # ⑩ 结构：每局一条 end_kyoku，全局一条 end_game（ダブロン也不会重复）
    check("end_kyoku 数 == 局数", len(evs(events, "end_kyoku")), 2)
    check("end_game 只有一条", len(evs(events, "end_game")), 1)
    check("最后一局是流局", evs(events, "end_kyoku")[-1]["reason"], "exhaustive_draw")

    # ⑪ 自检计数（数字，不是问题）
    t = info["stats"]
    check("hora 对账：立直棒守恒 1/1", t["kyoku_hora_balance_ok"], 1)
    check("hora 对账：赢家点数 1/1", t["hora_winner_ok"], 1)
    check("流局 sum=0 1/1", t["ryukyoku_sum_zero"], 1)
    check("kyoku 数", info["kyoku"], 2)
    check("第一局起手分", info["start_scores"], [25000] * 4)


# ---------------------------------------------------------------- 负向验证


def _mutate(old, new, count=1):
    """把 build() 出来的 XML 里 old 换成 new（count 次），返回新 XML。"""
    xml = build()
    assert xml.count(old) >= count, f"fixture 里找不到 {old!r}"
    return xml.replace(old, new, count)


def test_negative():
    print("\n▶ 负向验证：故意改坏，检查器必须报出来（检查器失灵这里就变红）")

    def expect(name, needle, xml=None):
        events, issues, info = convert_game(xml or build(), game_id="mut")
        hit = any(i.startswith(needle) for i in issues)
        check(f"【负向】{name} -> {needle}", hit, True)
        return issues

    # ① 打出一张手里没有的牌（37 是 seat3 的 1p，seat0 手里只有 36）
    #    这条同时验证座位归属：座位错一格，整局的打牌都会「不在手里」。
    expect("打出不在手里的牌", "discard_not_in_hand", _mutate("<D36/>", "<D37/>"))

    # ② 起手不是 13 张
    bad_init = INIT1.replace(f'hai2="{HANDS1[2]}"',
                             f'hai2="{",".join(HANDS1[2].split(",")[:12])}"')
    expect("起手少一张", "init:hai2_not_13", build(init1=bad_init))

    # ③ 同一种牌出现 5 张（136 张牌堆被破坏）：把 seat3 的 5s(89) 换成
    #    与 seat0 暗槓重复的 N(120) -> N 出现 5 次
    bad_init2 = INIT1.replace(f'hai3="{HANDS1[3]}"',
                              f'hai3="{HANDS1[3].replace(",89", ",120")}"')
    expect("同种牌 >4 张", "tile_over_4", build(init1=bad_init2))

    # ④ 和了点数对不上：赢家只拿到 1000，但 ten 说 1000 + 1 根棒 = 2000
    expect("赢家收的点数不对", "agari:winner_points",
           _mutate('sc="250,20,250,0,240,-10,250,0"',
                   'sc="250,10,250,0,240,-0,250,0"'))

    # ⑤ 立直棒不守恒：几条 deltas 加起来不等于 1000×(立直+供托)
    expect("立直棒不守恒", "agari:balance",
           _mutate('sc="250,20,250,0,240,-10,250,0"',
                   'sc="250,30,250,0,240,-10,250,0"'))

    # ⑥ 跨局分数接不上：下一局起手分被改掉 1000
    expect("跨局分数接不上", "score_break",
           _mutate('ten="270,250,230,250"', 'ten="270,250,240,250"'))

    # ⑦ 立直宣言牌被荣：**不许**发 reach（天凤不成立、不收 1000 点）
    xml = build().replace('<E72/>', '<E72/>')  # no-op, 保持结构
    xml = xml.replace('<F63/><REACH who="2" ten="250,250,240,250" step="2"/>', '<F63/>')
    events, issues, info = convert_game(xml, game_id="mut")
    check("【负向】宣言牌被荣 -> 不发 reach", len(evs(events, "reach")), 0)
    check("【负向】宣言牌被荣 -> 不发 reach_accepted", len(evs(events, "reach_accepted")), 0)
    check("【负向】宣言牌被荣 -> 记成计数",
          info["stats"].get("reach_ronned_on_declaration"), 1)

    # ⑧ 坏 XML 不崩
    events, issues, info = convert_game("<mjloggm><INIT seed=", game_id="x")
    check("【负向】坏 XML 不抛异常且报问题", any(i.startswith("xml:") for i in issues), True)


def test_dan_rate_missing():
    print("\n▶ dan/rate 缺失（天凤不公开段位）不能崩")
    un = '<UN n0="a" n1="b" n2="c" n3="d" sx="M,M,M,M"/>'
    events, issues, info = convert_game(build(un=un), game_id="nodan")
    sg = events[0]
    check("start_game 仍在", sg["type"], "start_game")
    check("dans 缺失 -> 四个 None", sg["dans"], [None, None, None, None])
    check("rates 缺失 -> 四个 None", sg["rates"], [None, None, None, None])
    check("没有 UN 也不崩", convert_game(
        build(un=""), game_id="x")[0][0]["dans"], [None, None, None, None])

    # 只有 2 个值（截断）也要补齐成 4 个
    un2 = '<UN n0="a" n1="b" n2="c" n3="d" dan="10,11" rate="1500.0"/>'
    sg2 = convert_game(build(un=un2), game_id="x")[0][0]
    check("dan 只有 2 个 -> 补齐 4 个", sg2["dans"], [10, 11, None, None])
    check("rate 只有 1 个 -> 补齐 4 个", sg2["rates"], [1500.0, None, None, None])

    # 一个值都不合法也不崩
    un3 = '<UN n0="a" n1="b" n2="c" n3="d" dan="?,-,x,1" rate="nan,?,3,4.5"/>'
    sg3 = convert_game(build(un=un3), game_id="x")[0][0]
    check("非法 dan -> None 兜底", sg3["dans"], [None, None, None, 1])


def test_sanma():
    print("\n▶ 三麻（hai3 为空）跳过，不污染四麻数据集")
    init = INIT1.replace(f'hai3="{HANDS1[3]}"', 'hai3=""').replace(
        'ten="250,250,250,250"', 'ten="350,350,350,0"')
    events, issues, info = convert_game(build(init1=init), game_id="3p")
    check("三麻被识别", info.get("sanma"), True)
    check("三麻不产出事件", events, [])
    check("三麻 info 仍带 dan/rate（可用于统计）", info.get("dans"), [16, 17, 18, 20])
    # dan/rate 缺失的三麻也不能崩
    un = '<UN n0="a" n1="b" n2="c" sx="M,M,M"/>'
    events, issues, info = convert_game(build(init1=init, un=un), game_id="3p2")
    check("三麻 + 缺 dan/rate 不崩", info.get("sanma"), True)


def test_tile_and_call_decoding():
    print("\n▶ 牌名 / m= 位布局（十进制！）")
    check("1m", tile_name(0), "1m")
    check("9p", tile_name(68), "9p")
    check("E", tile_name(108), "E")
    check("P(白)", tile_name(124), "P")
    check("C(中)", tile_name(135), "C")
    check("赤5m 16 -> 5mr（标准 MJAI，不是 0m）", tile_name(16), "5mr")
    check("赤5p 52 -> 5pr", tile_name(52), "5pr")
    check("赤5s 88 -> 5sr", tile_name(88), "5sr")

    # 真牌谱里抓出来的常数：m 是**十进制**
    c = decode_call(21511, 1)
    check("chi: kind/target/tile", (c["kind"], c["target"], c["tile"]), ("chi", 0, 9))
    check("chi: consumed 原始 id", c["consumed_raw"], [40, 44])
    c = decode_call(33801, 2)
    check("pon: kind/target/tile", (c["kind"], c["target"], c["tile"]), ("pon", 3, 22))
    c = decode_call(30720, 0)
    check("ankan: kind/tile", (c["kind"], c["tile"]), ("ankan", 30))
    # 🔴 回归：`m="48170"` 是「2 号碰白(P)」；按十六进制会解成 kakan of 4s
    #    （tools/mjlog_to_mjai.py 骨架的 bug 就在这里）
    c = decode_call(48170, 2)
    check("48170 是碰 白，不是 kakan（十进制回归）",
          (c["kind"], c["tile"], c["target"]), ("pon", 31, 0))
    # 真牌谱：`<N who="1" m="60511"/>` = 吃 8s（被吃的是 100），手里出 95(6s)/98(7s)
    c = decode_call(60511, 1)
    check("chi 8s: called/consumed",
          (c["kind"], c["called_raw"], c["consumed_raw"], c["target"]),
          ("chi", 100, [95, 98], 0))


def test_dan_rate_stats():
    print("\n▶ dan/rate 分布统计（Mac 要的数字）")
    s = _dan_rate_stats(
        [[16, 17, None, None], [10, 11, 12, 18], [None, None, None, None],
         [20, 20, 20, 20]],
        [[2100.0, 2200.0, None, None], [1500.0, 1600.0, 1700.0, 1800.0],
         [None] * 4, [2200.0, 2300.0, 2400.0, 2500.0]],
    )
    check("局数", s["games"], 4)
    check("4 家都有 dan 的局数", s["games_all4_dan"], 2)
    check("一个 dan 都没有的局数", s["games_no_dan"], 1)
    check("dan 直方图", s["dan_hist"],
          {10: 1, 11: 1, 12: 1, 16: 1, 17: 1, 18: 1, 20: 4})
    check("4 家 dan>=10 的局数", s["games_all4_dan_ge10"], 2)
    check("rate 总数", s["rate_n"], 10)
    check("rate min/max", (s["rate_min"], s["rate_max"]), (1500.0, 2500.0))
    check("rate 中位数（偶数个取中间两个的均值）", s["rate_median"], 2150.0)


def test_real_data():
    """真牌谱抽样（没下载数据就跳过，不算失败）。"""
    print("\n▶ 真牌谱抽样回归（_data/tenhou_scc）")
    cands = [ROOT / "_data" / "tenhou_scc",
             ROOT.parent / "potatomaj" / "_data" / "tenhou_scc"]
    d = next((c for c in cands if c.is_dir()), None)
    if d is None:
        print(f"  ⏭ 跳过：没有 {cands[0]} 也没 {cands[1]}")
        return
    files = sorted(d.glob("*.xml.gz"))[:20]
    n4 = n3 = bad = 0
    for fp in files:
        events, issues, info = convert_game(load_xml(fp), game_id=game_id_of(fp))
        if info.get("sanma"):
            n3 += 1
            continue
        n4 += 1
        if issues:
            bad += 1
            print(f"     · {fp.name}: {issues[:4]}")
    check("抽 20 个文件，4 麻转换 0 问题", bad, 0)
    check("抽 20 个文件里至少有一个四麻", n4 > 0, True)
    print(f"     （4 麻 {n4} 局 / 三麻 {n3} 局）")


def main():
    test_good()
    test_negative()
    test_dan_rate_missing()
    test_sanma()
    test_tile_and_call_decoding()
    test_dan_rate_stats()
    test_real_data()

    if FAILS:
        print(f"\n❌ tenhou_xml_to_mjai: {len(FAILS)} 项失败 -> {FAILS}")
        return 1
    print("\n✅ tenhou_xml_to_mjai 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
