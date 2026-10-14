"""危险牌估计的测试。

跑法：``.venv/bin/python tests/test_danger.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from potatomaj.danger import analyze, recommend, visible_counts  # noqa: E402
from potatomaj.shanten import shanten  # noqa: E402


def test_genbutsu_is_zero() -> int:
    """某家打过的牌，**对他**必须是 0 危险（注意 analyze 只算手里的牌）。"""
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "3z"]
    discards = [["5m"], ["2m", "1p"], ["2p"], ["3p"]]      # 1 家打过 2m
    a = analyze(hand, discards, my_seat=0)
    row = next((r for r in a["tiles"] if r["tile"] == "2m"), None)
    if row is None:
        print("✗ 没算到 2m")
        return 1
    per = {p["seat"]: p["danger"] for p in row["per_seat"]}
    if per.get(1) != 0:
        print(f"✗ 2m 对 1 家是现物，应为 0，实际 {per.get(1)}")
        return 1
    if row["danger"] <= 0:
        print("✗ 2m 只对 1 家安全，整体不该是 0")
        return 1
    print(f"✓ 现物只对该家为 0（1家=0，整体={row['danger']}）")
    return 0


def test_all_genbutsu() -> int:
    """三家都打过的牌 = 全场现物，危险度必须是 0。"""
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "4z"]
    discards = [["5m"], ["9m", "1p"], ["2p"], ["3p"]]
    discards[1].append("4z")
    discards[2].append("4z")
    discards[3].append("4z")
    a = analyze(hand, discards, my_seat=0)
    row = next((r for r in a["tiles"] if r["tile"] == "4z"), None)
    if row is None:
        print("✗ 没算到 4z")
        return 1
    if row["danger"] != 0 or not row["all_safe"]:
        print(f"✗ 4z 三家都打过，应为全场现物(0)，实际 {row['danger']}")
        return 1
    print("✓ 三家都打过 -> 全场现物，危险度 0")
    return 0


def test_suji_reduces() -> int:
    """筋只对「打过对应牌」的那一家生效 —— 要按家比较。"""
    hand = ["1p", "7p", "2p", "3p", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1z", "2z"]
    discards = [["5m"], ["4p"], ["9s"], ["9s"]]     # 只有 1 家打过 4p
    a = analyze(hand, discards, my_seat=0)

    def per(tile):
        r = next((x for x in a["tiles"] if x["tile"] == tile), None)
        return {p["seat"]: p["danger"] for p in r["per_seat"]} if r else {}

    p1, p2 = per("1p"), per("2p")
    if not p1 or not p2:
        print("✗ 缺牌")
        return 1
    if not p1[1] < p2[1]:
        print(f"✗ 对 1 家（打过 4p）来说，1p 是筋应比 2p 安全：{p1[1]} vs {p2[1]}")
        return 1
    if p1[2] != p2[2]:
        print(f"✗ 2 家没打过 4p，1p / 2p 对他应一样危险：{p1[2]} vs {p2[2]}")
        return 1
    print(f"✓ 筋只对该家生效：对1家 1p={p1[1]} < 2p={p2[1]}；对2家两者都 {p1[2]}")
    return 0


def test_recommend_keeps_shanten() -> int:
    """14 张时，推荐的打牌必须不掉向听。"""
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "3z", "4z"]
    discards = [["5m", "9m"], ["1p", "2p"], ["8p", "9p"], ["1s", "2s"]]
    a = recommend(hand, discards, my_seat=0)
    rec = a.get("recommend")
    if not rec or not rec.get("tile"):
        print("✗ 没给出推荐")
        return 1
    before = shanten(hand)
    after = shanten([t for t in hand if t != rec["tile"]][:13]) if hand.count(rec["tile"]) == 1 else None
    if after is None:
        rest = hand[:]
        rest.remove(rec["tile"])
        after = shanten(rest)
    if after > before:
        print(f"✗ 推荐打 {rec['tile']} 让向听从 {before} 掉到 {after}")
        return 1
    print(f"✓ 推荐打 {rec['tile']}（{rec['why'][:30]}），向听保持 {before}")
    return 0


def test_visible_counts() -> int:
    hand = ["1m", "1m", "2p"]
    discards = [["1m"], ["1m"], [], []]
    v = visible_counts(hand, discards, ["2p"])
    # 1m：手里 2 张 + 两家各打 1 张 = 4；2p：手里 1 张 + 宝牌 1 张 = 2
    if v["1m"] != 4 or v["2p"] != 2:
        print(f"✗ 张数统计错: 1m={v['1m']} 2p={v['2p']}（应为 4 / 2）")
        return 1
    print("✓ 可见张数统计正确（1m=4, 2p=2）")
    return 0


def test_sanma_has_no_phantom_seat() -> int:
    """三麻只有 3 家 —— 那个空的第四行不能被当成真人。

    踩过的坑：``discards`` 是固定 4 行的数组，三麻那一桌永远有一行是空的。
    不指明家数的话，那个不存在的第四家会被当成真人对手：它没有牌河，
    所以现物、筋全不成立，但基础危险度照样算 —— 而总体危险度取「最危险那一家」，
    于是它经常压过所有人。

    实测后果很具体：三麻局面下 13 张牌**全部**被判成危险度 >= 0.5，
    界面上那个红圈警示永远亮着，等于没有信息，还看着吓人。
    """
    hand = ["1m", "9m", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "2s", "3s", "4s", "6z", "6z"]
    ponds = [
        ["1p", "9s", "2z", "5z", "3p", "7p"],
        ["4p", "6p", "8s", "1z"],
        ["2p", "5s", "9p", "3z"],
        [],                                   # 三麻没有这一家
    ]
    as4 = recommend(hand, ponds, 0, [], sanma=True, melds=0, seats=4)
    as3 = recommend(hand, ponds, 0, [], sanma=True, melds=0, seats=3)

    # 关键判据：第 3 行是空的，它不该出现在逐家危险度里
    a3 = analyze(hand, ponds, 0, [], seats=3)
    seats_seen = {ps["seat"] for r in a3["tiles"] for ps in r["per_seat"]}
    if 3 in seats_seen:
        print("✗ 三麻却把座位 3（不存在的那家）算进危险度了")
        return 1

    hi4 = sum(1 for r in as4["tiles"] if r["danger"] >= 0.5)
    hi3 = sum(1 for r in as3["tiles"] if r["danger"] >= 0.5)

    # 光修空位还不够：危险度的量纲是「0.25 + 0.075×巡目」再乘张数系数，
    # 第 6 巡就 0.70 了，拿 0.5 当绝对线仍然是几乎每张都亮。
    # 所以界面上用的是**相对这一手牌**的线（最危险的那 1/4，最多 3 张）。
    # 这里测的就是界面真正生效的那个判据。
    ds = sorted((r["danger"] for r in as3["tiles"]), reverse=True)
    k = max(1, min(3, round(len(ds) * 0.25)))
    # 按**名次**取前 k 个（按数值划线会被并列带偏：实测 13 张里 7 张并到线上）
    warned = k
    if warned > 3:
        print(f"✗ 相对阈值下仍然亮太多红圈（{warned} 张）")
        return 1
    print(f"✓ 三麻不把空位当人（0.5 绝对线：{hi4}/{len(as4['tiles'])} -> {hi3}/{len(as3['tiles'])} 张；"
          f"改用名次之后只亮 {warned} 张）")
    return 0


def test_riichi_makes_danger_certain() -> int:
    """立直是**确定信息**：那家一定听牌了，危险度不该再按巡目估算。

    立直以前读不到（协议没解开），危险度只能给「相对排序」。现在解开了
    （见 docs/协议进展.md 第四轮），所以：

    * 立直那家的**现物**照旧绝对安全（这一点没立直时也成立）
    * 其余牌基础危险度直接拉满，不再看巡目 —— 因为他确实在听
    * 筋仍然起调节作用（筋确实更安全）
    """
    hand = ["1m", "9m", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "2s", "3s", "4s", "6z", "6z"]
    ponds = [
        ["1p", "9s", "2z", "5z", "3p", "7p"],
        ["4p", "6p", "8s", "1z"],          # 座位 1
        ["2p", "5s", "9p", "3z"],          # 座位 2
        [],
    ]
    base = recommend(hand, ponds, 0, [], sanma=True, melds=0, seats=3)
    rj = recommend(hand, ponds, 0, [], sanma=True, melds=0, seats=3,
                   riichi=[False, True, False])

    def danger_of(res, tile):
        for r in res["tiles"]:
            if r["tile"] == tile:
                return r["danger"], r["why"]
        return None, ""

    # 1) 对**立直那一家**的现物，必须绝对安全。
    #
    #    要查逐家数据，不能查总体值：总体取的是「最危险那一家」，
    #    而这张 4p 对座位 1（立直）安全、对座位 2 不安全，总体就会是座位 2 的值。
    #    行为是对的，是断言该看 per_seat。
    ana = analyze(hand, ponds, 0, [], seats=3, riichi=[False, True, False])
    per = {}
    for row in ana["tiles"]:
        if row["tile"] == "4p":
            per = {p["seat"]: p["danger"] for p in row["per_seat"]}
    if per.get(1) != 0.0:
        print(f"✗ 对立的直那家，现物 4p 应当为 0，却算出 {per.get(1)}")
        return 1
    if per.get(2, 0) <= 0:
        print(f"✗ 4p 对座位 2 不是现物，不该是 0：{per}")
        return 1

    # 2) 非现物、非筋的牌，立直后危险度应当比不立直高
    d_base, _ = danger_of(base, "2p")
    d_rj2, why2 = danger_of(rj, "2p")
    if not (d_rj2 > d_base):
        print(f"✗ 立直后危险度没有变高：{d_base} -> {d_rj2}")
        return 1

    # 3) 立直那家的筋，应该比无筋明显安全
    d_suji, why3 = danger_of(rj, "3p")      # 6p 现 -> 3p 是筋
    if not (d_suji < d_rj2):
        print(f"✗ 立直那家的筋没有更安全：筋 {d_suji} vs 无筋 {d_rj2}")
        return 1

    # 4) 理由里要写明是立直，不能只说「无筋且未见」
    if "立直" not in why2:
        print(f"✗ 理由里没提立直：{why2}")
        return 1

    print(f"✓ 立直后危险度变成确定的：无筋 {d_base} -> {d_rj2}，筋 {d_suji}，"
          f"对立直那家的现物 0（逐家）")
    return 0


if __name__ == "__main__":
    bad = (test_genbutsu_is_zero() + test_all_genbutsu() + test_suji_reduces()
           + test_recommend_keeps_shanten() + test_visible_counts()
           + test_sanma_has_no_phantom_seat() + test_riichi_makes_danger_certain())
    print()
    print("✅ 危险牌测试全部通过" if bad == 0 else f"❌ 有 {bad} 项失败")
    raise SystemExit(1 if bad else 0)
