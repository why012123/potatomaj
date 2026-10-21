"""冻结老 encoder（486/560 维）的回归测试 + 维度分发。

2026-10-21 给特征末尾追加三块（match_state / called_tile / last_discard）后，
当前布局变成 3p 571 / 4p 645，而**所有存量权重**仍是 3p 486 / 4p 560。
为了让天梯 / 平台 / ELO 历史继续跑，旧实现被整份冻结在
:mod:`aimodel.encode_legacy`（**bug-for-bug，包括 visible_counts 的
double-count**），并由 :mod:`aimodel.feature_dispatch` 按 ``net.dim`` 分发。

本文件钉住四件事：

1. ``encode_legacy`` 的输出对固定局面**逐位等于旧金值**（sha256）；
2. 老布局里除 ``visible / ukeire / safe_exit`` 外的块与新 ``encode`` 逐位相等
   （新特征只追加在末尾）；
3. ``visible_counts`` 的旧 bug 仍在（否则 mac_rl 的输入分布会变）；
4. ``encoder_for`` 的三种分支：老维度 / 新维度 / 未知维度。

跑法：``.venv/bin/python aimodel/tests/test_legacy_encode.py``
      ``.venv/bin/python aimodel/tests/test_legacy_encode.py --record``（重录金值）
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import encode as new_encode  # noqa: E402
from aimodel import encode_legacy as legacy  # noqa: E402
from aimodel import feature_dispatch as FD  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from aimodel.engine import View  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def make_view(players: int, hand: str, **overrides) -> View:
    """只用老字段构造 View（老代码不认新字段，这正是要测的）。"""
    rule = S.rule_for(players)
    base = dict(
        player=0,
        hand=S.parse_hand(hand),
        drawn=None,
        melds=[],
        hand_red=[0] * 34,
        discards=[[] for _ in range(players)],
        melds_all=[[] for _ in range(players)],
        kita=[0] * players,
        scores=[rule.start_score] * players,
        dora_indicators=[0],
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=40,
        turn=5,
        riichi=[False] * players,
        ippatsu=[False] * players,
        is_dealer=True,
        last_discard=None,
        rule=rule,
    )
    base.update(overrides)
    return View(**base)


def fixed_views() -> list[tuple[str, View]]:
    """固定局面清单：覆盖 3p/4p、副露、拔北、立直、宝牌、分数。"""
    views: list[tuple[str, View]] = [
        ("3p_basic", make_view(3, "123456789m1123p")),
        ("3p_pon_kita", make_view(
            3, "234567p234s",
            melds=[S.Meld("pon", 22, 1)],
            melds_all=[[S.Meld("pon", 22, 1)], [], []],
            kita=[1, 0, 0],
            discards=[[9, 10], [], [11, 12]],
        )),
        ("3p_riichi", make_view(
            3, "234567p234s5s", drawn=22,
            discards=[[9, 9], [10, 10, 10], [11, 11]],
            riichi=[True, False, False],
            ippatsu=[True, False, False],
            scores=[25000, 45000, 30000],
            turn=9,
        )),
        ("3p_kan_dora", make_view(
            3, "234567p234s",
            melds=[S.Meld("minkan", 0, 1)],
            melds_all=[[S.Meld("minkan", 0, 1)], [], []],
            dora_indicators=[0, 5, 10],
            wall_left=18,
            turn=14,
            seat_wind=S.SOUTH,
        )),
        ("4p_basic", make_view(4, "123456789m1123p")),
        ("4p_chi", make_view(
            4, "456789m1122p",
            melds=[S.Meld("chi", 9, 1, called=10)],
            melds_all=[
                [S.Meld("chi", 9, 1, called=10)],
                [S.Meld("pon", 27, 0)],
                [], [S.Meld("pon", 28, 2)],
            ],
            discards=[[9, 10, 11], [12], [13, 14], [15]],
            kita=[0, 0, 0, 0],
        )),
        ("4p_pon_east", make_view(
            4, "123456789m11p",
            melds=[S.Meld("pon", 27, 2)],
            melds_all=[[S.Meld("pon", 27, 2)], [], [], []],
            kita=[0, 0, 0, 0],
            scores=[10000, 50000, 25000, 35000],
            turn=17,
        )),
        ("4p_winds", make_view(
            4, "123456789m1123p",
            round_wind=S.SOUTH,
            seat_wind=S.WEST,
            is_dealer=False,
            scores=[38000, 21000, 41000, 25000],
            dora_indicators=[33, 7],
            wall_left=8,
            turn=19,
        )),
    ]
    return views


def checksum_legacy() -> str:
    """``encode_legacy.encode`` 在固定局面上的逐位 sha256（float32 字节）。"""
    h = hashlib.sha256()
    for tag, v in fixed_views():
        x = legacy.encode(v)
        h.update(tag.encode())
        h.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    return h.hexdigest()


#: ⚠️ 2026-10-21 从 **HEAD（改动前）** 的 encode 录制。
#: 这个金值只用于老权重兼容；**不要**用新 encode 的输出去改它。
GOLDEN_LEGACY = "bf355c74688cb1ed59a6ebc31f90be1090709d3a6ee431c0c0fdc47e77c28789"


# ---------------------------------------------------------------------------
print("== 1) 冻结副本的身份 / 维度 / bug-for-bug ==")
check(legacy.DIM == 486, f"encode_legacy.DIM 应为 486，得到 {legacy.DIM}")
check(legacy.layout_for(3)[1] == 486, f"legacy 三麻维度应为 486，得到 {legacy.layout_for(3)[1]}")
check(legacy.layout_for(4)[1] == 560, f"legacy 四麻维度应为 560，得到 {legacy.layout_for(4)[1]}")
for p in (3, 4):
    blocks = legacy.feature_blocks(p)
    check("match_state" not in dict(blocks),
          f"legacy {p} 麻不该有 match_state（它是冻结的老布局）")
    check("called_tile" not in dict(blocks), f"legacy {p} 麻不该有 called_tile")
    check("last_discard" not in dict(blocks), f"legacy {p} 麻不该有 last_discard")
    check(blocks[-1] == ("tenpai", 1), f"legacy {p} 麻最后一块应仍是 tenpai：{blocks[-1]}")
    v = make_view(p, "123456789m1123p")
    x = legacy.encode(v)
    check(x.shape == (legacy.layout_for(p)[1],),
          f"legacy {p} 麻 encode 维度不对：{x.shape}")
    check(np.isfinite(x).all(), f"legacy {p} 麻 encode 有 NaN/inf")

# bug-for-bug：自己的副露/拔北仍算两遍（这是有意的，别“修”）
meld = S.Meld("pon", 22, 1)
v = make_view(3, "234567p234s",
              melds=[meld], melds_all=[[meld], [], []], kita=[1, 0, 0])
vis = legacy.visible_counts(v)
check(vis[22] == 6, f"legacy 的 visible_counts 必须保留 double-count：5s 应为 6，得到 {vis[22]}")
check(vis[S.NORTH] == 2, f"legacy 的拔北必须保留 double-count：NORTH 应为 2，得到 {vis[S.NORTH]}")
check(new_encode.visible_counts(v)[22] == 3,
      "新 encode 的 visible_counts 应已修复（5s=3）")
print("  ✅ 486/560、无新块、double-count 保留；新 encode 已修复")


print("\n== 2) 旧金值：encode_legacy 必须逐位不变 ==")
got = checksum_legacy()
if GOLDEN_LEGACY == "PENDING":
    check(False, "GOLDEN_LEGACY 未录制（用 --record 生成后写回常量）")
else:
    print(f"  当前 {got[:32]}…   金值 {GOLDEN_LEGACY[:32]}…")
    check(got == GOLDEN_LEGACY,
          "encode_legacy 的输出变了 ✗ 老权重的输入分布必须逐位不变")

print("\n== 3) 新布局 = 老布局 + 末尾三块（除三块外逐位相等）==")
CHANGED = {"visible", "ukeire", "safe_exit"}   # 依赖 visible_counts 修复
differ_visible = 0
for tag, v in fixed_views():
    players = v.rule.players
    xo = legacy.encode(v)
    xn = new_encode.encode(v)
    Oo, do = legacy.layout_for(players)
    On, dn = new_encode.layout_for(players)
    check(xn.shape == (dn,), f"{tag}: 新 encode 维度 {xn.shape} != {dn}")
    check(xo.shape == (do,), f"{tag}: 老 encode 维度 {xo.shape} != {do}")
    widths = dict(legacy.feature_blocks(players))
    bad = []
    for name, w in legacy.feature_blocks(players):
        if name in CHANGED:
            continue
        if not np.array_equal(xo[Oo[name]:Oo[name] + w], xn[On[name]:On[name] + w]):
            bad.append(name)
    check(not bad, f"{tag}: 老块被新 encode 改动了：{bad}")
    if not np.array_equal(xo[Oo["visible"]:Oo["visible"] + 34],
                          xn[On["visible"]:On["visible"] + 34]):
        differ_visible += 1
check(differ_visible >= 1,
      "至少有一个「带自己副露」的局面，新老 visible 应不同（修复生效）")
print(f"  ✅ 除 {sorted(CHANGED)} 外全部逐位相等；{differ_visible} 个局面 visible 因修复而变化")


print("\n== 4) feature_dispatch：按 net.dim 选 encoder ==")
check(FD.known_dims(3) == {486: "v1", 571: "v2"},
      f"3p known_dims 错：{FD.known_dims(3)}")
check(FD.known_dims(4) == {560: "v1", 645: "v2"},
      f"4p known_dims 错：{FD.known_dims(4)}")
check(FD.encoder_for(486, 3) is legacy, "encoder_for(486,3) 必须返回 encode_legacy")
check(FD.encoder_for(571, 3) is new_encode, "encoder_for(571,3) 必须返回 encode")
check(FD.encoder_for(560, 4) is legacy, "encoder_for(560,4) 必须返回 encode_legacy")
check(FD.encoder_for(645, 4) is new_encode, "encoder_for(645,4) 必须返回 encode")
for dim, players in ((999, 4), (560, 3), (486, 4), (0, 4)):
    try:
        FD.encoder_for(dim, players)
        check(False, f"encoder_for({dim},{players}) 应抛 ValueError，实际没抛")
    except ValueError as exc:
        check("未知特征维度" in str(exc) or "不是这两种布局" in str(exc),
              f"encoder_for({dim},{players}) 的报错不清楚：{exc}")

v4 = make_view(4, "123456789m1123p")
check(np.array_equal(FD.encode_for(560, 4, v4), legacy.encode(v4)),
      "encode_for(560,4) 必须等于 encode_legacy.encode")
check(np.array_equal(FD.encode_for(645, 4, v4), new_encode.encode(v4)),
      "encode_for(645,4) 必须等于 encode.encode")
check(np.array_equal(FD.action_features_for(560, 4, v4), legacy.action_features(v4)),
      "action_features_for(560,4) 必须等于 encode_legacy.action_features")
check(np.array_equal(FD.action_features_for(645, 4, v4), new_encode.action_features(v4)),
      "action_features_for(645,4) 必须等于 encode.action_features")
check(FD.encode_for(560, 4, v4).shape == (560,),
      "老 encoder 的 encode_for 形状应为 560")
check(FD.encode_for(645, 4, v4).shape == (645,),
      "新 encoder 的 encode_for 形状应为 645")
# 人数错也要炸
for bad_players in (2, 5):
    try:
        FD.known_dims(bad_players)
        check(False, f"known_dims({bad_players}) 应抛 ValueError")
    except ValueError:
        pass
print("  ✅ 老/新/未知 三分支正确；encode_for / action_features_for 走对模块")


# ---------------------------------------------------------------------------
if "--record" in sys.argv:
    print(f'\nGOLDEN_LEGACY = "{checksum_legacy()}"')
    raise SystemExit(0)

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
