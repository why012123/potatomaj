#!/usr/bin/env python
"""``encode_tile`` 的单元测试（dot V1 路线书第 ① 步）。

四条硬约束（Mac 要复核的就是这四条）：

1. **形状/类型**：``tiles == (34, F)``、``globals == (G,)``、都是 float32；
   F=20、G=21，通道名表与路线书逐项对齐。
2. **金值 sha256**：三个固定构造的 ``View``，把 tiles/globals 的字节哈希写死。
   以后谁悄悄改了语义，这里立刻红 —— 不是「输出看起来差不多」。
3. 🔴 **无信息泄漏**：从真实 ``Game`` 里取一个决策点的 View，**把别家手牌
   随机改掉**，该玩家视角的编码必须**逐位不变**（``np.array_equal``）。
   同时做**负向验证**：改自己手牌 / 自己的赤五，编码**必须变** ——
   证明第 3 条不是恒真。
4. **三麻 / 四麻**都能跑；三麻的第 3 个对手通道恒 0。

另外钉住：
* 危险度通道在「所有对手都立直」时与生产 ``tile_danger`` 逐位一致；
* 振听标记 ``furiten_wait`` 只在该牌确实是和牌张时亮；
* ``visible + remaining == 1``（同一个 4 张的分解）。

跑法::

    .venv/Scripts/python.exe aimodel/tests/test_encode_tile.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import RandomAgent  # noqa: E402
from aimodel.encode import tile_danger, visible_counts  # noqa: E402
from aimodel.encode_tile import (  # noqa: E402
    F,
    G,
    GLOBAL_FEATURES,
    TILE_CHANNELS,
    encode_tile,
)
from aimodel.engine import Game, View  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def counts(tiles: list[int]) -> list[int]:
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    return c


def digest(x: np.ndarray) -> str:
    """跨平台稳定的 float32 字节哈希（强制 little-endian）。"""
    a = np.ascontiguousarray(x).astype("<f4", copy=False)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 三个固定 View（金值 + 通道覆盖）
# ---------------------------------------------------------------------------


def _fixture_4p_mid() -> View:
    """四麻中盘：自己听 5p 且振听、自己牌河有 5p、两家立直、有宝牌/赤五。

    手牌 123m456m789m123p + 5p5p（刚摸到一张 5p）。
    去掉 drawn 的 5p 后是 13 张，听 5p；自己牌河里已经有 5p -> 振听。
    """
    rule = S.rule_for(4)
    hand = counts([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 13])
    v = View(
        player=1,
        hand=hand,
        drawn=13,
        melds=[],
        hand_red=[0] * 34,
        discards=[[31, 31, 4], [13, 33], [22, 22, 27], [30, 5, 6]],
        melds_all=[
            [S.Meld("pon", 31, from_player=1)],      # 下家碰白
            [],                                       # 自己
            [S.Meld("chi", 9, from_player=3, called=10)],  # 对家吃 1p2p3p
            [S.Meld("pon", 28, from_player=0)],      # 上家碰南
        ],
        kita=[0, 0, 0, 0],
        scores=[25000, 32000, 21000, 22000],
        dora_indicators=[8, 30],   # 9m -> 1m；北 -> 东（四麻）
        round_wind=S.EAST,
        seat_wind=S.SOUTH,
        wall_left=40,
        turn=7,
        riichi=[False, False, True, True],
        ippatsu=[False, False, False, False],
        is_dealer=False,
        last_discard=(2, 27),
        rule=rule,
        furiten=True,
        kyoku=2,
        honba=1,
        kyotaku=1,
        dealer_repeat=True,
        hands_left=3,
        ranks=(3, 1, 4, 2),
        tsumogiri=(
            (False, False, True),
            (False, True),
            (True, False, False),
            (False, False, False),
        ),
        called_tile=None,
    )
    v.hand_red[13] = 1  # 手里有一张赤 5p
    return v


def _fixture_4p_early() -> View:
    """四麻序盘：自己有副露（碰 9m）、亲家、无立直、手牌 11 张。"""
    rule = S.rule_for(4)
    hand = counts([0, 0, 2, 3, 4, 9, 9, 10, 18, 22, 33])
    v = View(
        player=0,
        hand=hand,
        drawn=None,
        melds=[S.Meld("pon", 8, from_player=3)],  # 碰 9m
        hand_red=[0] * 34,
        discards=[[33, 0], [5, 6], [18, 19, 20], [27, 28]],
        melds_all=[
            [S.Meld("pon", 8, from_player=3)],       # 自己
            [S.Meld("chi", 12, from_player=0, called=13)],  # 下家吃 4p5p6p
            [],
            [S.Meld("pon", 31, from_player=2)],      # 上家碰白
        ],
        kita=[0, 0, 0, 0],
        scores=[35000, 25000, 25000, 15000],
        dora_indicators=[3],   # 4m -> 5m（自己手里有 5m）
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=60,
        turn=4,
        riichi=[False, False, False, False],
        ippatsu=[False, False, False, False],
        is_dealer=True,
        last_discard=(3, 28),
        rule=rule,
        furiten=False,
        kyoku=1,
        honba=0,
        kyotaku=0,
        dealer_repeat=False,
        hands_left=4,
        ranks=(1, 2, 3, 4),
        tsumogiri=((False, True), (False, True), (True, False, False), (False, False)),
        called_tile=None,
    )
    v.hand_red[22] = 1  # 手里有一张赤 5s
    return v


def _fixture_3p() -> View:
    """三麻：自己拔过 1 张北、有一副露、下家立直、自己的 5p 振听。"""
    rule = S.rule_for(3)
    hand = counts([0, 8, 9, 10, 11, 18, 19, 20, 22, 22, 27])
    v = View(
        player=2,
        hand=hand,
        drawn=None,
        melds=[S.Meld("pon", 13, from_player=0)],  # 碰 5p
        hand_red=[0] * 34,
        discards=[[13, 14, 15], [22, 30], [31, 9]],
        melds_all=[
            [S.Meld("pon", 30, from_player=2)],      # 下家拔北？这里用副露占位
            [],
            [S.Meld("pon", 13, from_player=0)],      # 自己碰 5p
        ],
        kita=[0, 0, 1],                              # 自己拔过 1 张北
        scores=[40000, 38000, 32000],
        dora_indicators=[0],   # 三麻：1m -> 9m
        round_wind=S.EAST,
        seat_wind=S.WEST,
        wall_left=30,
        turn=9,
        riichi=[False, True, False],
        ippatsu=[False, False, False],
        is_dealer=False,
        last_discard=(0, 13),
        rule=rule,
        furiten=True,
        kyoku=3,
        honba=2,
        kyotaku=0,
        dealer_repeat=False,
        hands_left=1,
        ranks=(1, 3, 2),
        tsumogiri=((False, False, True), (False, True), (False, False)),
        called_tile=13,
    )
    v.hand_red[22] = 1  # 手里有一张赤 5s
    return v


def fixtures() -> list[tuple[str, View]]:
    return [
        ("4p_mid", _fixture_4p_mid()),
        ("4p_early", _fixture_4p_early()),
        ("3p", _fixture_3p()),
    ]


#: 金值：由当前实现算出来后写死；改任何通道/归一化都必须更新（并说明原因）。
GOLDEN: dict[str, tuple[str, str]] = {
    "4p_mid": (
        "3e8152ec35c365f7f0def1d744ce057b98405513c64d8a820b591a52ce2b60ca",
        "4d5ef840d23ff5ef9a34b0edd622bf651189f8daf92cbc8da8ff36c5c2bee4da",
    ),
    "4p_early": (
        "7856747773720da494d7aa323e3b5f6edbbe17137d6ab6c8b0cdb1d005aa49e0",
        "f4d76a4b94898503d53f78e5c80fe6a6d7018fae7fededca65736c4a37f53478",
    ),
    "3p": (
        "4728774a9cc3caa8634af248ab32ecd943965bfae19f3ba4f07405363c218d48",
        "a5528c2b9b5d90f90e811e8414f41a926fc71fc349d61b296f161fa431545c42",
    ),
}


# ---------------------------------------------------------------------------
# 1. 常量 / 形状
# ---------------------------------------------------------------------------


def test_shape_and_constants() -> None:
    print("\n== 1. 形状 / F=20 / G=21 / 通道表 ==")
    check(F == 20, f"路线书要求 F=20，实际 {F}")
    check(G == 21, f"全局标量应为 21（路线书「约 20」，seat/round 拆成两个），实际 {G}")
    check(len(TILE_CHANNELS) == F, "TILE_CHANNELS 长度必须等于 F")
    check(len(GLOBAL_FEATURES) == G, "GLOBAL_FEATURES 长度必须等于 G")
    check(len(set(TILE_CHANNELS)) == F, f"通道名不能重复：{TILE_CHANNELS}")

    expected = (
        "hand", "visible", "remaining", "dora", "aka", "yakuhai",
        "terminal_honor", "own_pond", "own_last_discard", "furiten_wait",
        "opp1_pond", "opp2_pond", "opp3_pond",
        "opp1_last_discard", "opp2_last_discard", "opp3_last_discard",
        "riichi_danger", "genbutsu", "opp_meld", "suji",
    )
    check(TILE_CHANNELS == expected,
          f"通道顺序与路线书不一致：\n  实际 {TILE_CHANNELS}\n  期望 {expected}")

    for name, v in fixtures():
        tiles, glob = encode_tile(v)
        print(f"  {name}: tiles{tiles.shape} globals{glob.shape}")
        check(tiles.shape == (34, F), f"{name}: tiles 形状 {tiles.shape} != (34,{F})")
        check(glob.shape == (G,), f"{name}: globals 形状 {glob.shape} != ({G},)")
        check(tiles.dtype == np.float32, f"{name}: tiles dtype {tiles.dtype} != float32")
        check(glob.dtype == np.float32, f"{name}: globals dtype {glob.dtype} != float32")
        check(np.isfinite(tiles).all(), f"{name}: tiles 有 NaN/Inf")
        check(np.isfinite(glob).all(), f"{name}: globals 有 NaN/Inf")
        check(float(tiles.min()) >= 0.0 and float(tiles.max()) <= 1.0,
              f"{name}: tiles 应全在 [0,1]，实际 [{tiles.min()}, {tiles.max()}]")
        check(float(glob.min()) >= -1.0 and float(glob.max()) <= 1.0,
              f"{name}: globals 应全在 [-1,1]，实际 [{glob.min()}, {glob.max()}]")

        # 同一个 View 调两次必须逐位相同（编码里不能有随机/集合顺序依赖）
        t2, g2 = encode_tile(v)
        check(np.array_equal(tiles, t2), f"{name}: tiles 不是确定性的")
        check(np.array_equal(glob, g2), f"{name}: globals 不是确定性的")

        # visible + remaining = 1（valid 牌位）；hand <= visible
        if v.rule.players == 4:
            check(np.allclose(tiles[:, 1] + tiles[:, 2], 1.0, atol=1e-7),
                  f"{name}: visible + remaining 应恒为 1")
        else:
            valid = set(S.SANMA_TYPES)
            for t in range(34):
                if t in valid:
                    check(abs(float(tiles[t, 1] + tiles[t, 2]) - 1.0) < 1e-7,
                          f"{name}: 三麻牌 {t} 的 visible + remaining 不是 1")
        check(np.all(tiles[:, 1] >= tiles[:, 0] - 1e-7),
              f"{name}: visible 应 >= hand")

    # 三麻：不存在的 2m..8m 整行必须为 0；第 3 个对手通道恒 0
    v3 = _fixture_3p()
    t3, _ = encode_tile(v3)
    for t in range(1, 8):
        check(np.all(t3[t] == 0.0), f"三麻不存在的牌 {t} 整行应为 0")
    check(np.all(t3[:, 12] == 0.0), "三麻 opp3_pond 通道应恒 0")
    check(np.all(t3[:, 15] == 0.0), "三麻 opp3_last_discard 通道应恒 0")
    print("  ✅ 形状 / dtype / 范围 / 确定性 / 三麻零行")


# ---------------------------------------------------------------------------
# 2. 金值 sha256
# ---------------------------------------------------------------------------


def test_golden() -> None:
    print("\n== 2. 金值 sha256（锁定语义）==")
    for name, v in fixtures():
        tiles, glob = encode_tile(v)
        got_t, got_g = digest(tiles), digest(glob)
        want_t, want_g = GOLDEN[name]
        print(f"  {name}: tiles={got_t}")
        print(f"  {name}: globals={got_g}")
        check(got_t == want_t,
              f"{name}: tiles 金值不匹配（语义被改了？拿到 {got_t}，期望 {want_t}）")
        check(got_g == want_g,
              f"{name}: globals 金值不匹配（拿到 {got_g}，期望 {want_g}）")


# ---------------------------------------------------------------------------
# 3. 通道含义抽查（防止「形状对但内容错位」）
# ---------------------------------------------------------------------------


def test_channel_semantics() -> None:
    print("\n== 3. 通道含义抽查 ==")
    v1 = _fixture_4p_mid()
    t1, g1 = encode_tile(v1)
    check(t1[13, 0] == 0.5, f"4p_mid: 手里 5p 两张，hand 通道应为 0.5，实际 {t1[13, 0]}")
    check(t1[13, 9] == 1.0,
          "4p_mid: 振听且 5p 是听牌张 -> furiten_wait 必须亮")
    check(np.count_nonzero(t1[:, 9]) == 1,
          f"4p_mid: 听牌张只应有 5p 一张，实际 {np.nonzero(t1[:, 9])[0]}")
    check(t1[13, 4] == 1.0, "4p_mid: 手里有赤 5p -> aka 通道必须亮")
    check(t1[0, 3] == 1.0 and t1[27, 3] == 1.0,
          "4p_mid: 宝牌指示 9m/北 -> 1m 和东应标记为宝牌")
    check(np.count_nonzero(t1[:, 3]) == 2,
          f"4p_mid: 宝牌应恰好 1m/东 两张，实际 {np.nonzero(t1[:, 3])[0]}")
    check(t1[27, 5] == 1.0, "4p_mid: 东是场风/自风 -> 役牌通道应亮")
    check(t1[31, 5] == 1.0, "4p_mid: 白是三元牌 -> 役牌通道应亮")
    check(t1[13, 18] == 0.0, "4p_mid: 对家有 1p2p3p 吃（不含 5p），opp_meld 应为 0")

    t1_no_riichi = encode_tile(dataclasses.replace(
        v1, riichi=[False, False, False, False]
    ))[0]
    check(np.all(t1_no_riichi[:, 16] == 0.0), "无立直家时 riichi_danger 应恒 0")
    check(np.all(t1_no_riichi[:, 17] == 0.0), "无立直家时 genbutsu 应恒 0")
    check(np.all(t1_no_riichi[:, 19] == 0.0), "无立直家时 suji 应恒 0")

    # 对手副露是公开信息，必须进入 opp_meld 通道
    v1_public = dataclasses.replace(
        v1,
        melds_all=[
            [S.Meld("pon", 31, from_player=1)],
            [],
            [S.Meld("chi", 9, from_player=3, called=10)],
            [S.Meld("pon", 28, from_player=0), S.Meld("pon", 13, from_player=1)],
        ],
    )
    t1_public = encode_tile(v1_public)[0]
    check(t1_public[13, 18] == 1.0,
          "在对手副露里加一张 5p，opp_meld 通道必须变亮（公开信息不能丢）")
    print("  ✅ 手牌/振听/赤/宝牌/役牌/立直相关通道")

    # 危险度：所有对手都立直时，与生产 tile_danger 的立直分支逐位一致
    v_all = dataclasses.replace(v1, riichi=[True, False, True, True])
    vis = visible_counts(v_all)
    t_all, _ = encode_tile(v_all)
    bad = []
    for t in range(34):
        got = float(t_all[t, 16])
        want = float(tile_danger(v_all, t, opp_riichi=[0, 2, 3], vis=vis))
        if abs(got - want) > 1e-7:
            bad.append((t, got, want))
    check(not bad, f"riichi_danger 与 tile_danger 不一致：{bad[:5]}（共 {len(bad)} 处）")
    print("  ✅ riichi_danger 在全员立直时与 tile_danger 逐位一致")


# ---------------------------------------------------------------------------
# 4. 🔴 无信息泄漏（Game 级：改别家手牌 -> 编码不变；改自己 -> 必须变）
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """拿到第一个 View 并做完泄漏检查后，终止这一局。"""


class _ProbeGame(Game):
    """在第一次 ``view_for`` 时偷偷改玩家状态并比较编码。

    这是**真正从 Game 状态派生 View** 的路径（不是手工构造 View），所以能
    验证 ``view_for`` + ``encode_tile`` 合起来没有把别家暗牌泄漏进来。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probed = False
        self.leak_ok = False
        self.negative_ok = False

    def view_for(self, p: int, called_tile: int | None = None) -> View:
        v = super().view_for(p, called_tile)
        if not self.probed:
            self.probed = True
            self._probe(v, p, called_tile)
            raise _Stop
        return v

    def _probe(self, v: View, p: int, called_tile: int | None) -> None:
        base_t, base_g = encode_tile(v)

        # --- 3a. 改别家手牌：View 里不该有这些暗牌 -------------------
        opps = [q for q in range(self.n) if q != p]
        saved = {}
        changed = False
        for q in opps:
            pl = self.players[q]
            saved[q] = list(pl.hand)
            rolled = [int(x) for x in np.roll(np.asarray(pl.hand, dtype=np.int64), 1)]
            if rolled != pl.hand:
                changed = True
            pl.hand = rolled
        # 至少一个对手的手牌真的变了，否则这条测试是假的
        check(changed, f"{self.n} 麻：改对手手牌时没有真的改动（测试无效）")

        v2 = super().view_for(p, called_tile)
        leak_t, leak_g = encode_tile(v2)
        self.leak_ok = (
            np.array_equal(base_t, leak_t) and np.array_equal(base_g, leak_g)
        )
        if not self.leak_ok:
            diff_t = np.nonzero(base_t != leak_t)
            diff_g = np.nonzero(base_g != leak_g)
            check(False,
                  f"{self.n} 麻：改别家手牌后编码变了！tiles 差异 {len(diff_t[0])} 处 "
                  f"{list(zip(diff_t[0][:5], diff_t[1][:5]))}，globals 差异 "
                  f"{len(diff_g[0])} 处 {diff_g[0][:5]}")
        else:
            print(f"  ✅ {self.n} 麻：改 {len(opps)} 家对手手牌，tiles/globals 逐位不变")

        for q in opps:
            self.players[q].hand = saved[q]

        # --- 3b. 改自己手牌：编码必须变（负向验证）-------------------
        own = self.players[p]
        saved_own = list(own.hand)
        t1 = next((t for t in range(34) if own.hand[t] > 0), None)
        t2 = next((t for t in range(34) if t != t1 and own.hand[t] < 4), None)
        check(t1 is not None and t2 is not None,
              f"{self.n} 麻：找不到可换的两张牌，负向验证无法进行")
        if t1 is not None and t2 is not None:
            own.hand[t1] -= 1
            own.hand[t2] += 1
            v3 = super().view_for(p, called_tile)
            neg_t, neg_g = encode_tile(v3)
            self.negative_ok = not np.array_equal(neg_t, base_t)
            check(self.negative_ok,
                  f"{self.n} 麻：改自己手牌后编码竟然没变（负向验证失败）")
            check(neg_t[t1, 0] != base_t[t1, 0] or neg_t[t2, 0] != base_t[t2, 0],
                  f"{self.n} 麻：自己手牌变了但 hand 通道没变")
            if self.negative_ok:
                print(f"  ✅ {self.n} 麻：改自己手牌 -> 编码必须变（负向验证通过）")
            own.hand = saved_own

        # --- 3c. 改自己的赤五：只有自己看得到的字段，编码也必须变 -----
        own = self.players[p]
        red_t = next(
            (t for t in range(34) if t % 9 == 4 and own.hand[t] > 0
             and not any(m.aka > 0 and t in m.tiles for m in own.melds)),
            None,
        )
        if red_t is None:
            print(f"  （{self.n} 麻：自己的手牌里没有可单独改的 5，跳过赤五负向子检查）")
        else:
            saved_red = list(own.red)
            own.red[red_t] = 0 if own.red[red_t] else 1
            expected = 1.0 if own.red[red_t] else 0.0
            v4 = super().view_for(p, called_tile)
            red_tiles, _ = encode_tile(v4)
            check(red_tiles[red_t, 4] == expected,
                  f"{self.n} 麻：改自己的赤五后 aka 通道期望 {expected}，"
                  f"实际 {red_tiles[red_t, 4]}（{red_t}）")
            check(red_tiles[red_t, 4] != base_t[red_t, 4],
                  f"{self.n} 麻：改自己的赤五后 aka 通道没变（负向验证失败）")
            if red_tiles[red_t, 4] != base_t[red_t, 4]:
                print(f"  ✅ {self.n} 麻：改自己的赤五 -> aka 通道变化（负向验证通过）")
            own.red = saved_red


def test_no_leak(players: int) -> None:
    print(f"\n== 4. 无信息泄漏（{players} 麻，Game 级）==")
    agents = [RandomAgent(seed=1000 + i) for i in range(players)]
    game = _ProbeGame(agents, seed=11 + players, rule=S.rule_for(players))
    try:
        game.play_hand(0)
    except _Stop:
        pass
    check(game.probed, f"{players} 麻：没有走到第一个 view_for，测试没跑")
    check(game.probed and game.leak_ok, f"{players} 麻：无泄漏检查没有通过")
    check(game.probed and game.negative_ok, f"{players} 麻：负向验证没有通过")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("encode_tile 单元测试（dot V1 per-tile）")
    print("=" * 72)
    test_shape_and_constants()
    test_golden()
    test_channel_semantics()
    test_no_leak(4)
    test_no_leak(3)
    print("\n" + "=" * 72)
    if FAILED:
        print(f"❌ {len(FAILED)} 条失败")
        for m in FAILED:
            print(f"  - {m.splitlines()[0]}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
