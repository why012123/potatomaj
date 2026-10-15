"""`encode_v2` 的纯单元测试 —— **不依赖长任务，也不碰接线**。

表示重构的第一步：把观测拆成

    tile   (C, 34)   逐张牌的通道（卷积/共享权重的天然形状）
    scalar (S,)      跟具体哪张牌无关的项（向听 / 巡目 / 点数 / 立直标记 …）

约定（和 Mac 定死的四条约束）：
* **不动老 `encode()`** —— 存量权重还在用 486/560 维，新旧并存
* **不加决策头** —— 加头把存量权重搞挂过一次
* `view.kita` 是**每家的长度 n**，不是 34（这里会专门测这条）
* 张量轴序 `(通道, 34)`，通道含义写进 `ENCODE_V2_CHANNELS` 的 docstring

**等价性由构造保证**（v2 是从 `encode()` 的输出切片出来的），
这个文件负责**把那个保证钉住**：逐通道比对，防止将来漂移。

跑法：``.venv/bin/python aimodel/tests/test_encode_v2.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent, RandomAgent  # noqa: E402
from aimodel.encode import (  # noqa: E402
    ENCODE_V2_CHANNELS, ENCODE_V2_SCALAR_DIM, encode, encode_v2, feature_blocks,
    layout_for,
)
from aimodel.engine import Game  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


class Spy:
    """把对局里出现过的 View 都收下来（各决策种类都要，不能只收 discard）。"""

    def __init__(self, cap: int = 400):
        self.views: list = []
        self.cap = cap

    def decide(self, decision, view):
        if len(self.views) < self.cap:
            self.views.append((decision.kind, view))
        return 0


def collect_views(players: int, seed: int = 0, cap: int = 400):
    spy = Spy(cap)
    agents = [spy] + [RandomAgent(seed * 100 + i) for i in range(1, players)]
    Game(agents, seed=seed, rule=S.rule_for(players)).play()
    return spy.views


print("== 通道表与常量必须自洽 ==")
names = list(ENCODE_V2_CHANNELS)
check(len(names) == len(set(names)), f"通道名不能重复：{names}")
check("yakuhai" in names, "yakuhai 的占位通道应当在表里（它非逐张牌，占位恒 0）")
for p in (3, 4):
    O, dim = layout_for(p)
    for n, w in feature_blocks(p):
        if w == 34:
            check(n in O, f"{p} 麻：块 {n} 在布局里找不到偏移")
    # 实际能填进 tile 的通道数 = 布局里真实存在的那些（三麻没有 opp3_*）
    present = [n for n in ENCODE_V2_CHANNELS if n in O and n != "yakuhai"]
    check(ENCODE_V2_SCALAR_DIM[p] > 0, f"{p} 麻的标量维应当 > 0")
    check(ENCODE_V2_SCALAR_DIM[p] + 34 * len(present) == dim,
          f"{p} 麻：标量 {ENCODE_V2_SCALAR_DIM[p]} + {len(present)}*34 "
          f"!= DIM {dim} —— v2 必须无重叠无遗漏地覆盖整个 encode()")
    print(f"  {p} 麻：{len(present)} 个逐张牌通道 + 标量 {ENCODE_V2_SCALAR_DIM[p]} 维"
          f" = {dim} 维 ✓")
check(len([n for n in names if n not in ("yakuhai",)]) == 15,
      "通道表里应有 15 个「可能真实存在」的逐张牌通道 + 1 个占位"
      "（四麻 15 个全在；三麻少 opp3_* 两个 -> 12 个）")

print("\n== 逐通道等价：v2 必须能无损重建 encode() ==")
bad_total = 0
tested = 0
for players in (3, 4):
    O, dim = layout_for(players)
    views = collect_views(players, seed=players)
    check(len(views) > 20, f"{players} 麻收到的 View 太少（{len(views)}）")
    kinds_seen = {k for k, _ in views}
    for kind, v in views:
        x = encode(v)
        tile, scalar = encode_v2(v)
        tested += 1
        # 形状
        if tile.shape != (len(ENCODE_V2_CHANNELS), 34):
            bad_total += 1
            print(f"    ❌ tile 形状 {tile.shape} != "
                  f"({len(ENCODE_V2_CHANNELS)}, 34)  kind={kind}")
            break
        if scalar.ndim != 1 or scalar.shape[0] != ENCODE_V2_SCALAR_DIM[players]:
            bad_total += 1
            print(f"    ❌ scalar 形状 {scalar.shape} != "
                  f"({ENCODE_V2_SCALAR_DIM[players]},)  kind={kind}")
            break
        # 重建：逐通道盖回原位 + 标量按块顺序盖回原位
        rebuilt = np.zeros(dim, dtype=np.float32)
        # 「宽 34 且逐张牌」= 进 tile 的那些（yakuhai 虽然宽 34，但它进 scalar）
        tnames = {n for n, w in feature_blocks(players)
                  if w == 34 and n != "yakuhai"}
        for i, nm in enumerate(ENCODE_V2_CHANNELS):
            if nm == "yakuhai" or nm not in O:
                continue          # yakuhai 进 scalar；opp3_* 三麻不存在
            a = O[nm]
            rebuilt[a : a + 34] = tile[i]
        off = 0
        for n, w in feature_blocks(players):
            if n in tnames:
                continue
            a = O[n]
            rebuilt[a : a + w] = scalar[off : off + w]
            off += w
        if off != ENCODE_V2_SCALAR_DIM[players]:
            bad_total += 1
            print(f"    ❌ 标量块游标走完是 {off}，期望 {ENCODE_V2_SCALAR_DIM[players]}")
            break
        if not np.array_equal(rebuilt, x):
            bad_total += 1
            diff = np.nonzero(rebuilt != x)[0]
            print(f"    ❌ 重建后与 encode() 不等价 kind={kind} "
                  f"差异 {len(diff)} 处，前几个下标 {diff[:6]}")
            break
    else:
        print(f"  ✅ {players} 麻 {len(views)} 个 View 无损重建"
              f"（决策种类 {len(kinds_seen)}：{sorted(kinds_seen)}）")
check(bad_total == 0, f"{bad_total} 处重建失败")

print("\n== 通道含义要对得上（不能错位）==")
O4, _ = layout_for(4)
v = collect_views(4, seed=7, cap=1)[0][1]
tile, _ = encode_v2(v)
i_hand = list(ENCODE_V2_CHANNELS).index("hand")
i_vis = list(ENCODE_V2_CHANNELS).index("visible")
# hand 通道 == encode() 的 hand 块
check(np.array_equal(tile[i_hand], encode(v)[O4["hand"] : O4["hand"] + 34]),
      "hand 通道与 encode() 的 hand 块不一致（通道错位了）")
check(np.array_equal(tile[i_vis], encode(v)[O4["visible"] : O4["visible"] + 34]),
      "visible 通道与 encode() 的 visible 块不一致")
# visible 必须 >= hand（可见包含自己手牌）
check(np.all(tile[i_vis] >= tile[i_hand] - 1e-6),
      "visible 应当 >= hand（可见计数含自己手牌）")
# yakuhai 占位通道必须恒 0
i_yak = list(ENCODE_V2_CHANNELS).index("yakuhai")
check(np.all(tile[i_yak] == 0.0), "yakuhai 是占位通道，必须恒 0")
# 三麻的 opp3_* 通道恒 0
v3 = collect_views(3, seed=8, cap=1)[0][1]
t3, _ = encode_v2(v3)
for nm in ("opp3_pond", "opp3_melds"):
    i = list(ENCODE_V2_CHANNELS).index(nm)
    check(np.all(t3[i] == 0.0), f"三麻的 {nm} 通道应当恒 0（只有 3 家）")
print("  ✅ hand / visible / yakuhai 占位 / 三麻 opp3 恒 0")

print("\n== kita 是每家的长度 n，不是 34（踩过的坑）==")
v4 = collect_views(4, seed=9, cap=1)[0][1]
check(len(v4.kita) == 4, f"四麻 view.kita 长度应为 4，得到 {len(v4.kita)}")
check(len(v3.kita) == 3, f"三麻 view.kita 长度应为 3，得到 {len(v3.kita)}")
# scalar 里含 kita 块（长度 players /4），不能因为它是 34 就被当成通道
check("kita" not in ENCODE_V2_CHANNELS,
      "kita 是「每家一个标量」，不该出现在逐张牌通道表里")
print(f"  ✅ kita 长度 = 人数（三麻 {len(v3.kita)} / 四麻 {len(v4.kita)}），"
      "且不在通道表里")

print("\n== 没加决策头 / 老 encode 不变 ==")
from aimodel.encode import HEAD_SIZE  # noqa: E402

check(set(HEAD_SIZE) == {"discard", "tsumo", "ron", "riichi", "call", "chi",
                         "kita", "kan", "kyuushu"},
      f"决策头集合不能变（加头会搞挂存量权重），得到 {sorted(HEAD_SIZE)}")
check(encode(v4).shape == (560,), f"四麻 encode() 仍应是 560 维，得到 {encode(v4).shape}")
check(encode(v3).shape == (486,), f"三麻 encode() 仍应是 486 维，得到 {encode(v3).shape}")
print("  ✅ 决策头 9 个未变；老 encode() 仍是 486/560")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print(f"\n✅ 全部通过（共比对 {tested} 个 View）")
