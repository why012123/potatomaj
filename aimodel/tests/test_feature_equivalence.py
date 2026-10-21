"""数据侧数值等价性安全网：**特征必须逐位一致**。

为什么数据侧的安全网形状和训练侧不同（mac 点名的）
================================================
```
训练侧：改造前后**权重逐位一致** ✓ （`test_step_equivalence.py`）
数据侧：改造前后**特征逐位一致** ✗
```
⚠️ 而数据侧的风险**更重** ✗：
**特征不一致 = 训练数据变了** ✓ 不是「算错了」那么轻 ✗
-> **所有下游模型都不可比** ✓（包括已发布的 maj-pro 系列 ✓ 以及三杠杆排序的全部结论 ✓）

跑法：
    .venv/bin/python aimodel/tests/test_feature_equivalence.py --record   # 录制金值
    .venv/bin/python aimodel/tests/test_feature_equivalence.py            # 校验
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.encode import action_features  # noqa: E402
from aimodel.fastshanten import (  # noqa: E402
    shanten, ukeire, ukeire_count, useful_tiles,
)
from aimodel.engine import View  # noqa: E402
from aimodel.encode import (  # noqa: E402
    ENCODE_V2_CHANNELS, encode, encode_v2, feature_blocks, layout_for,
)

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


# ---------------------------------------------------------------- 手牌清单
#: 覆盖各种形状：和了形 / 听牌 / 1向听 / 2向听 / 字牌多 / 副露 / 国士 / 七对
HANDS: list[tuple[tuple[int, ...], int, bool]] = [
    ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 0, True),      # 清一色雏形
    ((0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4), 0, True),          # 多刻子
    ((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), 0, True),          # 七对
    ((0, 8, 9, 17, 18, 26, 27, 27, 28, 28, 29, 29, 30, 30), 0, True),  # 国士
    ((0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 16), 0, True),    # 三组顺子
    ((27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31, 31), 0, True),  # 字牌刻子
    ((0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26), 0, True),  # 全孤张
    ((0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 1, True),        # 1 副露
    ((0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4), 2, True),          # 2 副露
    ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 0, False),     # 四麻
    ((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), 0, False),
    ((27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31, 31), 1, False),
]

def counts_of(tiles: tuple[int, ...]) -> list[int]:
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    return c

VISIBLE: list[tuple[int, ...]] = [
    tuple(counts_of(HANDS[0][0])),             # 只看到自己手牌（34 长度！）
    tuple(0 for _ in range(34)),              # 场上什么都没见
    tuple(1 if i % 3 == 0 else 0 for i in range(34)),   # 零散可见
]



# ---------------------------------------------------------------- View 构造
def make_view(players: int, hand_counts: list[int], turn: int = 5) -> View:
    rule = S.rule_for(players)
    return View(
        player=0, hand=list(hand_counts), drawn=None, melds=[],
        hand_red=[0] * 34,
        discards=[[] for _ in range(players)],
        melds_all=[[] for _ in range(players)],
        kita=[0] * players,
        scores=[rule.start_score] * players,
        dora_indicators=[0], round_wind=S.EAST, seat_wind=S.EAST,
        wall_left=40, turn=turn,
        riichi=[False] * players, ippatsu=[False] * players,
        is_dealer=True, last_discard=None, rule=rule,
    )


# ---------------------------------------------------------------- 校验和
def _feed(h: hashlib._Hash, tag: str, x) -> None:
    """把任意嵌套结构喂进哈希（**递归** —— `encode_v2` 返回的是元组）✓"""
    h.update(tag.encode())
    if isinstance(x, (int, np.integer)):
        h.update(np.int64(x).tobytes())
    elif isinstance(x, dict):
        for k in sorted(x):
            h.update(np.int64(k).tobytes())
            _feed(h, "v", x[k])
    elif isinstance(x, (list, tuple)):
        for i, y in enumerate(x):
            _feed(h, str(i), y)
    else:
        h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())


def checksum_shanten_layer() -> str:
    """纯 counts 层：shanten / useful_tiles / ukeire / ukeire_count ✓"""
    h = hashlib.sha256()
    for tiles, n_melds, sanma in HANDS:
        c = counts_of(tiles)
        for vis in VISIBLE:
            _feed(h, "shanten", shanten(c, n_melds))
            _feed(h, "useful", useful_tiles(c, n_melds, sanma=sanma))
            _feed(h, "ukeire", ukeire(c, vis, n_melds, sanma=sanma))
            _feed(h, "ukeire_n", ukeire_count(c, vis, n_melds, sanma=sanma))
    return h.hexdigest()


def checksum_feature_layer() -> str:
    """特征层：`encode_v2` + `action_features`（这是真正进训练数据的东西）✓"""
    h = hashlib.sha256()
    for tiles, n_melds, sanma in HANDS:
        players = 3 if sanma else 4
        v = make_view(players, counts_of(tiles))
        _feed(h, "encv2", encode_v2(v))
        _feed(h, "actfeat", action_features(v))
    return h.hexdigest()


#: **不依赖 visible_counts 的老块** —— 逐位不变的硬性清单。
#:
#: 2026-10-21 两件事同时发生：
#:   1. 末尾追加 match_state 17 / called_tile 34 / last_discard 34；
#:   2. 修 `visible_counts` 把「自己的副露 / 自己的拔北」算两遍的 bug。
#: (1) 只该追加，(2) 只该影响依赖 vis 的块：**visible / ukeire / safe_exit**
#: （safe_exit 也依赖 vis：tile_danger 对字牌用 vis[tile] 判剩余枚数）。
#: 其余老块必须逐位不变 —— 这里就是那条断言。
LEGACY_UNCHANGED_BLOCKS: tuple[str, ...] = (
    "hand", "hand_onehot", "hand_pair", "hand_triple", "melds", "own_pond",
    "opp1_pond", "opp2_pond", "opp3_pond", "opp1_melds", "opp2_melds",
    "opp3_melds", "dora_indicator", "dora_tile", "yakuhai", "shanten_onehot",
    "hand_dora", "aka", "wall_left", "turn", "hand_size", "scores",
    "score_lead", "riichi", "ippatsu", "kita", "meld_count", "seat_onehot",
    "round_wind", "is_dealer", "menzen", "tenpai",
)


def checksum_legacy_blocks() -> str:
    """老块（不含 visible / ukeire / safe_exit）的逐位校验和。

    金值是在**改动前**录的；新增特征只追加在末尾、visible_counts 修复只
    影响上面那三块，所以它必须原样不变。
    """
    h = hashlib.sha256()
    for tiles, n_melds, sanma in HANDS:
        players = 3 if sanma else 4
        v = make_view(players, counts_of(tiles))
        O, _ = layout_for(players)
        x = encode(v)
        widths = dict(feature_blocks(players))
        for name in LEGACY_UNCHANGED_BLOCKS:
            if name not in O:          # 三麻没有 opp3_*
                continue
            _feed(h, name, x[O[name] : O[name] + widths[name]])
    return h.hexdigest()


# ---------------------------------------------------------------- 金值
#: ⚠️ 在**优化前**录制 ✓ （数据侧安全网）
GOLDEN_SHANTEN = "86d7da623429aade65ed29c2196c2985af5909c2f78652319f9e2e78f8e0f451"
#: 2026-10-21 因新增 match_state / called_tile / last_discard（及 visible_counts
#: 修复）重录；**shanten 层金值没动** ✓
GOLDEN_FEATURE = "d9fecc629adc11c76e94f87ce569769b381867e4a7868f96a973ae38a93fc85e"
#: ⚠️ 这个值**不是** `--record` 重录的：它固定为 2026-10-21 改动前的实测值。
#: 它证明「老特征一个都没动」——重录它就等于把安全网拆了。
GOLDEN_LEGACY_BLOCKS = "642507c48e214d7387af040c17b8f80f99ec6e45614be3658af4b316460d4a8d"


def main(argv: list[str]) -> int:
    if "--record" in argv:
        print(f'GOLDEN_SHANTEN = "{checksum_shanten_layer()}"')
        print(f'GOLDEN_FEATURE = "{checksum_feature_layer()}"')
        # ⚠️ 故意**不**打印 GOLDEN_LEGACY_BLOCKS：那个值是改动前的锚，
        # 重录它就失去「老块逐位不变」的意义。
        print("# GOLDEN_LEGACY_BLOCKS 保持 2026-10-21 改动前的值，不重录")
        return 0

    print("== 1) shanten 层（shanten/useful_tiles/ukeire/ukeire_count）==")
    got_s = checksum_shanten_layer()
    print(f"  当前 {got_s[:32]}…   金值 {GOLDEN_SHANTEN[:32]}…")
    if GOLDEN_SHANTEN == "PENDING":
        FAILED.append("GOLDEN_SHANTEN 未录制")
    else:
        check(got_s == GOLDEN_SHANTEN, "shanten 层数值变了 ✗（数据侧改动必须逐位一致）")

    print("\n== 2) 特征层（encode_v2 + action_features）==")
    got_f = checksum_feature_layer()
    print(f"  当前 {got_f[:32]}…   金值 {GOLDEN_FEATURE[:32]}…")
    if GOLDEN_FEATURE == "PENDING":
        FAILED.append("GOLDEN_FEATURE 未录制")
    else:
        check(got_f == GOLDEN_FEATURE,
              "特征变了 ✗✗ **这等于训练数据变了** -> 所有下游模型不可比 ✗")

    print("\n== 2b) 老块逐位不变（只允许 visible / ukeire / safe_exit 变）==")
    got_l = checksum_legacy_blocks()
    print(f"  当前 {got_l[:32]}…   金值 {GOLDEN_LEGACY_BLOCKS[:32]}…")
    check(got_l == GOLDEN_LEGACY_BLOCKS,
          "不依赖 visible_counts 的老块变了 ✗（新特征只该追加在末尾；"
          "若动了老块，所有存量数据/权重都不可比）")

    print("\n== 3) 确定性（同输入两次一致）==")
    check(checksum_shanten_layer() == got_s, "shanten 层不确定 ✗")
    check(checksum_feature_layer() == got_f, "特征层不确定 ✗")
    check(checksum_legacy_blocks() == got_l, "老块校验和不确定 ✗")
    print("  两次一致 ✓")

    print("\n== 4) ⭐ 敏感性自检（防「测试恒真」）==")
    # 扰动一个手牌的一个计数 -> 两层校验和都必须变
    import aimodel.fastshanten as F
    orig_useful = F.useful_tiles

    def off_by_one(counts, n_melds=0, *, sanma=True, base=None):
        counts = list(counts)
        counts[0] = max(0, counts[0] - 1)      # 少算一张
        return orig_useful(counts, n_melds, sanma=sanma, base=base)

    F.useful_tiles = off_by_one
    try:
        pert_s = checksum_shanten_layer()
    finally:
        F.useful_tiles = orig_useful
    check(pert_s != got_s, "shanten 层校验和对「useful_tiles 少算一张」不敏感 ✗（测试无效）")
    print(f"  扰动 useful_tiles -> shanten 层变化: {pert_s != got_s} ✓")

    orig_af = sys.modules["aimodel.encode"].action_features

    def bad_af(view):
        r = np.array(orig_af(view), copy=True)
        r[0] += 1e-6                            # 极小扰动
        return r

    sys.modules["aimodel.encode"].action_features = bad_af
    try:
        # 注意：本模块已 from ... import action_features，需直接改本模块的引用
        globals()["action_features"] = bad_af
        pert_f = checksum_feature_layer()
    finally:
        sys.modules["aimodel.encode"].action_features = orig_af
        globals()["action_features"] = orig_af
    check(pert_f != got_f, "特征层校验和对 1e-6 扰动不敏感 ✗（测试无效）")
    print(f"  扰动 action_features 1e-6 -> 特征层变化: {pert_f != got_f} ✓")

    if FAILED:
        print(f"\n❌ {len(FAILED)} 条失败")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
