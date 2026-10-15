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
from aimodel.encode import ENCODE_V2_CHANNELS, encode_v2  # noqa: E402

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


# ---------------------------------------------------------------- 金值
#: ⚠️ 在**优化前**录制 ✓ （数据侧安全网）
GOLDEN_SHANTEN = "86d7da623429aade65ed29c2196c2985af5909c2f78652319f9e2e78f8e0f451"
GOLDEN_FEATURE = "b87614f62ca0c57e4b4d32f999c6481ad64a4cd9ece9c7902fda755549b7f3b6"


def main(argv: list[str]) -> int:
    if "--record" in argv:
        print(f'GOLDEN_SHANTEN = "{checksum_shanten_layer()}"')
        print(f'GOLDEN_FEATURE = "{checksum_feature_layer()}"')
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

    print("\n== 3) 确定性（同输入两次一致）==")
    check(checksum_shanten_layer() == got_s, "shanten 层不确定 ✗")
    check(checksum_feature_layer() == got_f, "特征层不确定 ✗")
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
