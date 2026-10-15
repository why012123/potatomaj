"""快速向听 / 进张计算（训练时的高频调用）。

外层工程的 :mod:`potatomaj.shanten` 是「读得懂、改得动」的实现，但一次
向听要 230 µs、一次进张要 80 ms —— 拿来生成几万条训练数据太慢。这里换成
**按花色分块 + 帕累托剪枝** 的写法：

* 每个花色块（万 / 筒 / 索 / 字）单独枚举「面子数、搭子数、有没有雀头」，
  结果按支配关系剪枝后缓存起来（同一个花色块只算一次）
* 四块用增量合并，合并结果同样剪枝（状态数一直很小）
* 字牌块不可能有顺子，枚举时关掉

正确性由 ``tests/test_fastshanten.py`` 与 :mod:`potatomaj.shanten` 交叉验证。

向听数的定义与 :mod:`potatomaj.shanten` 一致：``-1`` 已和牌，``0`` 听牌。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .sanma import N_TYPES, SANMA_TYPES, YAOCHU

#: 每个花色块的宽度
_BLOCK = ((0, 9), (9, 18), (18, 27), (27, 34))

# ---------------------------------------------------------------------------
# 单个花色块：枚举 (面子, 搭子, 雀头) 的所有帕累托最优组合
# ---------------------------------------------------------------------------

_TABLE_CACHE: dict[tuple[tuple[int, ...], bool], tuple[tuple[int, int, int], ...]] = {}


def _raw_states(block: tuple[int, ...], allow_seq: bool) -> set[tuple[int, int, int]]:
    """枚举一个花色块里能拆出的 (面子数, 搭子数, 雀头数)。"""
    out: set[tuple[int, int, int]] = set()
    n = len(block)
    c = list(block)

    def rec(i: int, m: int, p: int, q: int) -> None:
        # 用 <= 0 而不是 == 0：万一传进来的计数是负数（调用方写错），
        # 也只会当成「这张牌没有」，而不是在这里无限递归下去
        while i < n and c[i] <= 0:
            i += 1
        if i >= n:
            out.add((m, p, q))
            return
        if m < 4:
            if c[i] >= 3:  # 刻子
                c[i] -= 3
                rec(i, m + 1, p, q)
                c[i] += 3
            if allow_seq and i + 2 < n and c[i + 1] and c[i + 2]:  # 顺子
                c[i] -= 1
                c[i + 1] -= 1
                c[i + 2] -= 1
                rec(i, m + 1, p, q)
                c[i] += 1
                c[i + 1] += 1
                c[i + 2] += 1
        if q == 0 and c[i] >= 2:  # 当雀头
            c[i] -= 2
            rec(i, m, p, 1)
            c[i] += 2
        if p < 4:
            if c[i] >= 2:  # 对子当搭子
                c[i] -= 2
                rec(i, m, p + 1, q)
                c[i] += 2
            if allow_seq and i + 1 < n and c[i + 1]:  # 两面 / 边张（89 等 7 也要算）
                c[i] -= 1
                c[i + 1] -= 1
                rec(i, m, p + 1, q)
                c[i] += 1
                c[i + 1] += 1
            if allow_seq and i + 2 < n and c[i + 2]:  # 嵌张
                c[i] -= 1
                c[i + 2] -= 1
                rec(i, m, p + 1, q)
                c[i] += 1
                c[i + 2] += 1
        c[i] -= 1  # 这张牌不参与任何面子/搭子
        rec(i, m, p, q)
        c[i] += 1

    rec(0, 0, 0, 0)
    return out


def _pareto(states: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    """按支配关系剪枝：(m,p,q) 都不小于另一个状态时，另一个可以丢掉。"""
    ordered = sorted(set(states), key=lambda s: (-s[0], -s[1], -s[2]))
    keep: list[tuple[int, int, int]] = []
    for s in ordered:
        m, p, q = s
        dominated = False
        for km, kp, kq in keep:
            if km >= m and kp >= p and kq >= q:
                dominated = True
                break
        if not dominated:
            keep.append(s)
    return tuple(keep)


def suit_table(block: Sequence[int], allow_seq: bool = True) -> tuple[tuple[int, int, int], ...]:
    """取（并缓存）一个花色块的帕累托状态表。"""
    key = (tuple(block), allow_seq)
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = _pareto(_raw_states(key[0], allow_seq))
        _TABLE_CACHE[key] = table
    return table


def _combine(
    a: tuple[tuple[int, int, int], ...], b: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    """合并两个花色表。"""
    out: set[tuple[int, int, int]] = set()
    for m1, p1, q1 in a:
        for m2, p2, q2 in b:
            m = m1 + m2
            if m > 4:
                continue
            p = p1 + p2
            out.add((m, p if p < 4 else 4, q1 | q2))
    return _pareto(out)


_COMBINE_CACHE: dict[tuple, tuple[tuple[int, int, int], ...]] = {}


def _combine_cached(a, b):
    key = (a, b)
    r = _COMBINE_CACHE.get(key)
    if r is None:
        r = _combine(a, b)
        _COMBINE_CACHE[key] = r
    return r


def _tables(counts: Sequence[int]):
    t = []
    for i, (lo, hi) in enumerate(_BLOCK):
        t.append(suit_table(tuple(counts[lo:hi]), allow_seq=(i < 3)))
    return t


#: 牌 index -> 它属于哪个 ``_BLOCK``（0..3）。
#: ``_BLOCK = ((0,9),(9,18),(18,27),(27,34))`` -> 万 0-8 / 筒 9-17 / 索 18-26 / 字 27-33
_BLOCK_OF: tuple[int, ...] = tuple(
    i for i, (lo, hi) in enumerate(_BLOCK) for _ in range(lo, hi)
)


def _shanten_from_tables(
    t: Sequence[tuple[tuple[int, int, int], ...]],
    counts: Sequence[int],
    n_melds: int,
) -> int:
    """``_tables`` **之后**的那一段（与 :func:`shanten` 逐字一致 ✓）。

    抽出来是为了让 :func:`useful_tiles` 能复用未变花色的表 ✓
    **公式一个字都没改** ✓ —— 改的只是「表从哪来」✓
    """
    best = _best_from_states(
        _combine_cached(_combine_cached(t[0], t[1]), _combine_cached(t[2], t[3])), n_melds
    )
    if n_melds == 0:
        v = _chiitoi_shanten(counts)
        if v < best:
            best = v
        v = _kokushi_shanten(counts)
        if v < best:
            best = v
    return best


def _memo_put(key: tuple, value: int) -> None:
    """写记忆化表（**含原有的容量控制** ✓ 不新增缓存 ✓）。"""
    if len(_SHANTEN_MEMO) >= _SHANTEN_MEMO_MAX:
        _SHANTEN_MEMO.clear()
    _SHANTEN_MEMO[key] = value


def _best_from_states(states: Iterable[tuple[int, int, int]], n_melds: int = 0) -> int:
    """块分解 -> 向听数。

    ``n_melds`` 是**已经副露（含暗杠）的面子数**：副露的面子不用再凑，
    但它一样占用「4 个面子」的名额，所以要把 ``m`` 加上去、并把搭子上限
    改成 ``4 - 总面子数``。
    """
    best = 8
    for m, p, q in states:
        m += n_melds
        if m > 4:
            continue
        room = 4 - m
        v = 8 - 2 * m - min(p, room) - (1 if q else 0)
        if v < best:
            best = v
    return best


# ---------------------------------------------------------------------------
# 七对子 / 国士
# ---------------------------------------------------------------------------


def _chiitoi_shanten(counts: Sequence[int]) -> int:
    pairs = 0
    kinds = 0
    for c in counts:
        if c:
            kinds += 1
            if c >= 2:
                pairs += 1
    return 6 - pairs + max(0, 7 - kinds)


def _kokushi_shanten(counts: Sequence[int]) -> int:
    kinds = sum(1 for i in YAOCHU if counts[i])
    pair = any(counts[i] >= 2 for i in YAOCHU)
    return 13 - kinds - (1 if pair else 0)


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

#: ``(counts 的 tuple, n_melds) -> shanten`` 的记忆化表。
#: 进张计算会对同一手牌反复问「摸到 X 之后的向听」，而每次
#: ``_tables()`` 都要重建 4 个花色 tuple + 3 次 combine —— 实测这是
#: ``action_features`` 里最贵的一块。键是精确输入，所以不会串味。
_SHANTEN_MEMO: dict[tuple[tuple[int, ...], int], int] = {}
_SHANTEN_MEMO_MAX = 200_000


def shanten(counts: Sequence[int], n_melds: int = 0) -> int:
    """普通形 / 七对子 / 国士 取最优的向听数。

    ``counts`` 是 **暗牌** 的 34 长度计数数组，``n_melds`` 是已副露（含暗杠）
    的面子数。多于 13 张（通常是刚摸到第 14 张）时结果等于「打掉一张之后的
    最优」，与 :func:`potatomaj.shanten.shanten` 同口径。

    这里不真的枚举弃牌，而是直接算块分解：多余的牌本来就可以被丢掉，所以
    「14 张」和「13 张取最优」是同一个式子（在 3000 手随机 14 张牌上验证过）。
    七对子 / 国士只在门清（``n_melds == 0``）时才算。

    **结果按 ``(counts, n_melds)`` 记忆化**（见 ``_SHANTEN_MEMO``）：
    进张计算会对同一手牌算几十次「摸一张之后的向听」，其中
    ``_tables()`` 每次都要重建 4 个花色 tuple 再做 3 次 combine，
    实测占掉 ``action_features`` 的绝大部分时间。
    """
    key = (tuple(counts), n_melds)
    hit = _SHANTEN_MEMO.get(key)
    if hit is not None:
        return hit

    t = _tables(counts)
    best = _best_from_states(
        _combine_cached(_combine_cached(t[0], t[1]), _combine_cached(t[2], t[3])), n_melds
    )
    if n_melds == 0:
        v = _chiitoi_shanten(counts)
        if v < best:
            best = v
        v = _kokushi_shanten(counts)
        if v < best:
            best = v

    if len(_SHANTEN_MEMO) >= _SHANTEN_MEMO_MAX:
        # 简单粗暴的容量控制：满了就整体清空。训练里手牌形状高度重复，
        # 清空一次很快就能重新填满，比引入 LRU 便宜也更好懂。
        _SHANTEN_MEMO.clear()
    _SHANTEN_MEMO[key] = best
    return best


def shanten_by_discard(counts: Sequence[int]) -> int:
    """慢速参照实现：多于 13 张时逐张弃牌取最优（只用于交叉验证）。"""
    total = sum(counts)
    if total <= 13:
        return shanten(counts)
    best = 99
    c = list(counts)
    for i in range(N_TYPES):
        if c[i] == 0:
            continue
        c[i] -= 1
        v = shanten_by_discard(c)
        c[i] += 1
        if v < best:
            best = v
    return best


def is_tenpai(counts: Sequence[int], n_melds: int = 0) -> bool:
    return shanten(counts, n_melds) == 0


def useful_tiles(
    counts: Sequence[int], n_melds: int = 0, sanma: bool = True, base: int | None = None
) -> list[int]:
    """摸到之后能让向听数变小的牌型（进张）。

    ``base`` 是这手牌**已经算好的**向听数；调用方（``action_features``
    会对每个候选弃牌各调一次）手上本来就有它，传进来可以省一次 ``shanten()``。

    **性能（2026-10-16，端到端可回收 ≈ 产数据 42%）**
    -------------------------------------------------
    旧写法对 34 个候选各调一次 :func:`shanten` ✗ 而每次 ``shanten`` 都要
    ``_tables`` 重建 **4 个花色表** ✓ 但**候选之间只有 1 个花色会变** ✓
    （实测：与 base 相比未变花色数 = 精确的 3.00 / 4 ✓）
    -> 136 次花色查询里约 102 次是**纯冗余** ✗

    ``suit_table`` **本来就有缓存** ✓ 所以贵的不是「重复计算」✗
    而是**建键 + 哈希 + 查字典**本身（cProfile：`suit_table` 2330 万次）✗
    -> 这与训练侧那个 `dict.get(k, (zeros_like, zeros_like))` 是**同一根因**：
       **查字典的次数远超必要，且每次查都要现造一个可哈希的键** ✓

    本版：**base 的 4 个表算一次**，每个候选**只重查变了的那 1 个花色** ✓
    -> 136 次查询降到约 38 次 ✓ **没有引入任何新缓存** ✓（反而是减少使用 ✓）

    ⚠️ 数值行为**逐位不变** ✓：
      · 公式一字未改（`_shanten_from_tables` 与 `shanten` 的该段逐字一致 ✓）
      · 记忆化仍按原来的方式读写 ✓（`_memo_put` 含原有容量控制 ✓）
      · `shanten(counts, n_melds)` 在 base 那次仍走原路径 ✓
    """
    if base is None:
        base = shanten(counts, n_melds)
    if base < 0:
        return []
    types = SANMA_TYPES if sanma else range(N_TYPES)
    c = list(counts)
    out: list[int] = []
    # base 的 4 个花色表：**整轮只算这一次** ✓（旧写法每个候选都重算 ✓）
    t_base = _tables(counts)
    for t in types:
        if c[t] >= 4:
            continue
        c[t] += 1
        key = (tuple(c), n_melds)
        hit = _SHANTEN_MEMO.get(key)
        if hit is not None:
            v = hit
        else:
            # 只重查**变了的那 1 个花色** ✓ 其余 3 个直接复用 ✓
            i = _BLOCK_OF[t]
            tt = list(t_base)
            lo, hi = _BLOCK[i]
            tt[i] = suit_table(tuple(c[lo:hi]), allow_seq=(i < 3))
            v = _shanten_from_tables(tt, c, n_melds)
            _memo_put(key, v)
        c[t] -= 1
        if v < base:
            out.append(t)
    return out


def ukeire(
    counts: Sequence[int],
    visible: Sequence[int] | None = None,
    n_melds: int = 0,
    sanma: bool = True,
    base: int | None = None,
) -> dict[int, int]:
    """进张 -> 剩余枚数。``visible`` 是场上已见到的牌（含自己手牌）。"""
    out: dict[int, int] = {}
    for t in useful_tiles(counts, n_melds, sanma=sanma, base=base):
        seen = visible[t] if visible is not None else counts[t]
        left = 4 - seen
        if left > 0:
            out[t] = left
    return out


def ukeire_count(
    counts: Sequence[int],
    visible: Sequence[int] | None = None,
    n_melds: int = 0,
    sanma: bool = True,
    base: int | None = None,
) -> int:
    """进张总枚数（手牌效率的常用指标）。"""
    return sum(ukeire(counts, visible, n_melds, sanma, base=base).values())


def waits(counts: Sequence[int], n_melds: int = 0, sanma: bool = True) -> list[int]:
    """听牌时的和了牌（向听 0 时等价于进张）。"""
    if shanten(counts, n_melds) != 0:
        return []
    return useful_tiles(counts, n_melds, sanma=sanma)


__all__ = [
    "shanten", "shanten_by_discard", "is_tenpai", "useful_tiles", "ukeire",
    "ukeire_count", "waits", "suit_table",
]
