"""向听数与有效牌（听牌 = 0 向听，和牌 = -1 向听）。

这是「判断结果」窗口里真正有用的部分：光把牌认出来还不够，得告诉玩家
「离听牌还差几张、摸哪些牌有用」。

三麻只有 1m/9m 两种万子，但这里不写死规则 —— 直接把手里有什么传进来即可，
算法天然只看存在的牌，所以三麻/四麻都能用。
"""

from __future__ import annotations

from functools import lru_cache

from .tiles import ALL_TILE_IDS, TILE_IDS, normalize_tile_id, parse_tile

#: 34 种牌的顺序（不含赤5），用作计数数组的下标
_BASE_IDS: tuple[str, ...] = tuple(TILE_IDS)
_INDEX = {tid: i for i, tid in enumerate(_BASE_IDS)}
N_TYPES = len(_BASE_IDS)


# ---------------------------------------------------------------------------
# 快实现（可选）
#
# aimodel 里那份「按花色分块 + 帕累托剪枝 + 缓存」的向听/进张，比本文件的朴素
# 实现快三个数量级 —— 实测一次进张 **192 ms vs 0.2 ms**。差这么多是因为本文件
# 的写法为了好读，每张候选牌都要重新枚举一遍所有拆解方式。
#
# 平时每巡只要算一次，慢了也就是「建议晚一秒出来」；但**每张牌打掉的收益**要算
# 十几次，1.3 秒的纯 CPU 占用会一直跟游戏抢核，表现就是整个页面卡。
#
# 软接入：拿得到就用快的，拿不到就用本文件的 —— 两份实现的结果在
# aimodel/tests 里逐手交叉验证过（13 张 × 400 手、14 张 × 150 手零处不一致），
# 这里另有一份自己的交叉验证（tests/test_fastshanten.py）。
# ---------------------------------------------------------------------------
def _fast():
    try:
        from aimodel import fastshanten as F
        return F
    except Exception:
        return None


def counts_from_tiles(tiles: list[str] | tuple[str, ...]) -> list[int]:
    """把牌 id 列表转成 34 长度的计数数组（赤5 并入普通 5）。"""
    counts = [0] * N_TYPES
    for tid in tiles:
        base = normalize_tile_id(tid)
        idx = _INDEX.get(base)
        if idx is not None:
            counts[idx] += 1
    return counts


def _is_honor(i: int) -> bool:
    return i >= 27


def _suit_pos(i: int) -> int:
    return i % 9


# ---------------------------------------------------------------------------
# 普通形（4 面子 + 1 雀头）
# ---------------------------------------------------------------------------


def _decompose(counts: list[int], i: int, melds: int, partials: int, out: set[tuple[int, int]]) -> None:
    """枚举「面子 / 搭子」的所有拆法，收集 (面子数, 搭子数) 的可达组合。"""
    while i < N_TYPES and counts[i] == 0:
        i += 1
    if i >= N_TYPES:
        out.add((melds, partials))
        return
    if melds >= 4:
        out.add((melds, min(partials, 4)))
        return

    c = counts[i]
    # 刻子
    if c >= 3:
        counts[i] -= 3
        _decompose(counts, i, melds + 1, partials, out)
        counts[i] += 3
    if not _is_honor(i):
        pos = _suit_pos(i)
        # 顺子
        if pos <= 6 and counts[i + 1] and counts[i + 2]:
            counts[i] -= 1
            counts[i + 1] -= 1
            counts[i + 2] -= 1
            _decompose(counts, i, melds + 1, partials, out)
            counts[i] += 1
            counts[i + 1] += 1
            counts[i + 2] += 1
    if partials < 4:
        # 对子当搭子
        if c >= 2:
            counts[i] -= 2
            _decompose(counts, i, melds, partials + 1, out)
            counts[i] += 2
        if not _is_honor(i):
            pos = _suit_pos(i)
            # 两面 / 边张
            if pos <= 7 and counts[i + 1]:
                counts[i] -= 1
                counts[i + 1] -= 1
                _decompose(counts, i, melds, partials + 1, out)
                counts[i] += 1
                counts[i + 1] += 1
            # 嵌张
            if pos <= 6 and counts[i + 2]:
                counts[i] -= 1
                counts[i + 2] -= 1
                _decompose(counts, i, melds, partials + 1, out)
                counts[i] += 1
                counts[i + 2] += 1
    # 这张不要了
    counts[i] -= 1
    _decompose(counts, i, melds, partials, out)
    counts[i] += 1


def _shanten_normal(counts: list[int], fixed_melds: int = 0) -> int:
    """普通形向听数。

    ``fixed_melds`` 是已经碰/吃/杠出来的面子数 —— 那些牌**不在手牌里**，
    但它们各占一个面子格，所以「还需要凑的面子」从 4 减到 4-fixed_melds。
    """
    need = max(0, 4 - fixed_melds)
    base = 8 - 2 * fixed_melds          # 副露已经完成的面子直接抵扣向听
    best = base
    # 枚举雀头
    for i in range(N_TYPES):
        if counts[i] >= 2:
            counts[i] -= 2
            combos: set[tuple[int, int]] = set()
            _decompose(counts, 0, 0, 0, combos)
            counts[i] += 2
            for melds, partials in combos:
                usable = min(partials, max(0, need - melds))
                best = min(best, base - 2 * melds - usable - 1)
    # 无雀头（雀头本身还要占一张牌，所以不加那 1）
    combos = set()
    _decompose(counts, 0, 0, 0, combos)
    for melds, partials in combos:
        usable = min(partials, max(0, need - melds))
        best = min(best, base - 2 * melds - usable)
    return best


# ---------------------------------------------------------------------------
# 七对子 / 国士无双
# ---------------------------------------------------------------------------


def _shanten_chiitoi(counts: list[int]) -> int:
    pairs = sum(1 for c in counts if c >= 2)
    kinds = sum(1 for c in counts if c >= 1)
    return 6 - pairs + max(0, 7 - kinds)


#: 幺九牌（1/9 数牌 + 全部字牌）在计数数组里的下标
_YAOCHU = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def _shanten_kokushi(counts: list[int]) -> int:
    kinds = sum(1 for i in _YAOCHU if counts[i] >= 1)
    has_pair = any(counts[i] >= 2 for i in _YAOCHU)
    return 13 - kinds - (1 if has_pair else 0)


def _shanten_13(counts: list[int], fixed_melds: int = 0) -> int:
    """手牌的向听数（普通形 / 七对子 / 国士取最优）。

    有副露时七对子和国士都不成立（那两个役型必须门清），所以只算普通形。
    """
    if fixed_melds:
        return _shanten_normal(counts, fixed_melds)
    return min(_shanten_normal(counts), _shanten_chiitoi(counts), _shanten_kokushi(counts))


def melds_from_hand_size(slots: int) -> int:
    """从**手牌可见张数**反推副露数。

    碰/吃/杠出来的牌不在手牌行里，所以手牌行会短 3 的倍数：
        14/13 张 -> 0 副露   11/10 -> 1   8/7 -> 2   5/4 -> 3   2/1 -> 4
    """
    if slots <= 0:
        return 0
    return max(0, min(4, (14 - slots) // 3))


def shanten(tiles: list[str] | tuple[str, ...], melds: int = 0) -> int:
    """返回向听数：-1 = 已和牌，0 = 听牌，n = n 向听。

    多于 13 张（通常是刚摸到第 14 张）时，按「打掉一张之后的最优」来算 ——
    这样摸到一张没用的牌不会让向听数变差，符合直觉，也是各家牌理的通行口径。
    """
    counts = counts_from_tiles(tiles)
    total = sum(counts)
    # 加回副露占用的张数，才是「等效满手」
    effective = total + 3 * melds
    if effective == 14 and _is_agari(counts, melds):
        return -1
    if effective > 13:
        best = 99
        for i in range(N_TYPES):
            if counts[i] == 0:
                continue
            counts[i] -= 1
            best = min(best, _shanten_13(counts, melds))
            counts[i] += 1
        return best
    return _shanten_13(counts, melds)


def _is_agari(counts: list[int], melds: int = 0) -> bool:
    """是否已经和牌。

    ``melds`` 是副露数 —— 副露各占一个面子，所以手牌只需要再凑 4-melds 个面子 + 雀头。
    有副露时七对子 / 国士都不成立（必须门清），不再检查。
    """
    need_sets = max(0, 4 - melds)
    # 七对子 / 国士（只在门清时）
    if melds == 0 and sum(counts) == 14:
        if all(c in (0, 2) for c in counts) and sum(1 for c in counts if c == 2) == 7:
            return True
        yaochu = _YAOCHU
        # 注意：必须「全部 14 张都是幺九」才算国士，否则手里有 13 种幺九 +
        # 任意一张普通牌也会被误判成和牌
        if all(counts[i] >= 1 for i in yaochu) and sum(counts[i] for i in yaochu) == 14:
            return True
    if not melds and sums_ok(counts) is False:
        return False

    def melds_ok(c: list[int]) -> bool:
        i = 0
        while i < N_TYPES and c[i] == 0:
            i += 1
        if i >= N_TYPES:
            return True
        if c[i] >= 3:
            c[i] -= 3
            if melds_ok(c):
                c[i] += 3
                return True
            c[i] += 3
        if not _is_honor(i) and _suit_pos(i) <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1
            c[i + 1] -= 1
            c[i + 2] -= 1
            ok = melds_ok(c)
            c[i] += 1
            c[i + 1] += 1
            c[i + 2] += 1
            if ok:
                return True
        return False

    # 枚举雀头，剩下的是否正好凑成 need_sets 个面子
    def sets_ok(c: list[int], left: int) -> bool:
        if left == 0:
            return not any(c)
        i = next((k for k in range(N_TYPES) if c[k]), None)
        if i is None:
            return False
        if c[i] >= 3:
            c[i] -= 3
            if sets_ok(c, left - 1):
                c[i] += 3
                return True
            c[i] += 3
        if not _is_honor(i) and _suit_pos(i) <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            ok = sets_ok(c, left - 1)
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
            if ok:
                return True
        return False

    for i in range(N_TYPES):
        if counts[i] >= 2:
            counts[i] -= 2
            ok = sets_ok(counts, need_sets)
            counts[i] += 2
            if ok:
                return True
    return False


def sums_ok(counts: list[int]) -> bool:
    return True


# ---------------------------------------------------------------------------
# 有效牌
# ---------------------------------------------------------------------------


def useful_tiles(tiles: list[str] | tuple[str, ...], sanma: bool = True, melds: int = 0) -> list[str]:
    """摸到之后能让向听数变小的牌（进张）。

    ``sanma=True`` 时只考虑三麻实际存在的牌（万子只有 1m/9m）。
    """
    F = _fast()
    if F is not None:
        try:
            counts = counts_from_tiles(tiles)
            return [_BASE_IDS[i] for i in F.useful_tiles(counts, melds, sanma=sanma)]
        except Exception:
            pass          # 快实现出岔子就老实走下面的慢实现，结果不会错
    base = shanten(tiles, melds=melds)
    counts = counts_from_tiles(tiles)
    out: list[str] = []
    for tid, idx in _INDEX.items():
        if counts[idx] >= 4:
            continue
        if sanma:
            t = parse_tile(tid)
            if t.suit == "m" and t.num not in (1, 9):
                continue
        trial = list(tiles) + [tid]
        if shanten(trial, melds=melds) < base:
            out.append(tid)
    return out


def describe(tiles: list[str] | tuple[str, ...], sanma: bool = True) -> str:
    """给人看的一句话判断。"""
    from .tiles import sort_tiles

    n = shanten(tiles)
    if n < 0:
        return "已经和牌"
    hand_size = len([t for t in tiles])
    if hand_size not in (13, 14):
        return f"{hand_size} 张牌（不是 13/14 张，未做判断）"
    if n == 0:
        waits = useful_tiles(tiles, sanma)
        return "听牌，等：" + (" ".join(sort_tiles(waits)) if waits else "（无有效牌）")
    if n >= 6:
        return f"{n} 向听"
    ups = useful_tiles(tiles, sanma)
    head = f"{n} 向听"
    if ups:
        head += "，进张：" + " ".join(sort_tiles(ups))
    return head


# ---------------------------------------------------------------------------
# 统一判断入口（GUI 用）
# ---------------------------------------------------------------------------


def best_discards(tiles: list[str] | tuple[str, ...], sanma: bool = True, melds: int = 0) -> list[str]:
    """14 张时：打掉哪些牌能让向听数最好（通常就是「打这张听牌」）。"""
    tiles = list(tiles)
    if len(tiles) + 3 * melds <= 13:
        return []
    base = shanten(tiles, melds=melds)
    out: list[str] = []
    for tid in sorted(set(tiles)):
        rest = tiles[:]
        rest.remove(tid)
        if shanten(rest, melds=melds) == base:
            out.append(tid)
    return out


def discard_options(
    tiles: list[str] | tuple[str, ...],
    sanma: bool = True,
    melds: int = 0,
    seen: dict[str, int] | None = None,
) -> list[dict]:
    """14 张（等效）时：手里**每一种牌**打掉之后的「收益」。

    收益用**进张数**衡量 —— 打掉这张之后，能让向听数变小的牌还剩几枚。
    这是牌效里最硬的一个数：只看手牌和场上已见的牌，不含任何猜测。所以它
    既是独立的一份参考，也是**神经网络用不上时的兜底**（碰/杠过、权重缺失）。

    ``seen`` 是「除我手牌之外」的可见张数（四家牌河 + 宝牌指示牌）；
    传了它算出来的才是**剩余枚数**，不传就按理论的 4 张算。
    """
    from .tiles import sort_tiles

    tiles = list(tiles)
    if len(tiles) + 3 * melds != 14:
        return []

    F = _fast()
    if F is not None:
        try:
            return _discard_options_fast(F, tiles, sanma, melds, seen, sort_tiles)
        except Exception:
            pass          # 快实现出岔子就退回慢实现

    seen = {normalize_tile_id(k): v for k, v in (seen or {}).items()}
    base = shanten(tiles, melds=melds)

    out: list[dict] = []
    for tid in sorted(set(tiles)):
        rest = tiles[:]
        rest.remove(tid)
        after = shanten(rest, melds=melds)
        ups = useful_tiles(rest, sanma=sanma, melds=melds)
        mine = counts_from_tiles(rest)
        total = 0
        alive: list[str] = []
        for u in ups:
            idx = _INDEX.get(normalize_tile_id(u))
            have = mine[idx] if idx is not None else 0
            left = 4 - seen.get(normalize_tile_id(u), 0) - have
            if left <= 0:
                continue                   # 已经绝张：算「能改善」但一张都摸不到
            total += left
            alive.append(u)
        out.append({
            "tile": tid,
            "shanten": after,
            "loss": after - base,          # >0 说明这一打会退向听
            "ukeire": total,               # 剩余进张枚数
            "kinds": len(alive),
            "tiles": sort_tiles(alive),
        })
    # 先看掉不掉向听，再看进张多少
    out.sort(key=lambda r: (r["loss"], -r["ukeire"], r["tile"]))
    return out


def _discard_options_fast(F, tiles, sanma, melds, seen, sort_tiles) -> list[dict]:
    """同一件事的快路径：把「已见张数」做成一张表，逐张候选查表即可。

    慢实现每张候选都要重新枚举一遍所有拆解方式，14 张手牌要 1.3 秒；
    这里每张候选只做一次进张查询（0.2 ms 量级）。
    """
    counts = counts_from_tiles(tiles)
    #: 场上已见（不含我手牌）
    outside = [0] * N_TYPES
    for k, v in (seen or {}).items():
        idx = _INDEX.get(normalize_tile_id(k))
        if idx is not None:
            outside[idx] += int(v)

    base = F.shanten(counts, melds)
    out: list[dict] = []
    for tid in sorted(set(tiles)):
        idx = _INDEX[normalize_tile_id(tid)]
        rest = counts[:]
        rest[idx] -= 1
        # 进张看的是「剩余枚数」，所以可见张数要把自己剩下的手牌也算进去
        visible = [outside[i] + rest[i] for i in range(N_TYPES)]
        ups = F.ukeire(rest, visible, melds, sanma=sanma)
        after = F.shanten(rest, melds)
        out.append({
            "tile": tid,
            "shanten": after,
            "loss": after - base,
            "ukeire": sum(ups.values()),
            "kinds": len(ups),
            "tiles": sort_tiles(_BASE_IDS[i] for i in ups),
        })
    out.sort(key=lambda r: (r["loss"], -r["ukeire"], r["tile"]))
    return out


def analyze(tiles: list[str] | tuple[str, ...], sanma: bool = True, melds: int = 0) -> dict:
    """把「手牌 -> 判断」这一步收成一个函数，界面直接拿来显示。

    返回 ``{"text", "shanten", "useful", "discards", "waiting"}``：

    * 13 张：``useful`` 是进张；0 向听时 ``waiting`` 是听的牌
    * 14 张（刚摸到）：``discards`` 是建议打掉的牌，打完之后看 ``useful``
    """
    from .tiles import sort_tiles

    tiles = list(tiles)
    n = len(tiles)
    result: dict = {
        "shanten": None, "text": "", "useful": [], "discards": [], "waiting": [],
        "hand_size": n, "melds": melds,
    }
    # 有副露时，手牌行本来就短 3 的倍数 —— 判断标准要按「等效张数」算
    if n + 3 * melds not in (13, 14):
        result["text"] = (
            f"{n} 张牌 + {melds} 副露（等效 {n + 3 * melds} 张）—— 这个张数凑不出手牌，"
            "所以没做判断。正常应该是 13（别人回合）或 14（刚摸完）；"
            "对不上的话通常是副露没读全，或者拔北被错算成了副露（拔北不占格数）"
        )
        return result

    s = shanten(tiles, melds=melds)
    result["shanten"] = s
    if s < 0:
        result["text"] = "已经和牌"
        return result

    if n + 3 * melds == 14:
        drops = best_discards(tiles, sanma, melds=melds)
        result["discards"] = drops
        if drops:
            rest = tiles[:]
            rest.remove(drops[0])
            follow = _shanten_13(counts_from_tiles(rest), melds)
            ups = useful_tiles(rest, sanma, melds=melds)
            result["useful"] = ups
            tail = ""
            if follow == 0 and ups:
                tail = f"，之后听：{' '.join(sort_tiles(ups))}"
            elif ups:
                tail = f"，之后进张：{' '.join(sort_tiles(ups))}"
            result["text"] = f"{s} 向听　建议打：{' '.join(sort_tiles(drops))}{tail}"
        else:
            result["text"] = f"{s} 向听"
        return result

    ups = useful_tiles(tiles, sanma, melds=melds)
    result["useful"] = ups
    if s == 0:
        result["waiting"] = ups
        result["text"] = "听牌，等：" + (" ".join(sort_tiles(ups)) if ups else "（无有效牌）")
    else:
        result["text"] = f"{s} 向听"
        if ups:
            shown = sort_tiles(ups)
            more = "" if len(shown) <= 12 else f" 等 {len(shown)} 种"
            result["text"] += "，进张：" + " ".join(shown[:12]) + more
    return result
