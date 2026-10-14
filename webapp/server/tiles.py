"""牌张工具：引擎的牌下标 <-> 牌 id（``1m`` / ``0p`` / ``7z``）。

引擎内部用 0~33 的下标，前端用 ``potatomaj.tiles`` 的 ``1m`` 记法。
这一层只做转换，**不含任何规则**。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from aimodel import sanma as S
from potatomaj.tiles import ALL_TILE_IDS, parse_tile

__all__ = [
    "index_to_id",
    "id_to_index",
    "tile_cn",
    "hand_list",
    "red_list",
    "meld_json",
    "sort_key",
    "split_hand",
    "tile_asset",
]

#: 34 种牌型 -> id（不含赤 5）
_ID_BY_INDEX = tuple(ALL_TILE_IDS[:34])
#: id -> 下标（赤 5 映射到普通 5）
_INDEX_BY_ID = {tid: i for i, tid in enumerate(_ID_BY_INDEX)}
for _red in ("0m", "0p", "0s"):
    _INDEX_BY_ID[_red] = _INDEX_BY_ID["5" + _red[1]]


def index_to_id(index: int, red: bool = False) -> str:
    """牌下标 -> id。``red`` 为真且是 5 的时候返回赤 5 的 id。"""
    if index < 0 or index >= 34:
        raise ValueError(f"牌下标越界: {index}")
    if red and index in S.AKA_TYPES:
        return "0" + _ID_BY_INDEX[index][1]
    return _ID_BY_INDEX[index]


def id_to_index(tid: str) -> int:
    return _INDEX_BY_ID[tid.strip().lower()]


def tile_cn(index: int) -> str:
    return S.tile_cn(index)


def hand_list(counts: list[int], reds: list[int]) -> list[str]:
    """``hand`` / ``red`` 计数数组 -> 手牌 id 列表（赤 5 排在普通牌前面）。

    这是**手牌的唯一权威表示**：某张牌有几张、其中几张是赤 5 全部由引擎说，
    前端只负责画。
    """
    out: list[str] = []
    for t in range(34):
        n = counts[t]
        if n <= 0:
            continue
        r = min(reds[t], n) if reds else 0
        suffix = _ID_BY_INDEX[t][1]
        out.extend(["0" + suffix] * r)
        out.extend([_ID_BY_INDEX[t]] * (n - r))
    return out


def red_list(counts: list[int], reds: list[int]) -> list[str]:
    """手牌里的赤 5 是哪些（给前端加高亮用）。"""
    out: list[str] = []
    for t in range(34):
        r = min(reds[t], counts[t]) if reds else 0
        if r:
            out.extend(["0" + _ID_BY_INDEX[t][1]] * r)
    return out


def meld_json(meld: S.Meld, viewer: int = -1) -> dict:
    """一组副露 -> 前端可画的结构。

    ``tiles`` 已经按摆放顺序排好（吃是顺子、杠是 4 张）；
    ``called`` 是其中哪一张来自别家（前端要横放 / 侧放）。
    """
    tiles = list(meld.tiles)
    kind = meld.kind
    # 赤 5 的位置：引擎只给了「这组里有几张赤」，放在第一张同名牌上。
    red_positions: list[int] = []
    if meld.aka:
        placed = 0
        for i, t in enumerate(tiles):
            if t in S.AKA_TYPES and placed < meld.aka:
                red_positions.append(i)
                placed += 1
    ids = [index_to_id(t, red=(i in red_positions)) for i, t in enumerate(tiles)]

    if kind == "chi":
        # 吃：哪一张是别人打的。引擎记的是 from_player，这里用最小牌的位置兜底。
        called = tiles.index(meld.tile) if meld.tile in tiles else 0
    elif kind == "ankan":
        called = -1  # 暗杠四张全是自己的 -> 前端把中间两张扣过来
    else:
        called = 0   # 碰 / 明杠 / 加杠：第一张是别人打的
    return {
        "kind": kind,
        "tile": index_to_id(meld.tile),
        "tiles": ids,
        "red": red_positions,
        "from": meld.from_player,
        "aka": meld.aka,
        "called": called,
        "label": str(meld),
    }


def sort_key(tid: str) -> tuple[int, int]:
    """手牌排序：按牌型顺序，赤 5 排在普通 5 前面。"""
    t = parse_tile(tid)
    return (_INDEX_BY_ID[t.base_id], 0 if t.red else 1)


def split_hand(hand: list[str], drawn: str | None) -> tuple[list[str], str | None]:
    """把手牌分成「理好的 13 张 + 刚摸的那张」。

    理牌顺序**由服务端定下来** —— 否则每个客户端理出来的顺序可能不一样。
    """
    rest = list(hand)
    if drawn is None:
        return sorted(rest, key=sort_key), None
    want = "5" + drawn[1] if drawn.startswith("0") else drawn
    for i, t in enumerate(rest):
        if t == drawn:
            rest.pop(i)
            break
    else:
        for i, t in enumerate(rest):
            if t == want:
                rest.pop(i)
                break
    return sorted(rest, key=sort_key), drawn


@lru_cache(maxsize=64)
def tile_asset(tid: str, tiles_dir: str) -> str | None:
    """牌 id -> 静态资源的相对路径（优先 png，退回 svg）。"""
    d = Path(tiles_dir)
    for ext in (".png", ".svg"):
        p = d / f"{tid}{ext}"
        if p.exists():
            return f"tiles/{tid}{ext}"
    return None
