"""牌的表示与宝牌推导。

牌 id 采用日麻通用记法：
    1m..9m  万子 (man)
    1p..9p  筒子 (pin)
    1s..9s  索子 (sou)
    1z..7z  字牌 (honor)：1z=东 2z=南 3z=西 4z=北 5z=白 6z=发 7z=中
    0m/0p/0s 赤5（红宝牌），仅在需要区分赤牌时使用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

SUIT_ORDER = ("m", "p", "s", "z")
SUIT_CN = {"m": "万", "p": "筒", "s": "索", "z": "字"}

HONOR_IDS = {1: "东", 2: "南", 3: "西", 4: "北", 5: "白", 6: "发", 7: "中"}
HONOR_NAMES_EN = {1: "E", 2: "S", 3: "W", 4: "N", 5: "P", 6: "F", 7: "C"}

#: 34 种普通牌（不含赤牌），顺序即显示顺序
TILE_IDS: tuple[str, ...] = tuple(
    f"{n}{s}" for s in SUIT_ORDER for n in range(1, (8 if s == "z" else 10))
)

#: 赤5
RED_FIVE_IDS: tuple[str, ...] = ("0m", "0p", "0s")

ALL_TILE_IDS: tuple[str, ...] = TILE_IDS + RED_FIVE_IDS

#: 幺九牌（含字牌）
TERMINAL_HONOR_IDS: tuple[str, ...] = tuple(
    t for t in TILE_IDS if t[1] == "z" or t[0] in ("1", "9")
)


def is_valid_tile_id(tid: str) -> bool:
    return tid in ALL_TILE_IDS


def normalize_tile_id(tid: str) -> str:
    """把赤5映射回普通5（用于向听/和牌计算）。"""
    if tid in RED_FIVE_IDS:
        return "5" + tid[1]
    return tid


@dataclass(frozen=True)
class Tile:
    """一张牌。``red`` 为 True 表示赤5。"""

    num: int  # 1-9；字牌 1-7；赤5 的 num 仍是 5
    suit: str  # m/p/s/z
    red: bool = False

    def __post_init__(self) -> None:
        if self.suit not in SUIT_ORDER:
            raise ValueError(f"未知花色: {self.suit}")
        top = 7 if self.suit == "z" else 9
        if not 1 <= self.num <= top:
            raise ValueError(f"非法点数: {self.num}{self.suit}")

    @property
    def id(self) -> str:
        return ("0" if self.red else str(self.num)) + self.suit

    @property
    def is_honor(self) -> bool:
        return self.suit == "z"

    @property
    def is_terminal(self) -> bool:
        return self.suit != "z" and self.num in (1, 9)

    @property
    def is_terminal_honor(self) -> bool:
        return self.is_honor or self.is_terminal

    @property
    def is_simple(self) -> bool:
        return not self.is_terminal_honor

    @property
    def base_id(self) -> str:
        """去掉赤牌属性的 id。"""
        return str(self.num) + self.suit

    @property
    def label_cn(self) -> str:
        if self.suit == "z":
            return HONOR_IDS[self.num]
        num_cn = "一二三四五六七八九"[self.num - 1]
        return f"{num_cn}{SUIT_CN[self.suit]}"

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return ("赤" if self.red else "") + self.label_cn


def parse_tile(tid: str) -> Tile:
    """把 ``"0p"`` / ``"5p"`` / ``"7z"`` 解析成 :class:`Tile`。"""
    tid = tid.strip().lower()
    if len(tid) != 2 or tid[1] not in SUIT_ORDER or not tid[0].isdigit():
        raise ValueError(f"无法解析的牌: {tid!r}")
    suit = tid[1]
    num = int(tid[0])
    if num == 0:
        if suit == "z":
            raise ValueError("字牌没有赤牌")
        return Tile(5, suit, red=True)
    return Tile(num, suit)


def tile_id(num: int, suit: str, red: bool = False) -> str:
    return Tile(num, suit, red).id


def dora_from_indicator(indicator: str) -> str:
    """由宝牌指示牌推出宝牌。

    数牌 9 -> 1 循环；风牌 北(4z) -> 东(1z)；三元牌 中(7z) -> 白(5z)。
    赤5 指示牌按普通 5 处理。
    """
    t = parse_tile(indicator)
    if t.suit == "z":
        # 风牌：東→南→西→北→東；三元牌：白→發→中→白
        if t.num == 4:
            nxt = 1
        elif t.num == 7:
            nxt = 5
        else:
            nxt = t.num + 1
    else:
        nxt = 1 if t.num == 9 else t.num + 1
    return f"{nxt}{t.suit}"


def dora_set(indicators: Iterable[str]) -> list[str]:
    """多个指示牌 -> 对应的宝牌列表（未翻开的槽位请先过滤掉）。"""
    return [dora_from_indicator(i) for i in indicators if i]


def to_counts(tiles: Iterable[str]) -> dict[str, int]:
    """统计每种牌的张数（赤5并入普通5）。"""
    counts: dict[str, int] = {}
    for tid in tiles:
        key = normalize_tile_id(tid)
        counts[key] = counts.get(key, 0) + 1
    return counts


def sort_tiles(tiles: Iterable[str]) -> list[str]:
    order = {tid: i for i, tid in enumerate(ALL_TILE_IDS)}

    def key(tid: str) -> tuple[int, int]:
        t = parse_tile(tid)
        return (order.get(t.base_id, 99), 0 if t.red else 1)

    return sorted(tiles, key=key)


def tiles_to_mpsz(tiles: Iterable[str]) -> str:
    """输出 ``123m456p77z`` 形式的紧凑记法。"""
    counts = to_counts(tiles)
    out: list[str] = []
    for suit in SUIT_ORDER:
        nums = sorted(int(k[0]) for k in counts if k[1] == suit)
        if not nums:
            continue
        digits = "".join(
            str(n) * counts[f"{n}{suit}"] for n in nums
        )
        out.append(digits + suit)
    return "".join(out)


def iter_all_tiles() -> Iterator[str]:
    return iter(TILE_IDS)


def mpsz_to_tiles(text: str) -> list[str]:
    """把 ``123m456p77z`` / ``123m 456p 77z`` 这种紧凑记法拆成牌 id 列表。

    数字 0 表示赤5（``0m`` -> 赤5万）。
    """
    out: list[str] = []
    digits: list[str] = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        elif ch in SUIT_ORDER:
            for d in digits:
                out.append(d + ch)
            digits = []
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"无法解析的字符: {ch!r}（在 {text!r} 中）")
    if digits:
        raise ValueError(f"结尾的数字没有花色: {text!r}")
    return out
