"""MJAI 牌谱格式的读写。

[MJAI](https://github.com/Equim-chan/mjai) 是麻将 AI 之间的通用牌谱格式 ——
Mortal、akochan、mjai-reviewer 都用它，一行一个 JSON 事件。走这个格式的好处：
**将来不管拿到的是天凤牌谱、雀魂牌谱、还是别的工具产出的数据，都只需要一个转换器**。

牌的写法（MJAI 官方约定）::

    数牌   1m~9m / 1p~9p / 1s~9s
    赤 5   0m / 0p / 0s          ← 和本项目内部的下标约定一致
    字牌   東=E 南=S 西=W 北=N 白=P 發=F 中=C

（注意：MJAI 用字母表示字牌，本项目内部用 ``1z~7z``。转换在
:func:`tile_to_mjai` / :func:`tile_from_mjai`。）

用法::

    from aimodel import mjai
    events = mjai.read("game.json")          # 读
    mjai.write("out.json", events)           # 写
    events = mjai.export(actions, names)     # 把引擎的动作流导成 MJAI
    hands = list(mjai.iter_kyoku(events))    # 按局切出来
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from . import sanma as S

#: 字牌下标 -> MJAI 字母
HONOR_MJAI = {S.EAST: "E", S.SOUTH: "S", S.WEST: "W", S.NORTH: "N",
              S.HAKU: "P", S.HATSU: "F", S.CHUN: "C"}
MJAI_HONOR = {v: k for k, v in HONOR_MJAI.items()}

#: 风牌（场风/自风）用的名字
WIND_MJAI = {"东": "E", "南": "S", "西": "W", "北": "N",
             S.EAST: "E", S.SOUTH: "S", S.WEST: "W", S.NORTH: "N"}


# ---------------------------------------------------------------------------
# 牌名转换
# ---------------------------------------------------------------------------
def tile_to_mjai(tile: int, red: bool = False) -> str:
    """牌型下标 -> MJAI 牌名。``red=True`` 时 5m/5p/5s 写成 ``0m/0p/0s``。"""
    if tile >= 27:
        return HONOR_MJAI[tile]
    suit = "mps"[tile // 9]
    n = tile % 9 + 1
    if red and n == 5:
        return "0" + suit
    return f"{n}{suit}"


def tile_from_mjai(name: str) -> tuple[int, bool]:
    """MJAI 牌名 -> (牌型下标, 是否赤)。

    赤五的**两种写法都认**（2026-10-21 加，之前只认第一种）：

    * ``0m`` / ``0p`` / ``0s``   —— 本项目内部约定（那 40,000 局老数据用的是这个）
    * ``5mr`` / ``5pr`` / ``5sr`` —— **标准 MJAI**（Mortal / riichienv /
      以及新转出来的 5,717 局鳳凰数据用的是这个）

    两种写法返回同一个结果 ``(base + 4, True)``。

    ⚠️ 为什么之前会崩：新转换器 `tools/tenhou_xml_to_mjai.py` 按**标准**写 ``5mr``，
    而这里只认 ``0m`` -> `ValueError: 认不出的牌名: '5mr'`，
    整个抽取在**第一个含赤五的手牌**上就退出了 ✗
    这不是转换器写错，是这里的读入端不够宽 ✓
    """
    name = name.strip()
    if not name:
        raise ValueError("空牌名")
    if name in MJAI_HONOR:
        return MJAI_HONOR[name], False
    # 标准 MJAI 的赤五后缀：5m + r
    red_suffix = len(name) == 3 and name[-1] in "rR"
    if red_suffix:
        name = name[:-1]
    n, suit = name[:-1], name[-1].lower()
    if suit not in "mps":
        raise ValueError(f"认不出的牌名: {name!r}")
    base = "mps".index(suit) * 9
    if n == "0":                      # 赤 5（本项目写法）
        return base + 4, True
    try:
        v = int(n)
    except ValueError:
        raise ValueError(f"认不出的牌名: {name!r}") from None
    if not 1 <= v <= 9:
        raise ValueError(f"认不出的牌名: {name!r}")
    if red_suffix:
        if v != 5:
            raise ValueError(f"只有 5 号牌能带赤后缀: {name!r}")
        return base + 4, True
    return base + v - 1, False


def hand_to_mjai(counts: Iterable[int], reds: Iterable[int] | None = None) -> list[str]:
    """计数数组 -> MJAI 牌名列表。赤 5 会写在前面（MJAI 的习惯写法）。"""
    counts = list(counts)
    reds = list(reds) if reds is not None else [0] * 34
    out: list[str] = []
    for t in range(34):
        n_red = min(reds[t], counts[t])
        out.extend([tile_to_mjai(t, red=True)] * n_red)
        out.extend([tile_to_mjai(t)] * (counts[t] - n_red))
    return out


def meld_to_mjai(m: S.Meld) -> dict:
    """本项目的 :class:`~aimodel.sanma.Meld` -> MJAI 的副露描述。"""
    if m.is_chi:
        # MJAI 规定吃事件的 ``pai`` 是**被吃的那张**（打出来的那张），
        # 不一定等于顺子里最小的那张。
        called = m.called if getattr(m, "called", -1) >= 0 else m.tile + 2
        rest = [t for t in (m.tile, m.tile + 1, m.tile + 2) if t != called]
        return {"type": "chi", "pai": tile_to_mjai(called),
                "consumed": [tile_to_mjai(t) for t in rest]}
    kind = {"pon": "pon", "minkan": "kan", "ankan": "ankan", "kakan": "kakan"}[m.kind]
    n = 4 if m.is_kan else 3
    return {"type": kind, "pai": tile_to_mjai(m.tile),
            "consumed": [tile_to_mjai(m.tile)] * n}


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------
def read(path: str | Path) -> list[dict]:
    """读一个 MJAI 文件（每行一个 JSON）。空行、``#`` 注释会跳过。"""
    events: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        events.append(json.loads(line))
    return events


def write(path: str | Path, events: Iterable[dict]) -> None:
    lines = [json.dumps(e, ensure_ascii=False) for e in events]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 从引擎的动作流导出
# ---------------------------------------------------------------------------
def export(
    actions: Iterable[dict],
    names: list[str] | None = None,
    aka: bool = True,
    players: int = 3,
) -> list[dict]:
    """把 ``Game.on_action`` 收到的事件流导成 MJAI 牌谱。

    用法::

        acts = []
        g = Game(agents, seed=1)
        g.on_action = acts.append
        g.play()
        mjai.write("game.json", mjai.export(acts, names=["A", "B", "C"]))
    """
    n = players
    names = names or [f"P{i}" for i in range(n)]
    out: list[dict] = [{"type": "start_game", "names": names, "aka_flag": aka}]
    kyoku = 0
    for a in actions:
        k = a["kind"]
        if k == "start_hand":
            kyoku += 1
            tehais = [hand_to_mjai(a["hands"][p], a["reds"][p]) for p in range(n)]
            out.append({
                "type": "start_kyoku",
                "bakaze": WIND_MJAI.get(a["round_wind"], "E"),
                "kyoku": kyoku,
                "honba": 0,
                "kyotaku": a.get("sticks", 0),
                "oya": a["dealer"],
                "scores": list(a["scores"]),
                "tehais": tehais,
            })
            for d in a["dora"]:
                out.append({"type": "dora", "dora_marker": tile_to_mjai(d)})
        elif k == "draw":
            out.append({"type": "tsumo", "actor": a["player"],
                        "pai": tile_to_mjai(a["tile"], bool(a.get("red")))})
        elif k == "discard":
            out.append({"type": "dahai", "actor": a["player"],
                        "pai": tile_to_mjai(a["tile"]),
                        "tsumogiri": bool(a.get("tsumogiri"))})
        elif k == "riichi":
            out.append({"type": "reach", "actor": a["player"]})
            out.append({"type": "reach_accepted", "actor": a["player"]})
        elif k == "kita":
            out.append({"type": "kita", "actor": a["player"]})       # 三麻专用
        elif k == "call":
            c = a["call"]
            ev = {"type": {"chi": "chi", "pon": "pon", "kan": "kan",
                           "ankan": "ankan", "kakan": "kakan"}[c],
                  "actor": a["player"]}
            if c in ("chi", "pon", "kan"):
                ev["target"] = a["from_player"]
            # 吃的 pai 是「被吃的牌」，不是顺子起点
            ev["pai"] = tile_to_mjai(a.get("called", a["tile"]))
            ev["consumed"] = [tile_to_mjai(t) for t in a.get("consumed", [])]
            out.append(ev)
        elif k == "win":
            ev = {"type": "hora", "actor": a["winner"],
                  "pai": tile_to_mjai(a["tile"]),
                  "hora_tsumo": bool(a.get("tsumo"))}
            if not a.get("tsumo"):
                ev["target"] = a["loser"]
            out.append(ev)
        elif k == "ryukyoku":
            out.append({"type": "ryukyoku", "deltas": list(a.get("deltas", [])),
                        "tenpai": list(a.get("tenpai", []))})
        if k in ("win", "ryukyoku"):
            out.append({"type": "end_kyoku"})
    out.append({"type": "end_game"})
    return out


# ---------------------------------------------------------------------------
# 按局切开
# ---------------------------------------------------------------------------
def iter_kyoku(events: Iterable[dict]) -> Iterator[list[dict]]:
    """把牌谱按「局」切开，每局一个事件列表（从 start_kyoku 到 end_kyoku）。

    ⚠️ **不能只认 ``end_kyoku`` 当结束标志** —— 真实牌谱的导出器不一定每局都给
    这个事件（天鳳牌谱转 MJAI 的常见实现就不给）。只认它的话，多局会被并成一局、
    大部分局直接丢掉（实测：497 局里只切出 8 局）。

    所以规则是：**遇到下一个 ``start_kyoku`` 就把上一局收掉**；
    文件末尾剩的那一局也要收。``end_kyoku`` 有就切、没有也不影响。
    """
    cur: list[dict] = []
    for e in events:
        t = e.get("type")
        if t == "start_kyoku":
            if cur:                      # 上一局没给 end_kyoku 也要收
                yield cur
            cur = [e]
        elif cur:
            cur.append(e)
            if t == "end_kyoku":
                yield cur
                cur = []
    if cur:                              # 文件结尾的最后一局
        yield cur


def hora_of(kyoku: Iterable[dict]) -> dict | None:
    """一局里的和牌事件（没有就是流局）。"""
    for e in kyoku:
        if e.get("type") == "hora":
            return e
    return None


def ryukyoku_of(kyoku: Iterable[dict]) -> dict | None:
    for e in kyoku:
        if e.get("type") == "ryukyoku":
            return e
    return None


__all__ = [
    "HONOR_MJAI", "MJAI_HONOR", "WIND_MJAI",
    "tile_to_mjai", "tile_from_mjai", "hand_to_mjai", "meld_to_mjai",
    "read", "write", "export", "iter_kyoku", "hora_of", "ryukyoku_of",
]
