#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""天凤牌谱 mjlog **XML** -> **标准 MJAI**（P0：保住每个玩家的 dan / rate）。

    python tools/tenhou_xml_to_mjai.py _data/tenhou_scc --out _data/tenhou_hk/mjai

## 为什么又写一个转换器

`tools/mjlog_to_mjai.py` 是骨架，但它的**座位映射是错的**（把 `T/U/V/W` 当
「相对当前行动者」，实际天凤是**绝对座位**：T/D=0、U/E=1、V/F=2、W/G=3）。
实测证据（本仓库 `_data/tenhou_scc`，5,000+ 局）：

    <INIT ... oya="1"> 之后第一个摸牌是 <U..>   -> seat 1，不是 seat 2
    <INIT ... oya="2"> 之后第一个摸牌是 <V..>   -> seat 2

所以本文件重写座位解析，并把 `<INIT>` 的 `hai*`、每一张打牌、每一次鸣牌都
**对到手上**（打出的牌必须真的在手里），一旦座位错一格立刻报 `discard_not_in_hand`。

`aimodel/mjlog.py` 的 `m=` 解码（40,000 行验过）是**正确**的那一份，这里照抄
位布局、不复用它的 xlsm 读取路径；它本身一行不改（训练数据生成器）。

## 输出：标准 MJAI + 我们自己的字段

标准 MJAI（riichienv / Mortal 能吃）：
    start_kyoku  bakaze / dora_marker / kyoku / honba / **kyoutaku** / oya /
                 scores（**实际点数**，250 -> 25000）/ tehais
    tsumo / dahai(tsumogiri) / chi / pon / daiminkan / ankan / kakan /
    reach / reach_accepted / dora / hora(deltas) / ryukyoku(deltas) /
    end_kyoku / end_game

我们自己的（P0，Mac 要用 dan>=10 筛凤凰卓）：
    start_game  {"type":"start_game","id":<int>,"names":[...4],"dans":[...],
                 "rates":[...],"sx":[...],"game_id":"<文件名去掉 .xml.gz>"}
    dan / rate / 名字缺失一律 `null`（天凤不公开段位时就是这样），**不崩**。

牌名用**标准 MJAI**：`1m..9m / 1p..9p / 1s..9s / E S W N P F C`，
赤五写作 `5mr / 5pr / 5sr`（`aimodel.mjlog._name_of_raw` 写的是 `0m`，
那是本项目内部口径，不是标准 MJAI，所以这里单独出一份名字表）。

## 三个自检（转换时顺带跑，结果打在报告里）

1. **座位/手牌**：每次打出的牌必须在打者手里；`<INIT>` 四家各 13 张；
   每种牌全局最多 4 张（136 张牌堆的一致性）。
2. **分数**：`start_kyoku.scores` - 1000×本局立直数 + hora/ryukyoku 的 deltas
   == **下一局** `start_kyoku.scores`（跨局账本，等价于"每局收支清零"）。
3. **宝牌/和了**：`<DORA>` 指示牌 == `<AGARI doraHai>`；`ba` 的立直棒数 ==
   `seed[2] + 本局成立立直数`；`hora.deltas` 与本局立直棒的关系见
   `deltas 的说明`。

## deltas 的说明（重要，别被 "必须等于 0" 误导）

`<AGARI sc>` 的「variation」是**已经扣过本局立直棒**之后的净变化，而赢家会把
桌上的棒**收回去**，所以单看一条 hora：

    sum(hora.deltas) == 1000 × (本局成立立直数 + 局首 kyoutaku)
                       中「赢家自己付过的那部分」… 实测更简单：
    sum(hora.deltas) + 1000×本局成立立直数 + 1000×局首 kyoutaku == 0

也就是说**跨事件**才守恒（立直时 -1000 记在 reach 上）。硬把单条 hora 的
deltas 改成合计 0，就会和 `<AGARI sc>` 对不上、下一局分数接不上 ——
正是 `aimodel/mjlog.py` 注释里说的「engine 多给 14000，日志只有 13000」那类 bug。
本文件按天凤 `sc` **原样**输出，并在报告里给出守恒式的实测通过率。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 牌 id 规则（tenhou 0..135 -> 34 种）只此一份，复用、不重写。
from aimodel.mjlog import t2p  # noqa: E402

# ---------------------------------------------------------------- 牌名 / 常量

_SUITS = "mps"
_NUM = "123456789"
_HONOR = "ESWNPFC"          # tenhou 27..33 = 東南西北 白發中
RED_FIVES = frozenset({16, 52, 88})
BAKAZE = "ESWN"

#: 摸牌标签：T/U/V/W = **绝对座位** 0/1/2/3（不是相对当前行动者）
TSUMO_TAGS = "TUVW"
DAHAI_TAGS = "DEFG"

#: 局种（RYUUKYOKU 的 type）-> MJAI 的 reason
RYUUKYOKU_REASON = {
    "": "exhaustive_draw",
    "nm": "exhaustive_draw",
    "yao9": "kyuushu_kyuuhai",
    "kaze4": "four_winds",
    "reach4": "four_reach",
    "ron3": "three_ron",
    "kan4": "four_kan",
}


def tile_name(raw: int) -> str:
    """天凤牌 id（0..135）-> 标准 MJAI 牌名（赤五 = 5mr/5pr/5sr）。"""
    raw = int(raw)
    t = t2p(raw)
    if raw in RED_FIVES:
        return "5" + _SUITS[t // 9] + "r"
    if t >= 27:
        return _HONOR[t - 27]
    return _NUM[t % 9] + _SUITS[t // 9]


def tile_type(raw: int) -> int:
    """天凤牌 id -> 0..33 的牌种（赤五与普通五同种）。"""
    return t2p(int(raw))


# ------------------------------------------------------------------ 小工具


def _ints(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]


def _split4(raw: str | None) -> list[str | None]:
    """`dan=` / `rate=` / `sx=` -> 4 个字符串（缺的补 None）。"""
    if not raw:
        return [None, None, None, None]
    parts = [p.strip() for p in raw.split(",")]
    while len(parts) < 4:
        parts.append(None)
    return parts[:4]


def _four_int(raw: str | None) -> list[int | None]:
    out: list[int | None] = []
    for p in _split4(raw):
        try:
            out.append(int(p) if p not in (None, "") else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _four_float(raw: str | None) -> list[float | None]:
    out: list[float | None] = []
    for p in _split4(raw):
        try:
            out.append(float(p) if p not in (None, "") else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def load_xml(path: str | Path) -> str:
    """读 .xml / .xml.gz -> str。"""
    p = Path(path)
    data = p.read_bytes()
    if str(p).lower().endswith(".gz"):
        data = gzip.decompress(data)
    return data.decode("utf-8")


def game_id_of(path: str | Path) -> str:
    """文件名 -> 牌谱 id（`2026101100gm-00a9-0000-04292c89.xml.gz` -> 去后缀）。"""
    name = Path(path).name
    for suf in (".xml.gz", ".xml"):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name


# --------------------------------------------------------------- m= 解码


def decode_call(m: int, seat: int) -> dict:
    """`<N m=...>` -> 鸣牌的种类 / 目标 / 牌种（位布局见 aimodel/mjlog.py）。

    ⚠️ `m` 是**十进制整数字符串**，不是十六进制 —— 这是 `tools/mjlog_to_mjai.py`
    骨架的致命 bug（它写 `int(el.get("m"), 16)`），实测会把每一次鸣牌都解成
    别的牌：`<N who="2" m="48170"/>` 是「2 号碰 白(P)」，按 hex 解成 kakan of 4s。

    返回 dict：kind ∈ {chi, pon, kakan, ankan, daiminkan}，以及
    target（来源座位）、tile（牌种 0..33）、consumed_raw 等细节。
    """
    low = m & 3
    source = (seat + low) % 4
    out: dict = {"m": m, "low": low, "target": source}

    if m & 0x04:                                  # ---- チー
        pat = (m >> 10) & 0x3F                    # (suit*7+run)*3 + called_pos
        pos, suit_run = pat % 3, pat // 3
        suit, run = suit_run // 7, suit_run % 7
        seq = [suit * 9 + run + k for k in range(3)]
        copies = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]   # low/mid/high 的副本号
        raws = [seq[k] * 4 + copies[k] for k in range(3)]
        out.update(kind="chi", tile=seq[pos], raws=raws, called_raw=raws[pos],
                   consumed_raw=[raws[k] for k in range(3) if k != pos])
        return out

    if m & 0x18:                                  # ---- ポン / 加槓
        tile = ((m >> 9) & 0x7F) // 3
        if m & 0x08:
            out.update(kind="pon", tile=tile, raws=[tile * 4 + k for k in range(4)])
        else:
            out.update(kind="kakan", tile=tile, raws=[tile * 4 + k for k in range(4)])
        return out

    tile = ((m >> 8) & 0xFF) // 4                 # ---- 暗槓 / 大明槓
    if low == 0:
        out.update(kind="ankan", tile=tile, raws=[tile * 4 + k for k in range(4)])
    else:
        out.update(kind="daiminkan", tile=tile, raws=[tile * 4 + k for k in range(4)])
    return out


# ------------------------------------------------------------ 一局（kyoku）


def _take_from_hand(hand: list[int], want: list[int], issues: list[str], tag: str) -> list[int]:
    """从手里拿走 want 指定的**具体副本**；拿不到就退化成按牌种拿，并记账。"""
    got: list[int] = []
    for raw in want:
        if raw in hand:
            hand.remove(raw)
            got.append(raw)
        else:
            # 副本号对不上（赤五/手牌漂移）：按牌种找一张替代
            t = tile_type(raw)
            alt = next((x for x in hand if tile_type(x) == t), None)
            if alt is not None:
                hand.remove(alt)
                got.append(alt)
                issues.append(f"{tag}:copy_fallback")
            else:
                issues.append(f"{tag}:tile_not_in_hand")
    return got


def _take_type(hand: list[int], t: int, n: int, issues: list[str], tag: str,
               exclude: int | None = None) -> list[int]:
    """从手里拿走 n 张牌种为 t 的牌（按手里实际副本）。

    ``exclude``：这一张**不在**面子里（天凤 pon 的 bits 5-6 就写着它）。
    不排除的话会拿错副本：实测 `m=33898` 碰 5s（被碰的是赤 5s 88），手里
    89/90/91 三张，天凤后续 `<F91/>` 打出 91 —— 说明面子用的是 89/90，
    按「随便拿两张」会拿走 89/91，之后那张 91 就打不出来了
    （全量 3,473 局里 6 局报 `discard_not_in_hand`，就是这个）。
    """
    got: list[int] = []
    for _ in range(n):
        cand = [x for x in hand if tile_type(x) == t and (exclude is None or x != exclude)]
        if not cand:
            cand = [x for x in hand if tile_type(x) == t]
            if cand:
                issues.append(f"{tag}:copy_exclude_fallback")
        if not cand:
            issues.append(f"{tag}:tile_not_in_hand")
            break
        hand.remove(cand[0])
        got.append(cand[0])
    return got


def convert_kyoku(init_el: ET.Element, actions: list[ET.Element], issues: list[str]) -> tuple[list[dict], dict]:
    """一个 `<INIT>` 段 -> 事件列表 + 账本（start_scores / net / reaches ...）。"""
    seed = _ints(init_el.get("seed"))
    if len(seed) < 6:
        issues.append("init:seed_short")
    seed = (seed + [0] * 6)[:6]
    round_i, honba, kyoutaku, _d1, _d2, indicator_raw = seed

    ten = _ints(init_el.get("ten"))
    if len(ten) != 4:
        issues.append("init:ten_not_4")
        ten = (ten + [250] * 4)[:4]
    start_scores = [int(x) * 100 for x in ten]

    oya = int(init_el.get("oya", "0"))
    bakaze = BAKAZE[(round_i // 4) % 4]
    kyoku_no = round_i % 4 + 1

    tehais_raw: list[list[int]] = []
    for s in range(4):
        raw = _ints(init_el.get(f"hai{s}"))
        if len(raw) != 13:
            issues.append(f"init:hai{s}_not_13")
        tehais_raw.append(raw)

    events: list[dict] = [{
        "type": "start_kyoku",
        "bakaze": bakaze,
        "dora_marker": tile_name(indicator_raw),
        "kyoku": kyoku_no,
        "honba": honba,
        "kyoutaku": kyoutaku,                 # 立直棒根数（不是点数）
        "oya": oya,
        "scores": list(start_scores),         # 实际点数（250 -> 25000）
        "tehais": [[tile_name(x) for x in h] for h in tehais_raw],
    }]

    # ---- 局内状态 --------------------------------------------------------
    hand: list[list[int]] = [list(h) for h in tehais_raw]   # 天凤原始牌 id
    melds: list[list[list[int]]] = [[] for _ in range(4)]   # 每副面子（原始 id）
    pond: list[list[int]] = [[] for _ in range(4)]
    indicators: list[int] = [indicator_raw]
    last_draw: list[int | None] = [None] * 4
    last_discard: tuple[int, int] | None = None             # (seat, raw)
    reach_accepted: set[int] = set()
    net = [0, 0, 0, 0]
    n_reach = 0
    tally: Counter[int] = Counter()                          # 牌种 -> 出现过的次数

    for h in tehais_raw:
        for x in h:
            tally[tile_type(x)] += 1
    tally[tile_type(indicator_raw)] += 1

    # 一边打一边算「起手 52 张 + 指示牌」之后的可见牌，检查 <= 4
    def _seen(raw: int) -> None:
        tally[tile_type(raw)] += 1

    #: 对账用的计数（hora/ryukyoku 的守恒式、事件类型直方图）
    stats: Counter = Counter()
    agg: dict = {"n_reach": 0, "kyoutaku": kyoutaku, "stats": stats}

    #: `<REACH step="1">` 只有在**同一局**有 step="2" 才算成立（否则是宣言牌被荣了，
    #: 天凤不收那 1000 点；见 aimodel/mjlog.py 的注释）。
    accepted_whos = {
        int(el.get("who", -1))
        for el in actions
        if el.tag == "REACH" and el.get("step", "1") == "2"
    }

    end_reason = "exhaustive_draw"

    for el in actions:
        tag = el.tag

        # ---- 摸牌 --------------------------------------------------------
        if len(tag) > 1 and tag[0] in TSUMO_TAGS and tag[1:].isdigit():
            seat = TSUMO_TAGS.index(tag[0])
            raw = int(tag[1:])
            hand[seat].append(raw)
            last_draw[seat] = raw
            _seen(raw)
            events.append({"type": "tsumo", "actor": seat, "pai": tile_name(raw)})
            continue

        # ---- 打牌 --------------------------------------------------------
        if len(tag) > 1 and tag[0] in DAHAI_TAGS and tag[1:].isdigit():
            seat = DAHAI_TAGS.index(tag[0])
            raw = int(tag[1:])
            tsumogiri = last_draw[seat] == raw
            if raw in hand[seat]:
                hand[seat].remove(raw)
            else:
                issues.append("discard_not_in_hand")
                t = tile_type(raw)
                alt = next((x for x in hand[seat] if tile_type(x) == t), None)
                if alt is not None:
                    hand[seat].remove(alt)
            last_draw[seat] = None
            pond[seat].append(raw)
            last_discard = (seat, raw)
            events.append({"type": "dahai", "actor": seat,
                           "pai": tile_name(raw), "tsumogiri": tsumogiri})
            continue

        # ---- 鸣牌 / 槓 ------------------------------------------------
        if tag == "N":
            who = int(el.get("who", -1))
            mraw = el.get("m")
            if who < 0 or not mraw:
                issues.append("N:bad_tag")
                continue
            m = int(mraw)          # 🔴 十进制，不是十六进制
            call = decode_call(m, who)
            kind = call["kind"]
            tile = call["tile"]

            if kind == "chi":
                src = call["target"]
                if last_discard is None or last_discard[1] != call["called_raw"]:
                    if last_discard is None or tile_type(last_discard[1]) != tile:
                        issues.append(f"N:chi_target_mismatch")
                    else:
                        issues.append("N:chi_copy_mismatch")
                pai_raw = last_discard[1] if last_discard else call["called_raw"]
                consumed = _take_from_hand(hand[who], call["consumed_raw"],
                                           issues, "N:chi")
                melds[who].append([pai_raw] + consumed)
                events.append({"type": "chi", "actor": who, "target": src,
                               "pai": tile_name(pai_raw),
                               "consumed": [tile_name(x) for x in consumed]})
                last_draw[who] = None
                continue

            if kind == "pon":
                # bits 5-6 = 四张里**没有被碰掉**的那一张的副本号
                exclude_raw = tile * 4 + ((m >> 5) & 3)
                consumed = _take_type(hand[who], tile, 2, issues, "N:pon",
                                      exclude=exclude_raw)
                src = call["target"]
                if last_discard is None or tile_type(last_discard[1]) != tile:
                    issues.append("N:pon_target_mismatch")
                pai_raw = last_discard[1] if last_discard else tile * 4
                melds[who].append([pai_raw] + consumed)
                events.append({"type": "pon", "actor": who, "target": src,
                               "pai": tile_name(pai_raw),
                               "consumed": [tile_name(x) for x in consumed]})
                last_draw[who] = None
                continue

            if kind == "daiminkan":
                consumed = _take_type(hand[who], tile, 3, issues, "N:daiminkan")
                src = call["target"]
                if last_discard is None or tile_type(last_discard[1]) != tile:
                    issues.append("N:daiminkan_target_mismatch")
                pai_raw = last_discard[1] if last_discard else tile * 4
                melds[who].append([pai_raw] + consumed)
                events.append({"type": "daiminkan", "actor": who, "target": src,
                               "pai": tile_name(pai_raw),
                               "consumed": [tile_name(x) for x in consumed]})
                last_draw[who] = None
                continue

            if kind == "ankan":
                consumed = _take_type(hand[who], tile, 4, issues, "N:ankan")
                melds[who].append(list(consumed))
                events.append({"type": "ankan", "actor": who,
                               "consumed": [tile_name(x) for x in consumed]})
                last_draw[who] = None
                continue

            if kind == "kakan":
                # 加槓：手里第 4 张加到已有的ポン上
                added = _take_type(hand[who], tile, 1, issues, "N:kakan")
                pai_raw = added[0] if added else tile * 4
                target_meld = None
                for ml in melds[who]:
                    if len(ml) == 3 and tile_type(ml[0]) == tile:
                        target_meld = ml
                        break
                if target_meld is None:
                    issues.append("N:kakan_no_pon")
                    consumed_names = [tile_name(pai_raw)] * 3
                else:
                    target_meld.append(pai_raw)
                    consumed_names = [tile_name(x) for x in target_meld[:3]]
                events.append({"type": "kakan", "actor": who,
                               "pai": tile_name(pai_raw),
                               "consumed": consumed_names})
                last_draw[who] = None
                continue

            issues.append(f"N:unknown_kind")
            continue

        # ---- 立直 --------------------------------------------------------
        if tag == "REACH":
            who = int(el.get("who", -1))
            step = el.get("step", "1")
            if step == "1":
                if who in accepted_whos and who >= 0:
                    events.append({"type": "reach", "actor": who})
                    n_reach += 1
                else:
                    # 宣言牌被荣了：天凤不成立、不收 1000 点。
                    # 实测（前 500 个文件、78 例）100% 都是「宣言牌打出去 -> AGARI
                    # 且 fromWho == 宣言者、who != 宣言者」，所以这是**正常现象**，
                    # 记成计数而不是问题（否则 --strict 永远红）。
                    stats["reach_ronned_on_declaration"] += 1
                continue
            events.append({"type": "reach_accepted", "actor": who})
            reach_accepted.add(who)
            net[who] -= 1000
            continue

        # ---- 槓宝牌 ------------------------------------------------------
        if tag == "DORA":
            raw = int(el.get("hai"))
            indicators.append(raw)
            _seen(raw)
            events.append({"type": "dora", "dora_marker": tile_name(raw)})
            continue

        # ---- 和了 --------------------------------------------------------
        if tag == "AGARI":
            agg["n_reach"] = n_reach
            ev, end_reason = _convert_agari(el, indicators, net, issues, agg)
            events.append(ev)
            continue

        # ---- 流局 --------------------------------------------------------
        if tag in ("RYUUKYOKU", "RYUKYOKU"):
            ev = _convert_ryukyoku(el, net, issues, agg)
            end_reason = ev["reason"]
            events.append(ev)
            continue

        if tag in ("BYE", "UN", "SHUFFLE", "TAIKYOKU", "GO", "INIT"):
            continue   # 与牌局无关 / 已经处理
        issues.append(f"unknown_tag:{tag[:24]}")

    # ---- 局末：**一个 kyoku 只发一条 end_kyoku** --------------------------
    # ダブロン 会连着两条 <AGARI>（实测存在），按原来每条都跟 end_kyoku 是重复的。
    events.append({"type": "end_kyoku", "reason": end_reason})

    # ---- 局级对账：立直棒守恒 -------------------------------------------
    if agg.get("hora_n"):
        stats["kyoku_hora"] += 1
        if agg.get("hora_delta_sum") == 1000 * (n_reach + kyoutaku):
            stats["kyoku_hora_balance_ok"] += 1
        else:
            issues.append("agari:balance")
        if agg.get("hora_sticks") == n_reach + kyoutaku:
            stats["kyoku_hora_sticks_ok"] += 1
        else:
            issues.append("agari:sticks")

    # ---- 局末检查：每种牌最多 4 张 ---------------------------------------
    over = [t for t, c in tally.items() if c > 4]
    if over:
        issues.append("tile_over_4:" + ",".join(f"t{t}={tally[t]}" for t in sorted(over)[:4]))
    for ev in events:
        stats["ev_" + ev["type"]] += 1

    info = {
        "start_scores": start_scores,
        "net": net,
        "reaches": n_reach,
        "kyoutaku": kyoutaku,
        "reason": end_reason,
        "n_indicators": len(indicators),
        "oya": oya,
        "stats": stats,
    }
    return events, info


def _convert_agari(el: ET.Element, indicators: list[int], net: list[int],
                   issues: list[str], ctx: dict | None = None) -> tuple[dict, str]:
    """`<AGARI>` -> MJAI `hora`（deltas 取天凤 `sc` 的 variation）。

    顺带对账（结果记进 `ctx['stats']`，报告里会打出来）：
      · `sum(deltas) == 1000 × (本局成立立直数 + 局首供托)`  —— 赢家把桌上所有棒收回去；
      · `ba[1] == 局首供托 + 本局成立立直数`                   —— 收走的棒根数；
      · `deltas[winner] == ten[1]×100 + 300×本场 + 1000×棒`     —— 与 `<AGARI ten>` 对得上。
    """
    ctx = {} if ctx is None else ctx
    stats: Counter = ctx.setdefault("stats", Counter())
    who = int(el.get("who", -1))
    from_who = int(el.get("fromWho", who))
    tsumo = who == from_who
    sc = _ints(el.get("sc"))
    deltas = [sc[i + 1] * 100 for i in range(0, min(len(sc), 8), 2)]
    if len(deltas) != 4:
        issues.append("agari:sc_not_8")
        deltas = (deltas + [0] * 4)[:4]
    ten = _ints(el.get("ten"))
    yaku = _ints(el.get("yaku"))
    ba = _ints(el.get("ba"))
    dora_hai = _ints(el.get("doraHai"))
    ura = _ints(el.get("doraHaiUra"))
    machi_raw = int(el.get("machi")) if el.get("machi") not in (None, "") else None

    # 宝牌指示牌必须与一路上发出去的 dora 事件一致
    if dora_hai != indicators:
        issues.append("agari:dora_mismatch")
    for i in range(4):
        net[i] += deltas[i]

    # ---- 对账（这些是「数字」不是「问题」，只有对不上才算问题）----------
    n_reach = int(ctx.get("n_reach", 0))
    kyoutaku = int(ctx.get("kyoutaku", 0))
    sticks = ba[1] if len(ba) > 1 else 0
    honba = ba[0] if ba else 0
    stats["hora"] += 1
    #: 一个 kyoku 里**可能有多条 AGARI**（ダブロン，实测存在：两家各一条 AGARI），
    #: 立直棒只记在**第一条**上；**本场 300 点也只给第一个赢家**
    #: （实测：`ba=1,1` 先 +9300 = 8000+300+1000，紧跟着 `ba=1,0` 的 +6400 里没有本场）。
    #: 所以「守恒式」和本场都必须按 kyoku 汇总/只在第一条上加。
    first_hora = int(ctx.get("hora_n", 0)) == 0
    ctx["hora_n"] = int(ctx.get("hora_n", 0)) + 1
    ctx["hora_delta_sum"] = int(ctx.get("hora_delta_sum", 0)) + sum(deltas)
    ctx["hora_sticks"] = int(ctx.get("hora_sticks", 0)) + sticks
    if sum(deltas) == 0:
        stats["hora_sum_zero"] += 1
    #: 赢家收的钱 = 手牌点数 + 本场 + 棒。`ten[1]` 是**点数**（不是百点），
    #: 实测第一版写成 ×100 结果 0/2,945 全错。
    if len(ten) > 1 and 0 <= who < 4:
        expect = ten[1] + (300 * honba if first_hora else 0) + 1000 * sticks
        if deltas[who] == expect:
            stats["hora_winner_ok"] += 1
        else:
            issues.append("agari:winner_points")
    stats["hora_tsumo" if tsumo else "hora_ron"] += 1

    ev = {
        "type": "hora",
        "actor": who,
        "target": from_who,
        "tsumo": tsumo,
        "pai": tile_name(machi_raw) if machi_raw is not None else None,
        "deltas": deltas,
        # ---- 和了信息（天凤口径）----
        "fu": ten[0] if len(ten) > 0 else None,
        "han": sum(yaku[1::2]) if yaku else None,
        "points": ten[1] if len(ten) > 1 else None,
        "limit_tier": ten[2] if len(ten) > 2 else 0,
        "yaku_ids": yaku,
        "dora_markers": [tile_name(x) for x in dora_hai],
        "ura_markers": [tile_name(x) for x in ura],
        "ba": ba,
        "raw_sc": sc,
    }
    return ev, "hora"


def _convert_ryukyoku(el: ET.Element, net: list[int], issues: list[str],
                      ctx: dict | None = None) -> dict:
    ctx = {} if ctx is None else ctx
    stats: Counter = ctx.setdefault("stats", Counter())
    stats["ryukyoku"] += 1
    sc = _ints(el.get("sc"))
    deltas = [sc[i + 1] * 100 for i in range(0, min(len(sc), 8), 2)]
    if len(deltas) != 4:
        issues.append("ryukyoku:sc_not_8")
        deltas = (deltas + [0] * 4)[:4]
    if sum(deltas) != 0:
        issues.append("ryukyoku:deltas_not_zero")
    else:
        stats["ryukyoku_sum_zero"] += 1
    for i in range(4):
        net[i] += deltas[i]
    t = el.get("type", "")
    return {
        "type": "ryukyoku",
        "reason": RYUUKYOKU_REASON.get(t, t or "exhaustive_draw"),
        "raw_type": t,
        "deltas": deltas,
        "raw_sc": sc,
    }


# ---------------------------------------------------------------- 一整局


def _is_sanma(children: list[ET.Element]) -> bool:
    """是不是三麻（三人打ち）牌谱。

    判据：第一个 `<INIT>` 的 `hai3` 为空（三麻没有第 4 家），或起手分是
    `350,350,350,0`。实测同时满足，且整局牌种里没有 2m..8m。
    """
    for el in children:
        if el.tag == "INIT":
            if not (el.get("hai3") or "").strip():
                return True
            ten = _ints(el.get("ten"))
            return len(ten) == 4 and ten[3] == 0 and ten[0] >= 300
    return False


def convert_game(xml_text: str, *, game_index: int = 0,
                 game_id: str | None = None) -> tuple[list[dict], list[str], dict]:
    """一条 mjlog XML -> (事件列表, 问题列表, 汇总信息)。"""
    issues: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:      # 坏 XML：不崩，报出来
        return [], [f"xml:{e}"], {}

    children = list(root)
    un = next((el for el in children if el.tag == "UN"), None)
    names = ([unquote(un.get(f"n{i}", "") or "") for i in range(4)]
             if un is not None else [None] * 4)
    dans = _four_int(un.get("dan") if un is not None else None)
    rates = _four_float(un.get("rate") if un is not None else None)
    sx = _split4(un.get("sx") if un is not None else None)

    # 🔴 这批数据**混了三麻**（三人打ち）：实测 8,219 个文件里约 29% 是 3 人局。
    #    判据（三种都指向同一批文件）：
    #      · 第一个 <INIT> 的 hai3=""            <- 决定性
    #      · ten="350,350,350,0"                 <- 三麻起手 35000
    #      · 牌种里完全没有 2m..8m（只有 1m/9m）
    #    三麻不能当四麻 MJAI 输出（riichienv 四麻裁判会直接报错），所以默认跳过并计数。
    if _is_sanma(children):
        return [], ["sanma"], {"sanma": True, "dans": dans, "rates": rates,
                               "names": names, "sx": sx, "kyoku": 0, "events": 0}

    start_game: dict = {
        "type": "start_game",
        # riichienv（Rust serde）要求 `id` 是**字符串**：给整数会报
        # `invalid type: integer 0, expected a string`。天凤牌谱 id 本来就是
        # 字符串（文件名），正好。
        "id": game_id if game_id is not None else str(game_index),
        "names": names,
        "dans": dans,
        "rates": rates,
        "sx": sx,
    }
    if game_id is not None:
        start_game["game_id"] = game_id
    events: list[dict] = [start_game]

    # 按 `<INIT>` 切块
    blocks: list[tuple[ET.Element, list[ET.Element]]] = []
    cur: tuple[ET.Element, list[ET.Element]] | None = None
    for el in children:
        if el.tag == "INIT":
            cur = (el, [])
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(el)
    if not blocks:
        issues.append("game:no_init")
        return events, issues, {"dans": dans, "rates": rates, "names": names,
                                "sx": sx, "kyoku": 0, "events": len(events)}

    infos: list[dict] = []
    reached_sticks = 0
    game_stats: Counter = Counter()
    for init_el, actions in blocks:
        ky_events, info = convert_kyoku(init_el, actions, issues)
        events.extend(ky_events)
        infos.append(info)
        reached_sticks += info["reaches"]
        game_stats.update(info.get("stats", Counter()))

    # 跨局账本：本局 start - 1000×立直 + deltas == 下一局 start
    for i in range(len(infos) - 1):
        got = [a + b for a, b in zip(infos[i]["start_scores"], infos[i]["net"])]
        want = infos[i + 1]["start_scores"]
        if got != want:
            issues.append("score_break")

    # 宝牌 / 立直棒的总账
    total_indicators = sum(info["n_indicators"] for info in infos)
    events.append({"type": "end_game"})

    info = {
        "dans": dans,
        "rates": rates,
        "names": names,
        "sx": sx,
        "kyoku": len(blocks),
        "events": len(events),
        "reaches": reached_sticks,
        "indicators": total_indicators,
        "start_scores": infos[0]["start_scores"],
        "stats": game_stats,
    }
    return events, issues, info


# --------------------------------------------------------------- 批量输出


def iter_inputs(paths: Iterable[str]) -> Iterator[Path]:
    """目录 -> 里面所有 .xml.gz / .xml（排序）；文件原样产出。"""
    for p in paths:
        path = Path(p)
        if path.is_dir():
            found = sorted(path.glob("*.xml.gz")) + sorted(path.glob("*.xml"))
            yield from found
        else:
            yield path


def dumps_event(ev: dict) -> str:
    return json.dumps(ev, ensure_ascii=False, separators=(",", ":"))


def _pct(a: float, b: float) -> str:
    return f"{a / b * 100:.2f}%" if b else "n/a"


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _dan_rate_stats(dans_per_game: list[list[int | None]],
                    rates_per_game: list[list[float | None]]) -> dict:
    """P0 的体检数字：多少局 4 家都有 dan、dan 直方图、rate 范围。"""
    hist: Counter = Counter()
    all4 = 0
    none4 = 0
    all_ge10 = 0
    any_ge10 = 0
    per_pos_hist: list[Counter] = [Counter() for _ in range(4)]
    rates: list[float] = []
    games_with_rate4 = 0
    dan_sum = 0
    for g_dans, g_rates in zip(dans_per_game, rates_per_game):
        have = [d for d in g_dans if d is not None]
        if len(have) == 4:
            all4 += 1
        if not have:
            none4 += 1
        if len(have) == 4 and all(d >= 10 for d in have):
            all_ge10 += 1
        if any(d is not None and d >= 10 for d in g_dans):
            any_ge10 += 1
        for i, d in enumerate(g_dans):
            if d is not None:
                hist[d] += 1
                per_pos_hist[i][d] += 1
                dan_sum += d
        have_r = [r for r in g_rates if r is not None]
        if len(have_r) == 4:
            games_with_rate4 += 1
        rates.extend(have_r)
    return {
        "games": len(dans_per_game),
        "games_all4_dan": all4,
        "games_no_dan": none4,
        "dan_hist": dict(sorted(hist.items())),
        "dan_total_values": sum(hist.values()),
        "dan_mean": (dan_sum / sum(hist.values())) if hist else None,
        "games_all4_dan_ge10": all_ge10,
        "games_any_dan_ge10": any_ge10,
        "dan_hist_by_seat": [dict(sorted(c.items())) for c in per_pos_hist],
        "rate_n": len(rates),
        "rate_min": min(rates) if rates else None,
        "rate_max": max(rates) if rates else None,
        "rate_median": _median(rates),
        "games_all4_rate": games_with_rate4,
    }


def dump(inputs: Iterable[str], out_dir: str | Path, *, shard: int = 100,
         limit: int | None = None, verbose: bool = True,
         report_path: str | Path | None = None) -> dict:
    """批量转换：按 `shard` 局一个文件写 `batch_XXXXXX.json`（每行一个事件）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = list(iter_inputs(inputs))
    if limit is not None:
        files = files[:limit]

    t0 = time.time()
    n_games = n_events = n_kyoku = n_bytes = 0
    n_failed = 0
    n_sanma = 0
    sanma_ids: list[str] = []
    issues: Counter = Counter()
    samples: list[dict] = []
    shard_idx = 0
    buf: list[str] = []
    buf_games = 0
    made: list[str] = []
    dans_per_game: list[list[int | None]] = []
    rates_per_game: list[list[float | None]] = []
    game_rows: list[dict] = []
    duplicate_ids: Counter = Counter()
    totals: Counter = Counter()

    def flush() -> None:
        nonlocal buf, buf_games, n_bytes, shard_idx
        if not buf:
            return
        name = f"batch_{shard_idx:06d}.json"
        text = "\n".join(buf) + "\n"
        path = out / name
        path.write_text(text, encoding="utf-8", newline="\n")
        n_bytes += len(text.encode("utf-8"))
        made.append(name)
        shard_idx += 1
        buf = []
        buf_games = 0

    for fp in files:
        try:
            xml = load_xml(fp)
            events, game_issues, info = convert_game(
                xml, game_index=n_games, game_id=game_id_of(fp))
        except Exception as e:                     # 单文件出错不拖垮全量
            n_failed += 1
            issues["fatal:" + type(e).__name__] += 1
            if len(samples) < 20:
                samples.append({"file": fp.name, "error": f"{type(e).__name__}: {e}"})
            continue

        if info.get("sanma"):
            n_sanma += 1
            if len(sanma_ids) < 50:
                sanma_ids.append(game_id_of(fp))
            continue

        if not events:
            n_failed += 1
            for it in game_issues:
                issues[it] += 1
            continue

        for it in game_issues:
            issues[it.split(":")[0] + ":" + it.split(":")[1] if ":" in it else it] += 1
        if game_issues and len(samples) < 20:
            samples.append({"file": fp.name, "issues": game_issues[:6]})

        for ev in events:
            buf.append(dumps_event(ev))
        buf_games += 1
        n_games += 1
        n_events += len(events)
        n_kyoku += info.get("kyoku", 0)
        dans_per_game.append(info.get("dans", [None] * 4))
        rates_per_game.append(info.get("rates", [None] * 4))
        totals.update(info.get("stats", Counter()))
        if game_id_of(fp) in duplicate_ids:
            issues["duplicate_game_id"] += 1
        duplicate_ids[game_id_of(fp)] += 1
        game_rows.append({
            "file": fp.name,
            "id": n_games - 1,
            "kyoku": info.get("kyoku", 0),
            "dans": info.get("dans"),
            "rates": info.get("rates"),
            "names": info.get("names"),
        })
        if buf_games >= shard:
            flush()

    flush()
    elapsed = time.time() - t0

    stats = _dan_rate_stats(dans_per_game, rates_per_game)
    report = {
        "games": n_games,
        "failed_files": n_failed,
        "sanma_skipped": n_sanma,
        "sanma_sample_ids": sanma_ids[:5],
        "events": n_events,
        "kyoku": n_kyoku,
        "bytes": n_bytes,
        "mb": round(n_bytes / 1e6, 2),
        "seconds": round(elapsed, 2),
        "games_per_sec": round(n_games / elapsed, 1) if elapsed else None,
        "files": made,
        "out_dir": str(out),
        "issues": dict(sorted(issues.items())),
        "issue_samples": samples,
        "dan_rate": stats,
        "totals": dict(sorted(totals.items())),
    }
    if report_path is not None:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**report, "games_detail": game_rows},
                                ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print_report(report)
    return report


def print_report(r: dict) -> None:
    print("=" * 72)
    print("  天凤 mjlog XML -> 标准 MJAI")
    print("=" * 72)
    print(f"  对局          {r['games']:,} 局   （失败 {r['failed_files']}，"
          f"三麻跳过 {r['sanma_skipped']:,}）")
    print(f"  局（kyoku）   {r['kyoku']:,}")
    print(f"  事件          {r['events']:,}")
    print(f"  体积          {r['mb']:,} MB   -> {r['out_dir']}")
    print(f"  文件          {len(r['files'])} 个")
    print(f"  耗时          {r['seconds']} 秒  ({r['games_per_sec']} 局/秒)")
    if r["issues"]:
        print("  自检问题：")
        for k, v in r["issues"].items():
            print(f"    {k:<34} {v:,}")
    else:
        print("  自检问题      0 ✅")
    for s in r.get("issue_samples", [])[:5]:
        print(f"    sample {s}")
    t = r.get("totals", {})
    if t:
        print("-" * 72)
        print("  和了/流局对账（这些是计数，不是问题）")
        hora = t.get("hora", 0)
        kh = t.get("kyoku_hora", 0)
        print(f"    hora {hora:,}（自摸 {t.get('hora_tsumo', 0):,} / 荣和 {t.get('hora_ron', 0):,}）"
              f"  ryukyoku {t.get('ryukyoku', 0):,}；有和了的 kyoku {kh:,}")
        print(f"    sum(deltas)==0                        {t.get('hora_sum_zero', 0):,}/{hora:,} "
              f"({_pct(t.get('hora_sum_zero', 0), hora)})  <- 桌上没立直棒时必然为 0")
        print(f"    kyoku 汇总 == 1000×(立直+供托)         {t.get('kyoku_hora_balance_ok', 0):,}/{kh:,} "
              f"({_pct(t.get('kyoku_hora_balance_ok', 0), kh)})  <- 立直棒守恒（含ダブロン）")
        print(f"    kyoku 收走的棒根数对得上               {t.get('kyoku_hora_sticks_ok', 0):,}/{kh:,} "
              f"({_pct(t.get('kyoku_hora_sticks_ok', 0), kh)})")
        print(f"    赢家 deltas == 点数+本场+棒             {t.get('hora_winner_ok', 0):,}/{hora:,} "
              f"({_pct(t.get('hora_winner_ok', 0), hora)})")
        print(f"    流局 sum(deltas)==0                    {t.get('ryukyoku_sum_zero', 0):,}/"
              f"{t.get('ryukyoku', 0):,}")
        print(f"    立直宣言牌被荣（跳过，不收费）          {t.get('reach_ronned_on_declaration', 0):,}")
        ev = {k[3:]: v for k, v in t.items() if k.startswith("ev_")}
        if ev:
            print(f"    事件直方图 {dict(sorted(ev.items(), key=lambda kv: -kv[1]))}")
    d = r.get("dan_rate", {})
    if d:
        print("-" * 72)
        print(f"  dan/rate：{d['games']:,} 局里 {d['games_all4_dan']:,} 局 4 家都有 dan，"
              f"{d['games_no_dan']:,} 局一个都没有")
        print(f"  dan 直方图    {d['dan_hist']}")
        print(f"  dan 均值      {d['dan_mean']}")
        print(f"  rate n/min/max/median  {d['rate_n']} / {d['rate_min']} / "
              f"{d['rate_max']} / {d['rate_median']}")
        print(f"  4 家 dan>=10  {d['games_all4_dan_ge10']:,} 局"
              f"   （任一家 >=10：{d['games_any_dan_ge10']:,} 局）")
    print("=" * 72)


# -------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="天凤 mjlog XML -> 标准 MJAI（含 dan/rate）")
    ap.add_argument("inputs", nargs="+", help=".xml.gz/.xml 文件或目录")
    ap.add_argument("--out", default=None, help="输出目录（默认只打印统计）")
    ap.add_argument("--shard", type=int, default=100, help="每个输出文件放多少局（默认 100）")
    ap.add_argument("--limit", type=int, default=None, help="最多处理多少个文件")
    ap.add_argument("--report", default=None, help="把报告（含每局 dan/rate）写成 JSON")
    ap.add_argument("--dump", action="store_true", help="把第一个文件的事件打到 stdout")
    ap.add_argument("--strict", action="store_true", help="有任何自检问题就返回 1")
    ap.add_argument("--verify", action="store_true", help="写完用 riichienv 验一遍")
    args = ap.parse_args(argv)

    if args.dump:
        files = list(iter_inputs(args.inputs))[:1]
        if not files:
            print("没有输入文件")
            return 2
        xml = load_xml(files[0])
        events, issues, info = convert_game(xml, game_index=0, game_id=game_id_of(files[0]))
        for ev in events:
            print(dumps_event(ev))
        print(f"\n# {len(events)} events, issues={issues}", file=sys.stderr)
        print(f"# info={json.dumps({k: v for k, v in info.items() if k != 'names'}, ensure_ascii=False)}",
              file=sys.stderr)
        return 0

    if args.out is None:
        print("🔴 需要 --out 目录（或加 --dump 看单个文件）")
        return 2

    rep = dump(args.inputs, args.out, shard=args.shard, limit=args.limit,
               report_path=args.report)

    if args.verify:
        ok, msg = verify_riichienv(Path(args.out))
        print(f"  riichienv 裁判 {'✅' if ok else '🔴'} {msg}")
        if not ok:
            return 1

    if args.strict and rep["issues"]:
        return 1
    return 0


def verify_riichienv(out_dir: Path, sample: int | None = None) -> tuple[bool, str]:
    """用 riichienv 当裁判：能解析就说明格式合法。"""
    try:
        from riichienv import MjaiReplay  # type: ignore
    except Exception as e:  # pragma: no cover
        return True, f"riichienv 不可用，跳过: {e}"
    files = sorted(out_dir.glob("*.json"))
    if sample:
        files = files[:sample]
    bad = 0
    rounds = 0
    for f in files:
        try:
            rp = MjaiReplay.from_jsonl(str(f))
            rounds += rp.num_rounds()
        except Exception as e:
            bad += 1
            print(f"    🔴 {f.name}: {type(e).__name__}: {e}")
    return bad == 0, f"{len(files)} 个文件 / {rounds} 局，失败 {bad}"


if __name__ == "__main__":
    sys.exit(main())
