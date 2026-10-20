#!/usr/bin/env python
"""mjlog XML -> MJAI 事件流。天凤 `sc/raw` 抓下来的牌谱用它转成现有管线的输入形状。

    python tools/mjlog_to_mjai.py _data/scraw/logs/xxx.xml.gz --json

## 为什么需要它

现有管线（`aimodel/mjlog.py`）读的是 **`GL1.xlsm` 表格里的行**（`parse_sheet`/`parse_row`），
**不是 mjlog XML**。而天凤 `sc/raw` 抓下来的是 mjlog XML。
-> 中间缺一个转换器，这个文件就是它。

## 牌 id

mjlog 里所有牌都是 **`tile*4 + 赤标记`**（0..135）。转成 MJAI 名字用
`mjlog._name_of_raw(raw)`（**复用现有实现，不重写一份** —— 重写就有第二份出错的机会）。

## 座位

`T` / `D` 是**自家**；`U`/`V`/`W` 和 `E`/`F`/`G` 是**其余三家**，字母序对应「上一个动作之后
的下一家」起的第 1/2/3 家。所以必须**跟踪当前行动者**，不能直接把字母当座位号。

`<INIT oya="N">` 给出起家；之后按 摸->打 循环推进。
鸣牌（`<N who=...>`）会改变下一家 —— 这里按 `who` 显式处理。

## 输出的事件（与 `mjlog.parse_row` 的既有形状一致）

    start_kyoku  {kyoku, honba, oya, scores, tehais}
    dora         {dora_marker}
    tsumo        {actor, pai}
    dahai        {actor, pai, tsumogiri}
    chi/pon/kan  {actor, target, pai, consumed}
    reach        {actor}
    reach_accepted {actor}
    end_kyoku    {reason}

## 验收（不要只看"跑通了"）

① 每局起手 13 张 × 4 与 `<INIT hai0..hai3>` 逐张相同
② 摸牌序列与 `T/U/V/W` 逐张相同
③ 打牌序列与 `D/E/F/G` 逐张相同
④ 和了局：`<AGARI>` 的 who/machi 与事件流一致
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimodel.mjlog import _name_of_raw  # noqa: E402

#: 摸牌标签：'T' 是自家，'U'/'V'/'W' 是其余三家（相对当前行动者的下一家起）
TSUMO_TAGS = ("T", "U", "V", "W")
#: 打牌标签：'D' 是自家，'E'/'F'/'G' 是其余三家
DAHAI_TAGS = ("D", "E", "F", "G")


def _tile(raw: int) -> str:
    """mjlog 的牌编号 -> MJAI 牌名（复用现成实现）。"""
    return _name_of_raw(raw)


def _ints(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]


def parse_game(xml: str) -> list[dict]:
    """一条牌谱 -> 事件列表。"""
    root = ET.fromstring(xml)
    events: list[dict] = []

    # 当前行动者。mjlog 用「相对字母」，所以必须自己维护这个数。
    seat = 0
    kyoku_no = 0
    honba = 0
    scores = [25000] * 4

    for el in root:
        tag = el.tag

        if tag == "INIT":
            oya = int(el.get("oya", "0"))
            seed = _ints(el.get("seed"))          # [局, 本场, 供托, 骰子1, 骰子2, 宝牌指示牌]
            kyoku_no = seed[0] if seed else 0
            honba = seed[1] if seed else 0
            scores = _ints(el.get("ten")) or scores
            tehais = [
                [_tile(v) for v in _ints(el.get(f"hai{i}"))] for i in range(4)
            ]
            events.append({
                "type": "start_kyoku",
                "kyoku": kyoku_no,
                "honba": honba,
                "oya": oya,
                "scores": scores,
                "tehais": tehais,
            })
            if len(seed) >= 6:
                events.append({"type": "dora", "dora_marker": _tile(seed[5])})
            seat = oya
            continue

        if tag == "REACH":
            who = int(el.get("who", seat))
            events.append({"type": "reach", "actor": who})
            continue

        if tag == "DORA":
            events.append({"type": "dora", "dora_marker": _tile(int(el.get("hai")))})
            continue

        if tag == "N":
            who = int(el.get("who", seat))
            m = int(el.get("m", "0"), 16) if el.get("m") else 0
            # m 的编码：低 2 位是相对 who 的「从谁那里」，接着是鸣牌种类
            from_who = (who + (m & 0x3)) % 4
            kind_bits = (m >> 2) & 0x7
            kind = {4: "chi", 5: "pon", 6: "kan"}.get(kind_bits, "pon")
            # 被鸣的两张（chi 需要具体哪两张）：高字节给出牌的编号
            raw = (m >> 10) & 0xFF
            pai = _tile(raw)
            if m & 0x4:                                # 赤
                pai = "5mr" if pai.startswith("5m") else pai
            events.append({"type": kind, "actor": who, "target": from_who,
                           "pai": pai, "consumed": []})
            # 鸣牌之后轮到 who 打牌
            seat = who
            continue

        if tag == "AGARI":
            who = int(el.get("who", -1))
            events.append({"type": "hora", "actor": who,
                           "deltas": _delta_from_scores(el.get("sc"), scores)})
            events.append({"type": "end_kyoku", "reason": "hora"})
            continue

        if tag in ("RYUUKYOKU", "UN"):
            if tag == "RYUUKYOKU":
                events.append({"type": "end_kyoku", "reason": "ryukyoku"})
            continue

        if tag == "TAIKYOKU":
            continue

        # 摸牌 / 打牌：**值是合在标签名里的**（`<D56/>`、`<T12/>`），
        # 不是 `<D p="56"/>` —— 所以 tag[0] 是字母、tag[1:] 是牌编号。
        # （我第一版按 `el.get("p")` 取，结果 dahai/tsumo 事件数是 0 ✗）
        if len(tag) > 1 and tag[0] in TSUMO_TAGS and tag[1:].isdigit():
            off = TSUMO_TAGS.index(tag[0])
            actor = seat if off == 0 else (seat + off) % 4
            events.append({"type": "tsumo", "actor": actor, "pai": _tile(int(tag[1:]))})
            seat = actor
            continue

        if len(tag) > 1 and tag[0] in DAHAI_TAGS and tag[1:].isdigit():
            off = DAHAI_TAGS.index(tag[0])
            actor = seat if off == 0 else (seat + off) % 4
            events.append({"type": "dahai", "actor": actor, "pai": _tile(int(tag[1:])),
                           "tsumogiri": off == 0})
            # ⚠️ **打牌之后轮到下一家** —— 这里原来写成 `seat = actor` ✗
            #    那会让下一张 `<T>` 还落在打牌者身上，之后整局的座位全部错位 ✗
            #    （症状：11 局只解出 423 次摸牌 ≈ 38/局，正常约 70）
            seat = (actor + 1) % 4
            continue

    return events


def _delta_from_scores(sc: str | None, before: list[int]) -> list[int]:
    if not sc:
        return [0, 0, 0, 0]
    vals = _ints(sc)
    after = vals[1::2][:4]      # sc 是 "before,after,before,after,..."
    if len(after) != 4:
        return [0, 0, 0, 0]
    return [a - b for a, b in zip(after, before)]


def load_xml(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".gz":
        return gzip.decompress(data).decode("utf-8")
    return data.decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mjlog XML -> MJAI 事件流")
    ap.add_argument("xml", help=".xml 或 .xml.gz")
    ap.add_argument("--json", action="store_true", help="输出 JSON（每行一个事件）")
    ap.add_argument("--check", action="store_true", help="做验收①：起手 13×4 张数")
    args = ap.parse_args(argv)

    xml = load_xml(Path(args.xml))
    events = parse_game(xml)

    if args.check:
        print(f"  事件数 {len(events)}")
        kinds: dict[str, int] = {}
        for e in events:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        print(f"  类型统计 {kinds}")
        ok = True
        for i, e in enumerate(events):
            if e["type"] != "start_kyoku":
                continue
            lens = [len(h) for h in e["tehais"]]
            flag = "✅" if all(n == 13 for n in lens) else "🔴"
            print(f"  {flag} 第{i}个 start_kyoku: 四家起手张数 {lens}")
            ok = ok and all(n == 13 for n in lens)
        print(f"  -> {'✅ 起手都是 13 张' if ok else '🔴 有起手不是 13 张'}")
        return 0 if ok else 1

    if args.json:
        for e in events:
            print(json.dumps(e, ensure_ascii=False))
    else:
        print(f"  {len(events)} 个事件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
