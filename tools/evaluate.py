#!/usr/bin/env python
"""对 samples/ 里的已知截图跑一遍识别，给出可复现的准确率报告。

用法::

    .venv/bin/python tools/evaluate.py            # 跑到控制台
    .venv/bin/python tools/evaluate.py --markdown  # 输出 markdown 表格（写进 README）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from potatomaj import vision as V  # noqa: E402
from potatomaj.classify import TemplateLibrary  # noqa: E402
from potatomaj.recognize import _accept, recognize  # noqa: E402

#: 与 recognize() 的默认值保持一致（评测必须用实际输出口径）
ACCEPT_SCORE = 0.26
ACCEPT_MARGIN = 0.04
from potatomaj.tiles import sort_tiles  # noqa: E402

#: 逐张肉眼核对过的真值（三麻截图）
HAND_TRUTH: dict[str, str] = {
    "sanma1": "1m 2p 3p 8p 8p 9p 0s 6s 6s 8s 1z 6z 6z",
    "sanma2": "1m 2p 3p 8p 8p 9p 0s 6s 6s 8s 1z 6z 6z",
    "sanma3": "1m 2p 3p 8p 8p 9p 0s 6s 6s 8s 1z 6z 6z",
    "sanma5": "1m 1p 3p 7p 2s 2s 3s 4s 4s 7s 9s 9s 2z 3z",
    "sanma6": "1m 1p 1p 3p 5p 1s 1s 2s 3s 6s 8s 1z 2z 6z",
}

#: 左上角宝牌指示牌真值
DORA_TRUTH: dict[str, list[str]] = {
    "sanma1": ["1z"],
    "sanma2": ["1z", "6p"],
    "sanma3": ["1z", "6p"],
    "sanma5": ["3p"],
    "sanma6": ["8s"],
}

#: 牌河真值 —— 全部是「把每张牌单独放大到 3~4 倍、去掉任何标签」之后
#: 逐张肉眼读出来的，不是照抄识别结果（这点很重要，否则就是循环论证）
SELF_POND_TRUTH: dict[tuple[int, int], str] = {   # (行, 列)
    (0, 0): "9s", (0, 1): "8p", (0, 2): "2z", (0, 3): "5p", (0, 4): "7p", (0, 5): "1m",
    (1, 0): "7p", (1, 1): "9p", (1, 2): "8s", (1, 3): "2s", (1, 4): "2s",
}

TOP_POND_TRUTH: dict[tuple[int, int], str] = {
    (0, 1): "1p", (0, 2): "5p", (0, 3): "1m", (0, 4): "9s", (0, 5): "8s",
    (1, 0): "9p", (1, 1): "8p", (1, 3): "6z", (1, 4): "4p",
    # (1,2) 被上一排挡住一半，读不准，不计入
}

#: 左家牌河真值：(列, 行)
LEFT_POND_TRUTH: dict[tuple[int, int], str] = {
    (0, 0): "3z", (0, 1): "6p", (0, 2): "2z", (0, 3): "1s",
    (1, 0): "1p", (1, 1): "6z", (1, 2): "7z", (1, 3): "8s", (1, 4): "9p", (1, 5): "1p",
}

POND_SOURCE = "sanma3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "samples"))
    ap.add_argument("--templates", default=str(ROOT / "templates"))
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--debug-dir", default=str(ROOT / "build"))
    args = ap.parse_args()

    samples = Path(args.samples)
    lib = TemplateLibrary(args.templates)
    rows = []
    hand_ok = hand_total = 0
    dora_ok = 0
    t_all = 0.0

    for name, truth in HAND_TRUTH.items():
        path = samples / f"{name}.webp"
        if not path.exists():
            path = samples / f"{name}.png"
        if not path.exists():
            continue
        rgb = V.load_rgb(path)
        t0 = time.time()
        res = recognize(rgb, lib)
        dt = time.time() - t0
        t_all += dt
        exp = sorted(sort_tiles(truth.split()))
        got = sorted(sort_tiles(res.hand))
        right = sum(1 for a, b in zip(got, exp) if a == b)
        hand_ok += right
        hand_total += len(exp)
        dok = res.dora_indicators == DORA_TRUTH.get(name, [])
        dora_ok += dok
        rows.append((name, right, len(exp), got == exp, dok, res.dora_indicators, dt))
        if args.debug_dir:
            from potatomaj.recognize import render_debug

            render_debug(rgb, res, Path(args.debug_dir) / f"rec_{name}.png")

    # 三家牌河逐格准确率 —— 直接以 recognize() 的实际输出为准，
    # 避免「评测一套、实际输出另一套」
    pond_right = pond_total = 0
    pond_wrong = 0
    pond_detail: list[str] = []
    src = samples / f"{POND_SOURCE}.webp"
    if src.exists():
        from potatomaj.detect import Region
        from potatomaj.layout import find_anchor, read_pond, read_side_pond

        rgb = V.load_rgb(src)
        anchor = find_anchor(rgb)
        res = recognize(rgb, lib)
        face = V.tile_face_mask(rgb, 160.0, 0.34)
        jobs = [
            ("自己", Region.POND_SELF,
             {(r, c): b for r, c, b in read_pond(rgb, anchor, "self", face=face)}, SELF_POND_TRUTH),
            ("对家", Region.POND_TOP,
             {(r, c): b for r, c, b in read_pond(rgb, anchor, "top", face=face)}, TOP_POND_TRUTH),
            ("上家", Region.POND_LEFT,
             {(c, r): b for r, c, b in read_side_pond(rgb, anchor, "left", face=face)}, LEFT_POND_TRUTH),
        ]
        for label, region, cells, truth in jobs:
            got_tiles = [t for t in res.tiles if t.region is region]
            correct = rejected = wrong = 0
            for key, want in sorted(truth.items()):
                pond_total += 1
                if key not in cells:
                    pond_detail.append(f"{label}{key} 未定位到")
                    continue
                want_box = cells[key]
                near = min(
                    got_tiles,
                    key=lambda t: (t.box.cx - want_box.cx) ** 2 + (t.box.cy - want_box.cy) ** 2,
                    default=None,
                )
                if near is None:
                    pond_detail.append(f"{label}{key}={want} 未定位到")
                    continue
                if near.tile_id == want:
                    correct += 1
                elif near.tile_id is None:
                    rejected += 1
                    pond_detail.append(f"{label}{key}={want} 判为不确定(?)")
                else:
                    wrong += 1
                    pond_detail.append(f"{label}{key} 错报: 应为{want} 报成{near.tile_id}")
            pond_right += correct
            pond_wrong += wrong
            print(f"  {label}牌河: 正确 {correct}/{len(truth)}  判? {rejected}  错报 {wrong}")

    if args.markdown:
        print("| 截图 | 手牌 | 宝牌指示牌 | 耗时 |")
        print("|---|---|---|---|")
        for name, right, total, exact, _dok, dora, dt in rows:
            mark = "✅" if exact else f"{right}/{total}"
            print(f"| `{name}` | {mark} {total} 张 | {' '.join(dora) or '—'} | {dt:.2f}s |")
        print()
        print(f"- 手牌逐张准确率：**{hand_ok}/{hand_total} = {hand_ok / max(1, hand_total):.1%}**")
        print(f"- 宝牌指示牌：**{dora_ok}/{len(rows)} 张截图完全正确**")
        if pond_total:
            print(f"- 三家牌河（{POND_SOURCE} 逐格核对）：**{pond_right}/{pond_total} = {pond_right / pond_total:.1%}**"
                  f"，其中错报 {pond_wrong} 张，其余不确定的会输出 `?` 而不是猜")
        print(f"- 平均单张识别耗时：{t_all / max(1, len(rows)):.2f}s")
        return 0

    print(f"模板库: {len(lib)} 张牌面")
    for name, right, total, exact, dok, dora, dt in rows:
        flag = "✅" if exact else "❌"
        print(f"  {name:8s} 手牌 {flag} {right}/{total}   宝牌 {'✅' if dok else '❌'} {dora}   {dt:.2f}s")
    print(f"\n手牌逐张准确率: {hand_ok}/{hand_total} = {hand_ok / max(1, hand_total):.1%}")
    print(f"宝牌指示牌全对: {dora_ok}/{len(rows)}")
    if pond_total:
        print(f"三家牌河逐格:   正确 {pond_right}/{pond_total} = {pond_right / pond_total:.1%}   错报 {pond_wrong} 张")
        for d in pond_detail:
            print(f"    · {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
