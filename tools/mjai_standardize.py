# -*- coding: utf-8 -*-
"""把「我们格式的 MJAI」转成「标准 MJAI」，好喂给 Mortal / riichienv。

背景（2026-10-21 实测）：
    riichienv 的 MjaiReplay.from_jsonl() 读不了我们 _data/pro4/mjai/*.json，
    报 missing field `kyoutaku`，补上又报 missing field `dora_marker`。

我们写出的 vs 标准，差三处：
    ① riichi_sticks  -> kyoutaku
    ② dora_markers: ["4m"] -> dora_marker: "4m"
    ③ scores: [250,250,250,250]（单位百点）-> [25000,25000,25000,25000]（实际点数）

用法：
    python -m tools.mjai_standardize <src文件或目录> <dst目录> [--dry]

设计原则：
    · **不改** aimodel/mjlog.py（那是训练数据生成器，改了就要重建 49 分钟的数据集）
    · 只做格式转换，**不动语义**：tile 名、事件顺序、deltas 一律原样
    · 转完用 riichienv 当裁判验一遍（见 --verify）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 我们自己的口径：scores 是「百点」；标准 MJAI 是实际点数
SCORE_UNIT = 100

# 需要改名的键（只在这些事件类型上改，避免误伤别的字段）
RENAME = {
    "start_kyoku": [("riichi_sticks", "kyoutaku")],
}
# 需要改形的键：list -> 单个首元素
SINGULAR = {
    "start_kyoku": [("dora_markers", "dora_marker")],
}
# hora 里的 dora_markers **不要动**。
# 实测（2026-10-21）：把它改成标量 dora_marker 会让消费方把它当成额外的指示牌，
# 一致率没变（8/29），说明改动无效；而删掉重复的 dora 事件后一致率 28% -> 48%。
SINGULAR_OPTIONAL: dict[str, list[tuple[str, str]]] = {}


def convert_event(ev: dict, state: dict | None = None) -> dict | None:
    """单个事件：我们的 -> 标准。返回新 dict（不改入参）；返回 None 表示该事件应当丢弃。

    ``state`` 用来跨事件记账（当前这一局的宝牌指示牌），调用方对同一个文件要传同一个 dict。
    """
    if state is None:
        state = {}
    t = ev.get("type")
    out = dict(ev)

    did_rename = False
    for old, new in RENAME.get(t, []):
        if old in out and new not in out:
            out[new] = out.pop(old)
            did_rename = True

    for old, new in SINGULAR.get(t, []) + SINGULAR_OPTIONAL.get(t, []):
        if old in out and new not in out:
            v = out.pop(old)
            if isinstance(v, list):
                out[new] = v[0] if v else None
            else:
                out[new] = v

    # scores：百点 -> 点数。**必须整场一致**，所以只对携带完整四家 scores 的事件做。
    # 🔴 只在本事件确实是「我们的格式」时才 ×100 —— 判据是「刚刚把 riichi_sticks 改名了」。
    #    否则同一个文件跑两遍会把 25000 再 ×100 变成 2500000 ✓
    #    （这是新写的回归测试当场抓到的，不是我自己想到的）
    if t == "start_kyoku" and did_rename and isinstance(out.get("scores"), list):
        out["scores"] = [int(s) * SCORE_UNIT for s in out["scores"]]

    # 记下这一局的宝牌指示牌
    if t == "start_kyoku":
        state["indicator"] = out.get("dora_marker")

    # 🔴 我们每一局都会在 start_kyoku 之后**再发一条 dora 事件**，内容一样。
    # 标准 MJAI 里 start_kyoku 已经带了 dora_marker，dora 事件只留给**杠宝牌**。
    # 不去重的话消费方会当成两张指示牌 -> 番数翻倍。
    #   实测：保留 -> 一致率 28%，删掉 -> 48%（29 手和牌，天凤原始记录为基准）
    if t == "dora" and state.get("dedup_dora", True):
        marker = out.get("dora_marker")
        if marker is not None and marker == state.get("indicator"):
            return None
        state["indicator"] = marker

    return out


def convert_file(src: Path, dst: Path) -> tuple[int, int, int]:
    """返回 (读入行数, 写出行数, 被改动/丢弃的事件数)。"""
    n = 0
    out_n = 0
    changed = 0
    state: dict = {}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8", newline="\n") as oh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            ev = json.loads(line)
            new = convert_event(ev, state)
            if new is None:
                changed += 1
                continue
            if new != ev:
                changed += 1
            oh.write(json.dumps(new, ensure_ascii=False, separators=(",", ":")) + "\n")
            out_n += 1
    return n, out_n, changed


def verify(dst: Path) -> tuple[bool, str]:
    """用 riichienv 当裁判。没装就跳过（返回 None 语义）。"""
    try:
        from riichienv import MjaiReplay  # type: ignore
    except Exception as e:  # pragma: no cover
        return None, f"riichienv 不可用，跳过裁判: {e}"
    try:
        rp = MjaiReplay.from_jsonl(str(dst))
        return True, f"num_rounds={rp.num_rounds()}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true", help="转完用 riichienv 验一遍")
    ap.add_argument("--dry", action="store_true", help="只转换，不写盘（打印统计）")
    a = ap.parse_args(argv)

    src = Path(a.src)
    dst = Path(a.dst)
    files = sorted(src.glob("*.json")) + sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    if not files:
        print(f"🔴 没找到输入: {src}")
        return 2

    tot_lines = tot_changed = 0
    bad = 0
    for f in files:
        out = (dst / f.name) if src.is_dir() else dst
        if a.dry:
            # 只统计，不写盘
            n = ch = 0
            st: dict = {}
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    new = convert_event(json.loads(line), st)
                    if new is None or new != json.loads(line):
                        ch += 1
        else:
            n, _, ch = convert_file(f, out)
        tot_lines += n
        tot_changed += ch
        print(f"  {f.name}: {n} 行, 改动/丢弃 {ch} 个事件")

        if a.verify and not a.dry:
            ok, msg = verify(out)
            flag = "✅" if ok else ("⚠️" if ok is None else "🔴")
            print(f"      裁判 {flag} {msg}")
            if ok is False:
                bad += 1

    print(f"\n  合计 {len(files)} 个文件 / {tot_lines} 行 / 改动 {tot_changed} 个事件")
    if a.verify and bad:
        print(f"  🔴 {bad} 个文件裁判不通过")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
