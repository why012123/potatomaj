# -*- coding: utf-8 -*-
"""tools/mjai_standardize 的回归测试。

背景（2026-10-21 实测）：我们自己写出的 MJAI **不符合标准 MJAI**，
riichienv / Mortal 这类消费方读不了 ——
    missing field `kyoutaku`      <- 我们写 riichi_sticks
    missing field `dora_marker`   <- 我们写 dora_markers: ["4m"]
另外 scores 是「百点」不是点数，且每局多出一条与 start_kyoku 重复的 dora 事件
（不去重的话消费方按两张指示牌算，番数翻倍）。

这个测试**不需要 riichienv**（只测纯函数），装了 riichienv 的机器可以另外跑
`python -m tools.mjai_standardize ... --verify` 做端到端裁判。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mjai_standardize import convert_event  # noqa: E402

FAILS = []


def check(name, got, want):
    if got == want:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}\n       得 {got!r}\n       期望 {want!r}")
        FAILS.append(name)


START = {
    "type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
    "riichi_sticks": 0, "dice": [3, 3], "scores": [250, 250, 250, 250],
    "oya": 0, "dora_markers": ["4m"], "tehais": [["7s"], ["9m"], ["1m"], ["9p"]],
}


def main():
    print("▶ mjai_standardize")

    # ① 三处必须改：kyoutaku / dora_marker / scores 单位
    st = {}
    got = convert_event(dict(START), st)
    check("riichi_sticks -> kyoutaku", got.get("kyoutaku"), 0)
    check("riichi_sticks 已删", "riichi_sticks" in got, False)
    check("dora_markers -> dora_marker（取首元素）", got.get("dora_marker"), "4m")
    check("dora_markers 已删", "dora_markers" in got, False)
    check("scores 百点 -> 点数", got.get("scores"), [25000, 25000, 25000, 25000])
    check("其它字段原样保留", got.get("tehais"), START["tehais"])
    check("type 不变", got.get("type"), "start_kyoku")

    # ② 紧跟着的重复 dora 事件要丢掉
    st2 = {}
    convert_event(dict(START), st2)
    dup = {"type": "dora", "dora_marker": "4m"}
    check("与 start_kyoku 重复的 dora 被丢弃", convert_event(dup, st2), None)

    # ③ 【负向验证】杠宝牌（指示牌不同）**不能**丢 —— 这是「去重」和「乱删」的区别
    kan = {"type": "dora", "dora_marker": "7p"}
    kept = convert_event(kan, st2)
    check("杠宝牌（不同指示牌）保留", kept is not None and kept.get("dora_marker"), "7p")

    # ④ 【负向验证】hora 里的 dora_markers 不能动
    #    实测：改成标量后消费方会当成额外指示牌，但一致率没变化（8/29）-> 改动无效，撤掉
    hora = {"type": "hora", "actor": 3, "target": 3, "tsumo": True, "pai": "3s",
            "han": 3, "fu": 20, "dora_markers": ["4m"], "ura_markers": []}
    got_h = convert_event(dict(hora), {})
    check("hora.dora_markers 保持原样", got_h.get("dora_markers"), ["4m"])
    check("hora 没有被塞进 dora_marker", "dora_marker" in got_h, False)

    # ⑤ 【负向验证】不是 start_kyoku 的事件不许被改名
    pon = {"type": "pon", "actor": 1, "target": 0, "pai": "5z",
           "consumed": ["5z", "5z"], "riichi_sticks": 9}
    got_p = convert_event(dict(pon), {})
    check("pon 上的 riichi_sticks 不被动", got_p.get("riichi_sticks"), 9)
    check("pon 没有被塞进 kyoutaku", "kyoutaku" in got_p, False)

    # ⑥ 幂等：已经标准了的事件再过一遍不变
    st3 = {}
    once = convert_event(dict(START), st3)
    st4 = {}
    twice = convert_event(dict(once), st4)
    check("已是标准格式再过一遍结果不变", twice, once)

    # ⑦ 入参不被修改（纯函数）
    raw = dict(START)
    convert_event(raw, {})
    check("入参未被就地修改", raw, START)

    if FAILS:
        print(f"❌ mjai_standardize: {len(FAILS)} 项失败 -> {FAILS}")
        return 1
    print("✅ mjai_standardize 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
