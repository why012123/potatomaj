#!/usr/bin/env python
"""把 `run_all.py` 的测试分片跑 —— 为了适配「重计算 35~46 秒被回收」的本机约束。

背景
====
全套 18 个测试文件约 **96 秒**（实测逐文件计时），而本机会话里的重计算进程
**35~46 秒被回收**。所以 `run_all.py` 一次跑不完。

这个脚本**不替代** `run_all.py`（它是权威入口，测试清单也从它读）。
它只是让「跑完全部」这件事在本机可行：分成 N 片，每片单独一次调用。

用法
====
    python -m tools.run_tests_shard --list          # 看分片方案
    python -m tools.run_tests_shard 1               # 跑第 1 片
    python -m tools.run_tests_shard 1 --groups 3    # 用 3 片的方案（默认 3）
    python -m tools.run_tests_shard --all-groups    # 依次跑全部片（可能超过窗口）

退出码：0 = 本片全绿
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TESTS_DIR = HERE / "aimodel" / "tests"
ROOT = Path(__file__).resolve().parents[1]


def load_test_list() -> list[str]:
    """自动发现 test_*.py —— 和 run_all.py 现在的做法一致。

    不要解析 run_all.py 里的列表：它已经从「硬编码清单」改成
    「自动发现 + DOCS 当说明」（原因是硬编码会让新测试**静默不跑**）。
    所以这里也直接 glob，保证两边永远不会不同步。
    """
    names = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))
    if not names:
        raise SystemExit(f"在 {TESTS_DIR} 里没找到 test_*.py")
    return names


#: 实测耗时（秒），用来把慢的摊开、别让某一片爆掉。
#: 数字变了不影响正确性，只影响分片是否均衡。
COST = {
    "test_eval_resolution.py": 42.0,
    "test_engine.py": 19.0,
    "test_chi_4p.py": 22.0,
    "test_models.py": 11.0,
    "test_4p.py": 7.0,
    "test_logs.py": 3.0,
    "test_fastshanten.py": 1.1,
}
DEFAULT_COST = 0.6


def plan(names: list[str], groups: int) -> list[list[str]]:
    """贪心装箱：最贵的先放，装进当前最空的一片。"""
    ordered = sorted(names, key=lambda n: -COST.get(n, DEFAULT_COST))
    bins: list[list[str]] = [[] for _ in range(groups)]
    load = [0.0] * groups
    for n in ordered:
        i = load.index(min(load))
        bins[i].append(n)
        load[i] += COST.get(n, DEFAULT_COST)
    # 片内按 run_all 的原始顺序跑，输出好看
    order = {n: i for i, n in enumerate(names)}
    for b in bins:
        b.sort(key=lambda n: order[n])
    return bins


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分片跑测试（适配杀窗口）")
    ap.add_argument("shard", nargs="?", type=int, default=1)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--list", action="store_true", help="只打印分片方案")
    ap.add_argument("--all-groups", action="store_true", help="依次跑全部片")
    args = ap.parse_args(argv)

    names = load_test_list()
    bins = plan(names, args.groups)

    if args.list:
        for i, b in enumerate(bins, 1):
            tot = sum(COST.get(n, DEFAULT_COST) for n in b)
            print(f"  第 {i} 片（约 {tot:.0f}s，{len(b)} 个）:")
            for n in b:
                print(f"      {n}")
        print(f"  合计 {sum(sum(COST.get(n, DEFAULT_COST) for n in b) for b in bins):.0f}s"
              f" / {len(names)} 个文件 / {args.groups} 片")
        return 0

    if args.all_groups:
        bad = 0
        for i in range(1, args.groups + 1):
            print(f"\n{'='*70}\n▶ 第 {i}/{args.groups} 片\n{'='*70}")
            if main([str(i), "--groups", str(args.groups)]) != 0:
                bad += 1
        print(f"\n{'='*70}")
        print("✅ 全部分片通过" if bad == 0 else f"❌ {bad} 片有失败")
        return 1 if bad else 0

    if not (1 <= args.shard <= args.groups):
        raise SystemExit(f"片号要在 1..{args.groups} 之间")

    group = bins[args.shard - 1]
    failed: list[str] = []
    t0 = time.perf_counter()
    for name in group:
        path = TESTS_DIR / name
        print(f"  ▶ {name}", flush=True)
        r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        if r.returncode != 0:
            failed.append(name)
    dt = time.perf_counter() - t0
    print(f"  第 {args.shard}/{args.groups} 片：{len(group)} 个文件，"
          f"{dt:.0f}s，" + ("全部通过 ✅" if not failed else f"失败：{failed} ❌"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
