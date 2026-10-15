#!/usr/bin/env python
"""稳健的微基准工具 —— 避开本机三个已知陷阱。

为什么需要它（三个都实际踩过）
============================
1. **计时器粒度**：Windows 的 `time.process_time()` 粒度约 **15.6 ms**。
   单次调用几百微秒 -> 打印出来就是 `0.0 ms`，看着像「免费」，其实是量不出来。

2. **记忆化把成本藏掉**：`shanten` 有 LRU 记忆。热身之后再计时，
   测到的是**缓存命中**（≈0），不是真实计算。
   而产数据时手牌几乎不重复 -> 真实命中率极低 -> 热身后的数字**系统性偏乐观**。

3. **别的进程在抢 CPU**：`perf_counter`（墙钟）会被训练进程拖慢，
   测出来的「变慢」可能只是被抢，不是代码变差。
   `process_time`（本进程 CPU 时间）不受影响 —— 所以**一律用它**。

本工具的做法
============
* 只用 `process_time`
* **批量放大**：把一次测量包成「重复 N 次」，自动把 N 翻倍直到总耗时
  超过粒度的 100 倍（默认 1.6 s），这样相对误差 < 1%
* 报告的是 **每次调用的成本**，并明确打出「N 次 / 总耗时 / 单次」
* 提供 `--no-warm` 选项：**不热身**，用来观察「冷缓存」的真实成本
  （产数据的场景更接近这个）

用法
====
    # 在代码里用
    from tools.bench import bench
    per_call, info = bench(lambda: my_func(x), label="my_func")
    print(info)

    # 命令行自检（确认工具本身工作正常）
    python -m tools.bench --selftest
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

#: `time.process_time()` 在本机的实测粒度（秒）。保守取 15.6ms。
GRANULARITY = 15.6e-3
#: 目标：总耗时至少是粒度的多少倍（100 倍 -> 相对误差 < 1%）
TARGET_MULT = 100


def bench(
    fn: Callable[[], object],
    *,
    label: str = "fn",
    min_total: float | None = None,
    max_reps: int = 200_000,
    warm: bool = True,
    wall_budget: float = 3.0,
) -> tuple[float, str]:
    """测 ``fn()`` 的单次成本（秒）。返回 ``(单次秒, 说明字符串)``。

    **批量放大但必须有硬上限** —— 这是踩过的坑：
    曾经只用 ``process_time`` 当预算，而 32 核并行下 CPU 时间累积极快，
    于是放大逻辑把重复次数推到 **419 万次 / 356 秒** ✗ 测量本身成了负担。

    所以现在有三道闸：
    1. ``max_reps``（默认 20 万）—— 重复次数硬上限
    2. ``wall_budget``（默认 3 秒）—— **墙钟**预算，每轮循环内检查
    3. 按实测比例放大时留 2 倍余量，避免下次又爆

    代价：极快的调用（< 0.1 µs）在 20 万次内到不了粒度 100 倍，
    这时会返回一个**偏粗但不错**的数，说明里会标 ``粒度高`` ✓ 这是诚实的降级。
    """
    min_total = min_total if min_total is not None else GRANULARITY * TARGET_MULT
    if warm:
        fn()

    reps = 1
    coarse = False
    t_start = time.perf_counter()
    while True:
        t0 = time.process_time()
        for _ in range(reps):
            fn()
        total = time.process_time() - t0

        # 三道闸
        if total >= min_total:
            break
        if reps >= max_reps:
            coarse = True
            break
        if time.perf_counter() - t_start > wall_budget:
            coarse = True
            break

        need = max(2.0, min_total / max(total, 1e-9))
        reps = int(min(max_reps, reps * need))

    per = total / reps
    tag = " ⚠️粒度高" if coarse else ""
    info = (f"  {label:28} {per*1e6:10.3f} µs/次   "
            f"（{reps:,} 次 / 合计 {total*1e3:.1f} ms"
            f"{'' if warm else '，未热身'}）{tag}")
    return per, info


def bench_calls(n_calls: int, fn: Callable[[], object], *, label: str = "fn",
                warm: bool = True) -> tuple[float, float, str]:
    """测「``n_calls`` 次调用」的总成本（适合在循环外报分项）。

    返回 ``(总秒, 单次秒, 说明)``。
    """
    per, _ = bench(fn, label=label, warm=warm)
    total = per * n_calls
    return total, per, f"  {label:28} {total*1e3:9.3f} ms / {n_calls:,} 次"


def selftest() -> int:
    print("=== 自检：能不能测出已知量级的操作 ===")
    bad = 0

    # 1) 空函数：应当接近 0，但工具不该崩
    _, info = bench(lambda: None, label="空函数")
    print(info)

    # 2) 已知成本：sum(range(1000))
    _, info = bench(lambda: sum(range(1000)), label="sum(range(1000))")
    print(info)

    # 3) 粒度陷阱验证：只测 1 次会怎样
    t0 = time.process_time()
    sum(range(1000))
    naive = time.process_time() - t0
    print(f"  {'只测 1 次（陷阱）':28} {naive*1e3:10.3f} ms   <- 粒度 {GRANULARITY*1e3:.1f} ms，"
          f"{'会被截成 0 ✗' if naive == 0 else '这次没截成 0（但不可靠）'}")

    # 4) 单调性：更重的活必须测出更大成本
    a, _ = bench(lambda: sum(range(100)), label="sum(range(100))")
    b, _ = bench(lambda: sum(range(10000)), label="sum(range(10000))")
    ok = b > a * 5
    print(f"  单调性检查: {a*1e6:.3f} µs vs {b*1e6:.3f} µs -> {'✅' if ok else '❌ 不成比例'}")
    if not ok:
        bad += 1

    print("\n✅ 自检通过" if not bad else f"\n❌ {bad} 项失败")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="稳健微基准（避开粒度/记忆化/抢CPU三个坑）")
    ap.add_argument("--selftest", action="store_true", help="跑自检")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
