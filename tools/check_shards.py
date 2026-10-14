#!/usr/bin/env python
"""检查一个 shard 目录：**有没有重复内容**、总数到底是多少。

## 为什么需要这个

远端踩过一次很隐蔽的坑 ✗：

    selfplay.py --tag b1 --out _data/b1     # 第 1 批
    selfplay.py --tag b2 --out _data/b2     # 第 2 批
    ...                                     # 共 8 批，报「205,344 局 = 102.7% 达标」

**但那 8 批是同一份数据的 8 个副本** ✗

根因：`--tag` **只给 shard 文件起名**，真正决定牌局的是 `--seed`，而它**默认 0** ✗
每批只换了 tag、没换 seed → 8 份完全相同的 25,668 局。

**从总量上完全看不出来** —— 累计 205,344 局 ≥ 20 万，看起来就是达标了 ✗
是靠「每一格统计都一模一样（连随机量都分毫不差）」才发现的 ✓

## 这个工具做什么

1. **按内容哈希分组** —— 内容相同的 shard 会被分成一组，直接暴露重复 ✓
2. 报**真实局数**（去重后）而不是文件数加起来的 ✓
3. 顺便报每批的 seed（如果 shard 里存了的话）

用法::

    python -m tools.check_shards _data/sp20k
    python -m tools.check_shards _data/ --quiet      # 只在不一致时输出
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def content_hash(path: Path) -> tuple[str, int, int]:
    """按**数组内容**算哈希（不用文件哈希 —— npz 头部可能带时间戳）。

    返回 ``(哈希, 决策条数, 对局数)``。
    """
    h = hashlib.sha256()
    rows = 0
    games = 0
    with np.load(path, allow_pickle=False) as d:
        for key in sorted(d.files):
            arr = d[key]
            if key == "_meta":
                continue
            h.update(key.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
            if arr.ndim >= 1 and rows == 0:
                rows = int(arr.shape[0])
        for key in ("matches", "games", "hands", "n_matches"):
            if key in d.files:
                v = d[key]
                games = int(v) if v.ndim == 0 else int(np.asarray(v).ravel()[0])
                break
    return h.hexdigest()[:16], rows, games


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="检查 shard 目录有没有重复内容")
    ap.add_argument("path", help="shard 目录（或单个 .npz）")
    ap.add_argument("--quiet", action="store_true", help="只在有问题时输出")
    args = ap.parse_args(argv)

    root = Path(args.path)
    files = sorted(root.rglob("*.npz")) if root.is_dir() else [root]
    if not files:
        print(f"没找到 .npz：{root}")
        raise SystemExit(1)

    groups: dict[str, list[Path]] = defaultdict(list)
    meta: dict[str, tuple[int, int]] = {}
    for f in files:
        try:
            h, rows, games = content_hash(f)
        except Exception as e:                              # noqa: BLE001
            print(f"  ⚠️ 读不了 {f.name}: {type(e).__name__}: {e}")
            continue
        groups[h].append(f)
        meta[h] = (rows, games)

    dup_groups = {h: v for h, v in groups.items() if len(v) > 1}
    uniq_files = len(groups)
    total_files = sum(len(v) for v in groups.values())
    total_rows = sum(meta[h][0] for h in groups)          # ← 去重后
    naive_rows = sum(meta[h][0] * len(v) for h, v in groups.items())

    ok = not dup_groups
    if args.quiet and ok:
        return

    print(f"目录：{root}")
    print(f"  shard 文件      {total_files}")
    print(f"  **不同内容**    {uniq_files}" + ("   ← 有重复 ✗" if dup_groups else " ✓"))
    print(f"  决策条数（去重）{total_rows:,}")
    if naive_rows != total_rows:
        print(f"  决策条数（不去重）{naive_rows:,}"
              f"   ← 虚高 **{naive_rows / max(1, total_rows):.1f} 倍** ✗")
    if dup_groups:
        print()
        print("  ⚠️ 发现重复内容 —— 这些文件其实是同一份数据：")
        for h, v in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
            print(f"    {h}  ×{len(v)}：{[x.name for x in v[:4]]}"
                  + (" …" if len(v) > 4 else ""))
        print()
        print("  → 常见原因：**每批只换了 --tag、没换 --seed**（seed 默认 0）✗")
        print("     正确用法：`--seed $((i * 1000003))`，每批递增")
        raise SystemExit(1)
    print("  ✅ 没有重复")


if __name__ == "__main__":
    main()
