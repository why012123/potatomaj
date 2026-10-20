#!/usr/bin/env python
"""从天凤 `sc/raw` 抓 **scc（鳳凰卓）** 牌谱 —— 串行、单会话。

    python tools/tenhou_scraw.py --out _data/scraw --limit 0

## 为什么自己写，不用 `houou-logs`

2026-10-21 实测：**这台机器上 github.com 不可达**（超时 / 连接被重置），
而 **tenhou.net 完全可达**（HTTP 200）。

而 `houou-logs` 只是个下载器 —— 它的源就是 `tenhou.net/sc/raw`。
所以直接抓源，少一个依赖，也绕开了 GitHub。

页面结构（从 `sc/raw/` 的 `sw()` 函数读出来的）：

    list.cgi        ->  JS：``list([{"file":"scc20251021.log.gz","size":123}, ...])``
    dat/<file>      ->  实际文件

筛选：``file`` 以 ``scc`` 开头（`sca`=個室 / `scb`=段位戦 / **`scc`=鳳凰卓** /
`scd`=雀荘戦 / `sce`=琥珀卓 / `scf`=公式大会）

## 🔴 三条硬约束

① **单会话、串行** —— 天凤只允许单会话下载，多开违反使用条款、会导致限制或封号。
   本脚本**不支持并发**（没有 --jobs 参数，故意的）。
② **落 `_data/`** —— 那里已被 `.gitignore` 覆盖，**牌谱绝不进 git**（仓库是 public）。
③ 已存在的文件**跳过**（可重复运行、断点续传）。

用户已明确：项目完全私人、无盈利，**风险由用户承担**（Mac 2026-10-21 转达）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://tenhou.net/sc/raw"
LIST = f"{BASE}/list.cgi"
DAT = f"{BASE}/dat"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def _get(url: str, timeout: float = 60.0) -> bytes:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def list_files() -> list[dict]:
    """从 `list.cgi` 解析出 `[{file, size}, ...]`。

    实测格式（2026-10-21）**不是 JSON，是 JS 对象字面量**：

        list([
        {file:'sca20261011.log.gz',size:66787},
        {file:'scc2026102023.html.gz',size:1331},
        ]);
        -> key 无引号 ✓ 字符串用**单引号** ✓ 末尾有分号 ✓

    共 497 项，其中 `scc*` 240 项。**注意 `scc` 的是 `.html.gz`** ——
    它是「当天牌谱的**列表**」（每行一个 `<a href=".../?log=...">牌譜</a>`），
    不是牌谱本身。真正的牌谱要再按那个链接去抓。
    """
    body = _get(LIST).decode("utf-8", "replace")
    m = re.search(r"list\(\s*(\[.*?\])\s*\)", body, re.S)
    if not m:
        raise SystemExit(f"list.cgi 里没找到 list([...])；前 300 字符：{body[:300]!r}")
    raw = m.group(1)
    # JS 字面量 -> JSON：给 key 加引号、单引号字符串换双引号
    j = re.sub(r"(\w+)\s*:", r'"\1":', raw)
    j = re.sub(r":\s*'([^']*)'", r': "\1"', j)
    return json.loads(j)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抓天凤 scc（鳳凰卓）牌谱，串行单会话")
    ap.add_argument("--out", default="_data/scraw", help="落到哪（默认 _data/scraw，已 gitignore）")
    ap.add_argument("--limit", type=int, default=0, help="最多下多少个（0 = 全部）")
    ap.add_argument("--prefix", default="scc", help="文件前缀（scc=鳳凰卓）")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  天凤 sc/raw -> scc（鳳凰卓）")
    print("=" * 70)

    files = list_files()
    mine = [f for f in files if str(f.get("file", "")).startswith(args.prefix)]
    total_bytes = sum(int(f.get("size") or 0) for f in mine)
    print(f"  list.cgi 共 {len(files)} 个文件；{args.prefix}* 有 {len(mine)} 个"
          f"（合计 {total_bytes / 1024 / 1024:.1f} MB）")
    if mine:
        print(f"  最早 {mine[0].get('file')}   最新 {mine[-1].get('file')}")
    print(f"  落到 {out.resolve()}")
    print()

    if args.limit:
        mine = mine[-args.limit:]          # 取最新的若干个
        print(f"  --limit {args.limit} -> 只下最新的 {len(mine)} 个")

    ok = skip = fail = 0
    for i, f in enumerate(mine, 1):
        name = str(f.get("file"))
        dst = out / name
        if dst.exists() and dst.stat().st_size > 0:
            skip += 1
            continue
        try:
            data = _get(f"{DAT}/{name}")
            dst.write_bytes(data)
            # 顺带校验一下能不能解压（.gz 的话）
            if name.endswith(".gz"):
                with gzip.open(dst, "rb") as fh:
                    fh.read(256)
            ok += 1
            print(f"  [{i}/{len(mine)}] ✅ {name}  {len(data) / 1024:.0f} KB")
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(mine)}] 🔴 {name}  {type(e).__name__}: {str(e)[:70]}")
        # ⚠️ 单会话、串行 —— 每次之间停一下，别把源打急
        time.sleep(1.0)

    print()
    print(f"  完成：新下 {ok}  跳过 {skip}  失败 {fail}   （共 {len(mine)}）")
    got = sum(p.stat().st_size for p in out.glob("*") if p.is_file())
    print(f"  目录合计 {got / 1024 / 1024:.1f} MB")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
