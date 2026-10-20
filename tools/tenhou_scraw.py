#!/usr/bin/env python
"""从天凤 `sc/raw` 抓 **scc（鳳凰卓）** 牌谱 —— 串行、单会话。

    # 第一级：抓「每小时一个的牌谱列表」（很小，240 个 ≈ 0.5 MB）
    python tools/tenhou_scraw.py --level 1 --out _data/scraw

    # 第二级：从那些列表里逐条抓真牌谱（慢，每条约 15 KB）
    python tools/tenhou_scraw.py --level 2 --out _data/tenhou_scc

## 两级结构（这是这个源最容易搞错的地方）

    list.cgi                 -> JS：``list([{file:'scc2026102023.html.gz',size:1331}, ...])``
    dat/<file>               -> 一级产物：**当天那一小时的牌谱列表**（HTML，每行一条对局）
      └─ 里面每行是 ``<a href="http://tenhou.net/0/?log=2026102023gm-...">牌譜</a>``
    http://tenhou.net/0/log/?<id>
                             -> 二级产物：**真牌谱**（``<mjloggm ver="2.3">`` XML，约 15 KB）

⚠️ 一级的 ``scc*.html.gz`` **不是牌谱**，是列表 ✗ —— 我第一次当成牌谱下了 5 个才发现 ✓

## 二级的 URL 只能是 `/0/log/?<id>` ✗

实测（2026-10-21）：

    http://tenhou.net/0/log/?<id>        -> 真 mjlog XML（15,278 B）✓
    http://tenhou.net/0/?log=<id>&tw=1   -> **天凤 Web 版网页**（8,152 B HTML）✗
    https://tenhou.net/0/?log=<id>       -> 同上 ✗

## 为什么自己写，不用 `houou-logs`

2026-10-21 实测这台机器：**github.com 不可达**（超时 / 连接被重置），
而 **tenhou.net 完全可达**。`houou-logs` 只是个下载器，源就是 `tenhou.net/sc/raw`，
所以直接抓源，少一个依赖，也绕开了 GitHub。

筛选：``file`` 以 ``scc`` 开头（`sca`=個室 / `scb`=段位戦 / **`scc`=鳳凰卓** /
`scd`=雀荘戦 / `sce`=琥珀卓 / `scf`=公式大会）

## 🔴 三条硬约束

① **单会话、串行** —— 天凤只允许单会话下载，多开违反使用条款、会导致限制或封号。
   本脚本**不支持并发**（没有 --jobs 参数，故意的）；每条之间还 sleep 1 秒。
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
LOG = "http://tenhou.net/0/log/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def _get(url: str, timeout: float = 60.0) -> bytes:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


# --------------------------------------------------------------------------
# 第一级：list.cgi -> 每个小时的 scc*.html.gz
# --------------------------------------------------------------------------


def list_files() -> list[dict]:
    """从 `list.cgi` 解析出 `[{file, size}, ...]`。

    实测格式（2026-10-21）**不是 JSON，是 JS 对象字面量**：

        list([
        {file:'sca20261011.log.gz',size:66787},
        {file:'scc2026102023.html.gz',size:1331},
        ]);
        -> key 无引号 ✓ 字符串用**单引号** ✓ 末尾有分号 ✓
    """
    body = _get(LIST).decode("utf-8", "replace")
    m = re.search(r"list\(\s*(\[.*?\])\s*\)", body, re.S)
    if not m:
        raise SystemExit(f"list.cgi 里没找到 list([...])；前 300 字符：{body[:300]!r}")
    raw = m.group(1)
    j = re.sub(r"(\w+)\s*:", r'"\1":', raw)
    j = re.sub(r":\s*'([^']*)'", r': "\1"', j)
    return json.loads(j)


def level1(out: Path, prefix: str, limit: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()
    mine = [f for f in files if str(f.get("file", "")).startswith(prefix)]
    total = sum(int(f.get("size") or 0) for f in mine)
    print(f"  list.cgi 共 {len(files)} 个；{prefix}* 有 {len(mine)} 个（合计 {total/1024/1024:.1f} MB）")
    if mine:
        print(f"  最早 {mine[0].get('file')}   最新 {mine[-1].get('file')}")
    if limit:
        mine = mine[-limit:]
        print(f"  --limit {limit} -> 只下最新 {len(mine)} 个")

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
            if name.endswith(".gz"):
                with gzip.open(dst, "rb") as fh:
                    fh.read(64)
            ok += 1
            if i % 40 == 0 or i == len(mine):
                print(f"  [{i}/{len(mine)}] ✅ {name}  {len(data)/1024:.0f} KB")
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(mine)}] 🔴 {name}  {type(e).__name__}: {str(e)[:70]}")
        time.sleep(1.0)
    print(f"  一级完成：新下 {ok} 跳过 {skip} 失败 {fail}")
    return 0 if fail == 0 else 1


# --------------------------------------------------------------------------
# 第二级：scc*.html.gz -> 逐条真牌谱
# --------------------------------------------------------------------------

_LOG_ID = re.compile(r"\?log=([0-9A-Za-z\-]+)")


def parse_log_ids(list_dir: Path) -> list[str]:
    """从一级产物里抽出全部对局 id，去重且保序。"""
    seen: dict[str, None] = {}
    for f in sorted(list_dir.glob("scc*.html.gz")):
        try:
            text = gzip.decompress(f.read_bytes()).decode("utf-8", "replace")
        except Exception:                            # noqa: BLE001
            continue
        for lid in _LOG_ID.findall(text):
            seen.setdefault(lid, None)
    return list(seen)


def level2(list_dir: Path, out: Path, limit: int, sleep: float, check: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    ids = parse_log_ids(list_dir)
    print(f"  一级产物里共 **{len(ids):,}** 条对局")
    if not ids:
        print("  🔴 没解析出对局 —— 先跑 --level 1")
        return 1
    if limit:
        ids = ids[:limit]
        print(f"  --limit {limit} -> 只抓前 {len(ids)} 条")

    ok = skip = fail = 0
    got_bytes = 0
    t0 = time.time()
    for i, lid in enumerate(ids, 1):
        dst = out / f"{lid}.xml.gz"
        if dst.exists() and dst.stat().st_size > 0:
            skip += 1
            got_bytes += dst.stat().st_size
            continue
        try:
            raw = _get(f"{LOG}?{lid}")
            if b"mjloggm" not in raw[:400]:
                # 不是牌谱（可能是网页 / 限流页）-> 不落盘，记失败
                fail += 1
                print(f"  [{i}/{len(ids)}] 🔴 {lid}  不是 mjlog（{len(raw)} B）")
                time.sleep(sleep)
                continue
            dst.write_bytes(gzip.compress(raw))
            ok += 1
            got_bytes += dst.stat().st_size
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(ids)}] 🔴 {lid}  {type(e).__name__}: {str(e)[:60]}")
        if i % check == 0 or i == len(ids):
            el = time.time() - t0
            rate = ok / el if el > 0 else 0
            print(f"  [{i}/{len(ids)}] 新下 {ok} 跳过 {skip} 失败 {fail} "
                  f"· {got_bytes/1024/1024:.1f} MB · {el/60:.1f} 分钟 · {rate:.2f} 条/秒")
        time.sleep(sleep)

    print()
    print(f"  二级完成：新下 {ok} 跳过 {skip} 失败 {fail}")
    print(f"  目录合计 {got_bytes/1024/1024:.1f} MB")
    return 0 if fail == 0 else 1


def verify(dir_: Path, n: int = 5) -> None:
    """抽几个 gz 解开看，并折算大约多少局。"""
    fs = sorted(dir_.glob("*.xml.gz"))
    if not fs:
        print(f"  {dir_} 里没有 .xml.gz")
        return
    total = sum(f.stat().st_size for f in fs)
    print(f"  {len(fs):,} 个文件，合计 {total/1024/1024:.1f} MB"
          f"（平均 {total/max(1,len(fs))/1024:.1f} KB/个）")
    good = 0
    for f in fs[:n]:
        try:
            text = gzip.decompress(f.read_bytes()).decode("utf-8", "replace")
            has = "<mjloggm" in text
            kyoku = text.count("<INIT")
            good += 1 if has else 0
            print(f"    {'✅' if has else '🔴'} {f.name}  {len(text):,} 字符  {kyoku} 局")
        except Exception as e:                       # noqa: BLE001
            print(f"    🔴 {f.name}  {type(e).__name__}")
    print(f"  抽检 {min(n, len(fs))} 个，{good} 个是可解的 mjlog")
    print(f"  折算：约 {len(fs) * 8:,} 局（按每个牌谱文件约 8 局估）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抓天凤 scc（鳳凰卓）牌谱，串行单会话")
    ap.add_argument("--level", type=int, default=1, choices=(1, 2),
                    help="1=抓每小时的列表；2=从列表逐条抓真牌谱")
    ap.add_argument("--out", default=None, help="落到哪（默认按 level：_data/scraw 或 _data/tenhou_scc）")
    ap.add_argument("--list-dir", default="_data/scraw", help="--level 2 时从哪个目录读一级产物")
    ap.add_argument("--limit", type=int, default=0, help="最多几个/几条（0 = 全部）")
    ap.add_argument("--prefix", default="scc", help="文件前缀（scc=鳳凰卓）")
    ap.add_argument("--sleep", type=float, default=1.0, help="每次请求之间停多久（秒）")
    ap.add_argument("--check", type=int, default=50, help="每多少条打印一次进度")
    ap.add_argument("--verify", action="store_true", help="只做抽检与折算，不下载")
    args = ap.parse_args(argv)

    print("=" * 70)
    print(f"  天凤 sc/raw · level {args.level} · scc（鳳凰卓）")
    print("=" * 70)

    if args.level == 1:
        out = Path(args.out or "_data/scraw")
        print(f"  落到 {out.resolve()}")
        return level1(out, args.prefix, args.limit)

    out = Path(args.out or "_data/tenhou_scc")
    print(f"  一级目录 {Path(args.list_dir).resolve()}")
    print(f"  落到     {out.resolve()}")
    if not args.verify:
        rc = level2(Path(args.list_dir), out, args.limit, args.sleep, args.check)
        if rc != 0:
            return rc
    verify(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
