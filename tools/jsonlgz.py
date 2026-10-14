#!/usr/bin/env python
"""读 gzip JSONL 分片的小工具（**容忍被截断的尾巴**）。

真正的实现在 ``webapp/server/recording.py`` —— 因为写入端也在那边，
两边共用一份逻辑才不会漂（这个坑踩过两次：先是 ``gzip.open`` 直接抛
``EOFError`` 把整个文件判死；换「自己管 GzipFile、出错 break」又把已经
解出来的几百 KB 一起丢了）。这里只是给不 import ``webapp`` 的工具一个入口。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.server.recording import iter_jsonl_gz  # noqa: E402,F401

__all__ = ["iter_jsonl_gz", "iter_dir"]


def iter_dir(games_dir: Path, pattern: str = "*.jsonl.gz") -> Iterator[tuple[dict, Path]]:
    for f in sorted(Path(games_dir).glob(pattern)):
        for rec in iter_jsonl_gz(f):
            yield rec, f
