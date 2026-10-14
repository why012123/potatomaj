#!/usr/bin/env python
"""土豆麻将 · 网页对战平台 —— 启动入口。

用法::

    python run.py                          # 默认 127.0.0.1:8790
    python run.py --host 0.0.0.0 --port 8790
    python -m webapp --ai-delay 0.5

环境变量（``MAJ_`` 前缀）见 :mod:`webapp.server.config`。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from webapp.server.app import main as run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
