#!/usr/bin/env python
"""有界的频道监听窗口。

为什么需要它：`ai_comm.py watch` 本身是无限循环，而在本机
「一个会话里无限跑」不现实（回合结束就没了）。这个包装把 watch
变成**可预算的窗口**：跑 N 秒、带时间戳、到点自己收工退出 0。

用法::

    python tools/ai_comm_watch_window.py --seconds 240 --interval 15
    python tools/ai_comm_watch_window.py --seconds 240 --once   # 只查一次就走

退出码：0 = 正常跑完窗口（不管有没有新消息）
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(Path(sys.executable))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="有界频道监听窗口")
    ap.add_argument("--seconds", type=float, default=240.0,
                    help="窗口长度（秒），到点自动收工")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="两次 fetch 之间的间隔（秒）")
    ap.add_argument("--once", action="store_true",
                    help="只查一次就退出（等价于 read）")
    args = ap.parse_args(argv)

    cmd = [PY, str(HERE / "ai_comm.py"), "watch", "--interval", str(args.interval)]
    if args.once:
        cmd.append("--once")

    t0 = time.time()
    print(f"[watch-window] 开始 {datetime.datetime.now():%H:%M:%S}  "
          f"窗口 {args.seconds:.0f}s / 间隔 {args.interval:.0f}s", flush=True)

    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    stopped = threading.Event()

    def killer() -> None:
        if not stopped.wait(args.seconds):
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=killer, daemon=True).start()

    new_msgs = 0
    assert p.stdout is not None
    for line in p.stdout:
        s = line.rstrip()
        if not s:
            continue
        if "【新消息】" in s:
            new_msgs += 1
            print(f"{datetime.datetime.now():%H:%M:%S} *** {s}", flush=True)
        elif "fetch 失败" in s or "新消息" in s:
            print(f"{datetime.datetime.now():%H:%M:%S} | {s}", flush=True)

    stopped.set()
    try:
        p.wait(timeout=10)
    except Exception:  # noqa: BLE001
        p.kill()
    dt = time.time() - t0
    print(f"[watch-window] 结束 {datetime.datetime.now():%H:%M:%S}  "
          f"实跑 {dt:.0f}s  新消息 {new_msgs} 条  exit={p.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
