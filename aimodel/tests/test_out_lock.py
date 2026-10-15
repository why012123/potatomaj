"""`acquire_out_lock` 的跨平台回归测试（**Windows 上原来会崩**）。

背景（server 在 Windows 实测踩到的真 bug）
========================================
`_owner_alive` 原来只接 `ProcessLookupError` / `PermissionError`
-> 而 **Windows 不支持信号 0**，`os.kill(pid, 0)` 抛的是 `OSError(WinError 87)` ✗
-> 异常冒出去，**`cmd_rl` 整个崩** ✗

后果特别重：`cmd_rl` **无条件**调 `acquire_out_lock`，而「陈旧锁」在一次
崩溃/被杀之后**必然出现** -> **等于「崩一次之后这台机器再也起不了 RL」** ✗

本测试用临时目录做隔离，**不碰任何真实权重**。
跑法：``.venv/bin/python aimodel/tests/test_out_lock.py``
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.train import acquire_out_lock  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="outlock_"))


def fresh(d: Path, pid_text: str | None = None) -> Path:
    """建一个 out 文件；给了 pid_text 就连带建一个锁。"""
    out = d / "w.npz"
    out.write_bytes(b"stub")
    if pid_text is not None:
        Path(str(out) + ".lock").write_text(pid_text, encoding="utf-8")
    return out


#: 一个几乎不可能存在的 pid（Windows 的 pid 是 4 的倍数且不会这么大）
DEAD_PID = 999_999_999

print("== 1) 没有锁 -> 正常拿到锁，且锁里写的是自己的 pid ==")
d = tmpdir()
try:
    out = fresh(d)
    lock = acquire_out_lock(out)
    check(lock is not None, "没有锁时应当拿到锁")
    check(lock is not None and lock.exists(), "锁文件应当被创建")
    if lock is not None:
        check(lock.read_text().strip() == str(os.getpid()),
              "锁里应当写自己的 pid")
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n== 2) ⭐ 陈旧锁（pid 已不存在）-> 必须**接管**，不能崩 ==")
print("     （这条就是 Windows 上原来崩的那条 ✓）")
d = tmpdir()
try:
    out = fresh(d, pid_text=str(DEAD_PID))
    try:
        lock = acquire_out_lock(out)
        check(lock is not None, "陈旧锁应当被接管（返回锁路径）")
        if lock is not None:
            check(lock.read_text().strip() == str(os.getpid()),
                  "接管后锁里应当换成自己的 pid")
        print("     -> 接管成功 ✓ 没崩 ✓")
    except Exception as e:  # noqa: BLE001
        check(False, f"陈旧锁不该抛异常，却抛了 {type(e).__name__}: {e}")
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n== 3) 活锁（pid 是自己）-> 必须**拒绝**（返回 None），不抢 ==")
d = tmpdir()
try:
    out = fresh(d, pid_text=str(os.getpid()))
    lock = acquire_out_lock(out)
    check(lock is None, f"自己是 owner 时应当拒绝（None），得到 {lock}")
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n== 4) 锁内容坏掉（不是数字）-> 当作陈旧，可接管 ==")
d = tmpdir()
try:
    out = fresh(d, pid_text="not-a-pid")
    lock = acquire_out_lock(out)
    check(lock is not None, "锁内容是垃圾时应当可接管")
finally:
    shutil.rmtree(d, ignore_errors=True)

print("\n== 5) 锁内容为空 -> 当作陈旧，可接管 ==")
d = tmpdir()
try:
    out = fresh(d, pid_text="")
    lock = acquire_out_lock(out)
    check(lock is not None, "空锁应当可接管")
finally:
    shutil.rmtree(d, ignore_errors=True)

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过（5 组 —— 第 2 组就是 Windows 上原来崩的那条）")
