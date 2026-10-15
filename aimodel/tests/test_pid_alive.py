"""`_pid_alive` 的跨平台测试，**含用假 windll 覆盖 Windows 分支**。

为什么需要这个文件（mac 的要求）
================================
`_pid_alive` 的 Windows 分支**未经实测** ✗ —— server 没有 Windows CI，
mac 无法验证 ✓ 所以它处于「写对了但没跑过」的状态 ✗（第七种形状：
「代码存在」≠「代码被执行过」✓）

对策：**在 Unix 上塞一个假的 `ctypes.windll`**，把 Windows 分支的**逻辑**走一遍 ✓
剩下没覆盖的只有真 `ctypes` 调用的行为 ✓ —— 那一层写清「未实测」✓

这样分清了「逻辑对不对」和「真 API 通不通」两件事 ✓
（和 `kl_anneal` 的分工同理：规则由单测覆盖、接线由静态测试覆盖 ✓）

跑法：``.venv/bin/python aimodel/tests/test_pid_alive.py``
"""

from __future__ import annotations

import ctypes
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import aimodel.train as T  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


#: STILL_ACTIVE
STILL_ACTIVE = 259


class FakeKernel32:
    """假的 kernel32。行为由构造参数决定。"""

    def __init__(self, handle: int, exit_code: int, ok: bool = True):
        self._handle = handle
        self._exit = exit_code
        self._ok = ok
        self.closed: list[int] = []

    def OpenProcess(self, access, inherit, pid):        # noqa: N802
        return self._handle

    def GetExitCodeProcess(self, h, code_ptr):          # noqa: N802
        # code_ptr 是 byref(c_ulong) -> 取 ._obj 写进去
        code_ptr._obj.value = self._exit
        return 1 if self._ok else 0

    def CloseHandle(self, h):                           # noqa: N802
        self.closed.append(h)
        return 1


class FakeWindll:
    def __init__(self, kernel32):
        self.kernel32 = kernel32


def with_fake_windll(kernel32):
    """上下文管理器：给 ctypes 装上假 windll，退出时还原。"""
    class _Ctx:
        def __enter__(self):
            self._had = hasattr(ctypes, "windll")
            self._old = getattr(ctypes, "windll", None)
            ctypes.windll = FakeWindll(kernel32)        # type: ignore[attr-defined]
            return self

        def __exit__(self, *exc):
            if self._had:
                ctypes.windll = self._old                # type: ignore[attr-defined]
            else:
                try:
                    del ctypes.windll                    # type: ignore[attr-defined]
                except AttributeError:
                    pass
            return False

    return _Ctx()


print("== A) Unix 分支（本机就是）==")
check(T._pid_alive(os.getpid()) is True, "自己的 pid 应当活着")
check(T._pid_alive(999_999_999) is False, "不存在的 pid 应当判死")
print("  ✅ 自己=True  不存在的=False")

print("\n== B) ⭐ Windows 分支：用假 windll 覆盖（这是本次新增的覆盖）==")

print("  B1) OpenProcess 返回 0（打不开句柄）-> 判死")
with with_fake_windll(FakeKernel32(handle=0, exit_code=0)):
    check(T._pid_alive(1234) is False, "句柄为 0 时应当判死")
print("      -> False ✓")

print("  B2) 句柄有效 + STILL_ACTIVE -> 判活")
k = FakeKernel32(handle=42, exit_code=STILL_ACTIVE)
with with_fake_windll(k):
    check(T._pid_alive(1234) is True, "STILL_ACTIVE 时应当判活")
check(k.closed == [42], f"拿到句柄后必须 CloseHandle（实际 closed={k.closed}）")
print(f"      -> True ✓  且 CloseHandle 被调用 {k.closed} ✓")

print("  B3) 句柄有效但退出码不是 STILL_ACTIVE -> 判死")
with with_fake_windll(FakeKernel32(handle=43, exit_code=0)):
    check(T._pid_alive(1234) is False, "退出码非 STILL_ACTIVE 时应当判死")
print("      -> False ✓")

print("  B4) GetExitCodeProcess 失败 -> 判死（保守方向的反面：这里是 API 明确失败）")
with with_fake_windll(FakeKernel32(handle=44, exit_code=STILL_ACTIVE, ok=False)):
    check(T._pid_alive(1234) is False, "GetExitCodeProcess 失败时应当判死")
print("      -> False ✓")

print("\n== C) ⭐ 最重要的一条：Windows 分支**自身出错**时必须保守返回 True ==")
print("     （误判「死了」-> 两个写者同时写 ✗；误判「活着」-> 只是拒绝启动 ✓）")


class ExplodingKernel32:
    def OpenProcess(self, *a):                          # noqa: N802
        raise RuntimeError("模拟 ctypes 调用炸了")

    def GetExitCodeProcess(self, *a):                    # noqa: N802
        raise RuntimeError("不该走到这")

    def CloseHandle(self, *a):                           # noqa: N802
        raise RuntimeError("不该走到这")


with with_fake_windll(ExplodingKernel32()):
    r = T._pid_alive(1234)
    check(r is True, f"Windows 分支抛异常时必须保守返回 True，得到 {r}")
print("      -> True ✓（保守方向正确 ✓）")


class BadByrefKernel32:
    """GetExitCodeProcess 拿到的东西不是预期类型 -> byref 可能炸。"""

    def OpenProcess(self, *a):                           # noqa: N802
        return 7

    def GetExitCodeProcess(self, h, ptr):                # noqa: N802
        raise TypeError("模拟 byref 用错")

    def CloseHandle(self, *a):                           # noqa: N802
        return 1


with with_fake_windll(BadByrefKernel32()):
    r = T._pid_alive(1234)
    check(r is True, f"ctypes 用法出错时也必须保守返回 True，得到 {r}")
print("      -> True ✓")

print("\n== D) 假 windll 撤掉后必须回到真实现（别把测试的猴补丁漏出去）==")
# ⚠️ 不能断言 `ctypes.windll is None` ✗ —— 本机就是 Windows，它**本来就存在** ✓
# （第一版这么写，在真 Windows 上必红 ✓ 自己的平台假设错了 ✓）
_left = getattr(ctypes, "windll", None)
check(not isinstance(_left, FakeWindll),
      f"测试结束后 ctypes.windll 不该还是假的 FakeWindll，得到 {type(_left).__name__}")
check(T._pid_alive(os.getpid()) is True, "撤掉后自己的 pid 仍应判活")
check(T._pid_alive(999_999_999) is False, "撤掉后不存在的 pid 仍应判死")
print(f"      ✅ 假 windll 已移除（现在是 {type(_left).__name__}）✓")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过（A~D —— Windows 分支的逻辑已覆盖；真 ctypes 调用仍未实测 ✓）")
