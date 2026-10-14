"""裸跑不许覆盖已训好的权重（P0-6 回归）。

审计证据：`python -m aimodel.train bc` 会用**默认的 256×256 新建网络**
直接覆盖 `weights/maj-normalv1.1.npz` —— 无备份、无确认。
一次手滑就丢掉一个训好的模型（8 个模型每个都要跑很久）。

规则（`train.guard_out_path`）：
* 目标不存在             -> 正常写
* 目标存在 + `--force`    -> 正常写（明确授权）
* 目标存在 + `--resume`   -> 正常写（本来就是接着它训）
* 目标存在 + 都没有       -> 改写成 `<name>.new.npz` 并提示

跑法：``.venv/bin/python aimodel/tests/test_weight_guard.py``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.train import guard_out_path  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def ns(out, **kw):
    return SimpleNamespace(out=str(out), **kw)


print("== 目标不存在：正常写，不改名 ==")
with tempfile.TemporaryDirectory() as td:
    target = Path(td) / "maj-normalv1.1.npz"
    got = guard_out_path(ns(target), resume=False)
    check(got == target, f"不存在时应原样返回，得到 {got}")
    print(f"  ✅ {target.name}（不存在）-> 原样")

    print("\n== 目标存在 + 都不是：必须改写成 .new.npz（止损核心）==")
    target.write_bytes(b"pretend-weights")
    got = guard_out_path(ns(target), resume=False)
    expect = target.with_name(target.stem + ".new.npz")
    check(got == expect, f"应改写成 {expect}，得到 {got}")
    check(got != target, "★ 绝不能返回原路径 —— 那就是覆盖")
    check(target.read_bytes() == b"pretend-weights", "原文件必须**没被动过**")
    print(f"  ✅ 已存在 + 无 --force/--resume -> {got.name}（原文件完好）")

    print("\n== 目标存在 + --force：允许覆盖 ==")
    got = guard_out_path(ns(target, force=True), resume=False)
    check(got == target, f"--force 时应原样返回，得到 {got}")
    print("  ✅ --force -> 原样（明确授权覆盖）")

    print("\n== 目标存在 + resume：允许接着训 ==")
    got = guard_out_path(ns(target), resume=True)
    check(got == target, f"--resume 时应原样返回，得到 {got}")
    print("  ✅ --resume -> 原样（本来就是接着它训）")

    print("\n== 多次裸跑不会越写越乱（.new.npz 再存在也还是它）==")
    expect.write_bytes(b"new")
    got = guard_out_path(ns(target), resume=False)
    check(got == expect, f"仍应指向 {expect.name}，得到 {got}")
    print("  ✅ 第二次裸跑仍写 .new.npz（不会又造一层 .new.new）")

    print("\n== 提示文字必须说清怎么办 ==")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        guard_out_path(ns(target), resume=False)
    msg = buf.getvalue()
    check("--force" in msg, f"提示里必须提 --force，实际：{msg!r}")
    check("--resume" in msg, f"提示里必须提 --resume，实际：{msg!r}")
    check(".new.npz" in msg, f"提示里必须说清新文件名，实际：{msg!r}")
    print("  ✅ 提示包含 --force / --resume / .new.npz")

print("\n== 真实权重文件绝不能被这个函数改动 ==")
real = ROOT / "aimodel" / "weights" / "maj-normalv1.1.npz"
if real.exists():
    before = real.stat().st_size
    guard_out_path(ns(real), resume=False)
    check(real.exists() and real.stat().st_size == before,
          "调用 guard 不能改动真实权重文件")
    print(f"  ✅ {real.name} 未被改动（{before} 字节）")
else:
    print(f"  （跳过：{real} 不存在）")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
