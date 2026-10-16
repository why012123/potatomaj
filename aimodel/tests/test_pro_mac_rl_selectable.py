"""回归测试：`maj-pro4v1.1_mac_rl` 必须**在平台可选列表里**（不只是文件存在）。

为什么需要它（2026-10-17 用户要求换权重时发现）：
    平台有**两条不同的路径** ✗ 只看一条会漏 ✓：
      · **载入**（`webapp/server/agents.py:85`）：`WEIGHTS_DIR / f"{name}.npz"`
        -> 只看**文件名** ✓
      · **列出可选**（`agents.py:200` 的 `add()`）：
        `spec = variants.get(name)` ✗ —— 名字**没注册就 KeyError** ✓
        而 `add()` 是 `except KeyError: return` ✗ -> **静默跳过** ✓
    -> 后果：权重文件在、sha256 对，**但界面上选不到** ✗
    -> 修法：在 `aimodel/variants.py` 注册 ✓（纯新增 ✓）

负向验证（铁律④）：
    把 `_register_pro_mac_rl()` 那次调用删掉（或让 `name` 拼错）
    -> 本测试必须**红** ✓
"""
from __future__ import annotations

import pathlib
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

NAME = "maj-pro4v1.1_mac_rl"


def main() -> int:
    failed: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'✅' if cond else '🔴'} {label}")
        if not cond:
            failed.append(label)

    from aimodel import variants

    print("─" * 74)
    print("① 注册表里查得到（这是「界面上选得到」的前提）")
    print("─" * 74)
    try:
        spec = variants.get(NAME)
        ok = True
    except KeyError:
        spec = None
        ok = False
    check(ok, f"`variants.get({NAME!r})` 不抛 KeyError")
    if spec is not None:
        check(spec.players == 4, f"players == 4（实际 {spec.players}）")
        check(spec.path.exists(), f"权重文件在：{spec.path.name}")

    print()
    print("─" * 74)
    print("② 平台的可选列表里有它（**这才是用户看得见的那一步**）")
    print("─" * 74)
    from webapp.server import agents as AG

    av = AG.available_models()
    names = [m["name"] if isinstance(m, dict) else str(m) for m in av]
    check(NAME in names, f"`available_models()` 里有 {NAME}")
    row = next((m for m in av if isinstance(m, dict) and m.get("name") == NAME), None)
    if row is not None:
        check(row.get("usable") is True, f"usable == True（实际 {row.get('usable')}）")
        check(not row.get("missing_heads"),
              f"不缺决策头（实际 {row.get('missing_heads')}）")

    print()
    print("─" * 74)
    print("③ 不能因为「文件在」就以为「能选」✗ —— 反面对照")
    print("─" * 74)
    # 一个**故意不存在于注册表**的名字：文件即使真在也不该出现在列表里
    fake = "maj-pro4v1.1_this_name_is_not_registered"
    try:
        variants.get(fake)
        check(False, "未注册的名字应当抛 KeyError")
    except KeyError:
        check(True, "未注册的名字抛 KeyError（-> add() 会静默跳过它）")
    check(fake not in names, "未注册的名字不在可选列表里")

    print()
    print("=" * 74)
    if failed:
        print(f"🔴 **失败 {len(failed)} 项**：")
        for f in failed:
            print(f"    · {f}")
        print("=" * 74)
        return 1
    print("✅ 全部通过")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
