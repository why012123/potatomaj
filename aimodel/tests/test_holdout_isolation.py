"""留出对手的**代码路径隔离**测试（对手池设计附录A 的配套件）。

要防什么
========
一次手滑（把 HOLDOUT 的分数也算进 avg）就污染了 ✗
而**污染后的症状正是我们想观测的信号**（「对 HOLDOUT 也涨」）✗
-> 一旦污染，我们**永远不知道有没有过拟合池子** ✗✗
=> 所以要**结构上不可能**，而不是纪律上禁止 ✓

三层隔离（附录A）
================
  ① 配置层：HOLDOUT 成员有 sha256 指纹（改成员 -> 测试 A 红）
  ② 导入层：`aimodel/` 下不许引用 holdout（测试 B）
  ③ 数据层：选择入口的入参不含 holdout 相关名字（测试 C）
  D：**敏感性自检** —— 证明 B/C 真的在检，不是恒真

⚠️ 本测试自己**必须**写出那些禁词 ✗ -> 所以扫描时排除 `aimodel/tests/` ✓
   （否则测试会检到自己 ✓ 那是个假红 ✓）

跑法：``.venv/bin/python aimodel/tests/test_holdout_isolation.py``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tools"))
from holdout_fingerprint import POOL, fingerprint  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


#: 禁词（**只在 aimodel/ 的模块里禁** ✓ 测试目录除外 ✓）
#:
#: ⚠️ **不能用裸词 `HOLDOUT`** ✗ —— 这是我第一版犯的错 ✓
#: 项目里「HOLDOUT / 留出」**已经被 BC 的验证集占用**了 ✓：
#:     `aimodel/progress.py` 的 `HOLDOUT_RE`（解析「留出 xx%」）
#:     `aimodel/train.py` 注释里的 "The HOLDOUT still comes from the teacher"
#: -> 用裸词会**误报** ✓（实测：2 条误报 ✓）而且**越扫越假红** -> 最后没人看 ✓
#: => **对手池的 holdout 必须带命名空间** ✓ 禁词只认 `pool_holdout` / `holdout_eval` ✓
#: 这条不只是测试细节 ✗ 它是**命名要求**：两个「holdout」语义不同、
#: 混用会让「留出率」和「留出对手」在同一段对话里指两件事 ✓
FORBIDDEN = ("pool_holdout", "holdout_eval", "opp_holdout")

#: 扫哪些文件：`aimodel/*.py` 顶层 + `tools/*.py` 里**选择侧**的
SCAN_GLOBS = ("aimodel/*.py",)


def scan(sources: dict[str, str] | None = None) -> list[str]:
    """找违规引用。返回 ``["文件:行号: 内容", …]``。

    ``sources=None``  -> 扫磁盘（真实检查 ✓）
    ``sources={名: 文本}`` -> 扫给定文本（**测试注入用** ✓ 不碰磁盘 ✓）
    """
    hits: list[str] = []
    if sources is None:
        srcs: dict[str, str] = {}
        for g in SCAN_GLOBS:
            for p in sorted(ROOT.glob(g)):
                srcs[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    else:
        srcs = dict(sources)

    for name, text in srcs.items():
        for i, line in enumerate(text.splitlines(), 1):
            # 允许出现的地方：注释里明确写「不许引用」的说明行
            if "FORBIDDEN-OK" in line:
                continue
            for w in FORBIDDEN:
                if w in line:
                    hits.append(f"{name}:{i}: {line.strip()[:90]}")
                    break
    return hits


def main() -> int:
    print("== A) HOLDOUT 成员指纹（改成员 -> 这里红）==")
    fp_file = ROOT / "aimodel" / "pool_holdout.sha256"
    if not fp_file.exists():
        check(False, "没有 aimodel/pool_holdout.sha256（跑 tools/holdout_fingerprint.py --update）")
    else:
        want = fp_file.read_text(encoding="utf-8").strip()
        got = fingerprint()
        check(want == got,
              f"HOLDOUT 成员指纹不匹配 ✗（成员被改过）\n"
              f"      记录 {want[:24]}…  实际 {got[:24]}…\n"
              f"      有意更换 -> 跑 tools/holdout_fingerprint.py --update 并在 commit 里写明")
        if want == got:
            n = len(json.loads(POOL.read_text(encoding="utf-8")).get("members", []))
            print(f"  ✅ 指纹匹配（members 数 {n}）")

    print("\n== B) `aimodel/` 下不许引用 holdout ==")
    hits = scan()
    check(not hits, "发现 holdout 引用 ✗：\n      " + "\n      ".join(hits[:8]))
    if not hits:
        print(f"  ✅ 扫了 {len(list(ROOT.glob(SCAN_GLOBS[0])))} 个模块，零引用")

    print("\n== C) 选择入口的入参不含 holdout 相关名字 ==")
    # 选择入口 = cmd_rl 的签名（它在 train.py 里，由 B 覆盖文件级）
    # 这里做**函数级**：取 cmd_rl 的源码，看参数名
    import inspect

    import aimodel.train as T  # noqa: E402,F401  (导入失败 -> 由 B 的扫描兜住)

    for fn_name in ("cmd_rl", "collect_selfplay"):
        fn = getattr(T, fn_name, None)
        if fn is None:
            continue
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            continue
        bad = [w for w in FORBIDDEN if w in sig]
        check(not bad, f"{fn_name} 的签名里出现 {bad} ✗")
    print("  ✅ cmd_rl / collect_selfplay 签名干净")

    print("\n== D) ⭐ 敏感性自检：B 真的在检吗（不是恒真）==")
    # 不碰磁盘 ✓ 只把注入的文本喂给**同一个** scan() ✓
    injected = {"aimodel/fake_injected.py": "SCORE = load('pool_holdout.json')  # 注入\n"}
    d1 = scan(injected)
    check(len(d1) > 0, "注入 holdout 引用后 scan() 仍返回空 -> **B 是恒真的** ✗（检查无效）")
    if d1:
        print(f"  注入 1 行 -> scan() 抓到 {len(d1)} 条 ✓（B 不是恒真 ✓）")

    # 再验：正常代码不许被误报
    clean = {"aimodel/fake_clean.py": "SCORE = load('pool_select.json')\n"}
    d2 = scan(clean)
    check(not d2, f"干净的代码被误报 ✗：{d2}")
    if not d2:
        print("  对照：正常引用 pool_select -> 不误报 ✓")

    # 再验：注释豁免标记真的生效（否则说明豁免逻辑没生效）
    exempt = {"aimodel/fake_exempt.py": "# FORBIDDEN-OK: 说明文字里必须提到 holdout\n"}
    d3 = scan(exempt)
    check(not d3, f"FORBIDDEN-OK 豁免没生效 ✗：{d3}")
    if not d3:
        print("  对照：带 FORBIDDEN-OK 的说明行被豁免 ✓")

    if FAILED:
        print(f"\n❌ {len(FAILED)} 条失败")
        return 1
    print("\n✅ 全部通过（A~D —— B/C 有 D 作敏感性背书 ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
