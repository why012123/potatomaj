"""仓库卫生：天鳳牌谱的保护规则**必须一直生效**。

## 为什么要有这条测试

这个仓库是**公开的**，而工作区里会有从 `GL1.xlsm` 抽出来的天鳳牌谱
（`_data/game_*.mjai.jsonl` 之类，几百个文件）✓
`.gitignore` 里那几条规则就是唯一的防线 ✗

**而它们掉过一次** ✗ —— 某次硬重置把规则抹了，之后几十轮没人发现 ✓
如果那期间谁跑了一次 `git add -A`，几百个牌谱就进了公开仓库 ✗✗

「补上规则」只解决当下 ✗ **必须有东西每次都检查它还在** ✓
—— 这就是这条测试存在的理由 ✓

## 检查三件事

1. **关键路径必须被忽略**（`git check-ignore` 逐条验证）
2. **关键路径必须没有被跟踪**（已经被提交进去的要立刻发现）
3. **`.gitignore` 里必须有对应规则文本**（防止规则被改成无效写法）

不在 git 仓库里（比如从 zip 解压出来的环境）会**跳过**，不算失败 ✓
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]

#: 必须被忽略的路径（牌谱数据 + 抽出来的中间产物）
MUST_IGNORE = [
    "GL1.xlsm",                       # 原始牌谱工作簿
    "任何其它.xlsm",
    "_data/x.jsonl",                  # 抽出的牌谱
    "_data/game_0001.mjai.jsonl",
    "data/tenhou_1.json",
    "aimodel/mjai_dump/a.json",       # mjlog.py --out 的默认落点
    "mjai_dump/a.json",
]

#: `.gitignore` 里必须出现的模式（少一条就算掉锁）
MUST_HAVE_RULES = ["*.xlsm", "_data/", "mjai_dump/"]


def _git(*args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""


def main() -> None:
    print("【仓库卫生：天鳳牌谱保护】")
    code, _ = _git("rev-parse", "--git-dir")
    if code != 0:
        print("  ⏭  不在 git 仓库里（从 zip 解压的环境？）—— 跳过")
        return

    failed: list[str] = []

    # ① 关键路径必须被忽略
    not_ignored = []
    for p in MUST_IGNORE:
        rc, _ = _git("check-ignore", "-q", p)
        if rc != 0:
            not_ignored.append(p)
    if not_ignored:
        failed.append(f"这些路径**没有被忽略**：{not_ignored}")
    else:
        print(f"  ✅ {len(MUST_IGNORE)} 条关键路径全部被忽略")

    # ② 关键路径必须没有被跟踪（已提交的立刻发现）
    _, tracked = _git("ls-files")
    bad_tracked = [
        f for f in tracked.splitlines()
        if f.lower().endswith(".xlsm")
        or "_data/" in f
        or "mjai_dump/" in f
        or "tenhou" in f.lower() and f.lower().endswith((".json", ".jsonl"))
    ]
    if bad_tracked:
        failed.append(f"这些**已经被提交进去了**（公开仓库！）：{bad_tracked[:5]}")
    else:
        print("  ✅ 没有任何牌谱类文件被跟踪")

    # ③ .gitignore 里必须有对应规则文本
    gi = ROOT / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    missing = [r for r in MUST_HAVE_RULES if r not in text]
    if missing:
        failed.append(f".gitignore 里缺少规则：{missing}")
    else:
        print(f"  ✅ .gitignore 含全部 {len(MUST_HAVE_RULES)} 条关键规则")

    if failed:
        print()
        for f in failed:
            print(f"  ❌ {f}")
        print()
        print("  → 这些规则掉过一次（硬重置抹的，几十轮没人发现）✗")
        print("     修法：把 .gitignore 里 *_data/ / *.xlsm / mjai_dump/ 那几行补回来")
        sys.exit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
