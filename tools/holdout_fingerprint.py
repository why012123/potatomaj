"""算 / 校验 HOLDOUT 池成员的指纹（对手池设计 §3 第 1 层）。

为什么需要它
============
「HOLDOUT 成员不可更换」这条纪律，靠人记是记不住的 ✗
-> 把成员名单的 sha256 存成文件 ✓ 改成员 -> 指纹不匹配 -> 测试红 ✓
-> 要**合法**更换 -> 必须**同时**改指纹文件 ✓
   而那次改动在 git 里**看得见** ✓
=> 「悄悄换尺子」变成「一次显式、有记录的改动」✓

用法
====
    python tools/holdout_fingerprint.py            # 校验（不一致则非零退出）
    python tools/holdout_fingerprint.py --update    # 换了成员之后更新指纹
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "aimodel" / "pool_holdout.json"
FP = ROOT / "aimodel" / "pool_holdout.sha256"


def fingerprint(path: Path = POOL) -> str:
    """**只对 members 算指纹** ✓ 不含 _readme/note ✓

    理由 ✗：说明文字改了不该让指纹变 ✓
    否则改一个错字就要动指纹文件 -> 指纹文件失去信号价值 ✓
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    members = data.get("members", [])
    # 规范化：排序后逐项拼接（顺序无关 -> 换顺序不算换尺子 ✓）
    norm = "\n".join(sorted(str(m) for m in members))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def main(argv: list[str]) -> int:
    if "--update" in argv:
        fp = fingerprint()
        FP.write_text(fp + "\n", encoding="utf-8")
        print(f"  已更新 {FP.relative_to(ROOT)}")
        print(f"  members 指纹 = {fp}")
        return 0

    want = FP.read_text(encoding="utf-8").strip() if FP.exists() else ""
    got = fingerprint()
    n = len(json.loads(POOL.read_text(encoding="utf-8")).get("members", []))
    print(f"  members 数 {n}")
    print(f"  记录指纹 {want or '（无）'}")
    print(f"  实际指纹 {got}")
    if not want:
        print("  ❌ 没有指纹文件 ✗（跑 --update 生成）")
        return 1
    if want != got:
        print("  ❌ 指纹不匹配 ✗ -> HOLDOUT 成员被改过 ✓")
        print("     若这是**有意的**：跑 --update 并**在 commit message 里写明换了谁、为什么**")
        print("     若这是**无意的**：把 members 改回去 ✗")
        return 1
    print("  ✅ 匹配（HOLDOUT 成员未被改动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
