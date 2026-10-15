"""打印某个权重的自检信息（供训练脚本收尾调用，也可单独用）。

用法: python tools/weight_check.py maj-pro4v1.1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else "maj-pro4v1.1"
    p = ROOT / "aimodel" / "weights" / f"{name}.npz"
    if not p.exists():
        print(f"  找不到权重：{p}")
        return 1
    import numpy as np

    d = np.load(p, allow_pickle=False)
    m = json.loads(bytes(d["_meta"]).decode())
    print(f"  权重   {p}")
    print(f"  steps  = {m.get('steps')}   lr = {m.get('lr')}   l2 = {m.get('l2')}")
    print(f"  dim    = {m.get('dim')}   hidden = {m.get('hidden')}")
    print(f"  kinds  = {m.get('kinds')}")
    print(f"  参数量 = {sum(d[k].size for k in d.files if k != '_meta')}")

    log = ROOT / "aimodel" / "weights" / "train_log.json"
    try:
        j = json.loads(log.read_text(encoding="utf-8"))
        bc = j.get("bc", {})
        if bc:
            bh = bc.get("best_holdout")
            print(
                f"  最近一次 BC：最优留出 "
                f"{bh:.1%} @ 第 {bc.get('best_epoch')} 轮 / 共 {bc.get('last_epoch')} 轮"
                if isinstance(bh, (int, float))
                else f"  最近一次 BC：{bc}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  （train_log.json 读不到：{type(exc).__name__}: {exc}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
