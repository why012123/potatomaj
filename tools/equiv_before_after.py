"""同机「改造前 vs 改造后」权重逐位对照 —— 重构安全网的**原始证据**。

为什么需要这个脚本（2026-10-16，安全网改了四次才说清）
====================================================
`test_step_equivalence.py` 是**跨机器**判据 ✓ 而它已降级为容差 1e-4 ✗
-> 它**不能**用来证明「某次重构是保值的」✗（1e-4 不是逐位 ✓）
-> 「保值」这个声明只能靠 **同一台机器上的前后对照** ✓ 本脚本就是干这个的 ✓

用法
====
1. 在**改造前**的工作树里：
       python tools/equiv_before_after.py dump  pre.npz
2. 在**改造后**的工作树里：
       python tools/equiv_before_after.py dump  post.npz
3. 任一工作树里：
       python tools/equiv_before_after.py cmp   pre.npz post.npz

`cmp` 的判据是 `np.array_equal`（逐位）✓ 输出最大绝对差与不相等元素数 ✓
**同机同代码路径下，期望 max_abs_diff == 0 且不相等元素数 == 0** ✓

⚠️ 注意（我第一次跑就踩到）：
`dump` 必须在**目标版本自己的工作树**里执行 ✗
脚本会把自身所在目录插进 `sys.path` ✓ 所以**在两个工作树里跑同一个脚本路径
会导入同一个版本的 aimodel** ✗ -> 那样的对照是假的 ✓
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 与 `test_step_equivalence.py` 保持同一套参数（口径一致才可比）✓
DIM4, N, BATCH, STEPS, SEED = 560, 1536, 256, 3, 123


def _data(seed: int = 7) -> dict:
    from aimodel.encode import ACTION_DIM, HEAD_SIZE
    from aimodel.net import KINDS

    rng = np.random.default_rng(seed)
    kind = np.array(KINDS * (N // len(KINDS) + 1))[:N]
    rng.shuffle(kind)
    mask = np.zeros((N, 34), dtype=np.float32)
    action = np.zeros(N, dtype=np.int64)
    for k in KINDS:
        m = kind == k
        sz = HEAD_SIZE[k]
        n_ok = 14 if k == "discard" else sz
        mask[m, :n_ok] = 1.0
        action[m] = rng.integers(0, n_ok, size=int(m.sum()))
    return {
        "X": rng.normal(size=(N, DIM4)).astype(np.float32), "mask": mask,
        "actfeat": rng.normal(size=(N, 34, ACTION_DIM)).astype(np.float32),
        "action": action, "logp": np.zeros(N, dtype=np.float32), "kind": kind,
        "adv": rng.normal(size=N).astype(np.float32),
        "ret": rng.normal(size=N).astype(np.float32),
    }


def dump(out_path: str) -> int:
    from aimodel.net import Net

    net = Net(dim=DIM4, hidden=(512, 512), seed=SEED, lr=3e-4)
    data = _data()
    for s in range(0, N, BATCH)[:STEPS]:
        sel = slice(s, s + BATCH)
        b = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        b["mode"] = "pg"
        net.step(b, clip=0.2, ent_coef=0.008, vf_coef=0.5,
                 ent_target=None, ent_coef_adapt=False)
    out = Path(out_path)
    np.savez(out, **{n: a for n, a in net.params()})
    print(f"  已写 {out}（{out.stat().st_size / 1024:.0f} KB）"
          f"  来自 {ROOT}")
    return 0


def cmp(a_path: str, b_path: str) -> int:
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    same_keys = set(a.files) == set(b.files)
    print(f"  参数数: {len(a.files)} vs {len(b.files)}   键一致={same_keys}")
    if not same_keys:
        print("  ❌ 键不一致，无法比较")
        return 1

    worst = 0.0
    nbad = 0
    tot = 0
    exact = True
    rows = []
    for k in a.files:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            print(f"  ❌ {k} 形状不同: {x.shape} vs {y.shape}")
            return 1
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        worst = max(worst, float(d.max()))
        nb = int((x != y).sum())
        nbad += nb
        tot += x.size
        if not np.array_equal(x, y):
            exact = False
        if nb:
            rows.append((float(d.max()), k, nb, x.size))

    print(f"  总元素 {tot:,}")
    print(f"  **逐位完全相同 (np.array_equal): {exact}**")
    print(f"  最大绝对差 = {worst:.17g}")
    print(f"  不相等的元素数 = {nbad}")
    if rows:
        print("  有差异的参数：")
        for mx, k, nb, sz in sorted(rows, reverse=True)[:8]:
            print(f"    {k:14} max={mx:.3e}  {nb}/{sz}")
    print()
    if exact:
        print("  ✅ 逐位一致 -> **这次重构是保值的**（同机同对照 ✓）")
        return 0
    print("  ❌ 有差异 -> 重构改变了数值 ✗")
    print("     先查：是不是 KL 分支被激活了（看 kl 是否从 0.0 变非 0）✓")
    print("     再查：搬运错误（逐头循环改批量最常见的错）✓")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "dump":
        return dump(argv[1])
    if len(argv) >= 3 and argv[0] == "cmp":
        return cmp(argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
