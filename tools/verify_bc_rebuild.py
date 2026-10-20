#!/usr/bin/env python
"""核「重抽后的 BC 数据集」是否符合 `chisel` 头的规格（只读，不写任何东西）。

    python tools/verify_bc_rebuild.py --old <旧.npz> --new <新.npz>

## 背景：为什么要重抽

`maj-pro4v1.2` 要给「吃」加一个 `chisel` 头（吃成什么样），
但旧数据集里 **chi 的 action 只有 0/1 两个值**（只有「吃不吃」，没有「吃哪一种」）：

    bc_all.npz: chi 决策 294,338 条，action = {0: 266,971, 1: 27,367}

所以新头没有标签可学 -> 必须从原始 mjai 日志重抽，
把 chi 的 action 从「二值」改成「该局面的第几种吃法」。
**其余 8 个头一行不动。**

## 验收四条（Mac 2026-10-20 定，第 ④ 条是他补的、也是最关键的一条）

    ① rows 与旧版**相同**（决策集合没变，只是 chi 的标签变细了）
    ② 七列 `X / mask / actfeat / logp / kind / player / ret` 的 sha256 **逐位相同**
    ③ **只有 `action` 一列不同**，且差异行**全部**是 `kind == "chi"`
    ④ 新 `action` 在 chi 行里**出现了 > 1 的值**

## 为什么第 ④ 条不能省

只核「action 列变了」不够 ✗ —— 若新旧都是 0/1、只是数值漂移了一点点 ✗
那**不是我们要的东西** ✓ 必须出现 >1 的取值 ✓ 才证明「第几种吃法」真的进来了 ✓
（所以本脚本会把 chi 行的 action **分布**打出来 ✓ 分布本身就能看出吃法是否真有多种 ✓）

## 退出码

0 = 四条全过；1 = 有任何一条不过（给 CI / 定时任务判断用）
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

#: 必须逐位相同的七列（`action` 是唯一允许变的那一列）
SAME_COLS = ("X", "mask", "actfeat", "logp", "kind", "player", "ret")

#: 唯一允许变的那一列
VAR_COL = "action"

PASS = 0
FAIL = 0


def ok(name: str, cond: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}   {detail}")
    return cond


def col_sha(a: np.ndarray) -> str:
    """一列的 sha256（按**实际字节**算；dtype 不同也算不同 ✓ 这是有意的）。"""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="核 BC 数据集重抽是否符合 chisel 规格")
    ap.add_argument("--old", required=True, help="旧版 npz（chi 的 action 是二值）")
    ap.add_argument("--new", required=True, help="重抽后的 npz（chi 的 action 是第几种吃法）")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("  BC 数据集重抽验收（chisel 头）")
    print("=" * 74)

    zo = np.load(args.old, allow_pickle=True)
    zn = np.load(args.new, allow_pickle=True)
    print(f"  旧: {args.old}")
    print(f"  新: {args.new}")
    print(f"  旧列: {list(zo.files)}")
    print(f"  新列: {list(zn.files)}")

    # ---- ⓪ 列集合必须先一致（否则后面都是废话）---------------------------
    print("\n⓪ 列集合")
    same_cols = set(zo.files) == set(zn.files)
    ok("新旧列集合相同", same_cols,
       f"缺 {set(zo.files) - set(zn.files)} / 多 {set(zn.files) - set(zo.files)}")
    if not same_cols:
        print("\n  列都不一样，后面不核了。")
        return 1

    # ---- ① rows 相同 ----------------------------------------------------
    print("\n① 行数")
    no, nn = len(zo[VAR_COL]), len(zn[VAR_COL])
    ok(f"rows 相同（旧 {no:,} / 新 {nn:,}）", no == nn, f"{no:,} vs {nn:,}")
    if no != nn:
        print("\n  行数不同 -> 决策集合变了 ✗ 与「只改 chi 标签」不符；后面跳过。")
        return 1

    # ---- ② 七列 sha256 逐位相同 -----------------------------------------
    print("\n② 七列 sha256（必须逐位相同）")
    for c in SAME_COLS:
        if c not in zo.files:
            ok(f"{c} 存在", False, "旧版没有这列")
            continue
        so, sn = col_sha(zo[c]), col_sha(zn[c])
        ok(f"{c:8} {so[:16]}… {'==' if so == sn else '!='} {sn[:16]}…",
           so == sn, "这一列不该变")

    # ---- ③ 只有 action 不同，且差异行全是 chi ----------------------------
    print("\n③ 只有 action 一列不同；差异行必须全是 kind=='chi'")
    ao, an = np.asarray(zo[VAR_COL]), np.asarray(zn[VAR_COL])
    diff = ao != an
    n_diff = int(diff.sum())
    ok(f"action 确实变了（{n_diff:,} 行 / 共 {no:,}：{n_diff / max(1, no) * 100:.1f}%）",
       n_diff > 0, "一行都没变 -> 重抽没生效")
    if n_diff:
        kinds = np.asarray(zo["kind"])
        bad = np.unique(kinds[diff])
        only_chi = set(bad.tolist()) <= {"chi"}
        ok("差异行全部是 kind=='chi'", only_chi,
           f"混进了 {sorted(set(bad.tolist()) - {'chi'})}")
        # 反向：chi 行是否**都**变了？（不强制，因为有些 chi 的 1/0 恰好等于新槽位）
        n_chi = int((kinds == "chi").sum())
        print(f"        （chi 行共 {n_chi:,}，其中 {n_diff:,} 行变了；"
              f"少数未变属正常：新槽位恰好等于旧值 0/1 时数值不变 ✓）")

    # ---- ④ 新 action 在 chi 行出现 > 1 的值 ------------------------------
    print("\n④ 新 action 在 chi 行里出现 > 1 的值（**最关键的一条**）")
    kinds_n = np.asarray(zn["kind"])
    chi_mask = kinds_n == "chi"
    old_chi_vals = np.unique(ao[chi_mask]) if chi_mask.any() else np.array([])
    new_chi_vals = np.unique(an[chi_mask]) if chi_mask.any() else np.array([])
    print(f"        旧版 chi 的 action 取值: {old_chi_vals.tolist()}")
    print(f"        新版 chi 的 action 取值: {new_chi_vals.tolist()}")
    ok("新版 chi 的 action 出现了 > 1 的值",
       bool((new_chi_vals > 1).any()) if new_chi_vals.size else False,
       "仍然只有 0/1 -> 说明只抽到了「吃不吃」，没抽到「吃哪一种」")

    # ---- 附：分布（人工看一眼就知道吃法是否真有多种）--------------------
    print("\n附：chi 行的 action 分布（新 vs 旧）")
    if chi_mask.any():
        for tag, arr in (("旧", ao), ("新", an)):
            vals, cnts = np.unique(arr[chi_mask], return_counts=True)
            dist = "  ".join(f"{int(v)}: {int(c):,}" for v, c in zip(vals, cnts))
            print(f"        {tag}  {dist}")
        n_multi = int((an[chi_mask] > 1).sum())
        print(f"        新：吃法槽位 > 1 的有 {n_multi:,} 行"
              f"（占 chi 行 {n_multi / max(1, chi_mask.sum()) * 100:.1f}%）")
    else:
        print("        （没有 kind=='chi' 的行？那这个数据集不该用来训 chisel 头）")

    print("\n" + "=" * 74)
    verdict = "四条全过" if FAIL == 0 else f"[FAIL] {FAIL} 条不过"
    print(f"  {verdict}   （通过 {PASS} / 失败 {FAIL}）")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
