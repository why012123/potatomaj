#!/usr/bin/env python
"""多 seed 复测：同一份配置跑 N 个种子，报 **mean ± SD**（而不是单个 Δ）。

## 为什么必须要这个工具

实测（2026-10-20，四麻 BC，25% 数据）：

    同数据 / 同 batch / 同轮数，**只换训练种子**
    配对赛差值 = **+2.051**（SE ±1.935，不显著）
    -> **运行间噪声的量级 ≈ 2 段位分** ✗

**而 arena 是确定性的**（同牌山）—— 所以配对赛给的 SE 只是「**评估精度**」，
**不含「训练本身的随机性」** ✗

    ⚠️ 确定性评估会给出一个**精确但有偏**的数 ✓
       很容易让人高估实验可靠度 ✗（本项目的形状⑯：精度掩盖了不确定性）

=> 任何「配置 A vs 配置 B」的比较，**每边至少 3 个种子** ✓ 报 mean ± SD ✗
   单 seed 的 ±4 分差异**什么都不能说明** ✓

## 用法

    # 比较「25% 数据」与「100% 数据」两个配置，各 3 个种子，对照同一个参考模型
    python tools/multiseed.py --bc-data _data/pro4/bc_all.npz \
        --ref aimodel/weights/frac_exp/frac100g.npz \
        --configs 0.25,1.0 --seeds 0,1,2 --epochs 2 --batch 256

    # 只跑一个配置、只量方差
    python tools/multiseed.py --bc-data ... --ref <模型> --configs 0.25 --seeds 0,1,2

## 输出

每个 (配置, 种子) 一个权重 + 一次配对赛；最后按配置汇总::

    配置 frac=0.25  N=3
      seed 0  Δ=-4.11  p=0.021
      seed 1  Δ=-2.06  p=0.145
      seed 2  Δ=-5.30  p=0.008
      **mean = -3.82   SD = 1.63   SEM = 0.94**

判据（预注册）：两个配置的 **mean 之差 > 2 × 合并 SD** 才算「有差异」✓
（比 p 值保守 ✓ 但先用这个 ✓ 因为它把「训练不确定性」显式算进去了 ✓）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(msg: str) -> None:
    print(f"❌ {msg}", flush=True)
    raise SystemExit(1)


def read_meta(path: Path) -> tuple[int, str, int]:
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        m = json.loads(bytes(z["_meta"]).decode())
    return int(m["dim"]), ",".join(str(h) for h in m["hidden"]), int(m.get("steps", 0))


def run(cmd: list[str], logpath: Path) -> str:
    logpath.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    logpath.write_text(out, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-10:])
        fail(f"命令失败（exit {proc.returncode}）：{' '.join(cmd)}\n{tail}")
    return out


def parse_paired(text: str) -> tuple[float, float]:
    d = re.search(r"配对差值（A−B）\s+(-?\d+(?:\.\d+)?)", text)
    p = re.search(r"单侧 p ≈ (\d+(?:\.\d+)?)", text)
    if not d or not p:
        raise ValueError("解析不出「配对差值 / 单侧 p」—— 判据不明，不能继续")
    return float(d.group(1)), float(p.group(1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="多 seed 复测：报 mean ± SD")
    ap.add_argument("--bc-data", required=True, help="BC 数据集 npz")
    ap.add_argument("--ref", required=True, help="所有种子都跟它配对赛（同一个参考 ✓）")
    ap.add_argument("--configs", default="1.0",
                    help="逗号分隔的 --bc-frac 值，例如 0.25,1.0")
    ap.add_argument("--seeds", default="0,1,2", help="逗号分隔的训练种子")
    ap.add_argument("--bc-games", type=int, default=40000, help="数据集中共有多少局")
    ap.add_argument("--bc-frac-seed", type=int, default=0, help="取子集的种子（各配置共用 ✓）")
    ap.add_argument("--model", default="maj-pro4v1.1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--outdir", default=str(ROOT / "_data" / "multiseed"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=1500,
                    help="每次配对赛上限（越小越快、CI 越宽 ✓）")
    ap.add_argument("--skip-train", action="store_true",
                    help="权重已存在就跳过训练（重跑配对赛时用）")
    args = ap.parse_args(argv)

    ref = Path(args.ref)
    if not ref.is_file():
        fail(f"参考模型不存在：{ref}")
    r_dim, r_hid, r_steps = read_meta(ref)
    print(f"参考模型：{ref}  dim={r_dim} hidden=[{r_hid}] steps={r_steps}", flush=True)

    configs = [float(x) for x in args.configs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    outdir = Path(args.outdir)
    py = sys.executable
    summary: dict[float, list[tuple[int, float, float]]] = {}

    for frac in configs:
        summary[frac] = []
        for seed in seeds:
            tag = f"f{frac:g}_s{seed}"
            ckpt = outdir / f"{tag}.npz"
            print(f"\n---- 配置 frac={frac:g}  seed={seed} ----", flush=True)

            if not (args.skip_train and ckpt.is_file()):
                run(
                    [
                        py, "-m", "aimodel.train", "bc", f"--model={args.model}",
                        "--bc-data", args.bc_data,
                        "--bc-frac", str(frac), "--bc-games", str(args.bc_games),
                        "--bc-frac-seed", str(args.bc_frac_seed),
                        "--epochs", str(args.epochs), "--hidden", args.hidden,
                        "--lr", str(args.lr), "--l2", str(args.l2),
                        "--batch", str(args.batch), "--seed", str(seed),
                        "--out", str(ckpt),
                        # ⚠️ --log 必须显式给到 outdir ✗
                        #    默认是 `aimodel/weights/train_log.json` ✓ 会**覆盖仓库里那个文件** ✗
                        #    （我 2026-10-20 就因为漏了它覆盖过一次 ✓ 已用 git checkout 恢复 ✓）
                        "--log", str(outdir / f"{tag}.json"),
                    ],
                    logpath=outdir / f"{tag}.stdout",
                )
            g_dim, g_hid, g_steps = read_meta(ckpt)
            if (g_dim, g_hid) != (r_dim, r_hid):
                fail(f"{tag} 架构与参考不一致：dim={g_dim} hidden=[{g_hid}] —— 不能比")
            print(f"    ✅ 架构一致 dim={g_dim} hidden=[{g_hid}] steps={g_steps}", flush=True)

            out = run(
                [
                    py, "-m", "aimodel.arena_test", "--a", str(ckpt), "--b", str(ref),
                    "--target-delta", "2", "--jobs", str(args.jobs),
                    "--max-games", str(args.max_games),
                ],
                logpath=outdir / f"{tag}.pair.txt",
            )
            try:
                diff, pval = parse_paired(out)
            except ValueError as e:
                fail(f"{e}（见 {outdir / f'{tag}.pair.txt'}）")
            print(f"    Δ = {diff:+.3f}   p ≈ {pval}", flush=True)
            summary[frac].append((seed, diff, pval))

    print("\n" + "=" * 70)
    print("汇总（Δ 是「该配置 − 参考模型」；负数 = 比参考弱）")
    print("=" * 70)
    for frac in configs:
        rows = summary[frac]
        if not rows:
            continue
        ds = [r[1] for r in rows]
        mean = statistics.fmean(ds)
        sd = statistics.stdev(ds) if len(ds) > 1 else float("nan")
        sem = sd / math.sqrt(len(ds)) if len(ds) > 1 else float("nan")
        print(f"\n  配置 frac={frac:g}   N={len(rows)}")
        for seed, d, p in rows:
            print(f"    seed {seed}   Δ={d:+7.3f}   p≈{p}")
        if len(ds) > 1:
            print(f"    **mean = {mean:+.3f}   SD = {sd:.3f}   SEM = {sem:.3f}**")
        else:
            print(f"    **mean = {mean:+.3f}（只有 1 个种子 ✗ 算不出 SD ✓ 不能下结论）**")

    print("\n判据（预注册）：两个配置的 mean 之差 > **2 × 合并 SD** 才算「有差异」✓")
    print("⚠️ 单 seed 的 ±4 分差异什么都不能说明 ✗（运行间噪声量级 ≈ 2）")


if __name__ == "__main__":
    main(sys.argv[1:])
