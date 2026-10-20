#!/usr/bin/env python
"""4 个种子的 RL 串联（seed 0/1/2/3），跑完一个立刻跑它的 1500 组配对赛。

    python tools/rl_seeds.py --seeds 0,1,2,3 --iters 20

## 为什么需要它

Mac 2026-10-21 07:1x 的排程：
「牌谱下完立刻接 RL 再跑 3 个种子（chunk01 算种子 0），
  每个跑完立刻 **1500 组**配对赛（不是 256），
  报 Δ/SE/单侧p/MDE/组数/sha256/steps，
  单模型不算数，要 3 个种子才能说 RL 有没有用。
  硬要求每步都报，不许静默。明显跑不完早点杀并报卡在哪。」

`rl_ladder.py` 一次只跑一个种子 ✓（`--seed N` 各自一个产物目录 ✗ 否则互相覆盖）
-> 这个脚本只做「按顺序串起来 + 每个种子之间留证据」✓ 不引入新逻辑 ✓

## 产物

```
_data/rl_set/seed{N}/            <- 每个种子一个目录（rl_ladder 用它自己的 --outdir）
    chunk01.npz / .json / .stdout / ladder.log / pair01.txt
_data/rl_set/SUMMARY.jsonl       <- 每个种子一行：sha256 / steps / Δ / SE / p / MDE / 组数
_data/rl_set/seed{N}.paired.txt  <- 配对赛原始输出（rl_ladder 已写 pair01.txt ✓ 这里再留一份）
```

## ⚠️ 顺序执行，不并行

4 个种子各约 2 小时（20 轮 RL 119 分钟 + 1500 组配对赛 140 秒）✗
并行会互相抢 CPU -> 每个都变慢 ✗ 且**配对赛的 σ 会不稳定** ✗
-> **一个跑完再下一个** ✓
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SETDIR = ROOT / "_data" / "rl_set"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    SETDIR.mkdir(parents=True, exist_ok=True)
    with (SETDIR / "driver.log").open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_steps(p: Path) -> int:
    """从 npz 的 `_meta`（UTF-8 JSON）里读 steps。"""
    import numpy as np

    z = np.load(p, allow_pickle=False)
    return int(json.loads(bytes(z["_meta"]).decode("utf-8")).get("steps", 0))


_PAIRED = re.compile(
    r"配对差值（A−B）\s*([+-]?[\d.]+).*?SE\s*±([\d.]+)", re.S
)
_PAIRED2 = re.compile(r"单侧 p ≈ ([\d.]+)")
_PAIRED3 = re.compile(r"本次 (\d+) 组能检出的最小效应（功效 80%）= ([\d.]+)")


def parse_paired(text: str) -> dict:
    """从 arena_test 的输出里抽五样。"""
    out: dict = {}
    m = _PAIRED.search(text)
    if m:
        out["delta"] = float(m.group(1))
        out["se"] = float(m.group(2))
    m = _PAIRED2.search(text)
    if m:
        out["p_one_sided"] = float(m.group(1))
    m = _PAIRED3.search(text)
    if m:
        out["groups"] = int(m.group(1))
        out["mde"] = float(m.group(2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多 seed RL 串联（每个种子跑完立刻配对赛）")
    ap.add_argument("--start", default="aimodel/weights/maj-pro4v1.1_mac_2ep.npz",
                    help="起点权重（所有种子用同一个）")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=1500)
    ap.add_argument("--model", default="maj-pro4v1.1")
    args = ap.parse_args(argv)

    SETDIR.mkdir(parents=True, exist_ok=True)
    summary = SETDIR / "SUMMARY.jsonl"
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    log("=" * 70)
    log(f"  多 seed RL 串联：seeds={seeds}  iters={args.iters}  配对赛={args.eval_games} 组")
    log(f"  起点 {args.start}")
    log(f"  产物 {SETDIR}")
    log("=" * 70)

    for i, seed in enumerate(seeds, 1):
        outdir = SETDIR / f"seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        log(f"---- 种子 {seed}（第 {i}/{len(seeds)} 个）开始 ----")
        log(f"     产物目录 {outdir}")
        t0 = time.time()
        r = subprocess.run(
            [PY, "-u", "tools/rl_ladder.py", args.start, "1", str(args.iters), "1",
             "--seed", str(seed), "--outdir", str(outdir),
             "--eval-games", str(args.eval_games), f"--model={args.model}"],
            cwd=str(ROOT),
        )
        took = time.time() - t0
        if r.returncode != 0:
            log(f"🔴 种子 {seed} 失败（exit {r.returncode}），用时 {took/60:.1f} 分钟 -> 停在这里")
            return 1

        ckpt = outdir / "chunk01.npz"
        pair = outdir / "pair01.txt"
        rec: dict = {"seed": seed, "minutes": round(took / 60, 1)}
        if ckpt.exists():
            rec["sha256"] = sha256_of(ckpt)
            try:
                rec["steps"] = read_steps(ckpt)
            except Exception:                       # noqa: BLE001
                pass
        if pair.exists():
            rec.update(parse_paired(pair.read_text(encoding="utf-8", errors="replace")))
        with summary.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"✅ 种子 {seed} 完成（{took/60:.1f} 分钟）")
        log(f"     {json.dumps(rec, ensure_ascii=False)}")

    log("=" * 70)
    log("  全部种子完成。逐行：")
    for line in summary.read_text(encoding="utf-8").splitlines():
        log("    " + line)
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
