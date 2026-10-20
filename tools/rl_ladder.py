#!/usr/bin/env python
"""RL 长跑：分块 + 每块之间配对赛 + 不涨就早停。

## 为什么有 Python 版（原先是 ``tools/rl_ladder.sh``）

server 那台机器**没有 bash** ✗（只有 ``dash``）✓
而 ``.sh`` 版用了 ``set -euo pipefail`` 和 ``[[ ]]`` ✓ dash 一个都不认 ✗
-> 实测结果是「**起不来**」✗ 不是「跑歪」✓（两者要分清 ✓ 起不来至少是安全的 ✓）

与其让人去移植 shell 兼容层 ✗ 不如**直接用 Python 写一遍** ✓
（本仓库的正式工具本来就是 Python ✓ 少一种语言少一类坑 ✓）

## 为什么不用 RL 内部的「最优」

实测它内部评测 **+11.3**，而可信配对赛只有 **+1.89** ✓ **虚高 6 倍** ✗
-> 内部那个「新高」是噪声 ✓ 不能作为进度依据 ✗

所以：每 N 轮存一个检查点 ✓ 然后用**同一协议**（同牌山 + SPRT）配对赛 ✓
-> 得到**可信的进度曲线** ✓

## ⚠️ 三个必须记住的坑（v1 踩过，白跑 3.4 小时）

1. **``--model`` 只选「规格」，不选「权重」** ✗
   它决定 dim / kinds / 得分条件 ✓ 权重从哪来**由 ``--out`` 决定** ✓
2. **``--hidden`` 默认 ``256,256``** ✗
   ``build_net`` 只在「``--out`` 不存在」时才用它新建 ✓
3. ⭐ **``--out`` 是对端：既「读哪份」也「写哪份」** ✗
   ``--out`` 存在 -> ``Net.load(--out)`` ✓
   ``--out`` 不存在 -> **静默新建一个随机网络** ✗✗
   -> 所以**每一块都必须先把上一块拷到 ``--out``** ✓
   否则「阶梯」会变成「10 次互不相干的随机重训」✗
   铁证（v1）：chunk01 与 chunk02 的 ``steps`` **都是 20,118** ✗ 各训各的 ✓

**所以本脚本每次开跑前都：先 ``cp`` ✓ 再断言 dim/hidden 与起点一致 ✓ 不一致立刻退出 ✓**
（宁可什么都不做 ✓ 也不要安静地产出一个错的东西 ✓ —— 与形状⑳同源 ✓）

## 用法

    # 只做起点自检、不训练（几秒，交接前必跑）
    python tools/rl_ladder.py aimodel/weights/maj-pro4v1.1_mac_rl.npz 0

    # 正式跑：8 块 × 20 轮，连续 2 块不涨就停
    python tools/rl_ladder.py aimodel/weights/maj-pro4v1.1_mac_rl.npz 8 20 2

## 判据

每块与**当前最好的那一块**配对赛（不是只比上一块 ✓）
- ``p < 0.05`` 且 差值为正 -> 新的最好 ✓ 继续
- 否则 -> 不涨计数 +1 ✓ 连续 ``patience`` 次就停 ✓
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "_data" / "rl_ladder"
LOG = OUTDIR / "ladder.log"


def log(msg: str = "") -> None:
    print(msg, flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def fail(msg: str) -> None:
    log(f"❌ {msg}")
    raise SystemExit(1)


def read_meta(path: Path) -> tuple[int, str, int]:
    """返回 (dim, hidden 字符串, steps)。hidden 用来做**架构指纹**。"""
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        if "_meta" not in z.files:
            raise KeyError(f"{path} 里没有 _meta")
        m = json.loads(bytes(z["_meta"]).decode())
    return (
        int(m["dim"]),
        ",".join(str(h) for h in m["hidden"]),
        int(m.get("steps", 0)),
    )


def parse_paired(text: str) -> tuple[float, float]:
    """从 arena_test 的输出里取「配对差值」与「单侧 p」。取不到就抛。"""
    d = re.search(r"配对差值（A−B）\s+(-?\d+(?:\.\d+)?)", text)
    p = re.search(r"单侧 p ≈ (\d+(?:\.\d+)?)", text)
    if not d or not p:
        raise ValueError("配对赛结果解析不出「配对差值 / 单侧 p」—— 判据不明，不能继续")
    return float(d.group(1)), float(p.group(1))


#: 子进程输出里哪些行要**实时**转发到 ladder.log。
#: 训练每轮会打几十行（逐头熵之类），全转发会淹掉日志 ✗；
#: 只挑「进度行」（`[N/M] ...`）与 `PPO` 那行 —— 这两行足以判断「还剩多久」✓
_PROGRESS = re.compile(r"^\s*(\[\d+/\d+\]|PPO )")


def run(cmd: list[str], logpath: Path | None = None, stream: bool = True) -> str:
    """跑子进程，**边跑边转发进度行**；失败就带尾部输出报错（不吞）✓

    ⚠️ 原来用 ``subprocess.run(capture_output=True)`` ✗
       那会把子进程输出**全部缓存到结束才落盘** ✗
       -> 长跑中途 ``chunk01.stdout`` 是 0 字节、``ladder.log`` 只有块边界 ✗
       -> **判断不了「还剩多久」** ✗ 2026-10-21 夜里因此两次说不清进度 ✗
       现在改成 ``Popen`` + 逐行读：进度行**实时**经 ``log()`` 打出，
       同时**立即**写进 logpath（每行 flush，不是结束后一次性写）✓

    ⚠️ 干跑验证（2026-10-21 06:0x，1 块 x 1 轮）：
       ``ladder.log`` 里出现了 ``PPO  policy_loss=...`` 与
       ``[1/1] 86727 决策  363s  对教师 段位分 +9.2 ...`` —— 这两行原本是看不到的 ✓
       ``chunk01.stdout`` 在运行中途就有 1,012 B（原来要等结束才有）✓
    """
    if logpath is not None:
        logpath.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    fh = logpath.open("w", encoding="utf-8") if logpath is not None else None
    rc = -1
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if fh is not None:
                fh.write(line)
                fh.flush()                    # ⚠️ 必须 flush，否则还是攒到最后 ✗
            if stream and _PROGRESS.match(line):
                log("      " + line.rstrip())
        rc = proc.wait()
    finally:
        if fh is not None:
            fh.close()

    out = "".join(lines)
    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-12:])
        fail(f"命令失败（exit {rc}）：{' '.join(cmd)}\n    尾部输出：\n{tail}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RL 阶梯：分块 + 块间配对赛 + 早停")
    ap.add_argument("start", help="起点权重 npz")
    ap.add_argument("chunks", type=int, nargs="?", default=10, help="块数")
    ap.add_argument("iters", type=int, nargs="?", default=20, help="每块 RL 轮数")
    ap.add_argument("patience", type=int, nargs="?", default=2, help="连续几块不涨就停")
    ap.add_argument("--model", default="maj-pro4v1.1", help="规格名（只影响 dim/kinds/得分条件）")
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--eval-matches", type=int, default=80)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4, help="配对赛并行度")
    ap.add_argument("--max-games", type=int, default=6000, help="配对赛最多多少组")
    ap.add_argument("--nice", type=int, default=10, help="POSIX 上降优先级；0 = 不降")
    args = ap.parse_args(argv)

    start = Path(args.start)
    if not start.is_file():
        fail(f"起点不存在：{start}")
    try:
        s_dim, s_hid, s_steps = read_meta(start)
    except Exception as e:                       # noqa: BLE001
        fail(f"读不出起点 {start} 的 _meta：{type(e).__name__}: {e}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    log(f"=== RL ladder (py) : {args.chunks} chunks x {args.iters} iters, start={start} ===")
    log(f"    起点身份: dim={s_dim}  hidden=[{s_hid}]  steps={s_steps}")
    log(f"    早停: 连续 {args.patience} 块不涨就停（判据 = 配对赛 p<0.05 且差值为正）")

    if args.nice and hasattr(os, "nice"):
        try:
            os.nice(int(args.nice))
            log(f"    已降优先级 nice={args.nice}")
        except OSError:
            log("    ⚠️ 降优先级失败，照常跑")
    else:
        log("    本平台没有 os.nice，不降优先级" if args.nice else "    不降优先级")

    py = sys.executable
    best, best_name, stale = start, "start", 0

    for c in range(1, args.chunks + 1):
        tag = f"{c:02d}"
        ckpt = OUTDIR / f"chunk{tag}.npz"
        log("")
        log(f"---- chunk {c} / {args.chunks} : RL {args.iters} iters（从 {best_name} 续）----")

        # ⭐ 先 cp 再断言：`--out` 不存在会让 build_net 静默新建随机网络 ✗
        shutil.copyfile(best, ckpt)
        g_dim, g_hid, _ = read_meta(ckpt)
        if g_dim != s_dim or g_hid != s_hid:
            fail(
                f"chunk{tag} 架构不一致：dim={g_dim} hidden=[{g_hid}] "
                f"vs 起点 dim={s_dim} hidden=[{s_hid}] —— 架构降级了，停"
            )
        log(f"    ✅ 架构自检通过: dim={g_dim} hidden=[{g_hid}]")

        t0 = time.time()
        run(
            [
                py, "-m", "aimodel.train", "rl", f"--model={args.model}", "--resume",
                "--iters", str(args.iters), "--matches", str(args.matches),
                "--eval-matches", str(args.eval_matches),
                "--ppo-epochs", str(args.ppo_epochs),
                "--out", str(ckpt), "--log", str(OUTDIR / f"chunk{tag}.json"),
            ],
            logpath=OUTDIR / f"chunk{tag}.stdout",
        )
        log(f"  chunk {c} done, took {int(time.time() - t0)}s, weights {ckpt}")

        # 训后**再断言一次**：写坏也不许静默 ✓
        g_dim, g_hid, g_steps = read_meta(ckpt)
        if g_dim != s_dim or g_hid != s_hid:
            fail(f"训完架构变了：dim={g_dim} hidden=[{g_hid}] —— 停")
        log(f"    ✅ 训后自检: dim={g_dim} hidden=[{g_hid}] steps={g_steps}")

        # 配对赛（可信判据）：本块 vs **当前最好的那一块** ✓
        log(f"  paired test: chunk {c} vs {best_name} ...")
        out = run(
            [
                py, "-m", "aimodel.arena_test", "--a", str(ckpt), "--b", str(best),
                "--target-delta", "2", "--jobs", str(args.jobs),
                "--max-games", str(args.max_games),
            ],
            logpath=OUTDIR / f"pair{tag}.txt",
        )
        for line in out.splitlines():
            if any(k in line for k in ("配对差值", "95% 单侧", "SPRT 判决")):
                log("    " + line.strip())

        try:
            diff, pval = parse_paired(out)
        except ValueError as e:
            fail(f"{e}（见 {OUTDIR / f'pair{tag}.txt'}）")

        if pval < 0.05 and diff > 0:
            log(f"    🎉 显著更强: 差值 {diff:+.3f}（p≈{pval}）-> 成为新的最好")
            best, best_name, stale = ckpt, f"chunk{tag}", 0
        else:
            stale += 1
            log(f"    ➖ 没涨: 差值 {diff:+.3f}（p≈{pval}）-> 不换最好；连续不涨 {stale}/{args.patience}")
            if stale >= args.patience:
                log("")
                log(f"=== 早停：连续 {args.patience} 块没有显著变强 -> 继续跑无意义 ===")
                break

    log("")
    log(f"=== 结束。最好的一块 = {best_name}（{best}）===")
    log("    ⚠️ 这份**没有**替换任何已部署的权重 ✗ 要不要换是单独的决定 ✓")


if __name__ == "__main__":
    main(sys.argv[1:])
