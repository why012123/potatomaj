#!/usr/bin/env python
"""宽度扫描：判断「加宽有没有用」= 瓶颈是容量还是数据/信号。

背景
====
有人提议换 PyTorch 写 ~10M 参数的大模型。先要回答一个更便宜的问题：

    在**同样的数据、同样的 epoch 数**下，把网络加宽，留出率还涨不涨？

* 涨   -> 容量是瓶颈，值得上更大模型（那时再考虑 PyTorch/Transformer）
* 不涨 -> 瓶颈在数据与信号，换更大的网络是白费

为什么不能在会话里跑
====================
本机实测：会话里的重计算进程 **35~46 秒被回收**（纯 sleep 进程能活 120 秒以上）。
所以这个脚本必须**人工启动**。

用法
====
    python -m tools.width_scan                      # 默认 512/1024/2048 各 2 轮
    python -m tools.width_scan --epochs 3
    python -m tools.width_scan --widths 512,1024     # 只跑两个
    python -m tools.width_scan --resume              # 断点续跑（跳过已完成的宽度）

输出
====
    _runlogs/width_scan.json     每个宽度每轮的训练/留出一致率 + 耗时
    _runlogs/width_scan.log      一行行的进度

判读
====
看**留出集一致率随宽度**的变化，而不是训练一致率（后者只会因为容量大而更高）。
另外看**耗时比**：宽度翻倍参数量涨约 4 倍，如果留出率只涨 0.2 个点，
投入产出比就该打个问号。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from aimodel import sanma as S  # noqa: E402
from aimodel import variants  # noqa: E402
from aimodel.net import Net  # noqa: E402
from aimodel.encode import layout_for  # noqa: E402
from aimodel.train import collect_bc, train_bc  # noqa: E402

LOG_DIR = ROOT / "_runlogs"
JSON_OUT = LOG_DIR / "width_scan.json"
LOG_OUT = LOG_DIR / "width_scan.log"


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_OUT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_bc(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=False)
    return {k: d[k] for k in d.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="宽度扫描：容量 vs 数据")
    ap.add_argument("--bc-data", default="_data/bc_tv13.npz")
    ap.add_argument("--widths", default="512,1024,2048",
                    help="逗号分隔的隐层宽度（每层同宽，两层）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--holdout-hands", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="跳过 json 里已完成的宽度")
    args = ap.parse_args(argv)

    bc_path = ROOT / args.bc_data
    if not bc_path.exists():
        say(f"找不到数据集 {bc_path}")
        return 1
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    done: dict = {}
    if JSON_OUT.exists():
        try:
            done = json.loads(JSON_OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            done = {}
    results: dict = done.get("widths", {})

    say(f"=== 宽度扫描开始：widths={widths} epochs={args.epochs} "
        f"batch={args.batch} lr={args.lr} l2={args.l2} ===")

    say(f"加载 BC 数据集 {bc_path.name} …")
    t0 = time.perf_counter()
    data = load_bc(bc_path)
    say(f"  数据集 {len(data['X']):,} 条，耗时 {time.perf_counter()-t0:.1f}s")

    # 留出集：用教师现打（和训练数据不同源），这样留出率是「泛化」而不是「背下来」
    # 注意 --holdout-hands 越大越慢；200 手够看趋势。
    rule = S.rule_for(args.players)
    dim = layout_for(args.players)[1]
    # 必须显式指定 spec —— collect_bc 不传 spec 会默认 DEFAULT_MODEL（三麻），
    # 三麻的 Game 只认 3 个 agent，四麻会直接 ValueError。
    spec = variants.ModelSpec(
        name=f"widthscan{args.players}p", players=args.players,
        reward="normal", style="normal",
    )
    say(f"生成留出集（{args.holdout_hands} 手教师对局，{args.players}麻）…")
    t0 = time.perf_counter()
    holdout = collect_bc(args.holdout_hands, seed=args.seed + 555_555,
                         verbose=False, spec=spec)
    say(f"  留出集 {len(holdout['X']):,} 条，耗时 {time.perf_counter()-t0:.1f}s")

    for w in widths:
        key = str(w)
        if args.resume and key in results and results[key].get("epochs_done", 0) >= args.epochs:
            say(f"--- 宽度 {w}：已完成 {results[key]['epochs_done']} 轮，跳过 ---")
            continue
        hidden = (w, w)
        net = Net(dim=dim, hidden=hidden, seed=args.seed, lr=args.lr, l2=args.l2)
        npar = net.n_params()
        say(f"--- 宽度 {w}（{npar:,} 参数）开始，{args.epochs} 轮 ---")
        t0 = time.perf_counter()
        hist, best_net, info = train_bc(
            net, data, epochs=args.epochs, batch_size=args.batch,
            seed=args.seed, verbose=False, holdout=holdout, patience=0,
        )
        dt = time.perf_counter() - t0
        last = hist[-1]
        row = {
            "width": w,
            "params": npar,
            "epochs_done": args.epochs,
            "seconds": round(dt, 1),
            "train_acc": last["与教师一致率"],
            "holdout_acc": last["留出集一致率"],
            "best_holdout": info["best_holdout"],
            "best_epoch": info["best_epoch"],
            "hist": [{k: v for k, v in r.items()} for r in hist],
        }
        results[key] = row
        say(f"    参数量 {npar:,}  耗时 {dt:.0f}s  "
            f"训练 {last['与教师一致率']:.2%}  留出 {last['留出集一致率']:.2%}  "
            f"（最优留出 {info['best_holdout']:.2%} @ {info['best_epoch']} 轮）")
        JSON_OUT.write_text(
            json.dumps({"widths": results, "config": vars(args)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    say("=== 汇总（留出集一致率随宽度）===")
    say(f"{'宽度':>8}{'参数量':>14}{'耗时(s)':>10}{'训练':>10}{'留出':>10}{'最优留出':>12}")
    for w in widths:
        r = results.get(str(w))
        if not r:
            continue
        say(f"{r['width']:>8}{r['params']:>14,}{r['seconds']:>10.0f}"
            f"{r['train_acc']:>10.2%}{r['holdout_acc']:>10.2%}{r['best_holdout']:>12.2%}")
    say(f"结果已存 {JSON_OUT}")

    # 给一句判读
    vals = [(w, results[str(w)]["best_holdout"]) for w in widths if str(w) in results]
    if len(vals) >= 2:
        (w0, h0), (w1, h1) = vals[0], vals[-1]
        d = h1 - h0
        if d > 0.005:
            verdict = (f"留出率随宽度上升 {d:+.2%}（{w0}->{w1}）-> "
                       f"**容量可能是瓶颈**，值得考虑更大模型")
        elif d < -0.005:
            verdict = (f"留出率随宽度**下降** {d:+.2%}（{w0}->{w1}）-> "
                       f"加宽在过拟合，**瓶颈是数据**")
        else:
            verdict = (f"留出率基本不变（{d:+.2%}，{w0}->{w1}）-> "
                       f"**瓶颈是数据/信号，加宽无用**，换 PyTorch 是白费")
        say("判读：" + verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
