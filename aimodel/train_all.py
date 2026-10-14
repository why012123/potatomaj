"""八个模型一起训。

每个模型 = ① 跟自己的专属教师做行为克隆 + ② 用自己的得分条件做自对弈 PPO。

用法::

    # 八个全训（已训好的会自动跳过）
    python -m aimodel.train_all

    # 只训某几个
    python -m aimodel.train_all --models maj-flash,maj-mountain4

    # 三个并行（默认 2），每轮预算调大一点
    python -m aimodel.train_all --jobs 3 --hands 2000 --iters 12

    # 重训（忽略已有权重）
    python -m aimodel.train_all --force

每个模型自己的日志写在 ``weights/logs/<模型名>.log``，跑完打印一张总表。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import rewards, variants

LOGS = variants.WEIGHTS_DIR / "logs"


def train_one(spec: variants.ModelSpec, args) -> tuple[str, bool, float]:
    """跑一个模型的 行为克隆 + 自对弈。返回 (名字, 是否成功, 耗时)。"""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{spec.name}.log"
    t0 = time.perf_counter()
    stage = getattr(args, "stage", "all")
    if stage == "auto":
        # 有这个模型的权重就只补强化学习，没有就从克隆开始
        stage = "rl" if spec.path.exists() else "all"
    cmd = [
        sys.executable, "-u", "-m", "aimodel.train", stage,
        "--model", spec.name,
        "--hands", str(args.hands),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--matches", str(args.matches),
        "--eval-matches", str(args.eval_matches),
        "--ppo-epochs", str(args.ppo_epochs),
        "--seed", str(args.seed),
        "--hidden", str(args.hidden),
        "--pool-ratio", str(args.pool_ratio),
    ]
    # **限制 BLAS 线程**：numpy 默认每个进程开满核数个线程，并行跑 4 个模型时
    # 就是 32 个线程抢 8 个核，负载飙到 99、单进程反而更慢。
    # 实测 4 进程各 1 线程比各 8 线程总吞吐高 ~22%，单进程也快 ~30%。
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + f"\n# BLAS 线程数 = {args.threads}\n\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = time.perf_counter() - t0
    return spec.name, r.returncode == 0, dt


def tail(path: Path, n: int = 2) -> str:
    try:
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return " | ".join(lines[-n:])[:110]
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="批量训练所有模型")
    ap.add_argument("--models", type=str, default="", help="逗号分隔，默认全部八个")
    ap.add_argument("--jobs", type=int, default=2, help="并行几个（吃 CPU，别开太大）")
    ap.add_argument("--hands", type=int, default=1200, help="每个模型行为克隆的局数")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--iters", type=int, default=8, help="每个模型自对弈轮数")
    ap.add_argument("--matches", type=int, default=250, help="每轮自对弈场数")
    ap.add_argument("--eval-matches", type=int, default=120)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument(
        "--hidden", type=str, default="384,384",
        help="网络宽度。bc4 用的是 384,384（35 万参数）—— 别再用默认的 256,256 偷工减料",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="已有权重也重训")
    ap.add_argument("--version", type=str, default=None,
                    help="训哪一版（默认当前版）。例如 --version v1.2 会写 maj-*v1.2.npz")
    ap.add_argument("--pool-ratio", type=float, default=0.25,
                    help="自对弈对手池里其它风格模型的占比")
    ap.add_argument("--threads", type=int, default=1,
                    help="每个训练进程的 BLAS 线程数（并行跑多个模型时别开大，会互相抢）")
    ap.add_argument(
        "--stage", choices=["all", "bc", "rl", "auto"], default="auto",
        help="all=克隆+自对弈；bc=只克隆；rl=只自对弈（拿已有权重接着练）；"
             "auto=有权重就只练 RL（默认）",
    )
    args = ap.parse_args(argv)

    if args.models:
        specs = [variants.get(m) for m in args.models.split(",") if m.strip()]
    else:
        specs = variants.all_specs(args.version)

    todo = []
    for spec in specs:
        if args.stage in ("rl", "auto") and not spec.path.exists():
            if args.stage == "rl":
                print(f"⏭  {spec.name:<16} 还没有权重，跳过（先跑 bc）")
                continue
        if spec.path.exists() and not args.force and args.stage in ("all", "bc"):
            print(f"⏭  {spec.name:<16} 已有权重，跳过（--force 可重训）")
            continue
        todo.append(spec)

    if not todo:
        print("没有要训的模型。")
        return

    print(f"要训 {len(todo)} 个模型，并行 {args.jobs} 个")
    for s in todo:
        print(f"   {s.describe()}")
    print()

    t0 = time.perf_counter()
    results: list[tuple[str, bool, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, ok, dt in pool.map(lambda s: train_one(s, args), todo):
            results.append((name, ok, dt))
            flag = "✅" if ok else "❌"
            print(f"{flag} {name:<16} {dt / 60:5.1f} 分钟   {tail(LOGS / f'{name}.log')}")
            sys.stdout.flush()

    print(f"\n总耗时 {(time.perf_counter() - t0) / 60:.1f} 分钟")
    print(f"\n{'模型':<16}{'人数':<6}{'得分条件':<30}{'状态'}")
    for name, ok, dt in results:
        spec = variants.get(name)
        state = f"✅ {dt / 60:.1f} 分钟" if ok else "❌ 失败（看日志）"
        print(f"{name:<16}{spec.players:<6}"
              f"{rewards.MODE_CN.get(spec.reward, spec.reward):<30}{state}")
    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print(f"\n失败的模型: {', '.join(bad)}（日志在 {LOGS}/）")


if __name__ == "__main__":
    main()
