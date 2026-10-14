"""训练巡检：一眼看清八个模型训到哪了、有没有走歪。

每小时的「看一眼」就用它，输出格式固定，方便跟上一小时对比：

* 每个模型在哪个阶段（收集克隆数据 / 克隆训练 / 强化学习 / 完成）
* 克隆阶段看**留出集一致率**（学习有没有学到东西、有没有过拟合）
* 强化学习阶段看**对教师段位分**的走势（最近几次，别只看最高那次）
* 有没有偷偷退化：和「上一小时」比，段位分是不是在往下掉

用法::

    python -m aimodel.progress              # 看一次
    python -m aimodel.progress --tail 6     # 每次多显示几轮
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from . import variants

LOGS = variants.WEIGHTS_DIR / "logs"

#: 强化学习每轮那行：  [3/16] 32966 决策  137s  对教师 段位分 -4.1  平均顺位 1.96 …
ITER_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\d+) 决策\s+(\d+)s\s+对教师 段位分\s+(-?[\d.]+)\s+"
    r"平均顺位\s+([\d.]+)\s+和了率\s+([\d.]+)%\s+放铳率\s+([\d.]+)%"
)
HOLDOUT_RE = re.compile(r"epoch (\d+)/(\d+)\s+一致率=([\d.]+)%（留出 ([\d.]+)%）")
BCDATA_RE = re.compile(r"行为克隆数据：(\d+) 条决策 / (\d+) 局")
START_RE = re.compile(r"起始：对教师 段位分\s+(-?[\d.]+)\s+平均顺位\s+([\d.]+)")


#: 巡检哪一版（跟着 train_all --version 走）
VER: str | None = None


def read(name: str) -> str:
    p = LOGS / f"{name}.log"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def snapshot(tail: int = 4) -> None:
    print(f"\n{'═' * 78}")
    print(f"训练巡检  {time.strftime('%H:%M:%S')}")
    print("═" * 78)
    for spec in variants.all_specs(VER):
        txt = read(spec.name)
        size = f"{spec.path.stat().st_size / 1024:.0f}KB" if spec.path.exists() else "—"
        line = f"\n▸ {spec.name:<20} {spec.cn:<9} 权重 {size:>7}"

        data = BCDATA_RE.search(txt)
        holds = HOLDOUT_RE.findall(txt)
        iters = ITER_RE.findall(txt)
        start = START_RE.search(txt)

        if not txt.strip():
            print(line + "   ⏳ 还没开始")
            continue
        if data:
            line += f"  克隆 {int(data.group(2))} 局 / {int(data.group(1))} 决策"
        if holds:
            last = holds[-1]
            line += f"  留出 {last[3]}%（第 {last[0]}/{last[1]} 轮）"
        print(line)
        if not iters:
            if "② 自对弈强化学习" in txt:
                print("      └ 强化学习刚启动，还没跑完第一轮")
            else:
                print("      └ 还在克隆阶段，没进强化学习")
            continue
        if start:
            print(f"      └ 起始（纯克隆）段位分 {start.group(1)}")
        recent = iters[-tail:]
        vals = [float(x[4]) for x in iters]
        best = max(vals)
        print(
            "      └ 强化学习 "
            + "  ".join(f"{x[0]}/{x[1]}→{x[4]}" for x in recent)
            + f"   最好 {best:+.1f}  当前 {vals[-1]:+.1f}"
        )
        # 走歪的预警：最近三轮连续下滑，且已经比克隆底子还差
        if len(vals) >= 4:
            base = float(start.group(1)) if start else vals[0]
            if vals[-1] < vals[-2] < vals[-3] and vals[-1] < base - 3:
                print(f"      ⚠️ 连续下滑且低于克隆底子（{vals[-1]:+.1f} < {base:+.1f}）—— 该停下来看了")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="训练巡检")
    ap.add_argument("--tail", type=int, default=4, help="每模型显示最近几轮")
    ap.add_argument("--version", type=str, default=None, help="看哪一版（默认当前版）")
    args = ap.parse_args(argv)
    global VER
    VER = args.version
    snapshot(args.tail)
    ws = sorted(variants.WEIGHTS_DIR.glob(f"*{VER or variants.VERSION}.npz"))
    print(f"\n已生成 {len(ws)}/8 个 {VER or variants.VERSION} 权重：")
    for w in ws:
        print(f"   {w.name:<28} {w.stat().st_size / 1024:>6.0f} KB  "
              f"{time.strftime('%H:%M:%S', time.localtime(w.stat().st_mtime))}")


if __name__ == "__main__":
    main()
