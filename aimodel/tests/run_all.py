#!/usr/bin/env python
"""把 aimodel 的测试全部跑一遍。

用法::

    .venv/bin/python aimodel/tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    "test_sanma.py",        # 三麻役种 / 符 / 点数（含雀魂官方点数表逐格核对）
    "test_yaku.py",         # 逐个役种对照表（34 个役 + 易混组合，防「混一色判成清一色」那种错）
    "test_4p.py",           # 四麻：136 张 / 吃 / 北家自风 / 三色同顺 / 点数
    "test_teacher_4p.py",   # 四麻教师不能对第 4 家失明（P0-1 回归）
    "test_chi_4p.py",       # 四麻的「吃」必须真的被学到（P0-2 回归）
    "test_bc_holdout.py",   # BC 存「留出最优」而非「最后一轮」（P0-5 回归）
    "test_weight_guard.py", # 裸跑不许覆盖已训好的权重（P0-6 回归）
    "test_eval_resolution.py",  # 同牌山注入 / 置信区间 / 还要多少场（P1-4 回归）
    "test_exploration.py",  # 逐头熵 / 分头自适应 / 软标签 / tie-break（P1-1 回归）
    "test_models.py",       # 八个模型 + 四种得分条件
    "test_fastshanten.py",  # 快速向听，与既有实现交叉验证
    "test_engine.py",       # 对局引擎：守恒、规则、状态机
    "test_encode.py",       # 特征编码：全局特征 / 动作特征 / 掩码
    "test_net.py",          # 神经网络：数值梯度、掩码、存档、PPO 截断
    "test_logs.py",         # 牌谱管线：MJAI 读写 / 重放对拍 / 数据抽取
]


def main() -> int:
    failed: list[str] = []
    for name in TESTS:
        path = HERE / name
        print("=" * 68)
        print(f"▶ {name}")
        print("=" * 68)
        r = subprocess.run([sys.executable, str(path)])
        if r.returncode != 0:
            failed.append(name)
        print()
    print("=" * 68)
    if failed:
        print(f"❌ {len(failed)} 个测试文件失败：{', '.join(failed)}")
        return 1
    print(f"✅ {len(TESTS)} 个测试文件全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
