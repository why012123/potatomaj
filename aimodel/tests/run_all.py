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
#: `aimodel/` 根 —— 发现测试时要**递归到子目录** ✓
#: 2026-10-20 的教训：原来只 `HERE.glob("test_*.py")` 不递归 ✗
#: -> 放在 `aimodel/ui/` 下的三个测试**从来没被跑过** ✗
#:    而套件照样报「25 个文件全部通过」✗ —— 又一次「测试通过 != 测试没跑」✗
AIMODEL = HERE.parent
#: 登记表只用来写「这个测试是干什么的」，**不再决定跑哪些** ✗
#:
#: 之前这里是硬编码列表，只有列进来的才会跑 ✓ 后果：新增一个 test_*.py
#: **不会报错、也不会被跳过提示，而是静默不跑** ✗
#: 实测踩过：磁盘上已有 18 个 `test_*.py`，run_all 还一直报 16 个 ✓
#: （远端的 `test_danger_model.py` 和我加的 `test_repo_hygiene.py` 都被漏掉过 ✓）
DOCS = {
    "test_sanma.py":        "三麻役种 / 符 / 点数（含雀魂官方点数表逐格核对）",
    "test_yaku.py":         "逐个役种对照表（34 个役 + 易混组合，防「混一色判成清一色」那种错）",
    "test_4p.py":           "四麻：136 张 / 吃 / 北家自风 / 三色同顺 / 点数",
    "test_teacher_4p.py":   "四麻教师不能对第 4 家失明（P0-1 回归）",
    "test_chi_4p.py":       "四麻的「吃」必须真的被学到（P0-2 回归）",
    "test_chisel.py":       "chisel（吃成什么样）：老权重兜底 / 零初始化 / 梯度 / 端到端",
    "test_bc_holdout.py":   "BC 存「留出最优」而非「最后一轮」（P0-5 回归）",
    "test_weight_guard.py": "裸跑不许覆盖已训好的权重（P0-6 回归）",
    "test_eval_resolution.py": "同牌山注入 / 置信区间 / 还要多少场（P1-4 回归）",
    "test_exploration.py":  "逐头熵 / 分头自适应 / 软标签 / tie-break（P1-1 回归）",
    "test_kl_anchor.py":    "KL 锚定 + 最优权重回滚进内存（P1-2 回归）",
    "test_models.py":       "八个模型 + 四种得分条件",
    "test_fastshanten.py":  "快速向听，与既有实现交叉验证",
    "test_engine.py":       "对局引擎：守恒、规则、状态机",
    "test_encode.py":       "特征编码：全局特征 / 动作特征 / 掩码",
    "test_state_features.py": "整场状态特征：振听/场次/本场/供托/连任/剩余/顺位/摸切/叫牌张",
    "test_legacy_encode.py": "冻结老 encoder：486/560 逐位金值 + 维度分发（保老权重）",
    "test_net.py":          "神经网络：数值梯度、掩码、存档、PPO 截断",
    "test_logs.py":         "牌谱管线：MJAI 读写 / 重放对拍 / 数据抽取",
    "test_danger_model.py": "危险度模型接入：布局一致性 / 人数维度守卫 / 退回手写路径",
    "test_repo_hygiene.py": "仓库卫生：天鳳牌谱的保护规则必须一直生效",
    "test_external.py":    "外部 mjai AI 协议桥：正常/超时/崩溃/非法动作/确定性",
    "test_strong.py":      "规则强 agent：无役不鸣 / 有役该鸣 / 弃和安全 / 确定性 / 四三麻整局",
    "test_arena_external.py": "天梯接外部 AI（崩溃隔离）+ 抽人公平性（含 3 条负向）",
    "test_season_independent.py": "天梯每期独立：两条路径一致 / 每期重置 / done_total 不混 / 两条负向验证",
}

#: 明确不跑的（目前没有）。放这里而不是「不写进列表」✗ —— 那样会变成静默跳过 ✓
SKIP: set[str] = set()


def discover() -> list[str]:
    """**自动发现**所有 ``test_*.py`` —— 新增测试不会再被静默漏掉。

    ⚠️ **必须递归**到子目录，不能只扫 ``HERE``：
       2026-10-20 的教训 —— 原来只 ``HERE.glob("test_*.py")``（不递归），
       而放在 ``aimodel/ui/`` 下的三个测试（elo_store / elo_schedule / chi_options）
       **从来没被跑过**，套件却照样报「25 个文件全部通过」。
       又一次「测试通过」≠「测试没跑」。
    """
    found = sorted(
        p.relative_to(AIMODEL).as_posix()
        for p in AIMODEL.rglob("test_*.py")
        if "__pycache__" not in p.parts and p.name not in SKIP
    )
    unknown = [n for n in found if Path(n).name not in DOCS]
    if unknown:
        print("⚠️ 这些测试还没有说明（不影响运行，但请补进 DOCS）：")
        for n in unknown:
            print(f"   · {n}")
        print()
    return found


TESTS = discover()


def main() -> int:
    failed: list[str] = []
    for name in TESTS:
        path = AIMODEL / name
        print("=" * 68)
        print(f"▶ {name}" + (f"  —— {DOCS[Path(name).name]}" if Path(name).name in DOCS else ""))
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
