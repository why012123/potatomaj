"""aimodel —— 土豆麻将的「AI 决策」模块（日式三人麻将）。

第一阶段（``potatomaj`` 包）解决的是「把截图读成牌」；这一阶段解决的是
「读到牌之后该打哪张」。方法是用机器学习：先模仿一个启发式教师，
再用自对弈强化学习把策略练得比教师好。

模块划分：

* :mod:`aimodel.sanma`   —— 三麻规则内核：役种判定、符与点数计算
* :mod:`aimodel.fastshanten` —— 快取向听/进张（训练时的高频调用）
* :mod:`aimodel.engine`  —— 对局引擎：摸打、副露、拔北、立直、流局
* :mod:`aimodel.encode`  —— 局面 -> 特征向量（含动作掩码）
* :mod:`aimodel.net`     —— 纯 numpy 神经网络（策略/价值多头）
* :mod:`aimodel.agents`  —— 随机 / 启发式教师 / ML 三种 agent
* :mod:`aimodel.train`   —— 行为克隆 + 自对弈强化学习
* :mod:`aimodel.evaluate`—— 对战评测
* :mod:`aimodel.play`    —— 可运行的对局脚本
* :mod:`aimodel.advise`  —— 与截图识别对接的出牌建议

只有 ``numpy`` 一个依赖，不需要 GPU。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 工程根目录（``potatomaj/``）——为了能 import 外层包 ``potatomaj.tiles`` 等
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__version__ = "0.1.0"

__all__ = ["ROOT", "__version__"]
