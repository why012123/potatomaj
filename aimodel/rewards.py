"""四种「得分条件」—— 也就是四个模型的训练目标。

规则（怎么和牌、怎么算点）四套模型完全一样，**不同的只有它们在追求什么**：

===========  ==========================================================
``normal``   雀魂段位分：``(点数-返し点)/1000 + 顺位点 + 额外段位分``
``mountain`` 在上面基础上**额外重罚放铳**：``- 1.5 × 放铳点数/1000``
``flash``    额外奖励**和牌次数**：``+ 2.0 × 和了几局``（不管大小）
``huge``     额外奖励**番数**：``+ 0.25 × 累计番数``（小牌不值钱）
===========  ==========================================================

为什么这么设计：

* 段位分本身已经包含「赢多少、输多少、第几名」，是所有模型的底子 ——
  不挂这个底子，模型会学出「只要不点炮，垫底也无所谓」这种废策略。
* ``mountain`` 额外扣一笔放铳分，等于把「点炮」的痛感放大；
  它会在该弃和的时候更早弃和。
* ``flash`` 按**次数**给钱，逼它抢速度：小牌快和比大牌慢和划算。
* ``huge`` 按**番数**给钱，逼它做大牌：同样是和，三番和八番差别很大。

返回值的量级都归一到「段位分」附近（±45 上下），再按 :data:`SCALE` 压进
价值头的 tanh 范围。
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import MatchResult

#: 每种目标 × 人数的量纲（用来把奖励压到 ±1 附近）。
#:
#: **必须按人数分开**：三麻牌快、拔北加宝牌、放铳失点集中，量纲比四麻大
#: 一截（实测 normal 三麻 ±105 vs 四麻 ±80）。
#:
#: 取值方法（不是拍脑袋）：用四套教师风格各跑 400 场，对**同一批对局**
#: 算四种模式的奖励，取 ``|reward|`` 的 99.9 分位，向上取整到 10 的倍数。
#: 实测（改之前 -> 改之后，被剪到 ±1 的样本占比）：
#:
#: ===========  ==================  ==================
#: 模式          三麻                四麻
#: ===========  ==================  ==================
#: normal       6.4% -> 0.0%         0.1% -> 0.1%
#: mountain     2.8% -> 0.1%         0.2% -> 0.1%
#: flash        4.2% -> 0.1%         0.1% -> 0.1%
#: huge         0.0% -> 0.1%         0.0% -> 0.1%
#: ===========  ==================  ==================
#:
#: tanh 在边界梯度趋近 0，把 6.4% 的极端局剪平就是让这些局面学不动。
#: 注意：优势做过批内标准化，所以调大 SCALE 对策略梯度没有副作用，
#: 只影响价值头的拟合质量。
SCALE: dict[str, dict[int, float]] = {
    #       三麻    四麻
    "normal":   {3: 110.0, 4: 80.0},
    "mountain": {3: 250.0, 4: 200.0},
    "flash":    {3: 120.0, 4: 90.0},
    "huge":     {3: 140.0, 4: 100.0},
}

#: 老版本的单一 SCALE（只按模式分、不按人数）—— 只留作对照，不再使用
SCALE_LEGACY: dict[str, float] = {
    "normal": 80.0,
    "mountain": 180.0,
    "flash": 90.0,
    "huge": 160.0,
}

#: 各模式的权重，想调手感改这里就行
WEIGHTS: dict[str, dict[str, float]] = {
    "normal": {"rp": 1.0, "deal_in": 0.0, "agari": 0.0, "han": 0.0},
    # 第一版用的是 1.5 / 2.0 / 0.25，实测四个模型的风格差异太小（放铳率 15~17%
    # 挤在一起、副露 0.77 vs 0.57），所以对比拉大一倍左右，让风格真的被推出来。
    "mountain": {"rp": 1.0, "deal_in": 3.0, "agari": 0.0, "han": 0.0},
    "flash": {"rp": 1.0, "deal_in": 0.0, "agari": 4.0, "han": 0.0},
    # 0.6 推不动（番数那点收益被段位分淹没了，实测平均番数还不如「防守」高），
    # 提到 1.5：10 番 = 15 段位分，足够让它为打点让路。
    "huge": {"rp": 1.0, "deal_in": 0.0, "agari": 0.0, "han": 1.5},
}

MODES: tuple[str, ...] = ("normal", "mountain", "flash", "huge")

MODE_CN = {
    "normal": "段位分（标准）",
    "mountain": "段位分 − 3.0×放铳（防守）",
    "flash": "段位分 + 4.0×和了次数（速攻）",
    "huge": "段位分 + 1.5×累计番数（大牌）",
}


@dataclass
class RewardParts:
    """一场比赛里某个玩家的成绩拆解（也方便打日志看模型在追什么）。"""

    rank_points: float
    deal_in_points: float  # 放铳丢的点数（正数）
    agari_count: int  # 和了几局
    han_total: int  # 和牌累计番数
    win_points: float  # 和牌拿到的点数（正数）
    score_delta: int

    def describe(self) -> str:
        return (
            f"段位分 {self.rank_points:+.1f}  放铳 {self.deal_in_points:.0f}"
            f"  和了 {self.agari_count} 局 / {self.han_total} 番"
            f"  收入 {self.win_points:.0f}  总得失 {self.score_delta:+d}"
        )


def parts_of(result: MatchResult, player: int) -> RewardParts:
    """从对局结果里抽出这个玩家的各项成绩。"""
    st = result.stats[player]
    return RewardParts(
        rank_points=result.rank_points[player] if result.rank_points else 0.0,
        deal_in_points=float(sum(st.deal_in_score)),
        agari_count=st.tsumo + st.ron,
        han_total=int(sum(st.win_han)),
        win_points=float(sum(st.win_score)),
        score_delta=result.scores[player] - result.start_scores[player],
    )


def reward(mode: str, result: MatchResult, player: int) -> float:
    """按 ``mode`` 算这个玩家这一场的得分条件。"""
    w = WEIGHTS.get(mode)
    if w is None:
        raise ValueError(f"未知的得分条件: {mode}（可选 {MODES}）")
    p = parts_of(result, player)
    return (
        w["rp"] * p.rank_points
        - w["deal_in"] * p.deal_in_points / 1000.0
        + w["agari"] * p.agari_count
        + w["han"] * p.han_total
    )


def reward_vector(mode: str, result: MatchResult) -> list[float]:
    """一场比赛所有座位的奖励。"""
    return [reward(mode, result, p) for p in range(len(result.scores))]


def scale_for(mode: str, players: int = 3) -> float:
    """取 ``(模式, 人数)`` 对应的量纲。拿不到就退回三麻 / 老默认值。"""
    per_mode = SCALE.get(mode)
    if per_mode is None:
        return SCALE_LEGACY.get(mode, 80.0)
    return per_mode.get(int(players), per_mode.get(3, 80.0))


def norm(value: float, mode: str, players: int = 3) -> float:
    """压到 tanh 的输出范围（剪到 ±1）。

    ``players`` 必须传 —— 三麻/四麻量纲不同，只按模式分会把三麻的
    极端局面剪掉（实测 normal 三麻 6.4% 被剪到 ±1）。
    """
    scale = scale_for(mode, players)
    return float(max(-1.0, min(1.0, value / scale)))


def norm_reward(mode: str, result: MatchResult, player: int) -> float:
    """按 **这场对局实际的人数** 归一化（从 ``result.scores`` 取，不用猜）。"""
    return norm(reward(mode, result, player), mode, len(result.scores))


__all__ = [
    "MODES", "MODE_CN", "SCALE", "SCALE_LEGACY", "WEIGHTS", "RewardParts",
    "parts_of", "reward", "reward_vector", "norm", "norm_reward", "scale_for",
]
