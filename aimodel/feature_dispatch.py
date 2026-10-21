"""按**权重维度**选择 encoder —— 让老 486/560 维权重继续跑旧特征。

背景（2026-10-21）
==================
给特征末尾追加了三块（``match_state`` 17 / ``called_tile`` 34 /
``last_discard`` 34），当前布局变成 **3p 571 / 4p 645**。
所有存量权重（``maj-pro`` 系列 / ``mac_rl`` / ELO 历史）都是
**3p 486 / 4p 560** —— 直接拿新 ``encode()`` 去喂，第一层 matmul 就炸。

为了生产不停摆，把旧实现**整份冻结**在 :mod:`aimodel.encode_legacy`
（bug-for-bug，连 ``visible_counts`` 的 double-count 都不修）。这里只做一件事：

    net.dim == 老维度  ->  aimodel.encode_legacy
    net.dim == 新维度  ->  aimodel.encode

**调用方不要自己 import encode / encode_legacy 跑模型**，一律用
:func:`encode_for` / :func:`action_features_for`；训练与重抽新数据仍用
:mod:`aimodel.encode`（那是 v2，新布局）。
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from . import encode as _current
from . import encode_legacy as _legacy

#: 版本名（给日志 / 调试 / 存档 meta 用；顺序不代表新旧优先级）
VERSION_LEGACY = "v1"
VERSION_CURRENT = "v2"

#: 允许的人数。写死成 (3, 4) 是故意的：别的值多半是调用方搞错人数了。
_PLAYERS = (3, 4)


def _check_players(players: int) -> int:
    try:
        p = int(players)
    except (TypeError, ValueError):
        raise ValueError(f"players 必须是 3 或 4，得到 {players!r}") from None
    if p not in _PLAYERS:
        raise ValueError(f"players 必须是 3 或 4，得到 {p}")
    return p


def known_dims(players: int) -> dict[int, str]:
    """``{老维度: "v1", 新维度: "v2"}`` —— 四麻 ``{560, 645}`` / 三麻 ``{486, 571}``。"""
    p = _check_players(players)
    return {
        int(_legacy.layout_for(p)[1]): VERSION_LEGACY,
        int(_current.layout_for(p)[1]): VERSION_CURRENT,
    }


def encoder_for(dim: int, players: int) -> ModuleType:
    """按 ``dim`` + 人数返回 encoder 模块（``encode`` 或 ``encode_legacy``）。

    判据只有维度：老布局 = 旧 encoder，新布局 = 新 encoder。
    两边都不是 -> ``ValueError``（宁可当场炸，也不要把错布局喂进网络）。
    """
    p = _check_players(players)
    d = int(dim)
    old_dim = int(_legacy.layout_for(p)[1])
    new_dim = int(_current.layout_for(p)[1])
    if d == new_dim:
        return _current
    if d == old_dim:
        return _legacy
    raise ValueError(
        f"未知特征维度 {d}（{p} 人）：当前布局 {new_dim} 维、"
        f"冻结老布局 {old_dim} 维。"
        f"这个权重不是这两种布局之一 —— 要么特征改过但没登记，要么权重坏了；"
        f"训练侧请重训（python -m aimodel.train bc）。"
    )


def encode_for(dim: int, players: int, view) -> np.ndarray:
    """用与权重配套的 encoder 编码 ``view``（返回 ``np.ndarray``）。"""
    return encoder_for(dim, players).encode(view)


def action_features_for(dim: int, players: int, view) -> np.ndarray:
    """用与权重配套的 encoder 算动作特征（返回 ``(34, ACTION_DIM)``）。"""
    return encoder_for(dim, players).action_features(view)


__all__ = [
    "VERSION_LEGACY", "VERSION_CURRENT",
    "known_dims", "encoder_for", "encode_for", "action_features_for",
]
