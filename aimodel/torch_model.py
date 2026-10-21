"""dot V1：**per-tile 一维卷积网络**（PyTorch）。

路线书（``docs/dot-V1-技术路线-per-tile.md``）第三节定的结构::

    x (B, 34, F) --transpose--> (B, F, 34)
      Conv1d(F -> 64, k=3, pad=1) + BN + ReLU
      ResBlock(64) x 4     (Conv-BN-ReLU-Conv-BN + 通道注意力 + 残差)
      Conv1d(64 -> 64, k=3, pad=1) + BN + ReLU
      flatten (B, 64*34) --concat--> global(21)
      Linear(64*34+21 -> 512) + ReLU
      Linear(512 -> 256) + ReLU
      9 个策略头 + 价值头（tanh）

* 头名/宽度**直接来自** ``net.KINDS`` 与 ``encode.HEAD_SIZE`` —— 不手抄常数，
  以后加头/改宽度会自动跟上（目前 9 个：discard/tsumo/ron/riichi/call/chi/
  kita/kan/kyuushu）。
* 全局标量在**最后 FC 前** concat，不是一开始拼进 34×F —— 路线书明确要求。
* 参数量自检：构造时打印并 assert 在 1M ~ 3M（路线书的预注册判据之一）。

输入布局
--------
``forward(tiles, globals)`` 主接口是 ``(B, F, 34)``（Conv1d 通道在前）。
为了方便直接喂 :func:`aimodel.encode_tile.encode_tile` 的输出
``tiles[34, F]``，也接受 ``(B, 34, F)``（会自动 transpose）。
两种布局的结果必须逐位相同（单测钉住）。形状不对（通道数错 / 34 改成 35）
一律抛 ``ValueError`` —— 绝不静默 reshape。

输出
----
``(heads, value)``：

* ``heads``：``dict[str, Tensor]``，键为 :data:`KINDS`，每个形状 ``(B, HEAD_SIZE[k])``
* ``value``：``(B,)``，tanh 后落在 ``[-1, 1]``

与 numpy 版 :class:`aimodel.net.Net` 的 ``logits(h, kind)`` / ``value(h)`` 一样，
这里也提供 ``trunk`` / ``logits`` / ``value`` 三个分开的方法，方便逐步替换。

跑法::

    .venv/Scripts/python.exe aimodel/torch_model.py
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import HEAD_SIZE
from .encode_tile import F as TILE_FEATURES
from .encode_tile import G as GLOBAL_FEATURES
from .net import KINDS as _NUMPY_KINDS

#: 策略头顺序**唯一来源**是 ``net.KINDS``（生产在用）
KINDS: tuple[str, ...] = tuple(_NUMPY_KINDS)

if set(KINDS) != set(HEAD_SIZE):  # pragma: no cover - 只会在上游改错时炸
    raise RuntimeError(
        f"net.KINDS 与 encode.HEAD_SIZE 不一致：{sorted(KINDS)} vs {sorted(HEAD_SIZE)}"
    )

#: 每个牌位（34 种牌）
TILE_POSITIONS = 34

#: 路线书：C=64、4 个残差块、512 -> 256
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 4
DEFAULT_HIDDEN: tuple[int, int] = (512, 256)
DEFAULT_SE_REDUCTION = 4

#: 路线书的预注册参数量区间
MIN_PARAMS = 1_000_000
MAX_PARAMS = 3_000_000


class ChannelAttention(nn.Module):
    """SE 风格的通道注意力：全局池化 -> 降维 -> 升维 -> sigmoid 缩放。"""

    def __init__(self, channels: int, reduction: int = DEFAULT_SE_REDUCTION) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + 通道注意力 + 残差（最后再 ReLU）。"""

    def __init__(self, channels: int, reduction: int = DEFAULT_SE_REDUCTION) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.attn = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.attn(h)
        return F.relu(x + h, inplace=True)


class TileConvNet(nn.Module):
    """per-tile Conv1d 网络：输入 ``(B, F, 34)`` + ``(B, G)``，输出 9 头 + 价值。"""

    def __init__(
        self,
        in_channels: int = TILE_FEATURES,
        global_dim: int = GLOBAL_FEATURES,
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        se_reduction: int = DEFAULT_SE_REDUCTION,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or global_dim <= 0:
            raise ValueError(f"通道数必须 > 0，收到 in_channels={in_channels}, "
                             f"global_dim={global_dim}")
        if blocks < 1:
            raise ValueError(f"至少要 1 个残差块，收到 blocks={blocks}")
        if len(hidden) != 2:
            raise ValueError(f"路线书的 FC 是 512 -> 256，收到 hidden={tuple(hidden)}")

        self.in_channels = int(in_channels)
        self.global_dim = int(global_dim)
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.hidden = tuple(int(h) for h in hidden)
        self.kinds = list(KINDS)

        self.input = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels, se_reduction) for _ in range(blocks)]
        )
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        fc_in = channels * TILE_POSITIONS + global_dim
        self.fc1 = nn.Linear(fc_in, self.hidden[0])
        self.fc2 = nn.Linear(self.hidden[0], self.hidden[1])
        #: 头名/宽度来自生产定义，**不自己编**
        self.heads = nn.ModuleDict(
            {k: nn.Linear(self.hidden[1], int(HEAD_SIZE[k])) for k in KINDS}
        )
        self.value_head = nn.Linear(self.hidden[1], 1)

        self.param_count = self.n_params()
        if not (MIN_PARAMS <= self.param_count <= MAX_PARAMS):
            raise AssertionError(
                f"参数量 {self.param_count:,} 不在路线书要求的 "
                f"{MIN_PARAMS:,} ~ {MAX_PARAMS:,} 区间"
            )
        print(
            f"[TileConvNet] 参数量 {self.param_count:,}"
            f"（{MIN_PARAMS:,} ~ {MAX_PARAMS:,} ✓；C={channels}, "
            f"blocks={blocks}, F={in_channels}, G={global_dim}）"
        )

    # -- 参数统计 -----------------------------------------------------------
    def n_params(self) -> int:
        """可训练参数总数（不含 BN 的 running_mean/running_var）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_report(self, verbose: bool = True) -> str:
        """打印并返回逐层参数量报表。"""
        lines = [
            f"{'layer':<44} {'shape':<22} {'params':>12}",
            "-" * 82,
        ]
        for name, p in self.named_parameters():
            shape = "(" + ", ".join(str(d) for d in p.shape) + ")"
            lines.append(f"{name:<44} {shape:<22} {p.numel():>12,}")
        lines.append("-" * 82)
        lines.append(f"{'TOTAL':<44} {'':<22} {self.param_count:>12,}")
        lines.append(
            f"参数量范围：{MIN_PARAMS:,} <= {self.param_count:,} <= {MAX_PARAMS:,} ✓"
        )
        report = "\n".join(lines)
        if verbose:
            print(report)
        return report

    # -- 输入校验 -----------------------------------------------------------
    def _prepare(
        self, x: torch.Tensor, globals_: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x 必须是 torch.Tensor，收到 {type(x).__name__}")
        if not isinstance(globals_, torch.Tensor):
            raise TypeError(
                f"globals 必须是 torch.Tensor，收到 {type(globals_).__name__}"
            )
        if x.dim() != 3:
            raise ValueError(
                f"x 必须是 3 维 (B, F, 34)，收到 shape={tuple(x.shape)}"
            )
        # 主接口 (B, F, 34)；同时容忍 encode_tile 直接 stack 出来的 (B, 34, F)
        if x.shape[1] == self.in_channels and x.shape[2] == TILE_POSITIONS:
            pass
        elif x.shape[1] == TILE_POSITIONS and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"x 形状不对：应为 (B, {self.in_channels}, {TILE_POSITIONS})"
                f"（或 tile-major (B, {TILE_POSITIONS}, {self.in_channels})），"
                f"收到 shape={tuple(x.shape)}"
            )
        if globals_.dim() != 2:
            raise ValueError(
                f"globals 必须是 2 维 (B, {self.global_dim})，"
                f"收到 shape={tuple(globals_.shape)}"
            )
        if globals_.shape != (x.shape[0], self.global_dim):
            raise ValueError(
                f"globals 形状不对：应为 (B, {self.global_dim}) 且 B={x.shape[0]}，"
                f"收到 shape={tuple(globals_.shape)}"
            )
        return x.float(), globals_.float()

    # -- 前向 ---------------------------------------------------------------
    def trunk(self, x: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        """``(B, F, 34)`` + ``(B, G)`` -> ``(B, 256)``（最后 FC 之后）。"""
        x, globals_ = self._prepare(x, globals_)
        h = self.input(x)
        h = self.res_blocks(h)
        h = self.output(h)
        h = h.flatten(1)                       # (B, C*34)
        h = torch.cat([h, globals_], dim=1)    # 全局标量在**最后 FC 前**拼
        h = F.relu(self.fc1(h), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        return h

    def logits(self, h: torch.Tensor, kind: str) -> torch.Tensor:
        if kind not in self.heads:
            raise KeyError(f"未知决策头 {kind!r}；可用 {sorted(self.heads)}")
        return self.heads[kind](h)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value_head(h)).squeeze(-1)

    def forward(
        self, x: torch.Tensor, globals_: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        h = self.trunk(x, globals_)
        heads = {k: self.logits(h, k) for k in self.kinds}
        return heads, self.value(h)


__all__ = [
    "TileConvNet", "ChannelAttention", "ResBlock", "KINDS", "HEAD_SIZE",
    "TILE_POSITIONS", "DEFAULT_CHANNELS", "DEFAULT_BLOCKS",
    "MIN_PARAMS", "MAX_PARAMS",
]


if __name__ == "__main__":  # pragma: no cover - 手工看参数量用
    net = TileConvNet()
    net.param_report()
