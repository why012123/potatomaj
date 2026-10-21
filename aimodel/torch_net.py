"""PyTorch 版策略/价值网络：与手写 numpy 版 ``aimodel/net.py`` **结构等价**。

为什么单独一个文件
------------------
现有工程只依赖 ``numpy + Pillow``，``net.py`` 里的手写前向/反向已经能跑；
这里不是替换它，而是给「4060 上跑 PyTorch 训练」开一条并行通道。两条通道
必须能互换权重，否则平台（按 ``.npz`` 加载）就用不了新训练出来的模型。

结构（与 ``net.py`` 逐项对应）
------------------------------
::

    X(in_dim)
      -> Linear(W0,b0) -> ReLU
      -> Linear(W1,b1) -> ReLU        # 最后一层隐层**不接**激活
      -> 共享特征 h
           |-- Wp_discard(34) + 动作塔 Wa1/Wa2(actfeat) -> 34 路
           |-- Wp_tsumo(2) / ron(2) / riichi(2) / call(3)
           |-- Wp_chi(2) / kita(2) / kan(2) / kyuushu(2)
           |-- Wv(1) -> tanh -> 价值
           `-- [可选] Wc(3)：当前 main 的 chisel（吃成什么样）二级头

9 个策略头的名字和宽度直接来自 ``encode.HEAD_SIZE``，顺序来自 ``net.KINDS``，
不在本文件里另抄一份常数。默认输入维度是 **645**（开发中的 560+85 布局），
但构造参数 ``in_dim`` 可配；加载旧权重时以 ``_meta['dim']`` 为准。

权重互换
--------
``state_dict`` 的键名故意与 ``Net.params()`` / ``.npz`` 里的数组名**完全同名**
（``W0`` / ``b0`` / ``Wp_discard`` / ``Wv`` / ``Wa1`` ...）。因此：

* ``TorchNet.load_npz(path)``：读 numpy 存档 -> torch 参数；
* ``net.save_npz(path)``：torch 参数 -> numpy 存档，``_meta`` 原样保留并更新。

``_meta`` 里的 ``steps/dim/hidden/kinds/epochs_done/lr/l2/action_dim`` 全部保住；
未知的 meta 键和未知数组也会原样带过去（``extra_meta`` / ``extra_arrays``），
避免「转一圈回来平台不认」。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import ACTION_DIM, HEAD_SIZE
from .net import KINDS as _NUMPY_KINDS

#: 开发中的四麻特征布局：560（现状）+ match_state 17 + called_tile 34 + last_discard 34。
#: 这里只做默认值，**绝不写死**：训练脚本会从数据分片自动探测真实宽度。
DEFAULT_IN_DIM = 645

#: 默认隐层宽度，与 ``Net(hidden=(256, 256))`` 一致。
DEFAULT_HIDDEN: tuple[int, ...] = (256, 256)

#: 动作塔宽度，与 ``net.ACTION_HIDDEN`` 一致。
ACTION_HIDDEN = 32

#: chisel（「吃成什么样」）头的宽度；老权重没有它，故默认 0（不建）。
CHISEL_SIZE = 3

#: 9 个策略头，顺序与 numpy 版一致。
KINDS: tuple[str, ...] = tuple(_NUMPY_KINDS)

#: 存档里必须出现的 ``_meta`` 键（测试会逐项检查）。
REQUIRED_META_KEYS: tuple[str, ...] = (
    "steps", "dim", "hidden", "kinds", "epochs_done", "lr", "l2", "action_dim",
)


def _pair(v: int | Sequence[int] | None) -> tuple[int, ...]:
    if v is None:
        return DEFAULT_HIDDEN
    if isinstance(v, int):
        return (int(v),)
    return tuple(int(x) for x in v)


class TorchNet(nn.Module):
    """numpy ``Net`` 的 PyTorch 等价实现。

    参数名与 ``Net.params()`` 完全相同，所以 ``state_dict`` 和 ``.npz`` 之间
    只差一个 ``_meta``。初始化也刻意使用 numpy 的 ``default_rng(seed)`` 和
    同样的分布/顺序 —— 同 seed 下两条通道**逐位相同**，方便做一致性测试。
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden: int | Sequence[int] = DEFAULT_HIDDEN,
        kinds: Sequence[str] = KINDS,
        *,
        action_dim: int = ACTION_DIM,
        action_hidden: int = ACTION_HIDDEN,
        seed: int = 0,
        lr: float = 1e-3,
        l2: float = 1e-5,
        chisel_size: int = 0,
        meta: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = _pair(hidden)
        self.kinds = [str(k) for k in kinds]
        self.action_dim = int(action_dim)
        self.action_hidden = int(action_hidden)
        self.chisel_size = int(chisel_size)
        self.lr = float(lr)
        self.l2 = float(l2)

        # 与 numpy 版一致的「非参数状态」。Adam 的 m/v 不存档，steps 读档后归零。
        self.steps = 0
        self.total_steps = 0
        self.epochs_done = 0
        self.missing_heads: tuple[str, ...] = ()
        self.untrained_heads: set[str] = set()
        self.chisel_trained = False
        self.chisel_updates = 0

        # 未知 meta / 未知数组原样保留，save_npz 时再写回去。
        self.extra_meta: dict = dict(meta or {})
        self.extra_arrays: dict[str, np.ndarray] = {}

        self._init_parameters(seed)

    # -- 初始化（逐位复刻 numpy Net.__init__）-------------------------------
    def _init_parameters(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        sizes = [self.in_dim, *self.hidden]

        for i in range(len(self.hidden)):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            self._set_param(f"W{i}", w.astype(np.float32))
            self._set_param(f"b{i}", np.zeros(fan_out, dtype=np.float32))

        last = self.hidden[-1] if self.hidden else self.in_dim
        for k in self.kinds:
            if k not in HEAD_SIZE:
                raise ValueError(f"未知决策头 {k!r}；HEAD_SIZE 里有 {sorted(HEAD_SIZE)}")
            n = int(HEAD_SIZE[k])
            w = rng.normal(0.0, np.sqrt(2.0 / last), (last, n)) * 0.1
            self._set_param(f"Wp_{k}", w.astype(np.float32))
            self._set_param(f"bp_{k}", np.zeros(n, dtype=np.float32))

        wv = rng.normal(0.0, np.sqrt(2.0 / last), (last, 1)) * 0.1
        self._set_param("Wv", wv.astype(np.float32))
        self._set_param("bv", np.zeros(1, dtype=np.float32))

        wa1 = rng.normal(0.0, np.sqrt(2.0 / self.action_dim),
                         (self.action_dim, self.action_hidden))
        self._set_param("Wa1", wa1.astype(np.float32))
        self._set_param("ba1", np.zeros(self.action_hidden, dtype=np.float32))
        wa2 = rng.normal(0.0, np.sqrt(2.0 / self.action_hidden),
                         (self.action_hidden, 1)) * 0.1
        self._set_param("Wa2", wa2.astype(np.float32))
        self._set_param("ba2", np.zeros(1, dtype=np.float32))

        if self.chisel_size > 0:
            self._set_param("Wc", np.zeros((last, self.chisel_size), dtype=np.float32))
            self._set_param("bc", np.zeros(self.chisel_size, dtype=np.float32))

    def _set_param(self, name: str, arr: np.ndarray) -> None:
        """按名字挂一个 ``nn.Parameter``，保证 state_dict 键与 npz 数组同名。"""
        t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        setattr(self, name, nn.Parameter(t))

    # -- 前向 ---------------------------------------------------------------
    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., in_dim) -> (..., last_hidden)``；最后一层隐层不接 ReLU。"""
        h = x
        n = len(self.hidden)
        for i in range(n):
            # 注意：numpy 存档里的 W 是 (in, out)，PyTorch 的 F.linear 要
            # (out, in)。这里用 matmul 保持**参数形状与 npz 完全一致**，
            # 否则 load_npz/save_npz 就要转置，平台侧容易对不上。
            h = torch.matmul(h, getattr(self, f"W{i}")) + getattr(self, f"b{i}")
            if i + 1 < n:
                h = F.relu(h)
        return h

    def logits(
        self,
        h: torch.Tensor,
        kind: str,
        actfeat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """策略头输出；``discard`` 额外加上动作塔的每张牌分数。"""
        if kind not in self.kinds:
            raise KeyError(f"{kind!r} 不在这个模型的 heads={self.kinds} 里")
        z = torch.matmul(h, getattr(self, f"Wp_{kind}")) + getattr(self, f"bp_{kind}")
        if kind == "discard" and actfeat is not None:
            z = z + self.action_scores(actfeat)
        return z

    def action_scores(self, actfeat: torch.Tensor) -> torch.Tensor:
        """``(..., 34, action_dim) -> (..., 34)``，每张候选牌一个 logit 偏置。"""
        a1 = F.relu(torch.matmul(actfeat, self.Wa1) + self.ba1)
        return (torch.matmul(a1, self.Wa2) + self.ba2)[..., 0]

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.matmul(h, self.Wv) + self.bv)[..., 0]

    def chisel_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.chisel_size <= 0:
            raise RuntimeError("这个模型没有 chisel 头（chisel_size=0）")
        return torch.matmul(h, self.Wc) + self.bc

    # -- 推理辅助 -----------------------------------------------------------
    def policy(
        self,
        x: torch.Tensor,
        kind: str,
        mask: torch.Tensor,
        actfeat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """掩码归一化后的动作概率；支持单条 ``(in_dim,)`` 或批量 ``(N,in_dim)``。"""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        single_mask = mask.dim() == 1
        if single_mask:
            mask = mask.unsqueeze(0)
        if actfeat is not None and actfeat.dim() == 2:
            actfeat = actfeat.unsqueeze(0)
        z = self.logits(self.trunk(x), kind, actfeat)
        p = masked_softmax(z, mask)
        return p[0] if single else p

    @torch.no_grad()
    def value_of(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(self.trunk(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- 权重转换 -----------------------------------------------------------
    def state_arrays(self) -> dict[str, np.ndarray]:
        """``state_dict`` -> ``{npz 数组名: np.float32 ndarray}``。"""
        return {
            name: p.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, p in self.named_parameters()
        }

    def build_meta(self) -> dict:
        """合并未知 meta 并刷新已知键；``steps`` 写的是累计更新次数。"""
        meta = dict(self.extra_meta)
        meta.update(
            {
                "dim": int(self.in_dim),
                "hidden": [int(x) for x in self.hidden],
                "kinds": list(self.kinds),
                "steps": int(self.total_steps),
                "epochs_done": int(self.epochs_done),
                "lr": float(self.lr),
                "l2": float(self.l2),
                "action_dim": int(self.action_dim),
            }
        )
        if self.chisel_size > 0:
            meta.update(
                {
                    "chisel": int(self.chisel_size),
                    "chisel_trained": bool(self.chisel_trained),
                    "chisel_updates": int(self.chisel_updates),
                }
            )
        else:
            # 没有 chisel 参数时不能留一个「有 chisel」的 meta，否则平台会
            # 以为存档里有 Wc/bc。
            for key in ("chisel", "chisel_trained", "chisel_updates"):
                meta.pop(key, None)
        return meta

    def save_npz(self, path: str | Path) -> Path:
        """把 torch 参数写成平台能直接 ``Net.load`` 的 ``.npz``。

        先写临时文件再 ``os.replace``，避免平台在写一半时读到坏档。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.state_arrays()
        for name, arr in self.extra_arrays.items():
            if name not in arrays and name != "_meta":
                arrays[name] = np.asarray(arr)
        arrays["_meta"] = np.frombuffer(
            json.dumps(self.build_meta(), ensure_ascii=False).encode("utf-8"),
            dtype=np.uint8,
        )

        tmp = path.with_name(path.name + ".tmp.npz")
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **arrays)
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load_npz(cls, path: str | Path, *, strict: bool = True) -> "TorchNet":
        """读 numpy ``Net.save`` 的存档，返回等价的 ``TorchNet``。

        ``strict=True``（默认）：存档缺了参数（chisel 可选头除外）直接报错。
        这样可以避免「权重转了一半、剩下的随机初始化」这种静默错误。
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            if "_meta" not in files:
                raise ValueError(f"{path}: 不是 numpy Net 存档（缺少 _meta）")
            meta = json.loads(bytes(data["_meta"]).decode("utf-8"))

            if "W0" in files:
                inferred_dim = int(np.asarray(data["W0"]).shape[0])
            else:
                inferred_dim = int(meta.get("in_dim", DEFAULT_IN_DIM))
            dim = int(meta.get("dim", inferred_dim))

            if "hidden" in meta:
                hidden = tuple(int(x) for x in meta["hidden"])
            else:
                hidden = []
                i = 0
                while f"W{i}" in files:
                    hidden.append(int(np.asarray(data[f"W{i}"]).shape[1]))
                    i += 1
                hidden = tuple(hidden) or DEFAULT_HIDDEN

            if "kinds" in meta:
                kinds = [str(k) for k in meta["kinds"]]
            else:
                kinds = [k for k in KINDS if f"Wp_{k}" in files]
                for name in sorted(files):
                    if name.startswith("Wp_"):
                        k = name[3:]
                        if k not in kinds and f"bp_{k}" in files:
                            kinds.append(k)
                if not kinds:
                    raise ValueError(f"{path}: 找不到任何策略头参数")

            action_dim = int(meta.get(
                "action_dim",
                np.asarray(data["Wa1"]).shape[0] if "Wa1" in files else ACTION_DIM,
            ))
            has_chisel = "Wc" in files and "bc" in files
            if ("Wc" in files) != ("bc" in files):
                raise ValueError(f"{path}: chisel 头参数不完整（Wc/bc 只有一个）")
            chisel_size = int(np.asarray(data["Wc"]).shape[1]) if has_chisel else 0

            model = cls(
                in_dim=dim,
                hidden=hidden,
                kinds=kinds,
                action_dim=action_dim,
                seed=0,
                lr=float(meta.get("lr", 1e-3)),
                l2=float(meta.get("l2", 1e-5)),
                chisel_size=chisel_size,
                meta=meta,
            )

            params = dict(model.named_parameters())
            missing: list[str] = []
            for name, p in params.items():
                if name in files:
                    arr = np.asarray(data[name], dtype=np.float32)
                    if tuple(arr.shape) != tuple(p.shape):
                        raise ValueError(
                            f"{path}: 参数 {name} 形状不符：存档 {tuple(arr.shape)} "
                            f"vs 模型 {tuple(p.shape)}（dim={dim}, hidden={hidden}）"
                        )
                    with torch.no_grad():
                        p.copy_(torch.from_numpy(np.array(arr, copy=True)))
                elif name in ("Wc", "bc") and chisel_size == 0:
                    continue
                else:
                    missing.append(name)
            if missing and strict:
                raise ValueError(
                    f"{path}: 存档缺少 {len(missing)} 组参数（{', '.join(missing[:6])}…）。"
                    "如果只是旧权重没有新头，请把对应 kind 从 meta['kinds'] 里去掉；"
                    "否则说明结构真的变了，不能静默随机初始化。"
                )

            # 未知数组（例如以后的辅助头/优化器快照）原样保留。
            for name in files:
                if name == "_meta" or name in params:
                    continue
                model.extra_arrays[name] = np.array(data[name], copy=True)

        model.steps = 0
        model.total_steps = int(meta.get("steps", 0))
        model.epochs_done = int(meta.get("epochs_done", 0))
        added = [k for k in KINDS if k not in model.kinds]
        model.untrained_heads = set(added)
        model.missing_heads = tuple(added)
        if chisel_size > 0:
            model.chisel_updates = int(meta.get("chisel_updates", 0))
            trained = meta.get("chisel_trained", None)
            if trained is None:
                trained = model.chisel_updates > 0 or bool(np.any(model.Wc.detach().cpu().numpy()))
            model.chisel_trained = bool(trained)
        else:
            model.chisel_trained = False
            model.chisel_updates = 0
        return model

    @classmethod
    def from_numpy_net(cls, net) -> "TorchNet":
        """从 numpy ``Net`` 实例直接复制（不经过磁盘）。"""
        chisel_size = int(getattr(net, "Wc", np.zeros((0, 0))).shape[1]) if hasattr(net, "Wc") else 0
        model = cls(
            in_dim=int(net.dim),
            hidden=tuple(int(x) for x in net.hidden),
            kinds=list(net.kinds),
            action_dim=int(net.Wa1.shape[0]),
            seed=0,
            lr=float(net.lr),
            l2=float(net.l2),
            chisel_size=chisel_size,
        )
        src = dict(net.params())
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in src:
                    p.copy_(torch.from_numpy(np.array(src[name], dtype=np.float32, copy=True)))
        model.steps = int(getattr(net, "steps", 0))
        model.total_steps = int(getattr(net, "total_steps", 0))
        model.epochs_done = int(getattr(net, "epochs_done", 0))
        model.chisel_trained = bool(getattr(net, "chisel_trained", False))
        model.chisel_updates = int(getattr(net, "chisel_updates", 0))
        model.untrained_heads = set(getattr(net, "untrained_heads", set()))
        model.missing_heads = tuple(getattr(net, "missing_heads", ()))
        return model


# ---------------------------------------------------------------------------
# 数值工具（与 net.py 的 softmax_masked 语义一致）
# ---------------------------------------------------------------------------


def masked_softmax(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 ``mask > 0`` 归一化；全零掩码兜底成均匀分布（不能出 NaN）。"""
    z = z.masked_fill(mask <= 0, -1e9)
    p = F.softmax(z, dim=-1)
    valid = (mask > 0).any(dim=-1, keepdim=True)
    uniform = torch.full_like(p, 1.0 / p.shape[-1])
    return torch.where(valid, p, uniform)


def load_npz(path: str | Path, **kwargs) -> TorchNet:
    """模块级别名，方便 ``from aimodel.torch_net import load_npz``。"""
    return TorchNet.load_npz(path, **kwargs)


def save_npz(net: TorchNet, path: str | Path) -> Path:
    """模块级别名，方便脚本里直接调用。"""
    return net.save_npz(path)


__all__ = [
    "TorchNet",
    "KINDS",
    "HEAD_SIZE",
    "ACTION_DIM",
    "ACTION_HIDDEN",
    "CHISEL_SIZE",
    "DEFAULT_IN_DIM",
    "DEFAULT_HIDDEN",
    "REQUIRED_META_KEYS",
    "masked_softmax",
    "load_npz",
    "save_npz",
]
