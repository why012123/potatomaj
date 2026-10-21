"""PyTorch 训练循环：流式读 ``_data/pro4/tbc/*.npz``，训练 9 个策略头 + 价值头。

这是「并行通道」，不是把 ``train.py`` 改掉。目标很具体：**今天在 4060 上
能跑通一步真实训练**，并把 loss / 吞吐 / 显存打出来。因此：

* 数据按分片流式加载，一次只把**一个** npz 读进内存（379 片 / 267 MB，
  以后还会更大；一次性 ``np.load`` 全部不是这里的做法）；
* 每个 batch 只做一次前向 + 一次反向；9 个策略头按 ``kind`` 取子集，
  与 ``net.Net.step`` 的损失口径逐项对齐（BC：CE + label smoothing +
  熵奖励；PG：PPO clip + 熵奖励；价值：MSE；L2 只加在权重矩阵上）；
* 默认 BC，因为 ``_data/pro4/tbc`` 的 ``logp`` 恒为 0（人类牌谱没有旧策略
  log prob）；``--mode pg`` 也给自对弈数据留了口子；
* 只新写文件，不 import/修改训练侧任何既有模块的可变状态。

用法::

    python -m aimodel.train_torch --device cuda --steps 50 --shards 8

    # 今天的数据是 560 维；模型类默认 645，训练脚本会从数据自动探测宽度：
    python -m aimodel.train_torch --device cuda --steps 50 --shards 8 \\
        --hidden 512,512 --save _data/torch_smoke.npz

注意
----
* ``--steps`` 是**优化器步数**，不是 epoch；建议冒烟测试 <= 200。
* 每次只读一个分片，但分片内会按行打乱；跨分片按 epoch 重新洗牌。
* 这个文件没有实现 ``train.py`` 的 KL 锚定 / 分头自适应熵 / chisel 训练；
  默认参数下它们都不参与现有 numpy trainer 的损失，故本文件的 BC/PPO
  主损失与 ``Net.step`` 对齐。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .encode import ACTION_DIM, HEAD_SIZE
from .net import KINDS
from .torch_net import DEFAULT_IN_DIM, TorchNet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "_data" / "pro4" / "tbc"

KIND_TO_ID: dict[str, int] = {k: i for i, k in enumerate(KINDS)}
ID_TO_KIND: list[str] = list(KINDS)

#: 训练数据至少要有这些列。
REQUIRED_COLUMNS = ("X", "mask", "actfeat", "action", "kind", "ret")


# ---------------------------------------------------------------------------
# 参数 / 数据发现
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="PyTorch 训练器（行为克隆 / PPO），流式读 tbc 分片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="tbc 分片目录，或单个 .npz")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=50, help="优化器步数（不是 epoch）")
    ap.add_argument("--batch", type=int, default=256, help="minibatch 大小")
    ap.add_argument("--shards", type=int, default=8,
                    help="最多用几个分片；0 = 全部（按文件名排序后取前 N 个）")
    ap.add_argument("--in-dim", type=int, default=0,
                    help="输入维度；0 = 从数据自动探测（模型类默认 645）")
    ap.add_argument("--hidden", default="256,256", help="隐层宽度，逗号分隔")
    ap.add_argument("--mode", choices=["bc", "pg"], default="bc",
                    help="bc = 行为克隆（tbc 数据）；pg = PPO（自对弈数据）")
    ap.add_argument("--lr", type=float, default=None,
                    help="学习率；None = 续训用存档值 / BC 1e-3 / PG 3e-4")
    ap.add_argument("--l2", type=float, default=None,
                    help="L2（只作用在权重矩阵上，与 net.py 一致）；None = 续训用存档值 / 1e-4")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="熵奖励系数；None = BC 0.001 / PG 0.008")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO 截断（仅 --mode pg）")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="BC 软标签系数（仅 --mode bc；0 = 硬交叉熵）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU 线程数；默认 1，尽量不抢 RL/配对赛的 CPU")
    ap.add_argument("--resume", default=None, help="从 numpy .npz 权重继续训练")
    ap.add_argument("--save", default=None, help="训练结束写出的 numpy .npz")
    ap.add_argument("--force", action="store_true", help="--save 目标已存在时允许覆盖")
    ap.add_argument("--print-every", type=int, default=1, help="每多少步打一行")
    return ap


def find_shards(data: str | Path) -> tuple[list[Path], list[Path]]:
    """返回 ``(全部, 选中)``；目录按文件名排序，单文件直接返回。"""
    p = Path(data)
    if p.is_file():
        return [p], [p]
    if not p.exists():
        raise SystemExit(f"❌ 数据路径不存在：{p}")
    shards = [q for q in sorted(p.glob("*.npz")) if ".tmp." not in q.name]
    if not shards:
        shards = [q for q in sorted(p.rglob("*.npz")) if ".tmp." not in q.name]
    if not shards:
        raise SystemExit(f"❌ {p} 下没有 .npz 分片")
    return shards, shards


def parse_hidden(text: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x) for x in str(text).split(",") if x.strip())
    except ValueError as e:
        raise SystemExit(f"❌ --hidden 解析失败：{text!r}（例：256,256）") from e
    if not out:
        raise SystemExit("❌ --hidden 不能为空")
    return out


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("❌ --device cuda，但当前 torch 看不到 CUDA")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 分片流式读取
# ---------------------------------------------------------------------------


def _read_shard(path: Path, *, need_logp: bool) -> dict[str, np.ndarray]:
    """只读当前分片；校验列名、动作类别和形状。"""
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in REQUIRED_COLUMNS if k not in d.files]
        if need_logp and "logp" not in d.files:
            missing.append("logp")
        if missing:
            raise SystemExit(f"❌ {path} 缺列：{missing}")
        data = {
            "X": np.asarray(d["X"], dtype=np.float32),
            "mask": np.asarray(d["mask"], dtype=np.float32),
            "actfeat": np.asarray(d["actfeat"], dtype=np.float16),
            "action": np.asarray(d["action"], dtype=np.int64),
            "kind": np.asarray(d["kind"]),
            "ret": np.asarray(d["ret"], dtype=np.float32),
        }
        if need_logp:
            data["logp"] = np.asarray(d["logp"], dtype=np.float32)
    n = len(data["X"])
    if n == 0:
        return data
    if data["X"].ndim != 2:
        raise SystemExit(f"❌ {path}: X 应为 (N, dim)，实际 {data['X'].shape}")
    if data["mask"].shape[0] != n or data["action"].shape[0] != n \
            or data["kind"].shape[0] != n or data["ret"].shape[0] != n:
        raise SystemExit(f"❌ {path}: 各列行数不一致")
    unknown = sorted(set(str(k) for k in np.unique(data["kind"])) - set(KIND_TO_ID))
    if unknown:
        raise SystemExit(f"❌ {path}: 未知决策类别 {unknown}；已知 {list(KIND_TO_ID)}")
    return data


def summarize_shards(shards: Sequence[Path], device: torch.device | None = None) -> dict:
    """统计选中分片的总行数 / 类别 / 维度 / 磁盘大小。"""
    rows = 0
    kinds: set[str] = set()
    dims: set[int] = set()
    logp_max = 0.0
    for p in shards:
        with np.load(p, allow_pickle=False) as d:
            x = np.asarray(d["X"])
            rows += int(x.shape[0])
            dims.add(int(x.shape[1]))
            kind = np.asarray(d["kind"])
            if kind.size:
                kinds.update(str(k) for k in np.unique(kind))
            if "logp" in d.files:
                lp = np.asarray(d["logp"], dtype=np.float32)
                if lp.size:
                    logp_max = max(logp_max, float(np.max(np.abs(lp))))
    if len(dims) > 1:
        raise SystemExit(f"❌ 选中的分片特征维度不一致：{sorted(dims)}")
    size_mb = sum(p.stat().st_size for p in shards) / 1024 ** 2
    return {
        "rows": rows,
        "kinds": kinds,
        "dim": next(iter(dims)) if dims else 0,
        "size_mb": size_mb,
        "logp_max": logp_max,
    }


def shard_advantages(
    model: TorchNet,
    X: np.ndarray,
    ret: np.ndarray,
    device: torch.device,
    chunk: int = 4096,
) -> np.ndarray:
    """按分片算 ``ret - V`` 并标准化（对应 train.compute_advantages 的流式版）。

    与 ``train.py`` 的差别：那里对**整个数据集**标准化，这里对**当前分片**
    标准化。对自对弈 shard 来说两者都合理；对 BC 数据（logp 恒 0）不要用 PG。
    """
    vals: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            xb = torch.as_tensor(X[s:s + chunk], dtype=torch.float32, device=device)
            vals.append(model.value_of(xb).detach().cpu().numpy())
    v = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)
    adv = ret.astype(np.float32) - v.astype(np.float32)
    std = float(adv.std())
    out = (adv - adv.mean()) / (std + 1e-6)
    model.train()
    return out.astype(np.float32)


def _make_batch(data: dict[str, np.ndarray], sel: np.ndarray, mode: str) -> dict:
    kind_strs = data["kind"][sel]
    kind_ids = np.fromiter(
        (KIND_TO_ID[str(k)] for k in kind_strs), dtype=np.int64, count=len(sel)
    )
    batch = {
        "X": np.ascontiguousarray(data["X"][sel], dtype=np.float32),
        "mask": np.ascontiguousarray(data["mask"][sel], dtype=np.float32),
        "actfeat": np.ascontiguousarray(data["actfeat"][sel], dtype=np.float32),
        "action": np.ascontiguousarray(data["action"][sel], dtype=np.int64),
        "kind_ids": kind_ids,
        "ret": np.ascontiguousarray(data["ret"][sel], dtype=np.float32),
        "n": int(len(sel)),
    }
    if mode == "pg":
        batch["logp"] = np.ascontiguousarray(data["logp"][sel], dtype=np.float32)
        batch["adv"] = np.ascontiguousarray(data["adv"][sel], dtype=np.float32)
    return batch


def batch_stream(
    shards: Sequence[Path],
    batch_size: int,
    seed: int,
    mode: str,
    model: TorchNet,
    device: torch.device,
    expected_dim: int,
) -> Iterator[dict]:
    """无限流：一次只持有一个分片的数组，按 epoch 洗牌分片顺序。"""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(shards))
        for si in order:
            path = shards[int(si)]
            data = _read_shard(path, need_logp=(mode == "pg"))
            n = len(data["X"])
            if n == 0:
                continue
            if data["X"].shape[1] != expected_dim:
                raise SystemExit(
                    f"❌ {path}: X 维度 {data['X'].shape[1]}，与第一个分片 "
                    f"{expected_dim} 不一致"
                )
            if mode == "pg":
                data["adv"] = shard_advantages(model, data["X"], data["ret"], device)
            perm = rng.permutation(n)
            for s in range(0, n, batch_size):
                sel = perm[s:s + batch_size]
                if len(sel) < 2:
                    continue
                yield _make_batch(data, sel, mode)


def to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {"n": int(batch["n"])}
    for k in ("X", "mask", "actfeat", "ret"):
        out[k] = torch.as_tensor(batch[k], dtype=torch.float32, device=device)
    out["action"] = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    out["kind_ids"] = torch.as_tensor(batch["kind_ids"], dtype=torch.long, device=device)
    if "logp" in batch:
        out["logp"] = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    if "adv" in batch:
        out["adv"] = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    return out


# ---------------------------------------------------------------------------
# 损失（与 net.Net.step 的 BC / PG 口径对齐）
# ---------------------------------------------------------------------------


def compute_loss(
    model: TorchNet,
    batch: dict,
    *,
    mode: str,
    ent_coef: float,
    vf_coef: float,
    label_smoothing: float,
    clip: float,
    collect_per_kind: bool = False,
) -> tuple[torch.Tensor, dict, list[str]]:
    """一次前向 + 标量损失；返回 ``(loss, stats, 本批出现的头)``。

    数学对应 ``Net.step``：

    BC::

        L_policy = (1-eps) * [ sum_rows CE + ent_coef * sum_rows (p*log p) ] / N
        L_value  = vf_coef * mean((v-ret)^2)
        L        = L_policy + L_value

    PG::

        ratio = exp(clamp(logp - old_logp, -20, 20))
        L_policy = [ -sum_rows min(ratio*A, clip(ratio)*A)
                     + ent_coef * sum_rows (p*log p) ] / N

    L2 由调用方在 ``backward`` 之后只加到权重矩阵上（bias 不加），与
    ``Net.step`` 的 ``+ self.l2 * W`` 对齐。
    """
    X = batch["X"]
    mask = batch["mask"]
    action = batch["action"]
    ret = batch["ret"]
    kind_ids = batch["kind_ids"]
    actfeat = batch.get("actfeat")
    n = int(X.shape[0])

    h = model.trunk(X)
    keep = 1.0
    if mode == "bc" and label_smoothing > 0.0:
        keep = 1.0 - float(min(max(label_smoothing, 0.0), 0.999))

    loss_policy = X.new_zeros(())
    ce_sum = X.new_zeros(())          # sum_rows CE，用于显示 BC 的平均交叉熵
    surr_sum = X.new_zeros(())        # PG 每头均值之和，最后除以头数
    ent_rows_sum = X.new_zeros(())    # sum_rows entropy，用于显示
    ent_head_sum = X.new_zeros(())    # 每头平均熵之和
    correct = X.new_zeros(())
    rows_policy = 0
    head_n = 0
    present: list[str] = []
    per_kind_t: dict[str, dict] = {}

    for k in model.kinds:
        kid = KIND_TO_ID[k]
        idx = torch.nonzero(kind_ids == kid, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        present.append(k)
        size = int(HEAD_SIZE[k])
        hk = h.index_select(0, idx)
        afk = actfeat.index_select(0, idx) if (k == "discard" and actfeat is not None) else None
        z = model.logits(hk, k, afk)
        m = mask.index_select(0, idx)[:, :size]
        z = z.masked_fill(m <= 0, -1e9)
        logp_all = F.log_softmax(z, dim=-1)
        p = logp_all.exp()
        a = action.index_select(0, idx)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        ent_rows = -(p * logp_all).sum(dim=1)

        if mode == "bc":
            # 行为克隆：dy/dz = keep * (p - onehot) / N
            pol_head = -logp.sum()
            ce_sum = ce_sum + pol_head
        else:
            # PPO 截断目标：与 net.py 一样先用 unclipped <= clipped 判断，
            # 再对 min 项求导；这里直接让 autograd 处理 where 的梯度。
            old_logp = batch["logp"].index_select(0, idx)
            adv = batch["adv"].index_select(0, idx)
            ratio = torch.exp(torch.clamp(logp - old_logp, -20.0, 20.0))
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            use_un = unclipped <= clipped
            pol_head = -torch.where(use_un, unclipped, clipped).sum()
            # 显示口径与 net.py 的 policy_loss 一致：每头 -mean(...)，再平均。
            surr_sum = surr_sum + (-torch.where(use_un, unclipped, clipped).mean())

        # 熵奖励：d(-H)/dz = p*(log p + H)；对应目标 + ent_coef * sum(p log p)。
        loss_policy = loss_policy + keep * (
            pol_head + float(ent_coef) * (p * logp_all).sum()
        ) / n

        ent_rows_sum = ent_rows_sum + ent_rows.sum()
        ent_head_sum = ent_head_sum + ent_rows.mean()
        head_n += 1
        pred = z.argmax(dim=1)
        correct = correct + (pred == a).sum()
        rows_policy += int(idx.numel())
        if collect_per_kind:
            # 只在最后一步收集逐头明细，避免每步都多做 9×3 次 GPU 归约/同步。
            per_kind_t[k] = {
                "n": int(idx.numel()),
                "ce": (-logp.mean()).detach(),
                "entropy": ent_rows.mean().detach(),
                "acc": (pred == a).float().mean().detach(),
            }

    v = model.value(h)
    value_loss = F.mse_loss(v, ret)
    loss = loss_policy + float(vf_coef) * value_loss

    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float((surr_sum / max(1, head_n)).detach().cpu())
        if mode == "pg" else float((ce_sum / max(1, n)).detach().cpu()),
        "ce": float((ce_sum / max(1, n)).detach().cpu()) if mode == "bc" else None,
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float((ent_head_sum / max(1, head_n)).detach().cpu()),
        "entropy_rows": float((ent_rows_sum / max(1, n)).detach().cpu()),
        "acc": float((correct / max(1, rows_policy)).detach().cpu()),
        "n_policy": rows_policy,
        "heads_present": len(present),
        "per_kind": per_kind_t,
    }
    return loss, stats, present


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------


def _mem_mb(device: torch.device) -> tuple[float, float, float]:
    if device.type != "cuda":
        return (0.0, 0.0, 0.0)
    return (
        torch.cuda.memory_allocated(device) / 1024 ** 2,
        torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        torch.cuda.memory_reserved(device) / 1024 ** 2,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("❌ --steps 必须 > 0")
    if args.batch < 2:
        raise SystemExit("❌ --batch 必须 >= 2")
    if args.shards < 0:
        raise SystemExit("❌ --shards 不能为负（0 = 全部）")

    # 不要抢 RL / 配对赛的 CPU；GPU 计算不受这个限制。
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    # 4060 的 FP32 一致性验证要求关掉 TF32（否则和 numpy 的差会到 1e-3）。
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = pick_device(args.device)
    all_shards, selected = find_shards(args.data)
    if args.shards > 0:
        selected = selected[: args.shards]
    if not selected:
        raise SystemExit("❌ 没有选中任何分片")

    summary = summarize_shards(selected)
    data_dim = int(summary["dim"])
    if data_dim <= 0:
        raise SystemExit("❌ 分片里没有有效行")

    # -- 模型 / 学习率 / 正则 -------------------------------------------------
    if args.resume:
        model = TorchNet.load_npz(args.resume)
        if args.in_dim and int(args.in_dim) != model.in_dim:
            raise SystemExit(
                f"❌ --in-dim {args.in_dim} 与存档 dim={model.in_dim} 不一致"
            )
        if model.in_dim != data_dim:
            raise SystemExit(
                f"❌ 存档 dim={model.in_dim}，数据 dim={data_dim} —— "
                "换过特征布局的权重不能直接续训（先重新训练或做迁移）"
            )
        lr = float(args.lr if args.lr is not None else model.lr)
        l2 = float(args.l2 if args.l2 is not None else model.l2)
    else:
        in_dim = int(args.in_dim) if args.in_dim else data_dim
        if in_dim != data_dim:
            raise SystemExit(
                f"❌ --in-dim {in_dim} 与数据 dim={data_dim} 不一致"
                "（0 = 自动探测；默认模型类是 645，但今天这批 tbc 仍是 560）"
            )
        lr = float(args.lr if args.lr is not None else (3e-4 if args.mode == "pg" else 1e-3))
        l2 = float(args.l2 if args.l2 is not None else 1e-4)
        model = TorchNet(
            in_dim=in_dim,
            hidden=parse_hidden(args.hidden),
            kinds=KINDS,
            action_dim=ACTION_DIM,
            seed=args.seed,
            lr=lr,
            l2=l2,
        )
    model.lr = lr
    model.l2 = l2
    ent_coef = float(
        args.ent_coef if args.ent_coef is not None
        else (0.008 if args.mode == "pg" else 0.001)
    )
    model.to(device)
    model.train()

    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        dev_name = f"{prop.name} ({prop.total_memory / 1024 ** 3:.1f} GB, sm_{prop.major}{prop.minor})"
    else:
        dev_name = "CPU"

    missing_in_data = [k for k in model.kinds if k not in summary["kinds"]]
    extra_in_data = sorted(set(summary["kinds"]) - set(model.kinds))
    print("=" * 88)
    print("PyTorch 训练器  aimodel/train_torch.py")
    print("=" * 88)
    print(f"torch          {torch.__version__}   CUDA {torch.version.cuda or '无'}")
    print(f"device         {device.type}: {dev_name}")
    print(f"data           {len(selected)}/{len(all_shards)} 个分片 / "
          f"{summary['rows']:,} 行 / {summary['size_mb']:.1f} MB")
    print(f"                {Path(args.data)}")
    print(f"data dim       {data_dim}  (模型类默认 {DEFAULT_IN_DIM}；本脚本按数据自动探测)")
    print(f"model          in_dim={model.in_dim} hidden={model.hidden} "
          f"heads={len(model.kinds)} params={model.n_params():,}"
          + (f" chisel={model.chisel_size}" if model.chisel_size else ""))
    print(f"mode           {args.mode.upper()}  lr={lr:g}  l2={l2:g}  "
          f"ent={ent_coef:g}  vf={args.vf_coef:g}"
          + (f"  label_smooth={args.label_smoothing:g}" if args.mode == "bc" else f"  clip={args.clip:g}"))
    print(f"batch/steps    {args.batch} / {args.steps}"
          f"  (epochs_done={model.epochs_done}, total_steps={model.total_steps:,})")
    print(f"data kinds     {sorted(summary['kinds'])}")
    if missing_in_data:
        print(f"注意           这些头在本批数据里没有样本：{missing_in_data}")
    if extra_in_data:
        print(f"⚠️             数据里有模型没有的头（会被跳过）：{extra_in_data}")
    if args.mode == "pg" and summary["logp_max"] < 1e-8:
        print("⚠️             所有 logp≈0：这是 BC 数据，PPO 的 ratio 没有意义；建议 --mode bc")
    print("-" * 88, flush=True)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    stream = batch_stream(
        selected, args.batch, args.seed, args.mode, model, device, data_dim,
    )

    t_start = time.perf_counter()
    first_dt = 0.0
    first_n = 0
    rows_done = 0
    last_stats: dict | None = None
    total_loss = 0.0
    history: list[dict] = []
    pol_label = "ce" if args.mode == "bc" else "surr"

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        batch_np = next(stream)
        batch = to_device(batch_np, device)
        optimizer.zero_grad(set_to_none=True)

        loss, stats, present = compute_loss(
            model, batch,
            mode=args.mode,
            ent_coef=ent_coef,
            vf_coef=float(args.vf_coef),
            label_smoothing=float(args.label_smoothing),
            clip=float(args.clip),
            collect_per_kind=(step == args.steps),
        )
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"❌ step {step}: loss={float(loss)}，训练发散了")

        loss.backward()

        # L2：与 net.py 一样只加在**权重矩阵**上（W0/W1/Wp_*/Wv/Wa1/Wa2），
        # bias 不加；而且只有拿到梯度的参数才更新（缺席的头不会因 L2 被动）。
        l2_sq = torch.zeros((), device=device)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if name.startswith("W"):
                    p.grad.add_(p.detach(), alpha=l2)
                    l2_sq = l2_sq + p.detach().pow(2).sum()
        l2_pen = 0.5 * l2 * float(l2_sq.detach().cpu())

        optimizer.step()
        model.steps += 1
        model.total_steps += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step == 1:
            first_dt, first_n = dt, batch_np["n"]
        rows_done += batch_np["n"]
        total_loss = stats["loss"] + l2_pen
        last_stats = stats
        history.append({
            "loss": total_loss,
            "pol": stats["ce"] if args.mode == "bc" else stats["policy_loss"],
            "value": stats["value_loss"],
            "acc": stats["acc"],
        })

        if step == 1 or step % max(1, args.print_every) == 0 or step == args.steps:
            alloc, peak, reserved = _mem_mb(device)
            pol_val = stats["ce"] if args.mode == "bc" else stats["policy_loss"]
            print(
                f"step {step:3d}/{args.steps}  "
                f"loss={total_loss:.4f}  {pol_label}={pol_val:.4f}  "
                f"value={stats['value_loss']:.4f}  ent={stats['entropy']:.3f}  "
                f"acc={stats['acc']:.1%}  |  "
                f"{dt * 1000:.1f} ms  {batch_np['n'] / max(dt, 1e-9):,.0f} samp/s  |  "
                f"VRAM {alloc:.1f}/{peak:.1f} MB (res {reserved:.1f})"
                + (f"  heads={stats['heads_present']}" if step == 1 else ""),
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    steady_dt = max(elapsed - first_dt, 1e-9)
    steady_rows = max(rows_done - first_n, 0)
    alloc, peak, reserved = _mem_mb(device)
    print("-" * 88)
    print(f"完成：{args.steps} 步 / {rows_done:,} 样本 / {elapsed:.2f} s")
    print(f"平均吞吐：{rows_done / elapsed:,.0f} 样本/秒（含首步 CUDA 初始化）")
    print(f"稳态吞吐：{steady_rows / steady_dt:,.0f} 样本/秒（第 2 步起）")
    print(f"显存：当前 {alloc:.1f} MB / 峰值 {peak:.1f} MB / reserved {reserved:.1f} MB")
    print(f"最后一步：loss={total_loss:.4f}  "
          f"value={last_stats['value_loss'] if last_stats else float('nan'):.4f}  "
          f"ent={last_stats['entropy'] if last_stats else float('nan'):.3f}  "
          f"acc={last_stats['acc'] if last_stats else float('nan'):.1%}")
    if len(history) >= 4:
        w = min(10, len(history) // 2)
        head, tail = history[:w], history[-w:]

        def _avg(rows: list[dict], key: str) -> float:
            return sum(float(r[key]) for r in rows) / max(1, len(rows))

        print(
            f"趋势（前{w}/后{w}步均值）："
            f"loss {_avg(head, 'loss'):.4f} -> {_avg(tail, 'loss'):.4f}；"
            f"{pol_label} {_avg(head, 'pol'):.4f} -> {_avg(tail, 'pol'):.4f}；"
            f"value {_avg(head, 'value'):.4f} -> {_avg(tail, 'value'):.4f}；"
            f"acc {_avg(head, 'acc'):.1%} -> {_avg(tail, 'acc'):.1%}"
        )
    if last_stats and last_stats.get("per_kind"):
        print("逐头（最后一个 batch）：")
        for k, v in last_stats["per_kind"].items():
            ce = float(v["ce"].detach().cpu()) if torch.is_tensor(v["ce"]) else float(v["ce"])
            ent = float(v["entropy"].detach().cpu()) if torch.is_tensor(v["entropy"]) else float(v["entropy"])
            acc = float(v["acc"].detach().cpu()) if torch.is_tensor(v["acc"]) else float(v["acc"])
            print(f"  {k:<8} n={v['n']:4d}  ce={ce:.4f}  "
                  f"ent={ent:.3f}  acc={acc:.1%}")

    if args.save:
        out = Path(args.save)
        if out.exists() and not args.force:
            alt = out.with_name(out.stem + ".new.npz")
            print(f"\n⚠️ {out} 已存在，改写 {alt}（要覆盖原文件加 --force）")
            out = alt
        model.lr = lr
        model.l2 = l2
        model.save_npz(out)
        print(f"\n已保存 -> {out}（{model.n_params():,} 参数；"
              f"_meta steps={model.total_steps}, dim={model.in_dim}, "
              f"hidden={list(model.hidden)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
