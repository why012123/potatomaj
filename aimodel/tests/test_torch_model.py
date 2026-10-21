#!/usr/bin/env python
"""``torch_model.TileConvNet`` 的单元测试（dot V1 路线书第 ② 步）。

钉死：
* 结构 = 路线书：Conv1d(F->64) + 4×ResBlock(64) + Conv1d(64->64) +
  FC(34*64+21 -> 512 -> 256) + 9 头 + 价值头（tanh）；
* 头名/宽度 = ``net.KINDS`` / ``encode.HEAD_SIZE``，**不手抄**；
* 参数量 1M ~ 3M（构造时模块自己也会 assert 并打印）；
* 前向形状、两种输入布局（(B,F,34) / (B,34,F)）结果逐位一致；
* 一次反向能跑通（loss 有限、每个参数都有梯度、优化器 step 后参数有更新）；
* **负向验证**：通道数改错 / 34 改成 35 / 全局标量维度错 / batch 不匹配，
  必须抛 ``ValueError``，绝不静默 reshape；
* 有 CUDA 时做一次 GPU 前向+反向冒烟（没有就跳过，不算失败）。

跑法::

    .venv/Scripts/python.exe aimodel/tests/test_torch_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # noqa: BLE001
    print(f"SKIP: PyTorch 不可用（{type(exc).__name__}: {exc}）")
    raise SystemExit(0)

from aimodel.encode import HEAD_SIZE  # noqa: E402
from aimodel.encode_tile import F as TILE_F, G as GLOBAL_G  # noqa: E402
from aimodel.net import KINDS  # noqa: E402
from aimodel.torch_model import (  # noqa: E402
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    MAX_PARAMS,
    MIN_PARAMS,
    TILE_POSITIONS,
    TileConvNet,
)

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def expect_error(fn, label: str, exc_type=ValueError) -> bool:
    try:
        fn()
    except exc_type as e:
        print(f"  ✅ {label}: 按预期抛 {type(e).__name__}（{str(e)[:64]}…）")
        return True
    except Exception as e:  # noqa: BLE001
        check(False, f"{label}: 抛了 {type(e).__name__} 而不是 {exc_type.__name__}: {e}")
        return False
    check(False, f"{label}: **没有报错**（负向验证失败）")
    return False


def main() -> int:
    print("=" * 78)
    print("TileConvNet 单元测试（dot V1 per-tile Conv1d）")
    print("=" * 78)

    torch.manual_seed(0)
    net = TileConvNet()

    # ------------------------------------------------------------------
    print("\n== 1. 结构 / 头 / 参数量 ==")
    check(TILE_F == 20, f"输入通道 F 应为 20，实际 {TILE_F}")
    check(GLOBAL_G == 21, f"全局标量 G 应为 21，实际 {GLOBAL_G}")
    check(DEFAULT_CHANNELS == 64, f"路线书 C=64，实际 {DEFAULT_CHANNELS}")
    check(DEFAULT_BLOCKS == 4, f"路线书 4 个残差块，实际 {DEFAULT_BLOCKS}")
    check(net.in_channels == TILE_F and net.global_dim == GLOBAL_G,
          f"网络输入维度 {net.in_channels}/{net.global_dim} 与 encode_tile 不一致")
    check(len(net.res_blocks) == 4, f"残差块数应为 4，实际 {len(net.res_blocks)}")
    check(net.fc1.in_features == 64 * TILE_POSITIONS + GLOBAL_G,
          f"FC1 输入应为 64*34+21={64 * TILE_POSITIONS + GLOBAL_G}，"
          f"实际 {net.fc1.in_features}")
    check(net.fc1.out_features == 512 and net.fc2.out_features == 256,
          f"路线书 FC 是 512 -> 256，实际 {net.fc1.out_features} -> {net.fc2.out_features}")
    check(set(net.heads) == set(KINDS), f"策略头集合 {sorted(net.heads)} != {sorted(KINDS)}")
    for k in KINDS:
        check(net.heads[k].out_features == HEAD_SIZE[k],
              f"头 {k} 宽度 {net.heads[k].out_features} != HEAD_SIZE {HEAD_SIZE[k]}")
    check(set(KINDS) == set(HEAD_SIZE), "net.KINDS 与 encode.HEAD_SIZE 头集合不一致")
    print(f"  头：{[(k, HEAD_SIZE[k]) for k in KINDS]}")

    check(MIN_PARAMS <= net.param_count <= MAX_PARAMS,
          f"参数量 {net.param_count:,} 不在 {MIN_PARAMS:,} ~ {MAX_PARAMS:,}")
    check(net.param_count == sum(p.numel() for p in net.parameters() if p.requires_grad),
          "param_count 与 parameters() 实际总数不一致")
    print(f"  参数量：{net.param_count:,}（区间 {MIN_PARAMS:,} ~ {MAX_PARAMS:,} ✅）")

    # param_report：必须逐层列出，并且总数对得上
    report = net.param_report(verbose=False)
    check("fc1.weight" in report, "param_report 没有列出 fc1.weight")
    check("TOTAL" in report and f"{net.param_count:,}" in report,
          "param_report 的总数与 net.param_count 对不上")
    check("参数量范围" in report, "param_report 缺少参数量范围自检")
    check(len(report.splitlines()) >= 20, "param_report 的逐层行数太少")
    print(f"  param_report 共 {len(report.splitlines())} 行（逐层 + 总数 + 范围）")

    # ------------------------------------------------------------------
    print("\n== 2. 前向形状 / 两种输入布局 ==")
    B = 8
    x = torch.randn(B, TILE_F, TILE_POSITIONS)
    g = torch.randn(B, GLOBAL_G)
    net.eval()
    with torch.no_grad():
        heads, value = net(x, g)
    check(set(heads) == set(KINDS), f"forward 返回头 {sorted(heads)} != {sorted(KINDS)}")
    for k in KINDS:
        check(tuple(heads[k].shape) == (B, HEAD_SIZE[k]),
              f"{k} 输出形状 {tuple(heads[k].shape)} != ({B}, {HEAD_SIZE[k]})")
        check(torch.isfinite(heads[k]).all(), f"{k} 输出有 NaN/Inf")
    check(tuple(value.shape) == (B,), f"value 形状 {tuple(value.shape)} != ({B},)")
    check(torch.isfinite(value).all() and float(value.abs().max()) <= 1.0,
          f"value 应有限且 |value|<=1，实际 max={float(value.abs().max())}")
    print(f"  9 头：{ {k: tuple(heads[k].shape) for k in KINDS} }")
    print(f"  value：{tuple(value.shape)}，|value|max={float(value.abs().max()):.4f} ✅")

    # 直接喂 encode_tile 的 (34, F) 批量布局，也要得到逐位相同的结果
    with torch.no_grad():
        heads_tm, value_tm = net(x.transpose(1, 2).contiguous(), g)
    same = all(torch.equal(heads[k], heads_tm[k]) for k in KINDS) and torch.equal(value, value_tm)
    check(same, "(B,F,34) 与 (B,34,F) 两种布局的前向结果不一致")
    print("  (B,F,34) 与 (B,34,F) 两种布局逐位一致 ✅")

    # trunk / logits / value 分开调用也要对
    with torch.no_grad():
        h = net.trunk(x, g)
        z = net.logits(h, "discard")
        v = net.value(h)
    check(tuple(h.shape) == (B, 256), f"trunk 输出 {tuple(h.shape)} != ({B},256)")
    check(tuple(z.shape) == (B, 34), f"logits(discard) {tuple(z.shape)} != ({B},34)")
    check(torch.equal(v, value), "value(trunk) 与 forward 的 value 不一致")

    # ------------------------------------------------------------------
    print("\n== 3. 一次反向：loss 有梯度、参数有更新 ==")
    torch.manual_seed(1)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    heads, value = net(x, g)
    targets = {k: torch.randint(0, HEAD_SIZE[k], (B,)) for k in KINDS}
    value_target = torch.zeros(B)
    loss = sum(F.cross_entropy(heads[k], targets[k]) for k in KINDS)
    loss = loss + F.mse_loss(value, value_target)
    check(torch.isfinite(loss), f"loss 不是有限值：{loss}")
    opt.zero_grad(set_to_none=True)
    loss.backward()

    missing = [n for n, p in net.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    check(not missing, f"这些参数没有有限梯度：{missing[:8]}")
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    opt.step()
    changed = [n for n, p in net.named_parameters()
               if not torch.equal(before[n], p.detach())]
    check(bool(changed), "优化器 step 之后没有任何参数变化（反向没接上？）")
    print(f"  loss={float(loss):.4f}，{len(list(net.parameters()))} 个参数都有梯度，"
          f"step 后 {len(changed)} 个参数更新 ✅")

    # ------------------------------------------------------------------
    print("\n== 4. 负向验证：形状错必须报错（不静默 reshape）==")
    g_ok = torch.randn(B, GLOBAL_G)
    expect_error(lambda: net(torch.randn(B, TILE_F + 1, TILE_POSITIONS), g_ok),
                 "输入通道数 F+1")
    expect_error(lambda: net(torch.randn(B, TILE_F, TILE_POSITIONS + 1), g_ok),
                 "牌位从 34 改成 35")
    expect_error(lambda: net(torch.randn(B, TILE_POSITIONS + 1, TILE_F), g_ok),
                 "tile-major 牌位从 34 改成 35")
    expect_error(lambda: net(torch.randn(B, TILE_F), g_ok), "二维输入")
    expect_error(lambda: net(torch.randn(1, B, TILE_F, TILE_POSITIONS), g_ok), "四维输入")
    expect_error(lambda: net(torch.randn(B, TILE_F, TILE_POSITIONS),
                             torch.randn(B, GLOBAL_G + 1)), "全局标量 G+1")
    expect_error(lambda: net(torch.randn(B, TILE_F, TILE_POSITIONS),
                             torch.randn(B - 1, GLOBAL_G)), "batch 与 globals 不匹配")
    expect_error(lambda: net(torch.randn(B, TILE_F, TILE_POSITIONS), None),
                 "globals=None", TypeError)
    expect_error(lambda: net.logits(torch.randn(B, 256), "not_a_head"),
                 "未知策略头", KeyError)

    # ------------------------------------------------------------------
    print("\n== 5. CUDA 冒烟（有卡才跑）==")
    if not torch.cuda.is_available():
        print("  跳过：当前环境没有 CUDA")
    else:
        dev = torch.device("cuda")
        cuda_net = TileConvNet().to(dev)
        cuda_net.train()
        xc = torch.randn(64, TILE_F, TILE_POSITIONS, device=dev)
        gc = torch.randn(64, GLOBAL_G, device=dev)
        hc, vc = cuda_net(xc, gc)
        loss_c = sum(h.float().sum() * 0.01 for h in hc.values()) + vc.float().pow(2).sum()
        loss_c.backward()
        torch.cuda.synchronize()
        finite = all(torch.isfinite(p.grad).all()
                     for p in cuda_net.parameters() if p.grad is not None)
        check(torch.isfinite(loss_c) and finite, "CUDA 前向/反向出现 NaN/Inf")
        print(f"  ✅ CUDA（{torch.cuda.get_device_name(0)}）前向+反向通过")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"❌ {len(FAILED)} 条失败")
        for m in FAILED:
            print(f"  - {m.splitlines()[0]}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
