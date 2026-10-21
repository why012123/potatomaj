"""PyTorch 网络的测试：结构、前向一致性、numpy <-> torch 权重往返、GPU 冒烟。

最重要的不是「跑起来」，而是**能证明它和手写 numpy 版是同一个函数**：

* 初始化逐位相同（同 seed、同分布、同顺序）；
* 9 个策略头 + 价值头的 logits / 概率 / value 与 ``Net`` 的差 < 1e-4；
* ``load_npz -> save_npz -> Net.load`` 后参数与 ``_meta`` 逐位/逐键保住；
* **负向验证**：故意篡改一个权重，上面的一致性检查必须红；
* 没有 CUDA 时 GPU 段 **跳过而不是失败**。

跑法::

    .venv/Scripts/python.exe aimodel/tests/test_torch_net.py
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 这个测试是「PyTorch 专用」的；没装 torch 的环境（例如只跑 numpy 的 CI）
# 应当整体跳过，而不是把 run_all.py 弄红。
try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"SKIP: PyTorch 不可用（{type(exc).__name__}: {exc}）")
    print("      这个测试需要 torch；当前环境没有安装，跳过。")
    raise SystemExit(0)

from aimodel.encode import ACTION_DIM, HEAD_SIZE  # noqa: E402
from aimodel.net import KINDS, Net, softmax_masked  # noqa: E402
from aimodel.torch_net import (  # noqa: E402
    DEFAULT_IN_DIM,
    REQUIRED_META_KEYS,
    TorchNet,
    masked_softmax,
)

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def assert_forward_parity(
    np_net: Net,
    t_net: TorchNet,
    X: np.ndarray,
    mask: np.ndarray,
    actfeat: np.ndarray,
    *,
    tol: float = 1e-4,
    label: str = "",
) -> dict[str, float]:
    """逐头比较 numpy / torch 的 trunk、logits、softmax、value。"""
    X = np.ascontiguousarray(X, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    actfeat = np.ascontiguousarray(actfeat, dtype=np.float32)

    h_np = np_net.trunk(X)
    h_t_t = t_net.trunk(torch.from_numpy(X))
    h_t = h_t_t.detach().cpu().numpy()
    diffs: dict[str, float] = {"trunk": max_abs(h_np, h_t)}

    for k in t_net.kinds:
        if k not in np_net.kinds:
            continue
        size = int(HEAD_SIZE[k])
        af_np = actfeat if k == "discard" else None
        af_t = torch.from_numpy(actfeat) if k == "discard" else None
        z_np = np_net.logits(h_np, k, af_np)
        z_t = t_net.logits(h_t_t, k, af_t).detach().cpu().numpy()
        diffs[f"logits:{k}"] = max_abs(z_np, z_t)

        m = mask[:, :size]
        p_np = np.stack([softmax_masked(z_np[i], m[i]) for i in range(len(X))])
        p_t = masked_softmax(
            t_net.logits(h_t_t, k, af_t), torch.from_numpy(m)
        ).detach().cpu().numpy()
        diffs[f"policy:{k}"] = max_abs(p_np, p_t)

    v_np = np_net.value(h_np)
    v_t = t_net.value(h_t_t).detach().cpu().numpy()
    diffs["value"] = max_abs(v_np, v_t)

    worst = max(diffs.values()) if diffs else 0.0
    if worst > tol:
        detail = "  ".join(
            f"{n}={v:.2e}" for n, v in sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        )
        raise AssertionError(f"{label} 前向不一致：max|Δ|={worst:.3e} > {tol:g}；{detail}")
    return diffs


# ---------------------------------------------------------------------------
print("=" * 78)
print("== 1. 结构 / 默认输入维度 ==")
print("=" * 78)
_sig = inspect.signature(TorchNet.__init__)
_default_dim = _sig.parameters["in_dim"].default
check(DEFAULT_IN_DIM == 645, f"DEFAULT_IN_DIM 应为 645，实际 {DEFAULT_IN_DIM}")
check(_default_dim == 645, f"TorchNet(in_dim=...) 默认值应为 645，实际 {_default_dim}")
check(len(KINDS) == 9, f"应当有 9 个策略头，实际 {len(KINDS)}: {KINDS}")
check(tuple(KINDS) == (
    "discard", "tsumo", "ron", "riichi", "call", "chi", "kita", "kan", "kyuushu",
), f"9 个头的顺序/名字不对：{KINDS}")

# 同 seed 初始化必须逐位一致（numpy 版与 torch 版共用同一套 RNG 调用顺序）。
np_small = Net(dim=48, hidden=(16, 16), seed=3)
t_small = TorchNet(in_dim=48, hidden=(16, 16), kinds=KINDS, action_dim=ACTION_DIM, seed=3)
_sd = t_small.state_dict()
for _name, _p in np_small.params():
    if _name in _sd:
        check(np.array_equal(np.asarray(_p), _sd[_name].detach().cpu().numpy()),
              f"初始化 {_name} 与 numpy 版不逐位相同")
print(f"  ✅ 9 个策略头：{list(KINDS)}")
print(f"  ✅ 宽度：{ {k: HEAD_SIZE[k] for k in KINDS} }")
print(f"  ✅ 默认 in_dim=645；同 seed 初始化与 numpy 逐位一致")
_shared = t_small.n_params()
print(f"  ✅ 参数量：共享 {_shared:,}；numpy 当前额外带可选 chisel "
      f"{np_small.n_params() - _shared:,}（本测试 torch 侧未启用）")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 2. 随机小网络：9 头 + value 的数值一致性 ==")
print("=" * 78)
rng = np.random.default_rng(0)
N = 32
X = rng.normal(size=(N, 48)).astype(np.float32)
mask = np.zeros((N, 34), dtype=np.float32)
for _k, _size in HEAD_SIZE.items():
    mask[:, :_size] = 1.0
actfeat = (rng.normal(size=(N, 34, ACTION_DIM)) * 0.5).astype(np.float32)
diffs = assert_forward_parity(np_small, t_small, X, mask, actfeat, label="small")
print(f"  trunk max|Δ|={diffs['trunk']:.3e}  value max|Δ|={diffs['value']:.3e}")
print("  策略头 logits / softmax max|Δ|：")
for _k in KINDS:
    print(f"    {_k:<8} logits={diffs[f'logits:{_k}']:.3e}  "
          f"policy={diffs[f'policy:{_k}']:.3e}")
check(max(diffs.values()) < 1e-4, f"小网络前向最大差 {max(diffs.values()):.3e} 应 < 1e-4")
print(f"  ✅ 全部 9 头 + value 都在 1e-4 内（max={max(diffs.values()):.3e}）")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 2b. 一步梯度：numpy Net.step 的手写反向 vs torch autograd ==")
print("=" * 78)
# 这一条比前向更强：它证明 BC 的 loss 口径（CE + entropy + label smoothing
# + value；L2=0）在两条通道上给出同一个梯度，而不只是「输出碰巧接近」。
from aimodel.train_torch import KIND_TO_ID, compute_loss  # noqa: E402

g_rng = np.random.default_rng(7)
GN = 36
g_net = Net(dim=48, hidden=(16, 16), seed=3, lr=0.0, l2=0.0)
g_torch = TorchNet.from_numpy_net(g_net)
g_X = g_rng.normal(size=(GN, 48)).astype(np.float32)
g_mask = np.zeros((GN, 34), dtype=np.float32)
for _k, _size in HEAD_SIZE.items():
    g_mask[:, :_size] = 1.0
g_kind = np.array([KINDS[i % len(KINDS)] for i in range(GN)])
g_action = np.array(
    [int(g_rng.integers(0, HEAD_SIZE[k])) for k in g_kind], dtype=np.int64
)
g_af = (g_rng.normal(size=(GN, 34, ACTION_DIM)) * 0.3).astype(np.float32)
g_ret = (g_rng.normal(size=GN) * 0.5).astype(np.float32)
g_adv = np.ones(GN, dtype=np.float32)
g_logp = np.zeros(GN, dtype=np.float32)

captured: dict = {}
_orig_apply = g_net._apply


def _capture(grads, **_kw):
    captured.update({k: v.copy() for k, v in grads.items()})


g_net._apply = _capture  # type: ignore[assignment]
g_np_stats = g_net.step(
    {"X": g_X, "mask": g_mask, "action": g_action, "logp": g_logp,
     "kind": g_kind, "adv": g_adv, "ret": g_ret, "mode": "bc", "actfeat": g_af},
    clip=0.0, ent_coef=0.01, vf_coef=0.5, label_smoothing=0.05,
)
g_net._apply = _orig_apply  # type: ignore[assignment]

g_batch = {
    "X": torch.from_numpy(g_X),
    "mask": torch.from_numpy(g_mask),
    "actfeat": torch.from_numpy(g_af),
    "action": torch.from_numpy(g_action),
    "kind_ids": torch.tensor([KIND_TO_ID[k] for k in g_kind], dtype=torch.long),
    "ret": torch.from_numpy(g_ret),
    "n": GN,
}
g_loss, g_stats, _present = compute_loss(
    g_torch, g_batch, mode="bc", ent_coef=0.01, vf_coef=0.5,
    label_smoothing=0.05, clip=0.2,
)
g_loss.backward()

check(abs(float(g_stats["value_loss"]) - float(g_np_stats["value_loss"])) < 1e-5,
      f"value_loss 不一致：torch {float(g_stats['value_loss']):.6f} vs "
      f"numpy {float(g_np_stats['value_loss']):.6f}")
check(abs(float(g_stats["entropy"]) - float(g_np_stats["entropy"])) < 1e-5,
      f"entropy 不一致：torch {float(g_stats['entropy']):.6f} vs "
      f"numpy {float(g_np_stats['entropy']):.6f}")

g_worst = 0.0
g_worst_name = ""
g_compared = 0
for _name, _p in g_torch.named_parameters():
    if _p.grad is None:
        continue
    _g = captured.get(_name)
    if _g is None:
        check(False, f"numpy step 没有给出 {_name} 的梯度")
        continue
    _d = max_abs(_p.grad.detach().cpu().numpy(), _g)
    g_compared += 1
    if _d > g_worst:
        g_worst, g_worst_name = _d, _name
check(g_worst < 5e-6,
      f"torch autograd 与 numpy 手写反向最大差 {g_worst:.3e}（{g_worst_name}）"
      " 应 < 5e-6")
print(f"  value_loss: torch={float(g_stats['value_loss']):.8f} "
      f"numpy={float(g_np_stats['value_loss']):.8f}")
print(f"  entropy   : torch={float(g_stats['entropy']):.8f} "
      f"numpy={float(g_np_stats['entropy']):.8f}")
print(f"  ✅ 比较了 {g_compared} 组参数梯度；最大绝对差 {g_worst:.3e}"
      f"（{g_worst_name}）—— BC loss 口径（含 entropy/label smoothing）一致")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 3. 真实权重：maj-pro4v1.1_mac_rl.npz（560 维 / 512×512） ==")
print("=" * 78)
weights = ROOT / "aimodel" / "weights" / "maj-pro4v1.1_mac_rl.npz"
real_ok = False
if not weights.exists():
    print(f"  SKIP: 找不到 {weights}")
else:
    np_real = Net.load(weights)
    t_real = TorchNet.load_npz(weights)
    check(t_real.in_dim == np_real.dim,
          f"load_npz 的 dim={t_real.in_dim} 应等于 numpy 的 {np_real.dim}")
    check(t_real.hidden == tuple(np_real.hidden),
          f"hidden 不一致：{t_real.hidden} vs {np_real.hidden}")
    check(t_real.kinds == list(np_real.kinds),
          f"kinds 不一致：{t_real.kinds} vs {np_real.kinds}")
    Xr = (rng.normal(size=(64, np_real.dim)) * 0.1).astype(np.float32)
    maskr = np.zeros((64, 34), dtype=np.float32)
    for _k, _size in HEAD_SIZE.items():
        maskr[:, :_size] = 1.0
    afr = (rng.normal(size=(64, 34, ACTION_DIM)) * 0.5).astype(np.float32)
    dr = assert_forward_parity(np_real, t_real, Xr, maskr, afr, label="mac_rl")
    print(f"  numpy: dim={np_real.dim} hidden={np_real.hidden} kinds={np_real.kinds}")
    print(f"  torch: dim={t_real.in_dim} hidden={t_real.hidden} "
          f"params={t_real.n_params():,} chisel={t_real.chisel_size}")
    print(f"  trunk max|Δ|={dr['trunk']:.3e}  value max|Δ|={dr['value']:.3e}")
    for _k in KINDS:
        print(f"    {_k:<8} logits={dr[f'logits:{_k}']:.3e}  "
              f"policy={dr[f'policy:{_k}']:.3e}")
    check(max(dr.values()) < 1e-4,
          f"mac_rl 前向最大差 {max(dr.values()):.3e} 应 < 1e-4")

    # 如果这个 checkout 里正好有真实 tbc 数据，再拿真样本跑一遍（不是随机特征）。
    _tbc = ROOT / "_data" / "pro4" / "tbc"
    _shards = sorted(_tbc.glob("*.npz")) if _tbc.is_dir() else []
    if _shards:
        with np.load(_shards[0], allow_pickle=False) as _d:
            _n = min(256, len(_d["X"]))
            Xd = np.asarray(_d["X"][:_n], dtype=np.float32)
            maskd = np.asarray(_d["mask"][:_n], dtype=np.float32)
            afd = np.asarray(_d["actfeat"][:_n], dtype=np.float32)
        dd = assert_forward_parity(np_real, t_real, Xd, maskd, afd,
                                   label="mac_rl real tbc")
        print(f"  真实 tbc 样本 {_n} 条（{_shards[0].name}）："
              f"trunk={dd['trunk']:.3e} value={dd['value']:.3e} "
              f"max={max(dd.values()):.3e}")
        check(max(dd.values()) < 1e-4,
              f"mac_rl 在真实 tbc 样本上的前向最大差 {max(dd.values()):.3e} 应 < 1e-4")
    else:
        print(f"  （本 checkout 没有 {_tbc}，真实 tbc 样本对比跳过；上面的随机特征已足够）")
    real_ok = True
    print(f"  ✅ 真实权重 9 头 + value 全部在 1e-4 内（max={max(dr.values()):.3e}）")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 4. 权重往返：numpy -> torch -> numpy（含 _meta / 未知键） ==")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    p1 = td / "numpy_weights.npz"
    p2 = td / "torch_weights.npz"

    net = Net(dim=48, hidden=(16, 16), seed=7, lr=3e-4, l2=1e-4)
    # 随机扰动所有权重，避免「全零/对称权重让错误看不出来」。
    for _name, _p in net.params():
        _p += rng.normal(0.0, 0.01, _p.shape).astype(_p.dtype)
    if hasattr(net, "Wc"):
        net.Wc[...] = rng.normal(0.0, 0.01, net.Wc.shape).astype(np.float32)
        net.bc[...] = rng.normal(0.0, 0.01, net.bc.shape).astype(np.float32)
        net.chisel_trained = True
        net.chisel_updates = 5
    net.epochs_done = 4
    net.total_steps = 123
    net.save(p1)

    # 写入一个未知 meta 键 + 一个未知数组，验证「转一圈不丢」。
    with np.load(p1, allow_pickle=False) as d:
        arrays = {k: np.array(d[k], copy=True) for k in d.files}
    meta = json.loads(bytes(arrays["_meta"]).decode("utf-8"))
    meta["custom_key"] = "keep-me"
    arrays["_meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    arrays["mystery"] = np.arange(5, dtype=np.float32)
    np.savez_compressed(p1, **arrays)

    t = TorchNet.load_npz(p1)
    check(t.in_dim == 48 and t.hidden == (16, 16), "load_npz 结构参数错误")
    check(t.total_steps == 123, f"load_npz steps 应为 123，实际 {t.total_steps}")
    check(t.epochs_done == 4, f"load_npz epochs_done 应为 4，实际 {t.epochs_done}")
    check(t.extra_meta.get("custom_key") == "keep-me",
          "load_npz 没有保留未知 _meta 键 custom_key")
    check("mystery" in t.extra_arrays, "load_npz 没有保留未知数组 mystery")

    _sd = t.state_dict()
    bad_params = [_name for _name, _p in net.params()
                  if _name in _sd and not np.array_equal(
                      np.asarray(_p), _sd[_name].detach().cpu().numpy())]
    check(not bad_params, f"numpy -> torch 参数不逐位一致：{bad_params[:5]}")
    if hasattr(net, "Wc"):
        check(t.chisel_size == 3, f"应识别出 chisel=3，实际 {t.chisel_size}")
        check(bool(t.chisel_trained) and t.chisel_updates == 5,
              "chisel_trained/chisel_updates 没有从 _meta 恢复")

    # torch -> numpy：改一下非权重状态，确认写回的是 torch 当前状态。
    t.epochs_done = 5
    t.total_steps = 456
    t.lr = 2e-4
    t.l2 = 2e-4
    t.save_npz(p2)

    with np.load(p2, allow_pickle=False) as d:
        meta2 = json.loads(bytes(d["_meta"]).decode("utf-8"))
        check("mystery" in d.files, "save_npz 没有把未知数组 mystery 写回去")
        check(np.array_equal(np.asarray(d["mystery"]), np.arange(5, dtype=np.float32)),
              "未知数组 mystery 的内容被改了")
    for _key in REQUIRED_META_KEYS:
        check(_key in meta2, f"save_npz 的 _meta 缺少必需键 {_key!r}")
    check(int(meta2.get("steps", -1)) == 456,
          f"_meta['steps'] 应为 456，实际 {meta2.get('steps')}")
    check(int(meta2.get("dim", -1)) == 48,
          f"_meta['dim'] 应为 48，实际 {meta2.get('dim')}")
    check(list(meta2.get("hidden", [])) == [16, 16],
          f"_meta['hidden'] 应为 [16,16]，实际 {meta2.get('hidden')}")
    check(list(meta2.get("kinds", [])) == list(KINDS),
          f"_meta['kinds'] 没有保住：{meta2.get('kinds')}")
    check(int(meta2.get("epochs_done", -1)) == 5,
          f"_meta['epochs_done'] 应为 5，实际 {meta2.get('epochs_done')}")
    check(abs(float(meta2.get("lr", -1)) - 2e-4) < 1e-12,
          f"_meta['lr'] 应为 2e-4，实际 {meta2.get('lr')}")
    check(abs(float(meta2.get("l2", -1)) - 2e-4) < 1e-12,
          f"_meta['l2'] 应为 2e-4，实际 {meta2.get('l2')}")
    check(int(meta2.get("action_dim", -1)) == ACTION_DIM,
          f"_meta['action_dim'] 应为 {ACTION_DIM}，实际 {meta2.get('action_dim')}")
    check(meta2.get("custom_key") == "keep-me", "save_npz 丢了未知 _meta 键")

    back = Net.load(p2)
    check(back.dim == 48 and back.hidden == (16, 16), "Net.load 结构参数错误")
    check(back.total_steps == 456 and back.epochs_done == 5,
          "Net.load 没恢复 steps/epochs_done")
    check(abs(back.lr - 2e-4) < 1e-12 and abs(back.l2 - 2e-4) < 1e-12,
          "Net.load 没恢复 lr/l2")
    back_sd = dict(back.params())
    bad_back = [_name for _name, _p in net.params()
                if _name in back_sd and not np.array_equal(np.asarray(_p), back_sd[_name])]
    check(not bad_back, f"torch -> numpy 参数不逐位一致：{bad_back[:5]}")
    if hasattr(net, "Wc"):
        check(bool(back.chisel_trained) and back.chisel_updates == 5,
              "chisel 状态没有在往返后保住")
    print(f"  ✅ 参数逐位一致；_meta 必需键 {list(REQUIRED_META_KEYS)} 全在")
    print(f"  ✅ 未知 _meta 键 custom_key / 未知数组 mystery 也保住了")
    print(f"  ✅ torch 当前 lr=2e-4 l2=2e-4 steps=456 epochs=5 写回后 Net.load 读到了")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 5. 负向验证：故意改权重，一致性检查必须红 ==")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    p = td / "base.npz"
    net = Net(dim=48, hidden=(16, 16), seed=11)
    net.save(p)
    t_ok = TorchNet.load_npz(p)
    t_bad = TorchNet.load_npz(p)
    # 只改一个权重元素，其他完全一样。
    with torch.no_grad():
        t_bad.Wp_discard[0, 0] += 1.0

    Xn = (rng.normal(size=(16, 48)) * 0.1).astype(np.float32)
    maskn = np.zeros((16, 34), dtype=np.float32)
    for _k, _size in HEAD_SIZE.items():
        maskn[:, :_size] = 1.0
    afn = (rng.normal(size=(16, 34, ACTION_DIM)) * 0.5).astype(np.float32)

    base = assert_forward_parity(net, t_ok, Xn, maskn, afn, label="base")
    caught = False
    try:
        assert_forward_parity(net, t_bad, Xn, maskn, afn, label="corrupted")
    except AssertionError as exc:
        caught = True
        print(f"  ✅ 篡改 Wp_discard[0,0] 后检查确实红了：{exc}")

    check(caught, "负向验证失败：篡改权重后一致性检查没有红")
    # 再确认这个「篡改」真的产生了可见差异，而不是恰好没影响输出。
    _h = t_bad.trunk(torch.from_numpy(Xn))
    _z_bad = t_bad.logits(_h, "discard", torch.from_numpy(afn)).detach().numpy()
    _h_ok = t_ok.trunk(torch.from_numpy(Xn))
    _z_ok = t_ok.logits(_h_ok, "discard", torch.from_numpy(afn)).detach().numpy()
    bad_diff = max_abs(_z_ok, _z_bad)
    check(bad_diff > 1e-4, f"篡改后的输出差只有 {bad_diff:.3e}，负向验证没有意义")
    print(f"  ✅ 干净权重 max|Δ|={max(base.values()):.3e}；"
          f"篡改后 discard logits max|Δ|={bad_diff:.3e}（> 1e-4）")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("== 6. CUDA（没有就跳过，不失败） ==")
print("=" * 78)
if not torch.cuda.is_available():
    print("  SKIP: torch.cuda.is_available() == False —— 这台机器没有可用 CUDA，"
          "GPU 段跳过（不算失败）")
else:
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"  GPU: {prop.name}  {prop.total_memory / 1024 ** 3:.1f} GB  sm_{prop.major}{prop.minor}")
    if weights.exists():
        src_cpu = TorchNet.load_npz(weights)
        src_gpu = TorchNet.load_npz(weights)
        in_dim = src_cpu.in_dim
    else:
        src_cpu = TorchNet.from_numpy_net(np_small)
        src_gpu = TorchNet.from_numpy_net(np_small)
        in_dim = src_cpu.in_dim
    src_gpu = src_gpu.to(dev)
    x = (rng.normal(size=(64, in_dim)) * 0.1).astype(np.float32)
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        y_cpu = src_cpu.trunk(x_t)
        v_cpu = src_cpu.value(y_cpu)
        y_gpu = src_gpu.trunk(x_t.to(dev))
        v_gpu = src_gpu.value(y_gpu)
    diff = max_abs(y_cpu.numpy(), y_gpu.detach().cpu().numpy())
    vdiff = max_abs(v_cpu.numpy(), v_gpu.detach().cpu().numpy())
    check(diff < 1e-4, f"GPU/CPU trunk 差 {diff:.3e} 应 < 1e-4")
    check(vdiff < 1e-4, f"GPU/CPU value 差 {vdiff:.3e} 应 < 1e-4")
    print(f"  CPU vs GPU：trunk max|Δ|={diff:.3e}  value max|Δ|={vdiff:.3e}")

    # 一次真正的 GPU optimizer step（不是只做前向）。
    src_gpu.train()
    opt = torch.optim.Adam(src_gpu.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    xb = torch.from_numpy(
        (rng.normal(size=(16, in_dim)) * 0.1).astype(np.float32)
    ).to(dev)
    loss = src_gpu.value(src_gpu.trunk(xb)).pow(2).mean()
    loss.backward()
    opt.step()
    torch.cuda.synchronize(dev)
    check(bool(torch.isfinite(loss)), "GPU optimizer step 的 loss 不是有限值")
    print(f"  ✅ GPU 前向一致；一次 Adam step OK（loss={float(loss):.6f}）")
    print(f"  显存：allocated={torch.cuda.memory_allocated(dev) / 1024 ** 2:.1f} MB  "
          f"peak={torch.cuda.max_memory_allocated(dev) / 1024 ** 2:.1f} MB")

# ---------------------------------------------------------------------------
print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED:
        print("   -", f)
    raise SystemExit(1)
print("✅ 全部通过")
