"""神经网络的测试：梯度、掩码、存档。

最重要的一条是**数值梯度校验**：手写反向传播最容易犯的错就是符号写反
（这个坑本项目真踩过 —— 策略梯度少了个负号，训练时 loss 一路往上涨）。
所以这里用有限差分把策略头、价值头、躯干各挑一个参数对一遍。

跑法：``.venv/bin/python aimodel/tests/test_net.py``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.encode import ACTION_DIM, DIM, HEAD_SIZE, action_index, action_mask  # noqa: E402
from aimodel.engine import Decision  # noqa: E402
from aimodel.net import Net, softmax_masked, softmax_rows  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def numeric_grad(net: Net, param: np.ndarray, idx: tuple, loss_fn, eps: float = 1e-3) -> float:
    orig = param[idx]
    param[idx] = orig + eps
    hi = loss_fn()
    param[idx] = orig - eps
    lo = loss_fn()
    param[idx] = orig
    return (hi - lo) / (2 * eps)


def analytic_grad(net: Net, X, mask, act, kind, adv, ret, actfeat=None) -> dict:
    """跑一次 step，把梯度抓出来。"""
    got: dict = {}
    original = net._apply

    def capture(grads, **kw):
        got.update({k: v.copy() for k, v in grads.items()})

    net._apply = capture  # type: ignore[assignment]
    net.step(
        {"X": X, "mask": mask, "action": act, "logp": np.zeros(len(X), dtype=np.float32),
         "kind": kind, "adv": adv, "ret": ret, "mode": "bc", "actfeat": actfeat},
        clip=0.0, ent_coef=0.0, vf_coef=0.5,
    )
    net._apply = original  # type: ignore[assignment]
    return got


print("== 数值梯度校验（行为克隆损失） ==")
rng = np.random.default_rng(0)
N = 24
net = Net(dim=48, hidden=(16, 16), seed=3, lr=0.0, l2=0.0)
# float32 的有限差分噪声太大（权重精度只有 7 位），梯度校验统一换成 float64
net.W = [w.astype(np.float64) for w in net.W]
net.b = [b.astype(np.float64) for b in net.b]
net.Wp = {k: v.astype(np.float64) for k, v in net.Wp.items()}
net.bp = {k: v.astype(np.float64) for k, v in net.bp.items()}
net.Wv = net.Wv.astype(np.float64)
net.bv = net.bv.astype(np.float64)
net.Wa1 = net.Wa1.astype(np.float64)
net.ba1 = net.ba1.astype(np.float64)
net.Wa2 = net.Wa2.astype(np.float64)
net.ba2 = net.ba2.astype(np.float64)
AF = rng.normal(size=(N, 34, ACTION_DIM)).astype(np.float64)
X = rng.normal(size=(N, 48)).astype(np.float32)
mask = np.zeros((N, 34), dtype=np.float32)
mask[:, :10] = 1.0  # 只有前 10 张牌是合法的
act = rng.integers(0, 10, N)
kind = np.array(["discard"] * N)
adv = np.ones(N, dtype=np.float32)
ret = rng.normal(size=N).astype(np.float32) * 0.3


def loss_of() -> float:
    """行为克隆损失 + 价值损失（与 step 内部一致，含动作塔）。"""
    acts, _ = net._forward(X)
    feat = acts[-1]
    z = feat @ net.Wp["discard"] + net.bp["discard"]
    a1 = np.maximum(AF @ net.Wa1 + net.ba1, 0.0)
    z = z + (a1 @ net.Wa2 + net.ba2)[..., 0]
    z = np.where(mask > 0, z, -1e9)
    p = softmax_rows(z)
    ce = -np.mean(np.log(np.maximum(p[np.arange(N), act], 1e-12)))
    v = np.tanh(feat @ net.Wv + net.bv)[:, 0]
    return float(ce + 0.5 * np.mean((v - ret) ** 2))


g = analytic_grad(net, X, mask, act, kind, adv, ret, AF)
for name, param, idx in [
    ("策略头 Wp", net.Wp["discard"], (0, 0)),
    ("策略头 bp", net.bp["discard"], (3,)),
    ("躯干 W0", net.W[0], (5, 2)),
    ("躯干 W1", net.W[1], (4, 7)),
    ("动作塔 Wa1", net.Wa1, (2, 5)),
    ("动作塔 ba1", net.ba1, (7,)),
    ("动作塔 Wa2", net.Wa2, (9, 0)),
    ("动作塔 ba2", net.ba2, (0,)),
    ("价值头 Wv", net.Wv, (2, 0)),
    ("价值头 bv", net.bv, (0,)),
]:
    key = {"策略头 Wp": "Wp_discard", "策略头 bp": "bp_discard", "躯干 W0": "W0",
           "躯干 W1": "W1", "价值头 Wv": "Wv", "价值头 bv": "bv",
           "动作塔 Wa1": "Wa1", "动作塔 ba1": "ba1",
           "动作塔 Wa2": "Wa2", "动作塔 ba2": "ba2"}[name]
    num = numeric_grad(net, param, idx, loss_of)
    ana = float(g[key][idx])
    rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
    check(rel < 0.05, f"{name}{idx}: 数值 {num:+.6e} vs 解析 {ana:+.6e}（相对误差 {rel:.1%}）")
    if rel < 0.05:
        print(f"  ✅ {name}{idx}: {ana:+.3e}（数值 {num:+.3e}）")

print("\n== 学习能力（能不能过拟合小样本） ==")
net = Net(dim=48, hidden=(32, 32), seed=1, lr=3e-3)
net.Wa1 = net.Wa1.astype(np.float32); net.ba1 = net.ba1.astype(np.float32)
net.Wa2 = net.Wa2.astype(np.float32); net.ba2 = net.ba2.astype(np.float32)
AF32 = AF.astype(np.float32)
for _ in range(150):
    net.step(
        {"X": X, "mask": mask, "action": act, "logp": np.zeros(N, dtype=np.float32),
         "kind": kind, "adv": adv, "ret": ret, "mode": "bc", "actfeat": AF32},
        clip=0.0, ent_coef=0.0, vf_coef=0.0,
    )
acts, _ = net._forward(X)
z = acts[-1] @ net.Wp["discard"] + net.bp["discard"]
a1 = np.maximum(AF32 @ net.Wa1 + net.ba1, 0.0)
z = z + (a1 @ net.Wa2 + net.ba2)[..., 0]
z = np.where(mask > 0, z, -1e9)
acc = float((z.argmax(1) == act).mean())
check(acc > 0.95, f"小样本过拟合应当接近 100%，得到 {acc:.0%}")
print(f"  ✅ 24 条样本过拟合到 {acc:.0%}")

print("\n== 动作掩码 ==")
p = softmax_masked(np.zeros(34, dtype=np.float32), mask[0])
check(abs(p.sum() - 1.0) < 1e-6, "掩码后的概率应当归一化")
check(all(p[i] > 0 for i in range(10)) and all(p[i] == 0 for i in range(10, 34)),
      "只有合法动作有概率")
allzero = softmax_masked(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
check(abs(allzero.sum() - 1.0) < 1e-6, "全零掩码也要能兜底，不能出 NaN")
d = Decision("discard", [3, 8, 20])
am = action_mask(d)
check(am[3] == 1 and am[8] == 1 and am[20] == 1 and am.sum() == 3, "打牌掩码按牌型来")
check(action_index(d, 1) == 8, "打牌动作下标就是牌型下标")
d2 = Decision("call", ["pass", "kan"])
am2 = action_mask(d2)
check(am2[0] == 1 and am2[2] == 1 and am2[1] == 0, "碰杠掩码按固定动作表来")
print("  ✅ 掩码与动作映射")

print("\n== 存档往返 ==")
net = Net(dim=DIM, hidden=(32, 32), seed=5)
x = np.random.default_rng(0).normal(size=DIM).astype(np.float32)
before = net.policy(x, "discard", np.ones(34, dtype=np.float32))
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "m.npz"
    net.save(path)
    net2 = Net.load(path)
after = net2.policy(x, "discard", np.ones(34, dtype=np.float32))
check(np.allclose(before, after, atol=1e-6), "存档再读回来，输出应当完全一致")
check(net2.dim == net.dim and net2.hidden == net.hidden, "结构参数应当一起存下来")
check(net.n_params() == net2.n_params(), "参数量一致")
print(f"  ✅ 存档往返一致（{net.n_params()} 参数）")

print("\n== PPO：优势为正应当抬高该动作的概率 ==")
net = Net(dim=48, hidden=(16, 16), seed=2, lr=5e-3, l2=0.0)  # 关掉 L2，才能断言「完全不动」
kind_pg = np.array(["discard"] * N)
adv_pos = np.full(N, 2.0, dtype=np.float32)
before = np.asarray([net.policy(X[i], "discard", mask[i])[act[i]] for i in range(N)])
logp_old = np.log(np.maximum(before, 1e-12)).astype(np.float32)
net.step(
    {"X": X, "mask": mask, "action": act, "logp": logp_old, "kind": kind_pg,
     "adv": adv_pos, "ret": ret, "mode": "pg"},
    clip=0.2, ent_coef=0.0, vf_coef=0.0,
)
after = np.asarray([net.policy(X[i], "discard", mask[i])[act[i]] for i in range(N)])
check(float(after.mean()) > float(before.mean()),
      f"正优势应当提高被选中动作的概率（{before.mean():.4f} -> {after.mean():.4f}）")
print(f"  ✅ 正优势：{before.mean():.4f} -> {after.mean():.4f}")
# 优势为负则应当压低
before2 = np.asarray([net.policy(X[i], "discard", mask[i])[act[i]] for i in range(N)])
logp2 = np.log(np.maximum(before2, 1e-12)).astype(np.float32)
net.step(
    {"X": X, "mask": mask, "action": act, "logp": logp2, "kind": kind_pg,
     "adv": np.full(N, -2.0, dtype=np.float32), "ret": ret, "mode": "pg"},
    clip=0.2, ent_coef=0.0, vf_coef=0.0,
)
after2 = np.asarray([net.policy(X[i], "discard", mask[i])[act[i]] for i in range(N)])
check(float(after2.mean()) < float(before2.mean()),
      f"负优势应当压低被选中动作的概率（{before2.mean():.4f} -> {after2.mean():.4f}）")
print(f"  ✅ 负优势：{before2.mean():.4f} -> {after2.mean():.4f}")
# 已经「比信任域还激进」时不再给梯度：
# ratio = exp(logp - logp_old) = e^3 > 1+eps 且优势为正 -> min 取截断项 -> 梯度为 0。
# 注意要直接看梯度，不能看参数变化 —— Adam 带动量，梯度为 0 时参数还会靠惯性走一点。
before3 = np.asarray([net.policy(X[i], "discard", mask[i])[act[i]] for i in range(N)])
captured: dict = {}
orig_apply = net._apply


def cap(grads, **kw):
    captured.update({k: v.copy() for k, v in grads.items()})


net._apply = cap  # type: ignore[assignment]
net.step(
    {"X": X, "mask": mask, "action": act,
     "logp": (np.log(np.maximum(before3, 1e-12)) - 3.0).astype(np.float32),
     "kind": kind_pg, "adv": np.full(N, 5.0, dtype=np.float32), "ret": ret, "mode": "pg"},
    clip=0.2, ent_coef=0.0, vf_coef=0.0,
)
net._apply = orig_apply  # type: ignore[assignment]
gmax = float(np.abs(captured["Wp_discard"]).max())
check(gmax < 1e-9, f"截断生效时策略头梯度应当为 0，得到 max|g| = {gmax:.3e}")
# 反过来：ratio 太低且优势为正时，min 取的是未截断项，梯度不为 0（该学还是要学）
captured.clear()
net._apply = cap  # type: ignore[assignment]
net.step(
    {"X": X, "mask": mask, "action": act,
     "logp": (np.log(np.maximum(before3, 1e-12)) + 3.0).astype(np.float32),
     "kind": kind_pg, "adv": np.full(N, 5.0, dtype=np.float32), "ret": ret, "mode": "pg"},
    clip=0.2, ent_coef=0.0, vf_coef=0.0,
)
net._apply = orig_apply  # type: ignore[assignment]
gmax2 = float(np.abs(captured["Wp_discard"]).max())
check(gmax2 > 1e-9, "ratio 低于下界且优势为正时应当照常给梯度（PPO 的 min 取未截断项）")
print(f"  ✅ 截断生效：超出上界时 max|g| = {gmax:.1e}；低于下界时 max|g| = {gmax2:.2e}")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
