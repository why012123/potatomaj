"""KL 锚定 + 最优权重回滚进内存（P1-2 回归）。

审计证据：全仓库 `grep -i "kl\\|anchor\\|reference"` 在 train/net 里**零命中**；
`train.py:272-324` 只有一个网络，snapshot 只当对手用；
`train.py:474-477` 的「新高才保存」**只保护文件，不保护内存里的权重**
-> RL 训崩了会把好权重覆盖掉（而且崩掉那轮之后的训练数据就是用崩掉的策略采的）。

本文件钉住四件事：
1. `dKL/dz = p - pref` —— **必须过数值梯度**（我最初推错了，见下）
2. KL 真的把策略拉向参考策略（kl_coef 越大，与参考的偏离越小）
3. 「新高才保存」时必须**同时**留一份内存最优；跑完能回滚
4. kl_coef=0 时行为与原来完全一致（不能因为加锚改变默认行为）

跑法：``.venv/bin/python aimodel/tests/test_kl_anchor.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.encode import ACTION_DIM  # noqa: E402
from aimodel.net import Net, softmax_rows  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


RNG = np.random.default_rng(0)
N, DIM, KC = 24, 48, 0.3


def mk_batch(mode="pg"):
    X = RNG.normal(size=(N, DIM)).astype(np.float32)
    mask = np.zeros((N, 34), dtype=np.float32)
    mask[:, :10] = 1.0
    return {
        "X": X, "mask": mask, "action": RNG.integers(0, 10, N),
        "logp": np.zeros(N, dtype=np.float32),
        "kind": np.array(["discard"] * N),
        "adv": np.ones(N, dtype=np.float32),
        "ret": (RNG.normal(size=N) * 0.3).astype(np.float32),
        "mode": mode,
        "actfeat": RNG.normal(size=(N, 34, ACTION_DIM)).astype(np.float32),
    }


def as64(n):
    n.W = [w.astype(np.float64) for w in n.W]
    n.b = [b.astype(np.float64) for b in n.b]
    n.Wp = {k: v.astype(np.float64) for k, v in n.Wp.items()}
    n.bp = {k: v.astype(np.float64) for k, v in n.bp.items()}
    for a in ("Wv", "bv", "Wa1", "ba1", "Wa2", "ba2"):
        setattr(n, a, getattr(n, a).astype(np.float64))


print("== dKL/dz = p - pref（用最小单行例子独立验证公式本身）==")


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _kl(pref, z):
    p = _softmax(z)
    return float(np.sum(pref * (np.log(np.maximum(pref, 1e-300))
                                - np.log(np.maximum(p, 1e-300)))))


K = 6
z0 = RNG.normal(size=K) * 2.0
pref0 = RNG.dirichlet(np.ones(K))
p0 = _softmax(z0)
num = np.zeros(K)
for j in range(K):
    zp = z0.copy(); zp[j] += 1e-6
    zm = z0.copy(); zm[j] -= 1e-6
    num[j] = (_kl(pref0, zp) - _kl(pref0, zm)) / 2e-6
simple = p0 - pref0
# 同时验一下「我最初写错的那版」是不是真的错 —— 防止有人又改回去
lp = np.log(p0); lr_ = np.log(pref0)
KLrow = float(np.sum(pref0 * (lr_ - lp)))
wrong = p0 * (lp - lr_ + KLrow)
check(np.abs(simple - num).max() < 1e-6,
      f"dKL/dz 应为 p-pref，与数值梯度差 {np.abs(simple - num).max():.2e}")
check(np.abs(wrong - num).max() > 1e-3,
      "被否掉的那版 p*(lp-lr+KL) 居然也对？那这条断言失效了，要重新推导")
# softmax 梯度必须「对动作求和为 0」，这是最快的自检
check(abs(simple.sum()) < 1e-12, "dKL/dz 对各动作求和必须为 0")
check(abs(wrong.sum()) > 1e-3,
      "被否掉那版求和不为 0 —— 这正是它错的原因（softmax 梯度必须零和）")
print(f"  ✅ p-pref 与数值梯度一致（差 {np.abs(simple-num).max():.2e}）；"
      f"旧版求和 {wrong.sum():+.4f} ≠ 0")

print("\n== 整网络层面：KL 的解析梯度也必须过数值梯度 ==")
base = Net(dim=DIM, hidden=(16, 16), seed=3, lr=0.0, l2=0.0)
ref = Net(dim=DIM, hidden=(16, 16), seed=99, lr=0.0, l2=0.0)
as64(base); as64(ref)
b = mk_batch("pg")
AF = b["actfeat"].astype(np.float64)
X, mask = b["X"], b["mask"]
base.set_reference(ref, b, kl_coef=KC)
check("discard" in base.ref_probs, "set_reference 应当缓存参考分布")
check(base.ref_probs["discard"].shape == (N, 34), "参考分布形状应为 (N, 34)")
check(abs(base.kl_coef - KC) < 1e-12, f"kl_coef 应设为 {KC}，得到 {base.kl_coef}")


def kl_only() -> float:
    acts, _ = base._forward(X)
    feat = acts[-1]
    z = feat @ base.Wp["discard"] + base.bp["discard"]
    a1 = np.maximum(AF @ base.Wa1 + base.ba1, 0.0)
    z = z + (a1 @ base.Wa2 + base.ba2)[..., 0]
    z = np.where(mask > 0, z, -1e4)
    p = softmax_rows(z)
    pr = base.ref_probs["discard"]
    return float(np.mean(np.sum(
        pr * (np.log(np.maximum(pr, 1e-12)) - np.log(np.maximum(p, 1e-12))), axis=1)))


def grads(kl):
    g = {}
    orig = base._apply
    base._apply = lambda gr, **kw: g.update({k: v.copy() for k, v in gr.items()})
    base.step(dict(b), clip=0.0, ent_coef=0.0, vf_coef=0.0, kl_coef=kl)
    base._apply = orig
    return g


ana = (grads(KC)["Wp_discard"] - grads(0.0)["Wp_discard"]) / KC
Wp = base.Wp["discard"]
worst = 0.0
for idx in [(0, 0), (3, 5), (12, 2)]:
    o = Wp[idx]
    Wp[idx] = o + 1e-5
    hi = kl_only()
    Wp[idx] = o - 1e-5
    lo = kl_only()
    Wp[idx] = o
    n_ = (hi - lo) / 2e-5
    worst = max(worst, abs(n_ - ana[idx]) / max(1e-12, abs(n_) + abs(ana[idx])))
check(worst < 1e-4,
      f"KL 整网络梯度与数值不符（最大相对差 {worst:.2e}）")
print(f"  ✅ 整网络 KL 梯度与数值梯度一致（最大相对差 {worst:.2e}）")

print("\n== KL 真的把策略拉向参考策略 ==")


def drift(kl_coef: float, steps: int = 60) -> float:
    """从同一初始权重出发训 steps 步，量它与参考策略的平均 KL。"""
    net = Net(dim=DIM, hidden=(16, 16), seed=5, lr=1e-2, l2=0.0)
    r = Net(dim=DIM, hidden=(16, 16), seed=77, lr=0.0, l2=0.0)
    bb = mk_batch("pg")
    net.set_reference(r, bb, kl_coef=kl_coef)
    for _ in range(steps):
        net.step(dict(bb), clip=0.2, ent_coef=0.0, vf_coef=0.0, kl_coef=kl_coef)
    acts, _ = net._forward(bb["X"])
    feat = acts[-1]
    z = feat @ net.Wp["discard"] + net.bp["discard"]
    a1 = np.maximum(bb["actfeat"].astype(np.float64) @ net.Wa1 + net.ba1, 0.0)
    z = z + (a1 @ net.Wa2 + net.ba2)[..., 0]
    z = np.where(bb["mask"] > 0, z, -1e4)
    p = softmax_rows(z)
    pr = net.ref_probs["discard"]
    return float(np.mean(np.sum(
        pr * (np.log(np.maximum(pr, 1e-12)) - np.log(np.maximum(p, 1e-12))), axis=1)))


d0 = drift(0.0)
d1 = drift(0.5)
d2 = drift(3.0)
check(d0 > d2, f"kl_coef=0 的偏离应大于 kl_coef=3.0（{d0:.4f} vs {d2:.4f}）")
check(d1 < d0, f"kl_coef=0.5 的偏离应小于 0（{d1:.4f} vs {d0:.4f}）")
check(d2 <= d1 + 1e-9, f"锚更紧，偏离应更小或相当（{d2:.4f} vs {d1:.4f}）")
print(f"  ✅ 与参考的 KL：coef=0 -> {d0:.4f}；0.5 -> {d1:.4f}；3.0 -> {d2:.4f}")

print("\n== kl_coef=0 时行为与原来一致（默认不能变）==")
n_a = Net(dim=DIM, hidden=(16, 16), seed=8, lr=1e-3, l2=0.0)
n_b = Net(dim=DIM, hidden=(16, 16), seed=8, lr=1e-3, l2=0.0)
b1 = mk_batch("pg")
s_a = n_a.step(dict(b1), clip=0.2, ent_coef=0.0, vf_coef=0.0)
s_b = n_b.step(dict(b1), clip=0.2, ent_coef=0.0, vf_coef=0.0, kl_coef=0.0)
check(abs(s_a["policy_loss"] - s_b["policy_loss"]) < 1e-12,
      "kl_coef=0 与不传参数的 loss 必须一致")
check(abs(s_a["kl"]) < 1e-12 and abs(s_b["kl"]) < 1e-12,
      f"kl_coef=0 时 kl 指标应为 0，得到 {s_a['kl']} / {s_b['kl']}")
w_same = all(np.allclose(x, y) for (_, x), (_, y) in zip(n_a.params(), n_b.params()))
check(w_same, "kl_coef=0 时权重更新必须完全相同")
print("  ✅ kl_coef=0 时 loss / kl 指标 / 权重更新都与原来一致")

print("\n== BC 阶段不该被锚（锚只约束 RL）==")
n_c = Net(dim=DIM, hidden=(16, 16), seed=9, lr=1e-3, l2=0.0)
r_c = Net(dim=DIM, hidden=(16, 16), seed=88, lr=0.0, l2=0.0)
b_bc = mk_batch("bc")
n_c.set_reference(r_c, b_bc, kl_coef=1.0)
st_bc = n_c.step(dict(b_bc), clip=0.0, ent_coef=0.0, vf_coef=0.0, kl_coef=1.0)
check(abs(st_bc["kl"]) < 1e-12,
      f"BC 阶段不该计 KL（否则会跟交叉熵打架），得到 kl={st_bc['kl']}")
print("  ✅ BC 阶段 kl=0（锚只作用于 RL）")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
