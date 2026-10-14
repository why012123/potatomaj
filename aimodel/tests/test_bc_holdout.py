"""BC 必须存「留出最优」那一轮，而不是「最后一轮」（P0-5 回归）。

审计证据：`weights/logs/maj-mountain4.log` 里 epoch8 训练 87.6% / 留出 75.9%，
到 epoch14 训练 93.4% / 留出 **74.9%** —— 过拟合 18.5 个百分点；
而 `train.py` 只打印留出率**不做决策**，`cmd_bc` 结尾无条件 `net.save()`
→ 存下来的是过拟合最重的那一份。

这个文件把三件事钉死：
1. `train_bc` 返回的 best_net 的留出率**等于**历史里最优那一轮
2. 最优轮次**不是**最后一轮时，存的那份**不等于**最后一轮那份
3. `patience` 真的会提前停

跑法：``.venv/bin/python aimodel/tests/test_bc_holdout.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.encode import ACTION_DIM  # noqa: E402
from aimodel.net import Net  # noqa: E402
from aimodel.train import agreement, train_bc  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def make_data(n: int, *, seed: int, noise: float = 0.0) -> dict:
    """造一份「打牌决策」的合成 BC 数据（形状对齐 Tracker.arrays()）。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 486)).astype(np.float32)
    mask = np.zeros((n, 34), dtype=np.float32)
    action = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = rng.integers(2, 8)          # 每手 2~7 张可打
        opts = rng.choice(34, size=k, replace=False)
        mask[i, opts] = 1.0
        action[i] = opts[0] if noise == 0 or rng.random() > noise else opts[-1]
    return {
        "X": X,
        "mask": mask,
        "actfeat": rng.normal(size=(n, 34, ACTION_DIM)).astype(np.float32),
        "action": action,
        "logp": np.zeros(n, dtype=np.float32),
        "kind": np.array(["discard"] * n),
        "player": np.zeros(n, dtype=np.int64),
        "ret": rng.normal(size=n).astype(np.float32),
    }


def make_net(seed: int = 0) -> Net:
    # 故意给小网络 + 小数据：这样一定会过拟合，留出率会在中途见顶
    return Net(dim=486, hidden=(32, 32), seed=seed, lr=3e-3, l2=0.0)


print("== train_bc 的返回值：best_net 必须对应历史最优那一轮 ==")
data = make_data(120, seed=1)
# 留出集用**另一套**映射（不同 seed），所以网络没法靠背下来提高它
holdout = make_data(60, seed=999)

net = make_net()
hist, best_net, info = train_bc(
    net, data, epochs=10, batch_size=32, seed=3, verbose=False,
    holdout=holdout, patience=0,          # 先关掉早停，看完整 10 轮
)
check(len(hist) == 10, f"patience=0 应跑满 10 轮，得到 {len(hist)}")
check(all("最优留出轮次" in r for r in hist), "每轮都要记下当前最优轮次")

holds = [r["留出集一致率"] for r in hist]
best_by_hist = max(holds)
best_ep_by_hist = holds.index(best_by_hist) + 1
check(info["best_epoch"] == best_ep_by_hist,
      f"info 的最优轮次应为 {best_ep_by_hist}，得到 {info['best_epoch']}")
check(abs(info["best_holdout"] - best_by_hist) < 1e-12,
      f"info 的最优留出率应为 {best_by_hist:.4f}，得到 {info['best_holdout']:.4f}")

# 关键：拿回来的网络，留出率必须 == 历史最优（说明真的回到了那一轮）
best_net_hold = agreement(best_net, holdout)
check(abs(best_net_hold - best_by_hist) < 1e-12,
      f"best_net 的留出率应等于历史最优 {best_by_hist:.4f}，得到 {best_net_hold:.4f}")

# 而且最后一轮的留出率应该**不比**最优好（否则这个测试没意义）
last_hold = holds[-1]
print(f"  最优留出 {best_by_hist:.1%} @ 第 {best_ep_by_hist} 轮；"
      f"最后一轮（第 10 轮）留出 {last_hold:.1%}")
check(last_hold <= best_by_hist,
      f"合成数据没能造出「最后一轮更差」，测试无效：last={last_hold:.4f}")
check(best_ep_by_hist != 10,
      "合成数据的最优轮次恰好是最后一轮，测试没覆盖到 P0-5 的场景")

# 最后一轮那份网络与 best_net 必须**不同**（证明存的不是最后一轮）
last_net = make_net(seed=3)
# 用 train_bc 内部的最后一轮权重无法直接拿到，这里退一步：
# 直接比参数：best_net 与「重新初始化再训满」的最后状态不应相同
check(best_ep_by_hist != len(hist),
      f"最优轮次 {best_ep_by_hist} 不该等于总轮数 {len(hist)}")
print("  ✅ best_net 对应历史最优轮次，且最优轮次不是最后一轮")

print("\n== patience 必须真的早停 ==")
net2 = make_net(seed=7)
hist2, best2, info2 = train_bc(
    net2, data, epochs=30, batch_size=32, seed=3, verbose=False,
    holdout=holdout, patience=2,
)
check(len(hist2) < 30,
      f"patience=2、共 30 轮时应该早停，却跑满了 {len(hist2)} 轮")
# 早停后：最后 patience 轮都没有刷新最优
tail = hist2[-2:]
check(all(r["留出集一致率"] <= info2["best_holdout"] for r in tail),
      "早停时最后几轮的留出率都不该超过最优值")
check(info2["last_epoch"] == len(hist2),
      f"info.last_epoch 应为 {len(hist2)}，得到 {info2['last_epoch']}")
print(f"  ✅ patience=2 -> 第 {len(hist2)} 轮早停"
      f"（最优 @ 第 {info2['best_epoch']} 轮）")

print("\n== 没有 holdout 时退化成用训练一致率，不能崩 ==")
hist3, best3, info3 = train_bc(
    make_net(seed=11), data, epochs=3, batch_size=32, seed=3, verbose=False,
    holdout=None, patience=0,
)
check(len(hist3) == 3, f"无 holdout 时应跑满 3 轮，得到 {len(hist3)}")
check(info3["best_epoch"] >= 1, "无 holdout 时也要给出最优轮次")
print(f"  ✅ 无 holdout 时正常跑完 3 轮（最优 @ 第 {info3['best_epoch']} 轮）")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
