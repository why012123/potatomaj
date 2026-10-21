"""`chisel`（吃成什么样）头：加载 / 零初始化 / 梯度 / 推理 / 端到端。

跑法：``.venv/bin/python aimodel/tests/test_chisel.py``

四条硬要求（对应交接单）：
  a) 老权重 ``maj-pro4v1.1_mac_rl.npz`` 能加载，``use_chisel=True`` 不崩（走兜底）
  b) 新头零初始化 -> chisel 的 argmax = 0
  c) 造一个 chipat 标签已知的小数据集，训几步后准确率 > 随机（1/3）
  d) 带 chisel 的 MLAgent 能跑完一整局四麻

另外钉住三件容易错的事：
  · 合法吃法顺序必须与数据侧 ``tenhou_bc._legal_chi_pairs`` **逐项相同**
  · chisel 的梯度**不许**回传躯干（否则会动到已训好的主策略）
  · 只有拿到逐吃法选项（``chi:a:b`` / ``Decision.pairs``）才记 chipat；
    stock engine 的二值选项下 chipat 必须保持 -1（不能把「心想的」当实际标签）
"""

from __future__ import annotations

import random
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import MLAgent, Tracker, legal_chi_pairs  # noqa: E402
from aimodel.encode import HEAD_SIZE  # noqa: E402
from aimodel.engine import Decision, Game  # noqa: E402
from aimodel.net import CHISEL_SIZE, KINDS, Net, softmax_rows  # noqa: E402
from aimodel.tests.test_chi_4p import chi_view  # noqa: E402
from aimodel.train import chisel_metrics  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


R4 = S.rule_for(4)
OLD_WEIGHTS = ROOT / "aimodel" / "weights" / "maj-pro4v1.1_mac_rl.npz"


def chi_decision(hand_str: str, tile: int, per_pair: bool, player: int = 1):
    """造一个吃牌决策；``per_pair=True`` 时给出逐吃法选项（WIP 格式）✓"""
    v = chi_view(hand_str, player=player)
    pairs = legal_chi_pairs(v.hand, tile)
    if not pairs:
        raise AssertionError(f"这手牌不能吃 {S.tile_cn(tile)}：{hand_str}")
    opts = ["pass", "chi"] if not per_pair else ["pass"] + [f"chi:{a}:{b}" for a, b in pairs]
    return Decision("chi", opts, tile=tile), v, pairs


# ---------------------------------------------------------------------------
# ① 合法吃法顺序：运行时实现 == 数据侧实现
# ---------------------------------------------------------------------------
print("== ① 合法吃法顺序（运行时 vs 数据侧 tenhou_bc）==")
# 数据侧模块会 import logcheck -> agents；这里只 import 一次，不参与生产路径 ✓
from aimodel.tenhou_bc import _legal_chi_pairs as _data_pairs  # noqa: E402

rng = random.Random(0)
mismatch = []
for tile in range(34):
    for _ in range(120):
        hand = [0] * 34
        for t in range(34):
            hand[t] = rng.choice([0, 0, 0, 1, 1, 2, 3, 4])
        if legal_chi_pairs(hand, tile) != _data_pairs(hand, tile):
            mismatch.append((tile, list(hand)))
check(not mismatch, f"合法吃法顺序与数据侧不一致 {len(mismatch)} 例：{mismatch[:2]}")
check(legal_chi_pairs([0] * 34, 27) == [], "字牌不能吃")
# 引擎的实现（engine._chi_pairs 按 start 升序）也必须同序 —— 不改引擎 ✓
check(legal_chi_pairs(S.parse_hand("223456m11p11s11z"), 2) == [(1, 3), (3, 4)],
      f"吃 3m 的顺序应为 [(1,3),(3,4)]，得到 "
      f"{legal_chi_pairs(S.parse_hand('223456m11p11s11z'), 2)}")
print(f"  ✅ 34 张牌 × 120 手随机牌逐项对拍（{34 * 120} 例）")

# ---------------------------------------------------------------------------
# ② a) 老权重：能加载 + use_chisel=True 不崩 + 走兜底
# ---------------------------------------------------------------------------
print("\n== ② a) 老权重（没有 chisel）必须能加载、不崩、走兜底 ==")
check(OLD_WEIGHTS.exists(), f"测试前提：老权重不存在 {OLD_WEIGHTS}")
if OLD_WEIGHTS.exists():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = Net.load(OLD_WEIGHTS)
    check(old.dim == 560 and old.hidden == (512, 512), f"老权重结构不对：{old.dim}/{old.hidden}")
    check("chi" in old.kinds, "老权重本身带 chi 头（本测试要测的正是缺 chisel）")
    check(old.chisel_trained is False, "老权重没有 chisel -> chisel_trained 必须 False")
    check(old.chisel_updates == 0, f"chisel_updates 应为 0，得到 {old.chisel_updates}")
    check(float(np.abs(old.Wc).max()) == 0.0, "老权重缺 chisel -> Wc 必须零初始化")
    check(float(np.abs(old.bc).max()) == 0.0, "老权重缺 chisel -> bc 必须零初始化")
    # ⚠️ 不能报「存档里缺参数」那种「模型坏了」的警告（那会让 44 个存量权重全炸）
    check(not caught, f"加载老权重不应有警告，实际：{[str(w.message)[:80] for w in caught]}")

    d, v, pairs = chi_decision("223456m11p11s11z", 2, per_pair=False)
    ag = MLAgent(old, seed=0, use_chisel=True)
    try:
        got = ag.decide(d, v)
        crashed = None
    except Exception as exc:                       # noqa: BLE001
        got, crashed = None, f"{type(exc).__name__}: {exc}"
    check(crashed is None, f"老权重 use_chisel=True 崩了：{crashed}")
    check(got in (0, 1), f"老权重 chi 决策应返回 0/1，得到 {got!r}")
    check(ag._chisel_active() is False, "chisel_trained=False 时 _chisel_active 必须 False")
    check(ag.last_chisel is None, "老权重不应走到 chisel 分支")
    check("chi_pair_engine" not in ag.counts,
          f"老权重不该出现「chisel 选出 pair 但引擎不认」计数：{ag.counts}")
    print("  ✅ 老权重加载 0 警告；chisel_trained=False；chi 决策交给引擎启发式")

# ---------------------------------------------------------------------------
# ③ b) 新头零初始化 -> argmax = 0
# ---------------------------------------------------------------------------
print("\n== ③ b) 新头零初始化：chisel argmax 必须为 0 ==")
net = Net(dim=48, hidden=(16, 16), seed=0)
check(net.Wc.shape == (16, CHISEL_SIZE), f"Wc 形状应为 (16,3)，得到 {net.Wc.shape}")
check(net.bc.shape == (CHISEL_SIZE,), f"bc 形状应为 (3,)，得到 {net.bc.shape}")
check(float(np.abs(net.Wc).max()) == 0.0, "新头 Wc 必须零初始化（不能随机）")
check(float(np.abs(net.bc).max()) == 0.0, "新头 bc 必须零初始化（不能随机）")
check("chisel" not in KINDS and "chisel" not in HEAD_SIZE,
      "chisel 绝不许进 KINDS / HEAD_SIZE（它不是 Decision.kind）")
check(any(n == "Wc" for n, _ in net.params()) and any(n == "bc" for n, _ in net.params()),
      "params() 必须包含 Wc / bc（存档 / Adam / copy 都靠它）")
_x = np.random.default_rng(0).normal(size=(64, 48)).astype(np.float32)
_z = net.chisel_logits(net.trunk(_x))
check(_z.shape == (64, 3), f"chisel_logits 形状应为 (64,3)，得到 {_z.shape}")
check(set(np.unique(_z.argmax(axis=1))).issubset({0}), "零初始化时 argmax 必须恒为 0")
print(f"  ✅ 参数量 {net.n_params():,}；Wc/bc 零初始化；64 个局面的 argmax 全是 0")

# 存档往返：独立头必须跟着走；老权重那套 meta（没有 chisel 键）也必须能读
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "c.npz"
    net.Wc[...] = np.arange(16 * 3, dtype=np.float32).reshape(16, 3) / 100.0
    net.bc[...] = [0.1, 0.2, 0.3]
    net.chisel_trained = True
    net.chisel_updates = 5
    net.save(p)
    back = Net.load(p)
    check(np.allclose(back.Wc, net.Wc) and np.allclose(back.bc, net.bc),
          "chisel 权重必须存档往返一致")
    check(back.chisel_trained is True and back.chisel_updates == 5,
          f"chisel 训练状态必须往返：{back.chisel_trained}/{back.chisel_updates}")
    # 手工抹掉 Wc/bc + chisel 的 meta 键，模拟「加头之前的老权重」
    data = dict(np.load(p, allow_pickle=False))
    import json

    meta = json.loads(bytes(data["_meta"]).decode())
    for k in ("chisel", "chisel_trained", "chisel_updates"):
        meta.pop(k, None)
    data["_meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    for name in ("Wc", "bc"):
        del data[name]
    np.savez_compressed(p, **data)
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        oldish = Net.load(p)
    check(not caught2, f"缺 chisel 的权重不应报警告：{[str(w.message)[:60] for w in caught2]}")
    check(float(np.abs(oldish.Wc).max()) == 0.0 and oldish.chisel_trained is False,
          "缺 chisel 的权重必须零初始化 + chisel_trained=False")
print("  ✅ chisel 权重/训练状态随存档往返；缺列的老权重零初始化且不报警")

# ---------------------------------------------------------------------------
# ④ c) 梯度真的通：已知标签的小数据集，几轮后准确率 > 1/3
# ---------------------------------------------------------------------------
print("\n== ④ c) 已知 chipat 标签的小数据集：训几步后准确率 > 1/3 ==")
N = 300
rng_np = np.random.default_rng(7)
X = rng_np.normal(size=(N, 32)).astype(np.float32)
kind = np.array(["chi"] * N)
mask = np.zeros((N, 34), dtype=np.float32)
mask[:, :HEAD_SIZE["chi"]] = 1.0
action = rng_np.integers(0, HEAD_SIZE["chi"], size=N).astype(np.int64)
# 标签 = 「在**固定的随机躯干特征**上线性可分」的三分类 ✓
# 躯干特征在本次训练中不会变（adv=0 / vf_coef=0 / ent_coef=0 / l2=0），
# 所以这个标签对 chisel 头是**确定可学**的 ✓
probe = Net(dim=32, hidden=(24,), seed=0, lr=0.05, l2=0.0)
H = probe.trunk(X)
W_true = np.random.default_rng(11).normal(size=(24, 3)).astype(np.float32)
labels = (H @ W_true).argmax(axis=1).astype(np.int8)
batch = {
    "X": X,
    "mask": mask,
    "actfeat": np.zeros((N, 34, 13), dtype=np.float16),
    "action": action,
    "logp": np.zeros(N, dtype=np.float32),
    "kind": kind,
    "adv": np.zeros(N, dtype=np.float32),      # 主策略头不动
    "ret": np.zeros(N, dtype=np.float32),
    "mode": "bc",
    "chipat": labels,
}
data = {k: v for k, v in batch.items() if k != "adv"}
data["ret"] = batch["ret"]
m0 = chisel_metrics(probe, data)
check(m0 is not None and m0["n"] == N, f"chisel_metrics 应统计 {N} 行，得到 {m0}")
base_acc = m0["acc"]

# —— 数值梯度校验（手写反向最容易把符号 / 1/n 写错）——————————————
# 用一个**同初始化**的独立网络，把 _apply 换成「只抓梯度、不更新权重」✓
probe_grad = Net(dim=32, hidden=(24,), seed=0, l2=0.0)
probe_grad.Wc = probe_grad.Wc.astype(np.float64)
probe_grad.bc = probe_grad.bc.astype(np.float64)
captured: dict = {}
_orig_apply = probe_grad._apply


def _capture(grads, **_kw):
    captured.update({k: v.copy() for k, v in grads.items()})


probe_grad._apply = _capture  # type: ignore[assignment]
probe_grad.step(batch, clip=0.0, ent_coef=0.0, vf_coef=0.0, chisel_coef=1.0)
probe_grad._apply = _orig_apply  # type: ignore[assignment]


def _chisel_loss() -> float:
    zz = probe_grad.chisel_logits(probe_grad.trunk(X))
    pp = softmax_rows(zz)
    return float(-np.mean(np.log(np.maximum(pp[np.arange(N), labels], 1e-12))))


worst_rel = 0.0
for param, idx in ((probe_grad.Wc, (0, 0)), (probe_grad.Wc, (3, 1)),
                   (probe_grad.bc, (2,))):
    orig = float(param[idx])
    eps = 1e-6
    param[idx] = orig + eps
    hi = _chisel_loss()
    param[idx] = orig - eps
    lo = _chisel_loss()
    param[idx] = orig
    num = (hi - lo) / (2 * eps)
    ana = float(captured["Wc" if param is probe_grad.Wc else "bc"][idx])
    worst_rel = max(worst_rel, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
check(worst_rel < 1e-4,
      f"chisel 的解析梯度与数值梯度不符（最大相对误差 {worst_rel:.2e}）")
print(f"  ✅ 数值梯度校验通过（最大相对误差 {worst_rel:.1e}）")

trunk_before = [w.copy() for w in probe.W]
head_before = probe.Wp["chi"].copy()
for _ in range(160):
    probe.step(
        batch, clip=0.0, ent_coef=0.0, vf_coef=0.0, chisel_coef=1.0,
    )
m1 = chisel_metrics(probe, data)
check(base_acc <= 1.0, "基线准确率算出来了")
check(probe.chisel_trained is True, "训过 chisel 后 chisel_trained 必须 True")
check(probe.chisel_updates == 160, f"chisel_updates 应为 160，得到 {probe.chisel_updates}")
check(m1 is not None and m1["acc"] > 1 / 3,
      f"chisel 准确率必须 > 随机 1/3，得到 {m1['acc'] if m1 else None:.1%}")
check(m1["acc"] > m1["majority"],
      f"chisel 准确率 {m1['acc']:.1%} 必须高于多数类基线 {m1['majority']:.1%}")
# 隔离性：梯度**不许**回传躯干 / 主策略头
check(all(np.array_equal(a, b) for a, b in zip(trunk_before, probe.W)),
      "chisel 梯度不许改躯干（已训好的主策略必须一点不动）")
check(np.array_equal(head_before, probe.Wp["chi"]),
      "chisel 梯度不许改 chi 主策略头")
print(f"  ✅ {base_acc:.1%} -> {m1['acc']:.1%}（多数类 {m1['majority']:.1%}，"
      f"160 步；躯干/chi 头逐位未动）")

# chisel_coef=0 时一点都不能动（默认路径 / 金值测试的安全网）
probe2 = Net(dim=32, hidden=(24,), seed=0, lr=0.05, l2=0.0)
probe2.step(batch, clip=0.0, ent_coef=0.0, vf_coef=0.0, chisel_coef=0.0)
check(float(np.abs(probe2.Wc).max()) == 0.0 and probe2.chisel_trained is False,
      "chisel_coef=0 时 Wc 必须保持 0、chisel_trained=False")
check(probe2.step(batch, clip=0.0, ent_coef=0.0, vf_coef=0.0)["chisel_n"] == 0,
      "chisel_coef 默认 0 -> chisel_n 必须是 0（整段跳过）")
print("  ✅ chisel_coef=0（默认）整段跳过：Wc 保持 0、chisel_trained=False")

# ---------------------------------------------------------------------------
# ⑤ 推理：逐吃法选项 -> chisel argmax 就是真正返回的那一对
# ---------------------------------------------------------------------------
print("\n== ⑤ 推理：有逐吃法选项时 chisel 真的选 pair；二值选项时保持原路径 ==")
net4 = Net(dim=560, hidden=(16,), seed=0)
net4.chisel_trained = True
net4.bc[...] = np.array([0.0, 6.0, 0.0], dtype=np.float32)  # 强烈偏好 slot 1
d, v, pairs = chi_decision("223456m11p11s11z", 2, per_pair=True)
check(len(pairs) >= 2, f"测试前提：这手牌至少有 2 种吃法，得到 {pairs}")
tr = Tracker()
ag = MLAgent(net4, seed=0, tracker=tr)
choice = ag.decide(d, v)
check(choice == 1 + 1,
      f"chisel 偏好 slot 1 -> 应返回 options[2]={d.options[2]!r}，得到 {d.options[choice]!r}")
check(tr.chipats == [1], f"Tracker 的 chipat 应记为 1，得到 {tr.chipats}")
check(tr.actions == [1], f"Tracker 的 action 应仍是 chi 头下标 1，得到 {tr.actions}")
check(ag.last_chisel and ag.last_chisel["honored"] is True,
      f"last_chisel 应标 honored=True：{ag.last_chisel}")

# `Decision.pairs`（另一 agent 正在加的字段）也要认；用 setattr 模拟
d2, v2, pairs2 = chi_decision("223456m11p11s11z", 2, per_pair=False)
try:
    d2.pairs = list(pairs2)                   # 新字段还没合并也不影响本测试
except Exception:                             # pragma: no cover
    pass
# 只有 pairs、没有逐吃法选项串时，无法把「第几种」安全地映回 options ✗
# -> 必须走原路径（二值），不能猜 ✓
tr2 = Tracker()
ag2 = MLAgent(net4, seed=0, tracker=tr2)
c2 = ag2.decide(d2, v2)
check(c2 == 1, f"二值选项 + pairs 时仍应返回「吃」（引擎按启发式），得到 {c2}")
check(tr2.chipats == [-1], f"引擎没给逐吃法选项时 chipat 必须是 -1，得到 {tr2.chipats}")
check(ag2.last_chisel and ag2.last_chisel["honored"] is False,
      f"未命中逐吃法选项时 honored 必须 False：{ag2.last_chisel}")

# use_chisel=False：彻底走原路径，连 chisel 都不算
tr3 = Tracker()
ag3 = MLAgent(net4, seed=0, tracker=tr3, use_chisel=False)
c3 = ag3.decide(d, v)
check(c3 == 1, f"use_chisel=False 应返回第一个 chi 选项，得到 {c3}")
check(tr3.chipats == [-1] and ag3.last_chisel is None,
      "use_chisel=False 不许走 chisel 分支")
print("  ✅ 逐吃法 -> 真选 pair + chipat；二值 -> 原路径 + chipat=-1；开关可关")

# ---------------------------------------------------------------------------
# ⑥ d) 端到端：带 chisel 的 MLAgent 跑完一整局四麻
# ---------------------------------------------------------------------------
print("\n== ⑥ d) 端到端：四麻整局 ==")
net_game = Net(dim=560, hidden=(32,), seed=1, lr=1e-3)
net_game.chisel_trained = True
agents = [
    MLAgent(net_game, seed=i, greedy=True, use_chisel=True) for i in range(4)
]
try:
    res = Game(agents, seed=5, rule=R4).play()
    crashed = None
except Exception as exc:                       # noqa: BLE001
    res, crashed = None, f"{type(exc).__name__}: {exc}"
check(crashed is None, f"带 chisel 的整局四麻崩了：{crashed}")
if res is not None:
    check(len(res.hands) > 0, "整局必须至少打完一局")
    check(len(res.deltas) == 4 and len(res.ranks) == 4, "四麻必须有 4 家结算")
    # ⚠️ sum(deltas) **不一定为 0**：立直棒会从点数里扣走留在场上 ✓
    #    所以只断言「≤ 0 且是 1000 的整数倍」（不可能凭空多出点数）✓
    check(sum(res.deltas) <= 0 and sum(res.deltas) % 1000 == 0,
          f"点数差应 ≤0 且为 1000 的整数倍（立直棒留在场上），得到 {sum(res.deltas)}")
    total = sum(a.counts.get("chi", 0) for a in agents)
    print(f"  ✅ 打完 {len(res.hands)} 局 / 4 家；chi 决策合计 {total} 次；未崩溃")

# ---------------------------------------------------------------------------
if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    for f in FAILED:
        print("   -", f)
    raise SystemExit(1)
print("\n✅ test_chisel 全部通过")
