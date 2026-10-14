"""探索坍缩的修复（P1-1 回归）。

审计证据：打牌头熵 **H=0.291**（上限 ln34≈3.53）、**argmax 与采样 87.7% 重合**、
`train.py` 的 ent_coef 从 0.01 降到 0.002（梯度贡献 <1%）。
根因之一：三个座位用**同一个确定性教师**，BC 是无标签平滑的硬交叉熵
-> 学出来是「一个确定性策略的三份镜像」。

本文件钉住四件事：
1. `net.step` 必须**逐头**报熵（老代码把 8 个头平均成一个数，看不出哪个塌了）
2. 分头自适应熵系数：熵低于目标要**调大**系数，高于目标要调小
3. 软标签（label smoothing）的梯度必须与解析式一致（数值梯度校验）
4. 教师 tie-break 随机化：默认**关**（保持确定性），开了才随机

跑法：``.venv/bin/python aimodel/tests/test_exploration.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import STYLES, HeuristicAgent  # noqa: E402
from aimodel.encode import ACTION_DIM  # noqa: E402
from aimodel.engine import Decision, View  # noqa: E402
from aimodel.net import KINDS, Net, softmax_rows  # noqa: E402
from aimodel.train import ENT_TARGETS, print_entropy_by_kind  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


RNG = np.random.default_rng(0)
N, DIM = 48, 48


def make_batch(kinds_mix=("discard", "call")):
    """造一个混了多种决策头的 batch。"""
    kinds = np.array([kinds_mix[i % len(kinds_mix)] for i in range(N)])
    X = RNG.normal(size=(N, DIM)).astype(np.float32)
    mask = np.zeros((N, 34), dtype=np.float32)
    act = np.zeros(N, dtype=np.int64)
    for i, k in enumerate(kinds):
        size = {"discard": 34, "call": 3}[k]
        m = RNG.integers(2, size + 1)
        opts = RNG.choice(size, size=m, replace=False)
        mask[i, opts] = 1.0
        act[i] = opts[0]
    return {
        "X": X, "mask": mask, "action": act,
        "logp": np.zeros(N, dtype=np.float32), "kind": kinds,
        "adv": np.ones(N, dtype=np.float32),
        "ret": RNG.normal(size=N).astype(np.float32),
        "mode": "pg",
        "actfeat": RNG.normal(size=(N, 34, ACTION_DIM)).astype(np.float32),
    }


print("== 逐头熵：必须分开报，不能只给一个平均数（P1-1）==")
net = Net(dim=DIM, hidden=(16, 16), seed=1, lr=1e-3, l2=0.0)
st = net.step(make_batch(("discard", "call")), clip=0.2, ent_coef=0.0, vf_coef=0.0)
check("entropy" in st, "整体熵要保留（向后兼容）")
check("entropy_by_kind" in st, "★ 必须提供逐头熵 entropy_by_kind，否则看不出哪个头塌了")
by = st.get("entropy_by_kind", {})
check(set(by) == {"discard", "call"},
      f"batch 里只有 discard/call，逐头熵就该只有这两个，得到 {sorted(by)}")
check(0.0 < by.get("discard", -1) < np.log(34) + 1e-6,
      f"打牌头熵应落在 (0, ln34]，得到 {by.get('discard')}")
check(all(v >= -1e-12 for v in by.values()), "熵不该为负")
# 4 个动作的头熵上限是 ln4，2 个动作的上限是 ln2 —— 用「合法动作数」验证量级合理
print(f"  逐头熵: {({k: round(v, 3) for k, v in by.items()})}")
print(f"  ✅ 逐头熵分开报（discard={by.get('discard'):.3f}, call={by.get('call'):.3f}）")

print("\n== 逐头打印函数不能崩，且要点出最不足的头 ==")
row = {"entropy_by_kind": {"discard": 0.10, "call": 0.50},
       "ent_coef_by_kind": {"discard": 0.01, "call": 0.01}}
import io  # noqa: E402
import contextlib  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print_entropy_by_kind(row)
out = buf.getvalue()
check("discard" in out and "call" in out, f"逐头打印要列出每个头，实际：{out!r}")
check("0.10/1.20" in out, f"要打印「实际/目标」，实际：{out!r}")
check("探索最不足" in out, f"熵远低于目标时要给出警告，实际：{out!r}")
# 空数据不能崩
with contextlib.redirect_stdout(io.StringIO()):
    print_entropy_by_kind({})
print("  ✅ 逐头打印 + 探索不足告警 + 空数据不崩")

print("\n== 分头自适应熵系数（P1-1）==")
# 学习率设 0，这样只观察系数怎么动，不受权重更新干扰
# 注意：目标熵必须 > 0，否则按设计**不会**设置系数（下面单独断这条）
net2 = Net(dim=DIM, hidden=(16, 16), seed=2, lr=0.0, l2=0.0)
targets = {"discard": 99.0, "call": 1e-3}  # discard 目标极高 -> 必调大；call 极低 -> 必调小
batch = make_batch(("discard", "call"))
net2.step(batch, clip=0.2, ent_coef=0.01, vf_coef=0.0,
          ent_target=targets, ent_coef_adapt=True)
after = dict(net2._ent_coef)
check(after, "开启自适应后 _ent_coef 应当有值")
check(after.get("discard", 0.0) > 0.01,
      f"discard 目标熵极高 -> 系数应被**调大**，得到 {after.get('discard')}")
check(after.get("call", 1.0) < 0.01,
      f"call 目标熵极低 -> 系数应被**调小**，得到 {after.get('call')}")
# 目标熵为 0 的头不该被自适应（否则会除零）
net2b = Net(dim=DIM, hidden=(16, 16), seed=20, lr=0.0, l2=0.0)
net2b.step(batch, clip=0.2, ent_coef=0.01, vf_coef=0.0,
           ent_target={"discard": 99.0, "call": 0.0}, ent_coef_adapt=True)
check("call" not in net2b._ent_coef,
      f"目标熵为 0 的头不该被自适应（除零风险），得到 {net2b._ent_coef}")
# 系数必须有上下界，不能指数爆炸
net3 = Net(dim=DIM, hidden=(16, 16), seed=3, lr=0.0, l2=0.0)
for _ in range(200):
    net3.step(batch, clip=0.2, ent_coef=0.01, vf_coef=0.0,
              ent_target=targets, ent_coef_adapt=True)
check(all(1e-4 <= v <= 1.0 for v in net3._ent_coef.values()),
      f"自适应系数必须夹在 [1e-4, 1.0]，得到 {net3._ent_coef}")
# 关闭自适应时系数不该被碰
net4 = Net(dim=DIM, hidden=(16, 16), seed=4, lr=0.0, l2=0.0)
net4.step(batch, clip=0.2, ent_coef=0.01, vf_coef=0.0, ent_target=targets,
          ent_coef_adapt=False)
check(not net4._ent_coef, f"关掉自适应时不该写 _ent_coef，得到 {net4._ent_coef}")
print(f"  ✅ 低于目标 -> 调大（discard {after.get('discard'):.4f}）；"
      f"高于目标 -> 调小（call {after.get('call'):.4f}）；有上下界")

print("\n== 目标熵必须分头设（二分类头不能跟打牌头用一个值）==")
check(set(ENT_TARGETS) <= set(KINDS),
      f"ENT_TARGETS 的键必须是真实决策头，多出：{set(ENT_TARGETS) - set(KINDS)}")
check(ENT_TARGETS["discard"] > ENT_TARGETS["tsumo"],
      "打牌头（34 动作）的目标熵必须远大于自摸头（2 动作）")
check(ENT_TARGETS["tsumo"] < 0.2,
      f"自摸头应当低熵（能和就和），得到 {ENT_TARGETS['tsumo']}")
check(all(v > 0 for v in ENT_TARGETS.values()), "目标熵必须为正")
print(f"  ✅ 打牌 {ENT_TARGETS['discard']} > 鸣牌 {ENT_TARGETS['call']} "
      f"> 自摸 {ENT_TARGETS['tsumo']}（分头设置生效）")

print("\n== 软标签的梯度必须等于解析式（数值梯度校验）==")
EPS = 0.05
net5 = Net(dim=DIM, hidden=(16, 16), seed=5, lr=0.0, l2=0.0)
for attr in ("W", "b"):
    setattr(net5, attr, [v.astype(np.float64) for v in getattr(net5, attr)])
for attr in ("Wp", "bp"):
    setattr(net5, attr, {k: v.astype(np.float64) for k, v in getattr(net5, attr).items()})
for attr in ("Wv", "bv", "Wa1", "ba1", "Wa2", "ba2"):
    setattr(net5, attr, getattr(net5, attr).astype(np.float64))

b5 = make_batch(("discard",))
b5["mode"] = "bc"          # ★ 软标签只在 BC 阶段生效
AF5 = b5["actfeat"].astype(np.float64)
X5 = b5["X"]
mask5 = b5["mask"]
act5 = b5["action"]


def smoothed_loss() -> float:
    acts, _ = net5._forward(X5)
    feat = acts[-1]
    z = feat @ net5.Wp["discard"] + net5.bp["discard"]
    a1 = np.maximum(AF5 @ net5.Wa1 + net5.ba1, 0.0)
    z = z + (a1 @ net5.Wa2 + net5.ba2)[..., 0]
    z = np.where(mask5 > 0, z, -1e9)
    p = softmax_rows(z)
    K = mask5.sum(axis=1)
    ce = -np.mean(np.log(np.maximum(p[np.arange(len(X5)), act5], 1e-12)))
    # 软标签损失 = (1-eps)*CE + eps*mean(log K)（第二项是常数）
    return float((1 - EPS) * ce + EPS * np.mean(np.log(K)))


got = {}
orig = net5._apply
net5._apply = lambda grads, **kw: got.update({k: v.copy() for k, v in grads.items()})
net5.step(dict(b5), clip=0.0, ent_coef=0.0, vf_coef=0.0, label_smoothing=EPS)
net5._apply = orig

Wp = net5.Wp["discard"]
ana = got["Wp_discard"]
worst = 0.0
for idx in [(0, 0), (3, 5), (7, 11)]:
    o = Wp[idx]
    Wp[idx] = o + 1e-3
    hi = smoothed_loss()
    Wp[idx] = o - 1e-3
    lo = smoothed_loss()
    Wp[idx] = o
    num = (hi - lo) / 2e-3
    rel = abs(num - ana[idx]) / max(1e-12, abs(num) + abs(ana[idx]))
    worst = max(worst, rel)
check(worst < 1e-4,
      f"软标签数值梯度与解析梯度不符（最大相对差 {worst:.2e}）"
      " —— 别用 -(eps/K)*coef*(1-K*p) 那种写法，它恒等于 0")
print(f"  ✅ 软标签梯度与解析式一致（最大相对差 {worst:.2e}）")

# 软标签 = 硬梯度乘 (1-eps)
net6 = Net(dim=DIM, hidden=(16, 16), seed=6, lr=0.0, l2=0.0)
g_hard = {}
net6._apply = lambda grads, **kw: g_hard.update({k: v.copy() for k, v in grads.items()})
net6.step(dict(b5), clip=0.0, ent_coef=0.0, vf_coef=0.0, label_smoothing=0.0)
net7 = Net(dim=DIM, hidden=(16, 16), seed=6, lr=0.0, l2=0.0)
g_smooth = {}
net7._apply = lambda grads, **kw: g_smooth.update({k: v.copy() for k, v in grads.items()})
net7.step(dict(b5), clip=0.0, ent_coef=0.0, vf_coef=0.0, label_smoothing=EPS)
ratio = np.abs(g_smooth["Wp_discard"]).sum() / max(1e-12, np.abs(g_hard["Wp_discard"]).sum())
check(abs(ratio - (1 - EPS)) < 1e-6,
      f"软标签梯度应是硬梯度的 (1-eps)={1-EPS}，实测比值 {ratio:.6f}")
print(f"  ✅ 软标签梯度 = 硬梯度 × (1-eps)（实测比值 {ratio:.6f}）")

print("\n== 教师 tie-break 随机化：默认必须关（保护别人的对拍基线）==")


def tie_view() -> View:
    """造一个「两张牌打出去向听/进张完全一样」的局面。"""
    c = list(S.parse_hand("1122334455667m"))
    return View(
        player=0, hand=c, drawn=None, melds=[], hand_red=[0] * 34,
        discards=[[] for _ in range(3)], melds_all=[[] for _ in range(3)],
        kita=[0] * 3, scores=[25000] * 3, dora_indicators=[0],
        round_wind=S.EAST, seat_wind=S.EAST, wall_left=60, turn=4,
        riichi=[False] * 3, ippatsu=[False] * 3, is_dealer=True,
        last_discard=None, rule=S.DEFAULT_RULE,
    )


v = tie_view()
opts = [t for t in range(34) if v.hand[t] > 0]
d = Decision("discard", list(opts))

# 默认：同一 seed 反复调，结果必须完全一致（确定性）
det = HeuristicAgent(7)
picks_default = {det._discard(d, v) for _ in range(30)}
check(len(picks_default) == 1,
      f"默认（tie_break_eps=0）教师必须是确定性的，却得到 {picks_default}")

# 打开 eps：同一 seed 也要可复现，但**多次不同 seed** 应当出现不同选择
eps_agent = HeuristicAgent(7, tie_break_eps=1e9)   # 1e9 -> 所有候选都算「等价」
rep = {HeuristicAgent(7, tie_break_eps=1e9)._discard(d, v) for _ in range(1)}
again = {HeuristicAgent(7, tie_break_eps=1e9)._discard(d, v) for _ in range(1)}
check(rep == again, "同一个 seed + 同样的调用顺序，结果必须可复现")
varied = set()
for s in range(40):
    ag = HeuristicAgent(s, tie_break_eps=1e9)
    varied.add(ag._discard(d, v))
check(len(varied) > 1,
      f"打开 tie-break 之后应当出现多种选择（打破确定性），只得到 {varied}")
check(hasattr(HeuristicAgent(0), "rng"), "教师应当持有 rng")
print(f"  ✅ 默认确定性（{len(picks_default)} 种）；开启后 40 个种子出 "
      f"{len(varied)} 种选择（{sorted(varied)[:6]}…）")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
