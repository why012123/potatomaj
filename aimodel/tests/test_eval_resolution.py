"""评测分辨率：同牌山注入 + 置信区间 + 「还要多少场」（P1-4 回归）。

审计证据（实测）：300 场配对检验的**单模型 95% 置信区间是 ±6.7~10.5 段位分**，
而要检测的效果只有 +3.65 —— 区间比效应还宽两倍。
要单侧 p<0.05、80% 功效需要 **2000~3000 场/模型**。

本文件钉住四件事：
1. `wall_for(seed)` 必须**确定性**（同 seed 同牌山）—— 不然「同牌山」是假的
2. 灌入牌山后 `Game` 真的用了它（不是偷偷重新随机）
3. `paired_stats` 的均值 / 标准误 / 置信区间 / 单侧 p 算对（用已知答案的合成数据）
4. **同牌山确实把配对差值的方差压下来** —— 这是 P1-4 的核心理由

跑法：``.venv/bin/python aimodel/tests/test_eval_resolution.py``
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.ab import Z95, matches_needed, paired_stats, wall_for  # noqa: E402
from aimodel.agents import RandomAgent  # noqa: E402
from aimodel.engine import Game  # noqa: E402
from aimodel.net import Net  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


R3 = S.DEFAULT_RULE

print("== 牌山生成必须确定性（同 seed 同牌山）==")
w1 = wall_for(12345, R3)
w2 = wall_for(12345, R3)
w3 = wall_for(12346, R3)
check(w1 == w2, "同一个 seed 必须给出完全相同的牌山")
check(w1 != w3, "不同 seed 应该给出不同的牌山（否则种子没生效）")
check(len(w1) == len(w2) and len(w1) > 0, "牌山不能为空")
check(all(isinstance(x, tuple) and len(x) == 2 for x in w1),
      "牌山元素应是 (牌型, 是否赤) 二元组")
print(f"  ✅ seed 12345 -> {len(w1)} 张，两次完全一致；12346 不同")

print("\n== 灌入的牌山必须真的被引擎使用 ==")


class FirstHand:
    """把第一局的起手记下来，用来比较两局是不是同一副牌。"""

    def __init__(self):
        self.tehais = None

    def decide(self, decision, view):
        if self.tehais is None and decision.kind == "discard":
            self.tehais = list(view.hand)
        return 0


def first_hand_of(wall, seed):
    spy = FirstHand()
    g = Game([spy, RandomAgent(1), RandomAgent(2)], seed=seed, rule=R3)
    g.rigged_wall = list(wall)
    g.play()
    return spy.tehais


wall = wall_for(999, R3)
# 同一个游戏种子、同一副牌山 -> 起手必须一样
a = first_hand_of(wall, 111)
b = first_hand_of(wall, 111)
check(a == b, "同牌山 + 同种子，起手必须相同")
# 换了牌山 -> 起手应当不同（否则说明 rigged_wall 没被消费）
c = first_hand_of(wall_for(1000, R3), 111)
check(a != c, "换牌山之后起手应当变化 —— 否则 rigged_wall 根本没生效")
print("  ✅ 同牌山起手一致；换牌山起手变化（rigged_wall 确实生效）")

print("\n== paired_stats 用已知答案核对 ==")
# 手算一组：差值 1,2,3,4,5 -> 均值 3；样本方差 = 2.5；se = sqrt(2.5/5)=0.7071
d = [1.0, 2.0, 3.0, 4.0, 5.0]
st = paired_stats(d)
check(abs(st["mean"] - 3.0) < 1e-12, f"均值应为 3.0，得到 {st['mean']}")
check(abs(st["se"] - math.sqrt(2.5 / 5)) < 1e-12,
      f"标准误应为 {math.sqrt(2.5/5):.6f}，得到 {st['se']:.6f}")
check(abs(st["lo"] - (3.0 - Z95 * st["se"])) < 1e-12, "置信区间下界算错")
check(abs(st["hi"] - (3.0 + Z95 * st["se"])) < 1e-12, "置信区间上界算错")
check(st["t"] > 0 and st["p_one_sided"] < 0.05,
      f"这组数据应当显著（t={st['t']:.2f}, p={st['p_one_sided']:.4f}）")
# 全 0 差值 -> 不该显著，也不能崩
st0 = paired_stats([0.0] * 10)
check(abs(st0["mean"]) < 1e-12, "全 0 差值均值应为 0")
check(st0["p_one_sided"] >= 0.5 or st0["se"] == 0.0, "全 0 差值不该显著")
# 空样本不能崩
ste = paired_stats([])
check(ste["n"] == 0, "空样本 n 应为 0")
print(f"  ✅ 均值/标准误/区间/p 手算一致；t={st['t']:.2f} p={st['p_one_sided']:.4f}")

print("\n== 「还要多少场」随标准误按 1/sqrt(n) 缩放 ==")
# 审计的实测口径：效应 +3.65，但 300 场时单模型 95% 区间是 ±6.7~10.5
# -> 配对 se 大约 3.4~5.4。取 se=4.0 / n=300 看要多少场。
need = matches_needed(3.65, 4.0, 300)
check(need is not None and need > 300,
      f"效应 3.65 / se=4.0 / n=300 应当需要**更多**场，得到 {need}")
check(need is not None and 2000 <= need <= 3000,
      f"按审计口径应当落在 2000~3000 场量级，得到 {need:.0f}")
# se 缩小一半 -> 需要 4 倍场次
need2 = matches_needed(3.65, 2.0, 300)
check(need2 is not None and abs(need2 - need / 4) < need * 0.01,
      f"se 减半应当只要 1/4 的场次：{need:.0f} -> {need2:.0f}")
check(matches_needed(0.0, 1.0, 100) is None, "效应为 0 时应返回 None")
check(matches_needed(3.65, 0.0, 100) is None, "标准误为 0 时应返回 None")
# 反过来：se 很小的时候，几十场就够
few = matches_needed(3.65, 0.5, 100)
check(few is not None and few < 100, f"se 很小时应当不需要更多场，得到 {few}")
print(f"  ✅ 效应 3.65：se=4.0 -> 约需 {need:,.0f} 场；se=2.0 -> 约需 {need2:,.0f} 场"
      f"（正好 1/4）；se=0.5 -> {few:.0f} 场")

print("\n== 核心理由：同牌山必须把配对方差压下来 ==")


def make_net(seed):
    return Net(dim=486, hidden=(16, 16), seed=seed, lr=1e-3, l2=0.0)


def run_pairs(n_deals: int, same_wall: bool) -> list[float]:
    """跑 n_deals 组，每组两副座位；返回每组的 rp(A) - rp(B)。"""
    from aimodel.agents import MLAgent

    na, nb = make_net(1), make_net(2)
    out = []
    for m in range(n_deals):
        deal_seed = 555000 + m
        w = wall_for(deal_seed, R3) if same_wall else None
        pair = []
        for half in range(2):
            gseed = deal_seed * 7 + half
            base = ["A", "B", "third"]
            seats = [base[(s - half) % 3] for s in range(3)]
            agents = []
            for s, tag in enumerate(seats):
                if tag == "A":
                    agents.append(MLAgent(na, seed=gseed * 3 + s, greedy=True))
                elif tag == "B":
                    agents.append(MLAgent(nb, seed=gseed * 3 + s, greedy=True))
                else:
                    agents.append(RandomAgent(gseed * 3 + s))
            g = Game(agents, seed=gseed, rule=R3)
            if same_wall and w is not None:
                g.rigged_wall = list(w)
            r = g.play()
            pair.append(r.rank_points[seats.index("A")]
                        - r.rank_points[seats.index("B")])
        out.append(sum(pair) / len(pair))
    return out


N_DEALS = 24
sim = run_pairs(N_DEALS, same_wall=True)
dif = run_pairs(N_DEALS, same_wall=False)


def var_of(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)


v_sim, v_dif = var_of(sim), var_of(dif)
print(f"  同牌山 方差={v_sim:8.2f}  se={math.sqrt(v_sim/N_DEALS):5.2f}")
print(f"  不同牌山 方差={v_dif:8.2f}  se={math.sqrt(v_dif/N_DEALS):5.2f}")
check(v_sim < v_dif,
      f"同牌山应当降低配对方差（同牌山 {v_sim:.2f} vs 不同牌山 {v_dif:.2f}）"
      " —— 如果这里失败，说明同牌山没起作用")
print("  ✅ 同牌山的配对方差确实更小（这正是 P1-4 的核心理由）")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过")
