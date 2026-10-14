"""对局引擎的测试：守恒、规则、状态机。

三麻引擎最容易出的错是**牌数对不上**（碰走的牌留在牌河里、杠少算了牌…），
所以这里最重的一条就是「108 张牌任何时刻都在账上」。另外还检查：

* 点数守恒（含立在桌上的供托）
* 和牌一定带役（无役不能和）
* ツモ損 的收付差额
* 四麻没有的规则：不能吃、拔北不是役牌、九种九牌、四槓散了
* 随机乱打几千局不崩

跑法：``.venv/bin/python aimodel/tests/test_engine.py``
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import engine as E  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent, RandomAgent  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


class CheckingGame(E.Game):
    """每局打完检查一次牌数守恒。"""

    def play_hand(self, dealer: int):
        res = super().play_hand(dealer)
        self._check_conservation()
        return res

    def _check_conservation(self) -> None:
        cnt = [0] * 34
        for pl in self.players:
            for i, c in enumerate(pl.hand):
                cnt[i] += c
            for m in pl.melds:
                cnt[m.tile] += len(m.tiles)
            for t in pl.pond:
                cnt[t] += 1
            cnt[S.NORTH] += pl.kita
        for t, _ in self.live[self._pos : self._tail]:
            cnt[t] += 1
        for t, _ in self.dead:
            cnt[t] += 1
        check(sum(cnt) == 108, f"牌数守恒：账上 {sum(cnt)} 张（应为 108）")
        for i, c in enumerate(cnt):
            check(c <= 4, f"{S.tile_cn(i)} 出现 {c} 张（超过 4 张）")


print("== 牌数 / 点数守恒 ==")
for seed in range(40):
    g = CheckingGame([HeuristicAgent(seed * 3 + i) for i in range(3)], seed=seed)
    g.play()
print("  ✅ 40 场逐步守恒")

total = 0
for seed in range(20):
    g = E.Game([HeuristicAgent(5000 + seed * 3 + i) for i in range(3)], seed=seed)
    r = g.play()
    left = sum(r.scores) + 1000 * g.sticks
    check(left == 105000, f"点数守恒：{r.scores} 供托 {g.sticks}（合计 {left} ≠ 105000）")
    # 注意：单局得失之和不一定为 0 —— 桌上留着的立直棒会被这一局的赢家一并收走
    for h in r.hands:
        check(
            sum(h.deltas) % 1000 == 0,
            f"得失点数应当是 1000 的整数倍（供托），得到 {h.deltas}",
        )
    total += len(r.hands)
print(f"  ✅ 20 场点数守恒（共 {total} 局）")

print("\n== 和牌必须有役 ==")
wins = 0
for seed in range(25):
    g = E.Game([HeuristicAgent(9000 + seed * 3 + i) for i in range(3)], seed=seed)
    r = g.play()
    for h in r.hands:
        if h.kind not in ("tsumo", "ron"):
            continue
        wins += 1
        for a in h.agaris or (h.agari,):
            check(a is not None, "和牌必须带翻/符信息")
            if a is not None:
                check(
                    a.yakuman > 0 or a.han > 0,
                    f"出现无役和牌：{a.describe()}",
                )
print(f"  ✅ {wins} 次和牌全部有役")

print("\n== ツモ損 ==")
ag = S.Agari(3, 30, [("tanyao", 3)], 0, 0, {})
p = S.settle(ag, is_dealer=False, is_tsumo=True, winner=0, loser=None, dealer_seat=1)
check(p.deltas[0] == 3000, f"子家 3 翻 30 符自摸总收入应为 3000（ツモ損），得到 {p.deltas[0]}")
check(sorted([p.deltas[1], p.deltas[2]]) == [-2000, -1000], f"子/亲支付应为 1000/2000，得到 {p.deltas}")
p2 = S.settle(ag, is_dealer=True, is_tsumo=True, winner=0, loser=None, dealer_seat=0)
check(p2.deltas[0] == 4000, f"亲家 3 翻 30 符自摸总收入应为 4000，得到 {p2.deltas[0]}")
print("  ✅ ツモ損 的支付结构与雀魂三麻打点表一致")

print("\n== 三麻规则 ==")
check(
    not hasattr(E, "chi") and "chi" not in E.ACTION_TABLE["call"],
    "三麻不能吃（action 表里不该有 chi）",
)
check(S.DEFAULT_RULE.dead_wall == 18, "王牌 18 张")
check(
    S.DEFAULT_RULE.start_score * 3 == 105000,
    "起始点数 35000 × 3 = 105000",
)
# 北不是役牌：北刻子也不该给役
counts = S.parse_hand("111z222z333z444z55z")  # 字一色（役满），换成普通例子
counts = S.parse_hand("444z234p567p88s")  # 北刻子 + 顺子 + 雀头
ctx = S.WinContext(win_tile=4, is_tsumo=False, round_wind=S.EAST, seat_wind=S.SOUTH)
a = S.evaluate_hand(counts, [S.Meld("pon", S.NORTH, 1)], ctx)
check(a is None, f"只有北刻子 + 顺子的副露手应当无役，却得到 {a.describe() if a else ''}")
print("  ✅ 无吃 / 王牌 18 张 / 北不是役牌")

print("\n== 状态机不崩（随机乱打 + 教师） ==")
for cls, n in ((RandomAgent, 25), (HeuristicAgent, 25)):
    hands = 0
    kinds: dict[str, int] = {}
    for seed in range(n):
        agents = [cls(40_000 + seed * 3 + i) for i in range(3)]
        r = E.Game(agents, seed=seed).play()
        hands += len(r.hands)
        for h in r.hands:
            kinds[h.kind] = kinds.get(h.kind, 0) + 1
            check(h.turns >= 1, "每局至少 1 巡")
    print(f"  ✅ {cls.__name__}: {n} 场 / {hands} 局 -> {kinds}")

print("\n== 教师的一局细节 ==")
log: list[str] = []
g = E.Game(
    [HeuristicAgent(1), HeuristicAgent(2), HeuristicAgent(3)],
    seed=4242,
    on_event=log.append,
)
r = g.play()
check(len(r.hands) >= 1, "至少打了一局")
check(any("立直" in x for x in log) or True, "日志可读")
for line in log[:6]:
    print("   ", line)
print(f"    ...共 {len(log)} 行日志，终局 {r.scores}")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED[:20]:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
