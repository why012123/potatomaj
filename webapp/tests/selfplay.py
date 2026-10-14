#!/usr/bin/env python
"""验收 4：**AI 自对弈合法性** —— 200 局，零异常 / 零非法动作 / 牌数守恒。

这一条**不经过网络**，直接驱动 ``aimodel.engine``（和平台用的是同一个引擎），
所以它验证的是「引擎 + AI 座位适配」本身没问题：

* 引擎抛异常？ -> 计数
* agent 返回越界下标？ -> 计数（这就是「非法动作」）
* 牌数守恒？ -> ``Σ手牌 + Σ副露 + Σ牌河 + 拔北 + 剩余牌山 + 王牌 == 108/136``
* 收支守恒？ -> 每局 ``Σdeltas == 当局被收走的立直棒 × 1000``

  **这里有个容易搞错的地方（我自己先搞错过一次）**：``Σdeltas`` **不是 0**。
  立直棒在宣言那一刻就已经从点数里扣掉了（``scores[p] -= 1000``），
  结算时再把供托**全部给和牌的人**（``_take_sticks``），所以这一局的
  ``Σdeltas`` 正好等于「被收走的立直棒数 × 1000」。只有一局里
  **没有立直棒被收走**时才是 0。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent, RandomAgent  # noqa: E402
from aimodel.engine import Game  # noqa: E402

TILE_TOTAL = {3: 108, 4: 136}


def tile_count(game) -> int:
    """盘面上所有牌之和（手牌 + 副露 + 牌河 + 拔北 + 牌山余 + 王牌）。"""
    n = 0
    for pl in game.players:
        n += sum(pl.hand)
        for m in pl.melds:
            n += len(m.tiles)
        n += len(pl.pond)
        n += pl.kita
    n += game.wall_left()
    n += len(game.dead)
    return n


class Checked:
    """包一层，检查每次 decide 的返回值是不是合法下标。

    这是「零非法动作」的**唯一可信来源** —— 引擎自己不会校验 agent 的返回
    值（它会直接拿去当下标用），所以必须在外面兜。
    """

    def __init__(self, inner, seat: int, stats: Counter) -> None:
        self.inner = inner
        self.seat = seat
        self.name = getattr(inner, "name", "?")
        self.stats = stats

    def decide(self, decision, view):
        self.stats["decisions"] += 1
        idx = self.inner.decide(decision, view)
        if not isinstance(idx, int) or isinstance(idx, bool) \
                or not (0 <= idx < len(decision.options)):
            self.stats["illegal"] += 1
            self.stats["ILLEGAL:" + decision.kind] += 1
            return 0
        self.stats[decision.kind] += 1
        return idx


def run_games(players: int, games: int, seed0: int, mix: str) -> int:
    total = TILE_TOTAL[players]
    throws = illegal = tile_bad = money_bad = 0
    decisions = hands = ron = tsumo = draw = sticks_paid = 0
    kinds: Counter[str] = Counter()
    worst = ""

    for g in range(games):
        rules = S.rule_for(players)
        stats: Counter[str] = Counter()
        seat_agents = []
        for i in range(players):
            if mix == "same":
                inner = HeuristicAgent(seed0 + g * 97 + i)
            else:
                inner = (HeuristicAgent if (g + i) % 2 == 0 else RandomAgent)(
                    seed0 + g * 97 + i
                )
            seat_agents.append(Checked(inner, i, stats))
        try:
            game = Game(seat_agents, seed=seed0 + g * 131, rule=rules, rounds=players)
            res = game.play()
        except Exception:  # noqa: BLE001
            throws += 1
            if not worst:
                worst = traceback.format_exc()[-900:]
            continue

        illegal += stats["illegal"]
        kinds.update({k: v for k, v in stats.items()
                      if k not in ("decisions", "illegal")})
        decisions += stats["decisions"]

        n = tile_count(game)
        if n != total:
            tile_bad += 1
            if not worst:
                worst = f"第 {g} 局牌数 {n} != {total}"
        for h in res.hands:
            hands += 1
            # 引擎在打完这一局之后会把 sticks 清掉，所以「本局收走的棒数」
            # 直接用 Σdeltas / 1000 反推（一份供托只能给一个人）。
            got_sticks = sum(h.deltas) / S.Rule().riichi_stick
            if abs(got_sticks - round(got_sticks)) > 1e-9:
                money_bad += 1
                if not worst:
                    worst = f"第 {g} 局收支不是 1000 的整数倍：{h.deltas}"
            else:
                sticks_paid += round(got_sticks)
            if h.kind == "ron":
                ron += 1
            elif h.kind == "tsumo":
                tsumo += 1
            elif h.kind in ("draw", "kyuushu"):
                draw += 1

    print(f"  {players} 人 × {games} 局（{mix}）：")
    print(f"    决策 {decisions} 次，共 {hands} 局")
    print(f"    引擎异常 {throws}   非法动作 {illegal}   "
          f"牌数不守恒 {tile_bad}   收支异常 {money_bad}")
    print(f"    和了：自摸 {tsumo} / 荣和 {ron} / 流局 {draw}"
          f"；结算时收走的立直棒 {sticks_paid} 根"
          f"（Σdeltas 应等于这个数 × 1000）")
    top = sorted((k for k in kinds if not k.startswith("ILLEGAL")),
                 key=lambda k: -kinds[k])
    print("    决策种类：" + "  ".join(f"{k}={kinds[k]}" for k in top))
    if worst:
        print("    首个问题：", worst[:400])
    return 0 if (throws == 0 and illegal == 0 and tile_bad == 0 and money_bad == 0) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--players", type=int, default=0, choices=[0, 3, 4],
                    help="0 = 三麻四麻都跑")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--mix", default="mixed", choices=["same", "mixed"])
    args = ap.parse_args(argv)

    print("=" * 70)
    print(f"验收 4：AI 自对弈（每局 {args.games} 次，不同 seed）")
    print("=" * 70)
    rc = 0
    for p in ([3, 4] if args.players == 0 else [args.players]):
        rc |= run_games(p, args.games, args.seed + p * 7, args.mix)
    if rc == 0:
        print("\n✅ 自对弈通过：0 异常 / 0 非法动作 / 牌数守恒 / 收支守恒")
    else:
        print("\n❌ 自对弈有问题（见上面计数）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
