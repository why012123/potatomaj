"""教师那条路也要和 aimodel 保持一致。

`ai.teacher_scores` 复刻了 `HeuristicAgent._discard` 的**循环结构**，
但打分调用的是教师自己的方法（`_danger` / `_pressure` / `_discard_cost` …）。
为什么不整段抄过来：抄一遍就会漂 —— 教师改了权重、加了新的考虑，
我这边不会报错，只会安静地给出不一样的建议。

这个测试就是钉住这一点：拿真引擎跑局面，同一手牌喂给两边，
**选出来的牌必须是同一张**。不一致就说明我这边镜像的那层出了偏差。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402
from aimodel.engine import Decision, Game  # noqa: E402
from potatomaj import ai as AI  # noqa: E402


class CaptureAgent(HeuristicAgent):
    """正常打（就是教师本人），但把每个打牌局面抄下来。"""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.views = []

    def decide(self, decision, view):
        if decision.kind == "discard":
            self.views.append((decision, view))
        return super().decide(decision, view)


def main() -> int:
    total = 0
    disagree = []
    for players in (3, 4):
        for seed in (1, 2, 3):
            agents = [CaptureAgent(seed * 10 + i) for i in range(players)]
            try:
                Game(agents, seed=seed, rule=S.rule_for(players), rounds=players).play()
            except Exception as exc:                       # noqa: BLE001
                disagree.append(f"{players}麻 seed{seed} 跑不起来: {exc}")
                continue

            for seat, ag in enumerate(agents):
                for decision, view in ag.views[::max(1, len(ag.views) // 8)][:8]:
                    total += 1
                    # 教师自己选的那张
                    want = decision.options[ag._discard(decision, view)]
                    # 我这边打分最高的那张
                    scores = AI.teacher_scores(view)
                    if not scores:
                        disagree.append(f"{players}麻 seed{seed} 座位{seat}: 我这边没给出候选")
                        continue
                    got = max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                    if got != want:
                        disagree.append(
                            f"{players}麻 seed{seed} 座位{seat} 第{view.turn}巡: "
                            f"教师选 {S.tile_cn(want)}，我选 {S.tile_cn(int(got))}")

    print(f"  比对了 {total} 个打牌局面")
    if disagree:
        print()
        for d in disagree[:8]:
            print("❌", d)
        if len(disagree) > 8:
            print(f"...还有 {len(disagree) - 8} 条")
        print()
        print(f"❌ 和教师不一致：{len(disagree)} 处")
        return 1
    print()
    print(f"✅ 教师那条路和 aimodel 选出的牌完全一致（{total} 个局面）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
