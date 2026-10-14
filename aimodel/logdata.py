"""从牌谱抽训练数据。

两条用途，正好补上这个项目最缺的两块：

1. **行为克隆** —— ``(当时可见的局面 → 该玩家的动作)``。
   现在的 BC 教师是手写启发式（向听→进张→危险度三条规则）；有了真实牌谱，
   就能换成**真人高手**的决策，起点完全不同。

2. **危险度拟合** —— ``(局面 + 打出的这张牌 → 会不会放铳)``。
   ``encode.tile_danger()`` 现在是手写常数（中张 0.62、幺九 0.22、筋牌打 0.65 折），
   这是整个项目最弱的一环：它读不到听牌形状，对默听和清一色完全没有防备。
   牌谱里的真实放铳记录正好能把它拟合出来。

## 怎么保证抽出来的数据能用

**特征走的是同一套 :func:`aimodel.encode.encode` / :func:`action_features`**，
产出结构也和 :class:`aimodel.agents.Tracker` 一致，所以能直接进
``train.train_bc`` / ``train_ppo``。

## 用法::

    python -m aimodel.logdata game.json --out human.npz --players 3
    python -m aimodel.logdata logs/ --out human.npz --limit 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import logcheck as LC
from . import mjai, sanma as S
from .agents import Tracker
from .encode import HEAD_SIZE, action_features, action_index, action_mask, encode
from .engine import Decision, Game, View


@dataclass
class Samples:
    """抽出来的两类样本。"""

    tracker: Tracker = field(default_factory=Tracker)
    #: 危险度样本：每条是 ``(局面特征, 动作特征, 打出的牌, 是否放铳)``
    danger_x: list[np.ndarray] = field(default_factory=list)
    danger_a: list[np.ndarray] = field(default_factory=list)
    danger_tile: list[int] = field(default_factory=list)
    danger_dealt: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracker)

    def arrays(self) -> dict:
        data = self.tracker.arrays()
        data["danger_X"] = (
            np.asarray(self.danger_x, dtype=np.float16)
            if self.danger_x else np.zeros((0, 0), dtype=np.float16)
        )
        data["danger_A"] = (
            np.asarray(self.danger_a, dtype=np.float16)
            if self.danger_a else np.zeros((0, 0), dtype=np.float16)
        )
        data["danger_tile"] = np.asarray(self.danger_tile, dtype=np.int32)
        data["danger_dealt"] = np.asarray(self.danger_dealt, dtype=np.int8)
        return data


class RecordingLogAgent(LC.LogAgent):
    """照牌谱出牌，同时把每个决策点记下来。"""

    def __init__(self, seat: int, script: list[tuple], samples: Samples,
                 problems: list | None = None) -> None:
        super().__init__(seat, script, problems)
        self.samples = samples

    def decide(self, decision: Decision, view: View) -> int:
        choice = super().decide(decision, view)
        kind = decision.kind
        if kind in HEAD_SIZE and choice < len(decision.options):
            x = encode(view)
            mask = action_mask(decision)[: HEAD_SIZE[kind]]
            head = action_index(decision, choice)
            af = action_features(view) if kind == "discard" else None
            self.samples.tracker.record(
                kind, x, mask, head, 0.0, view.player,
                label="", actfeat=af,
            )
            if kind == "discard":
                self.samples.danger_x.append(x)
                self.samples.danger_a.append(af)
                self.samples.danger_tile.append(int(decision.options[choice]))
                self.samples.danger_dealt.append(0)   # 打完这一局再回填
        return choice


def extract_kyoku(kyoku: list[dict], n: int = 3,
                  samples: Samples | None = None) -> Samples:
    """把一局牌谱重放一遍，抽出决策样本 + 放铳标签。"""
    samples = samples if samples is not None else Samples()
    rule = S.rule_for(n)
    scripts = LC.build_scripts(kyoku, n)
    problems: list = []
    agents = [RecordingLogAgent(i, scripts[i], samples, problems) for i in range(n)]

    game = Game(agents, seed=0, rule=rule)
    game.rigged_wall = LC.kyoku_to_wall(kyoku, n, rule)
    game.on_event = None

    #: 每一步 discard 在 samples.danger_* 里的下标，用来回填放铳标签
    marks: list[tuple[int, int, int]] = []   # (样本下标, 座位, 牌)
    n_danger = len(samples.danger_x)

    def dispatch(a: dict) -> None:
        for ag in agents:
            ag.on_action(a)
        if a.get("kind") == "discard":
            marks.append((len(samples.danger_x), a["player"], a["tile"]))

    game.on_action = dispatch
    res = game.play_hand(dealer := int(kyoku[0].get("oya", 0)))

    # ── 回填放铳标签：牌谱里那次荣和，是被谁打出来的哪张牌 ──
    hora = mjai.hora_of(kyoku)
    if hora is not None and not hora.get("hora_tsumo"):
        win_tile, _ = mjai.tile_from_mjai(hora["pai"])
        loser = hora.get("target")
        for i in range(len(samples.danger_x) - 1, n_danger - 1, -1):
            if samples.danger_tile[i] == win_tile and loser is not None:
                samples.danger_dealt[i] = 1
                break
    del dealer, res, marks
    return samples


def extract_file(path: str | Path, n: int = 3, limit: int | None = None,
                 samples: Samples | None = None) -> Samples:
    samples = samples if samples is not None else Samples()
    events = mjai.read(path)
    for i, kyoku in enumerate(mjai.iter_kyoku(events)):
        if limit and i >= limit:
            break
        extract_kyoku(kyoku, n=n, samples=samples)
    return samples


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="从 MJAI 牌谱抽训练数据")
    ap.add_argument("path", help="牌谱文件或目录")
    ap.add_argument("--out", default="human.npz", help="输出 npz")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--limit", type=int, default=None, help="每个文件最多几局")
    args = ap.parse_args(argv)

    p = Path(args.path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    samples = Samples()
    for f in files:
        extract_file(f, n=args.players, limit=args.limit, samples=samples)

    data = samples.arrays()
    np.savez_compressed(args.out, **data)
    n_d = len(data["danger_tile"])
    dealt = int(data["danger_dealt"].sum()) if n_d else 0
    print(f"牌谱 {len(files)} 个 → {args.out}")
    print(f"  决策样本 {len(samples)} 条（行为克隆用）")
    print(f"  危险度样本 {n_d} 条，其中放铳 {dealt} 条"
          f"（{dealt / n_d * 100:.2f}%）" if n_d else "  危险度样本 0 条")
    if n_d and dealt < 30:
        print("  ⚠️ 放铳正样本太少，拟合出来的危险度会不可靠 —— 多喂点牌谱")


if __name__ == "__main__":
    main()
