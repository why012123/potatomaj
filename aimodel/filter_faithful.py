"""Keep only kyoku whose replay is FAITHFUL to the log.

Why this exists
---------------
`aimodel/logdata.py` extracts behaviour-cloning and danger samples by REPLAYING
each hand through the engine (it shares `logcheck`'s path: ``Game.rigged_wall`` +
``LogAgent``). When the replay diverges from the log, the engine keeps playing
its own moves and those fabricated decisions get recorded as training data.

Measured divergence rate on GL1.xlsm: **5.5%** of kyoku (200-hand slice: 6.0%).
So roughly one in eighteen samples was bogus. This filter replays every kyoku and
keeps only the ones with ZERO `LogAgent` problems.

Usage
-----
    python aimodel/filter_faithful.py <src.json> <dst.json> [players]

`src` is any MJAI file (e.g. a batch dumped by ``python -m aimodel.mjlog``).
`dst` receives only the faithful kyoku, ready for ``python -m aimodel.logdata``.

Note: this reads `LogAgent.problems` rather than re-deriving the comparison. Do
NOT re-implement the action matching here -- `LogAgent.on_action` filters by
`a["player"]` and a hand-rolled equivalent silently gets it wrong (I did exactly
that once and drew a false conclusion from it).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimodel import logcheck as LC, mjai, sanma as S
from aimodel.engine import Game


def faithful_kyoku(kyoku: list[dict], n: int) -> bool:
    """Replay one kyoku; True when the engine's play matched the log exactly."""
    start = kyoku[0]
    scripts = LC.build_scripts(kyoku, n)
    agents = [LC.LogAgent(i, scripts[i]) for i in range(n)]
    game = Game(agents, seed=0, rule=S.rule_for(n))
    game.rigged_wall = LC.kyoku_to_wall(kyoku, n, S.rule_for(n))
    game.on_event = None

    def dispatch(a: dict) -> None:
        for ag in agents:
            ag.on_action(a)

    game.on_action = dispatch
    game.play_hand(int(start.get("oya", 0)))
    return all(not ag.problems for ag in agents)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    src, dst = argv[1], argv[2]
    n = int(argv[3]) if len(argv) > 3 else 4

    kept = dropped = 0
    out: list[str] = []
    for kyoku in mjai.iter_kyoku(mjai.read(src)):
        if faithful_kyoku(kyoku, n):
            kept += 1
            out.extend(json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                       for e in kyoku)
        else:
            dropped += 1

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    total = kept + dropped
    print(f"kyoku={total}  kept={kept}  dropped={dropped}  "
          f"({dropped / max(total, 1):.1%} dropped)")
    print("wrote:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
