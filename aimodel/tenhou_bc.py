"""Behaviour-cloning dataset from real Tenhou hands.

WHY THIS EXISTS
---------------
`train.collect_bc()` builds its dataset by letting the *heuristic teacher* play
games and recording what it did. That makes the teacher the ceiling: the cloned
policy can at best match a hand-written rule set. This module instead extracts
behaviour-cloning samples from the **40,000 real Tenhou hands** in ``GL1.xlsm``,
so the policy can imitate strong human play.

Measured on those logs (3,000-hand slice): per-seat win rate **20.98%**, riichi
19.72%, draw 16.6% -- i.e. Houou-level play, well above the heuristic teacher.

WHY IT REUSES logcheck
----------------------
Reconstructing a `View` from a Tenhou log (wall order, dead wall, calls,
riichi sticks) is already implemented and hard-won in `logcheck.py` /
`mjlog.py`. Writing a second replay would be a second chance to get the wall
order wrong. So we replay with `logcheck.LogAgent` and simply *also* record
each decision, by subclassing it.

Faithfulness matters: `logdata.py` shares that replay path and we measured that
about 5.5% of kyoku diverge (the engine keeps playing where the log ended). A
divergent hand would feed fabricated decisions into BC, so `--min-faithful`
drops any kyoku whose `LogAgent.problems` is non-empty.

OUTPUT FORMAT
-------------
Exactly the keys `train.Tracker.arrays()` produces, so the result is a drop-in
for `train.train_bc`::

    X        (n, 560)      float32   observation
    mask     (n, 34)       float32   legal actions for that decision kind
    actfeat  (n, 34, 13)   float16   per-tile action features (discard rows)
    action   (n,)          int64     chosen head index (== tile for discards)
    logp     (n,)          float32   always 0: human moves carry no model logp
    kind     (n,)          <U8       decision kind
    player   (n,)          int64     seat
    ret      (n,)          float32   value target (norm_reward of that seat)

`ret` is filled in afterwards from the game result, matching what
`collect_bc` does.

USAGE
-----
    python -m aimodel.tenhou_bc --src _data/final40k --out _data/tbc --shard 4
    python -m aimodel.tenhou_bc --src _data/final40k --out _data/tbc --limit 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import logcheck as LC
from . import mjai
from . import rewards
from . import sanma as S
from . import variants
from .encode import HEAD_SIZE, action_features, action_index, action_label, action_mask, encode
from .engine import Game

ROOT = Path(__file__).resolve().parent.parent
# `--hidden`-style commas are a shell hazard; hidden width is a constant here.
DEFAULT_MODEL = "maj-normal4v1.3"


def _legal_chi_pairs(hand, tile: int) -> list[tuple[int, int]]:
    """给定被吃的那张，返回**手里真有**的 (a,b) 组合，按 (a,b) 升序 ✓

    例：吃 4s、手里有 2s3s 与 5s6s -> [(2s,3s), (5s,6s)] ✓
    这个顺序就是 chipat 的编码基准 ✓ 推理端必须用同一个顺序 ✗
    """
    if tile >= 27:
        return []
    suit0 = (tile // 9) * 9
    out: list[tuple[int, int]] = []
    for lo in (tile - 2, tile - 1, tile):
        hi = lo + 2
        if lo < suit0 or hi >= suit0 + 9:
            continue
        others = [x for x in (lo, lo + 1, hi) if x != tile]
        if all(hand[x] > 0 for x in others):
            out.append((others[0], others[1]))
    return sorted(out)


class RecordingLogAgent(LC.LogAgent):
    """Replays the log exactly like `LogAgent`, and records each decision.

    The recording body is copied from `train.RecordingAgent.decide` on purpose:
    the dataset must be byte-compatible with what `train_bc` already consumes.
    """

    def __init__(self, seat: int, script, tracker, problems=None) -> None:
        super().__init__(seat, script, problems)
        self.tracker = tracker

    def backfill_chi(self, ev) -> None:
        """吃牌事件到达后，把「第几种吃法」回填到那一行 ✓

        engine 的事件里 `tile` = 顺子起点 ✓ `called` = 被吃的那张 ✓
        所以「另外两张」= (start, start+1, start+2) 去掉 called ✓
        """
        pairs = getattr(self, "_chi_pairs", None)
        row = getattr(self, "_chi_row", None)
        if not pairs or row is None or row >= len(self.tracker):
            return
        start, called = ev.get("tile"), ev.get("called")
        if start is None or called is None:
            return
        others = tuple(x for x in (start, start + 1, start + 2) if x != called)
        try:
            self.tracker.chipats[row] = pairs.index(others)
        except ValueError:
            pass

    def decide(self, decision, view) -> int:
        choice = super().decide(decision, view)
        # 吃：记下该局面**合法吃法**（按 (a,b) 升序）✓ 供「吃成什么样」标签用 ✓
        if decision.kind == "chi" and decision.tile is not None:
            self._chi_pairs = _legal_chi_pairs(view.hand, decision.tile)
            self._chi_row = len(self.tracker)
        else:
            self._chi_pairs, self._chi_row = [], None
        kind = decision.kind
        # Only record decisions that actually follow the log. When the replay
        # has diverged, `super().decide` returns 0 as a fallback and the move is
        # NOT what the human played -- recording it would poison the dataset.
        if kind in HEAD_SIZE and not self.problems:
            self.tracker.record(
                kind,
                encode(view),
                action_mask(decision)[: HEAD_SIZE[kind]],
                action_index(decision, choice),
                0.0,                      # humans carry no policy logp
                view.player,
                action_label(decision, choice),
                action_features(view) if kind == "discard" else None,
            )
        return choice


def extract_kyoku(kyoku, tracker, n: int, rule):
    """Replay one kyoku, recording into `tracker`. Returns (problems, result)."""
    start = kyoku[0]
    scripts = LC.build_scripts(kyoku, n)
    agents = [RecordingLogAgent(i, scripts[i], tracker) for i in range(n)]
    game = Game(agents, seed=0, rule=rule)
    game.rigged_wall = LC.kyoku_to_wall(kyoku, n, rule)
    game.on_event = None

    def dispatch(a, agents=agents):
        for ag in agents:
            ag.on_action(a)
        # 吃牌：回填「第几种吃法」✓
        if isinstance(a, dict) and a.get("call") == "chi":
            p = a.get("player")
            if isinstance(p, int) and 0 <= p < len(agents):
                agents[p].backfill_chi(a)

    game.on_action = dispatch
    res = game.play_hand(int(start.get("oya", 0)))
    problems = sum(len(ag.problems) for ag in agents)
    return problems, res


def _hand_rets(res, tracker, start_idx: int, mode: str) -> list[float]:
    """Value target per recorded decision, in recording order.

    NOTE ON THE VALUE TARGET
    ------------------------
    `train.collect_bc` uses `rewards.norm_reward(spec.reward, <MatchResult>, p)`,
    which is defined over a whole match: 段位分, 放銃点, 和了次数, 累计番数 all
    need `MatchResult.stats`, `rank_points` and `start_scores`. We replay ONE
    kyoku at a time, so those quantities do not exist here and pretending they
    do would be fabrication.

    What we can state exactly is the hand's own point swing, `res.deltas[p]`.
    Two modes:

      * ``delta`` (default) -- `res.deltas[p] / 1000.0`, i.e. "did this hand
        gain or lose me points, in units of 1000";
      * ``zero``            -- all zeros. Use this if you want the value head to
        contribute nothing rather than learn a target that is NOT the one the
        policy is eventually scored on.

    ⚠️ Neither equals `spec.reward`. If you want the real thing, the kyoku have
    to be reassembled into matches first (the log rows are hands, but
    `seed`/`oya` and the running scores identify their match). That is a
    follow-up, deliberately not guessed at here.
    """
    if mode == "zero":
        return [0.0] * (len(tracker) - start_idx)
    return [float(res.deltas[p]) / 1000.0
            for p in tracker.players[start_idx:]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tenhou -> behaviour-cloning dataset")
    ap.add_argument("--src", default=str(ROOT / "_data" / "final40k"),
                    help="dir of MJAI json files (from mjlog.py) or a single file")
    ap.add_argument("--out", default=str(ROOT / "_data" / "tbc"))
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="max kyoku (0 = all)")
    ap.add_argument("--shard", type=int, default=0,
                    help="write every N kyoku as one npz (0 = single file)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="reward spec used for the value target `ret`")
    ap.add_argument("--ret-mode", default="delta", choices=["delta", "zero"],
                    help="value target: per-hand point swing, or all zeros (see _hand_rets)")
    ap.add_argument("--max-faithful-problems", type=int, default=0,
                    help="drop kyoku with more LogAgent problems than this")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    rule = S.rule_for(args.players)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.src)
    files = sorted(glob.glob(str(src / "*.json"))) if src.is_dir() else [str(src)]
    if not files:
        print(f"no json files under {src}", file=sys.stderr)
        return 2

    tracker_cls = __import__("aimodel.train", fromlist=["Tracker"]).Tracker
    tracker = tracker_cls()

    n_kyoku = n_kept = n_dropped = 0
    since_shard = 0
    shard_no = 0
    t0 = time.perf_counter()
    pending_rets: list[float] = []

    def flush(shard_no: int) -> int:
        """Write the accumulated tracker to one npz; returns rows written."""
        data = tracker.arrays()
        data["ret"] = np.asarray(pending_rets, dtype=np.float32)
        rows = len(data["X"])
        if rows == 0:
            return 0
        path = out_dir / f"tbc_{shard_no:04d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)          # atomic: never leave a half file
        return rows

    total_rows = 0
    for f in files:
        if args.limit and n_kyoku >= args.limit:
            break
        for kyoku in mjai.iter_kyoku(mjai.read(f)):
            if args.limit and n_kyoku >= args.limit:
                break
            n_kyoku += 1
            start_idx = len(tracker)
            problems, res = extract_kyoku(kyoku, tracker, args.players, rule)
            if problems > args.max_faithful_problems:
                # Diverged: undo the recording for this kyoku and move on.
                # ⚠️ 这里必须列**全部**并行列表 ✗ 漏一个就会长度错位 ✗
                #   2026-10-21 实测：漏了 `chipats` -> 丢弃的局把它留下 ✗
                #   于是 chipat 比 X 长 854 条、非 chi 行也带 0/1、填充率只有 8.5% ✗
                for lst in (tracker.kinds, tracker.X, tracker.masks, tracker.actions,
                            tracker.logps, tracker.players, tracker.labels,
                            tracker.actfeats, tracker.chipats):
                    del lst[start_idx:]
                n_dropped += 1
                continue
            n_kept += 1
            pending_rets.extend(_hand_rets(res, tracker, start_idx, args.ret_mode))
            since_shard += 1
            if args.shard and since_shard >= args.shard:
                total_rows += flush(shard_no)
                shard_no += 1
                since_shard = 0
                tracker = tracker_cls()
                pending_rets = []
        if not args.quiet:
            print(f"  {Path(f).name}: kyoku={n_kyoku} kept={n_kept} "
                  f"dropped={n_dropped} rows={len(tracker.X)}")

    total_rows += flush(shard_no)
    dt = time.perf_counter() - t0
    print()
    print(f"kyoku read      {n_kyoku:,}")
    print(f"  faithful      {n_kept:,}  ({n_kept / max(n_kyoku, 1):.1%})")
    print(f"  dropped       {n_dropped:,}")
    print(f"decision rows   {total_rows:,}")
    print(f"output          {out_dir}")
    print(f"elapsed         {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

