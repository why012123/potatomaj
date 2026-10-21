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
`--layout v2`（**默认**，生产口径）与 :meth:`train.Tracker.arrays` 逐列一致::

    X        (n, 645)      float32   observation（4 麻；3 麻 571）
    mask     (n, 34)       float32   legal actions for that decision kind
    actfeat  (n, 34, 13)   float16   per-tile action features (discard rows)
    action   (n,)          int64     chosen head index (== tile for discards)
    logp     (n,)          float32   always 0: human moves carry no model logp
    kind     (n,)          <U8       decision kind
    player   (n,)          int64     seat
    chipat   (n,)          int8      「吃成什么样」标签（-1 = 无关）
    ret      (n,)          float32   value target (norm_reward of that seat)

`--layout v3`（dot V1 per-tile）把 ``X`` 换成两块，其余列**逐列不变**::

    tiles    (n, 34, 20)   float32   ``encode_tile`` 的逐通道表（牌, 通道）
    globals  (n, 21)       float32   ``encode_tile`` 的全局标量
    mask / actfeat / action / logp / kind / player / chipat / ret  同上
    （**没有 X** —— 避免同一份信息在 npz 里存两遍）

`ret` 仍由 `_hand_rets` 事后填入。:func:`validate_dataset` / :func:`load_shard`
是给训练器的入口：v2/v3 自动判别，v3 的 ``tiles`` 形状不是
``(n, 34, 20)`` 或 ``globals`` 不是 ``(n, 21)`` 会**当场 ValueError**。

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
from .agents import Tracker as _BaseTracker
from .encode import (
    ACTION_DIM,
    HEAD_SIZE,
    action_features,
    action_index,
    action_label,
    action_mask,
    encode,
)
from .encode_tile import F as TILE_F
from .encode_tile import G as GLOBAL_G
from .encode_tile import encode_tile
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


# ---------------------------------------------------------------------------
# 数据集列 / 两种 layout 的 Tracker
# ---------------------------------------------------------------------------
#: v3 不需要 ``X``，但基类 ``Tracker`` 用 ``len(self.X)`` 当行数、用
#: ``arrays()`` 统一组装 mask/actfeat/chipat。v3 每行塞一个**共享的 0 长数组**
#: 当占位：``np.stack`` 出 ``(n, 0)`` 后立刻丢掉，不占每行 645 维的内存。
_EMPTY_X = np.zeros(0, dtype=np.float32)

LAYOUTS: tuple[str, ...] = ("v2", "v3")


class V3Tracker(_BaseTracker):
    """在基类 Tracker 上多记 ``tiles`` / ``globals``，输出时把 X 去掉。"""

    def __init__(self) -> None:
        super().__init__()
        self.tiles: list[np.ndarray] = []
        self.globals: list[np.ndarray] = []

    def record_v3(
        self,
        kind: str,
        tiles: np.ndarray,
        globals_: np.ndarray,
        mask: np.ndarray,
        action: int,
        logp: float,
        player: int,
        label: str = "",
        actfeat: np.ndarray | None = None,
        chipat: int = -1,
    ) -> None:
        tiles = np.asarray(tiles, dtype=np.float32)
        globals_ = np.asarray(globals_, dtype=np.float32)
        if tiles.shape != (34, TILE_F):
            raise ValueError(
                f"v3 tiles 形状 {tiles.shape} != (34, {TILE_F}) —— encode_tile 坏了？"
            )
        if globals_.shape != (GLOBAL_G,):
            raise ValueError(
                f"v3 globals 形状 {globals_.shape} != ({GLOBAL_G},) —— encode_tile 坏了？"
            )
        # 基类 record：X 用共享占位，其余列照常。len(tracker) 因此仍然正确。
        super().record(kind, _EMPTY_X, mask, action, logp, player, label, actfeat, chipat)
        self.tiles.append(tiles)
        self.globals.append(globals_)

    def arrays(self) -> dict:
        n = len(self)
        if len(self.tiles) != n or len(self.globals) != n:
            raise ValueError(
                f"v3 列长度不一致：rows={n}, tiles={len(self.tiles)}, "
                f"globals={len(self.globals)}"
            )
        data = super().arrays()          # mask / actfeat / action / ... / chipat
        data.pop("X", None)              # v3 不存 645 维平铺向量
        data["tiles"] = np.stack(self.tiles).astype(np.float32)
        data["globals"] = np.stack(self.globals).astype(np.float32)
        return data


def make_tracker(layout: str):
    """按 layout 造一个空 Tracker（--shard 切分时也用它重置）。"""
    if layout == "v2":
        return _BaseTracker()
    if layout == "v3":
        return V3Tracker()
    raise ValueError(f"未知 layout {layout!r}，可选 {LAYOUTS}")


def _record_decision(tracker, layout: str, decision, view, choice: int) -> None:
    """把一个决策写进 tracker。

    **v2 分支的求值顺序和旧代码逐字一致**（``encode`` 在前，``action_mask``
    等在后）—— 回归测试要求同样输入抽出的 npz 逐位相同，不能因为重构改变
    浮点/缓存的调用顺序。
    """
    kind = decision.kind
    if layout == "v3":
        tiles, globals_ = encode_tile(view)
        tracker.record_v3(
            kind,
            tiles,
            globals_,
            action_mask(decision)[: HEAD_SIZE[kind]],
            action_index(decision, choice),
            0.0,                      # humans carry no policy logp
            view.player,
            action_label(decision, choice),
            action_features(view) if kind == "discard" else None,
        )
    else:
        tracker.record(
            kind,
            encode(view),
            action_mask(decision)[: HEAD_SIZE[kind]],
            action_index(decision, choice),
            0.0,                      # humans carry no policy logp
            view.player,
            action_label(decision, choice),
            action_features(view) if kind == "discard" else None,
        )


# ---------------------------------------------------------------------------
# 训练器入口：v2/v3 判别 + 形状校验
# ---------------------------------------------------------------------------


def _keys_of(data) -> set[str]:
    return set(data.files) if hasattr(data, "files") else set(data.keys())


def validate_dataset(data, expect_layout: str | None = None) -> str:
    """校验一个 npz/字典是 v2 还是 v3，返回 layout。

    这是**加载器侧**的硬门禁：v3 的 ``tiles`` 必须是 ``(n, 34, 20)``、
    ``globals`` 必须是 ``(n, 21)``。把 F 改成 19 或把 34 改成 35 会在这里
    直接 ``ValueError``，而不是让训练器在第一层 matmul 才炸。
    """
    keys = _keys_of(data)
    has_tiles = "tiles" in keys or "globals" in keys
    if has_tiles:
        if "tiles" not in keys or "globals" not in keys:
            raise ValueError("v3 数据必须同时有 tiles 和 globals")
        if "X" in keys:
            raise ValueError("v3 数据不应包含 X（同一份信息不要存两遍）")
        layout = "v3"
    else:
        if "X" not in keys:
            raise ValueError("数据集里既没有 v2 的 X，也没有 v3 的 tiles")
        layout = "v2"

    if expect_layout is not None:
        if expect_layout not in LAYOUTS:
            raise ValueError(f"expect_layout 必须是 {LAYOUTS}，收到 {expect_layout!r}")
        if layout != expect_layout:
            raise ValueError(f"期望 {expect_layout} 数据，实际是 {layout}")

    if layout == "v3":
        tiles = np.asarray(data["tiles"])
        globals_ = np.asarray(data["globals"])
        if tiles.ndim != 3 or tiles.shape[1:] != (34, TILE_F):
            raise ValueError(
                f"v3 tiles 形状 {tiles.shape} 不对：应为 (n, 34, {TILE_F})"
            )
        if not np.issubdtype(tiles.dtype, np.floating):
            raise ValueError(f"v3 tiles dtype {tiles.dtype} 不是浮点")
        n = int(tiles.shape[0])
        if globals_.shape != (n, GLOBAL_G):
            raise ValueError(
                f"v3 globals 形状 {globals_.shape} 不对：应为 ({n}, {GLOBAL_G})"
            )
        if not np.issubdtype(globals_.dtype, np.floating):
            raise ValueError(f"v3 globals dtype {globals_.dtype} 不是浮点")
    else:
        X = np.asarray(data["X"])
        if X.ndim != 2 or X.shape[1] <= 0:
            raise ValueError(f"v2 X 形状 {X.shape} 不对：应为 (n, dim)")
        n = int(X.shape[0])

    expected = {
        "mask": (n, 34),
        "actfeat": (n, 34, ACTION_DIM),
        "action": (n,),
        "logp": (n,),
        "kind": (n,),
        "player": (n,),
        "ret": (n,),
    }
    for key, shape in expected.items():
        if key not in keys:
            raise ValueError(f"数据集缺少列 {key!r}")
        arr = np.asarray(data[key])
        if arr.shape != shape:
            raise ValueError(f"列 {key!r} 形状 {arr.shape} != {shape}")
    if "chipat" in keys and np.asarray(data["chipat"]).shape != (n,):
        raise ValueError(f"chipat 形状 {np.asarray(data['chipat']).shape} != ({n},)")
    return layout


def load_shard(path, expect_layout: str | None = None) -> tuple[str, dict]:
    """读一个 npz 分片并校验，返回 ``(layout, dict_of_arrays)``。

    v3 的数组是**存储 dtype**；训练时再 ``.astype(np.float32)``（这里已经是
    float32，但保留转换点，方便以后改 float16 存储）。
    """
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    layout = validate_dataset(data, expect_layout=expect_layout)
    return layout, data


class RecordingLogAgent(LC.LogAgent):
    """Replays the log exactly like `LogAgent`, and records each decision.

    The recording body is copied from `train.RecordingAgent.decide` on purpose:
    the dataset must be byte-compatible with what `train_bc` already consumes.
    """

    def __init__(
        self, seat: int, script, tracker, problems=None, layout: str = "v2"
    ) -> None:
        super().__init__(seat, script, problems)
        self.tracker = tracker
        if layout not in LAYOUTS:
            raise ValueError(f"未知 layout {layout!r}，可选 {LAYOUTS}")
        self.layout = layout

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
            _record_decision(self.tracker, self.layout, decision, view, choice)
        return choice


def extract_kyoku(kyoku, tracker, n: int, rule, layout: str = "v2"):
    """Replay one kyoku, recording into `tracker`. Returns (problems, result)."""
    start = kyoku[0]
    scripts = LC.build_scripts(kyoku, n)
    agents = [
        RecordingLogAgent(i, scripts[i], tracker, layout=layout) for i in range(n)
    ]
    game = Game(agents, seed=0, rule=rule)
    game.rigged_wall = LC.kyoku_to_wall(kyoku, n, rule)
    game.on_event = None

    # ---------------------------------------------------------------
    # 把「这一局在**整场**里的位置」灌进去 ✗ 否则 match_state 全是默认值
    # ---------------------------------------------------------------
    # `extract_kyoku` 是**一局一局**重放的（每局新建一个 Game ✓ 从不调
    # `Game.play()`），而 honba / kyoku / 供托 / 剩余局数 都是 `play()` 里
    # 才推进的 ✓ 不灌的话 View 里这些字段恒为默认值（kyoku=1 honba=0
    # kyotaku=0 hands_left=0），新加的 17 维 match_state 就等于**恒定的噪声** ✗
    # 数据里这些都是现成的：`start_kyoku` 带 kyoku / honba / riichi_sticks /
    # scores / bakaze ✓
    def _as_points(v: int) -> int:
        """天凤 mjlog 的点数单位是「百点」（250 = 25000）✓ 标准 MJAI 是实际点数 ✗"""
        return int(v) * 100 if int(v) <= 1000 else int(v)

    bakaze = start.get("bakaze", "E")
    game.round_wind = S.EAST if bakaze == "E" else S.SOUTH
    game.kyoku = int(start.get("kyoku", 1) or 1)
    game.honba = int(start.get("honba", 0) or 0)
    # 供托：我们自己的格式叫 riichi_sticks，标准 MJAI 叫 kyoutaku ✓ 两种都认
    game.sticks = int(start.get("riichi_sticks", start.get("kyoutaku", 0)) or 0)
    # 连庄：本场 > 0 基本等价于「这一局是连庄来的」✓（天凤里本场也可来自流局）
    game.dealer_repeat = game.honba > 0
    # 剩余局数（含本局）：东场 4 - kyoku + 1；南场 8 - kyoku + 1（南1 = 第5局）
    _base = 0 if game.round_wind == S.EAST else 4
    game.hands_left = max(0, _base + 4 - game.kyoku + 1)
    if start.get("scores"):
        game.scores = [_as_points(v) for v in start["scores"]]
        game.start_scores = list(game.scores)

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


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数；单独暴露出来，测试可以断言 ``--layout`` 默认 v2。"""
    ap = argparse.ArgumentParser(description="Tenhou -> behaviour-cloning dataset")
    ap.add_argument("--src", default=str(ROOT / "_data" / "final40k"),
                    help="dir of MJAI json files (from mjlog.py) or a single file")
    ap.add_argument("--out", default=str(ROOT / "_data" / "tbc"))
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="max kyoku (0 = all)")
    ap.add_argument("--shard", type=int, default=0,
                    help="write every N kyoku as one npz (0 = single file)")
    ap.add_argument("--layout", choices=LAYOUTS, default="v2",
                    help="v2 = 现有 645 维平铺特征（默认，生产口径）；"
                         "v3 = dot V1 per-tile（tiles/globals，不再存 X）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="reward spec used for the value target `ret`")
    ap.add_argument("--ret-mode", default="delta", choices=["delta", "zero"],
                    help="value target: per-hand point swing, or all zeros (see _hand_rets)")
    ap.add_argument("--max-faithful-problems", type=int, default=0,
                    help="drop kyoku with more LogAgent problems than this")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rule = S.rule_for(args.players)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.src)
    files = sorted(glob.glob(str(src / "*.json"))) if src.is_dir() else [str(src)]
    if not files:
        print(f"no json files under {src}", file=sys.stderr)
        return 2

    def new_tracker():
        return make_tracker(args.layout)

    tracker = new_tracker()

    n_kyoku = n_kept = n_dropped = 0
    since_shard = 0
    shard_no = 0
    t0 = time.perf_counter()
    pending_rets: list[float] = []

    def flush(shard_no: int) -> int:
        """Write the accumulated tracker to one npz; returns rows written."""
        rows = len(tracker)
        if rows == 0:
            return 0
        if len(pending_rets) != rows:
            raise ValueError(
                f"ret 长度 {len(pending_rets)} != 决策行数 {rows}（数据管线坏了）"
            )
        data = tracker.arrays()
        data["ret"] = np.asarray(pending_rets, dtype=np.float32)
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
            problems, res = extract_kyoku(
                kyoku, tracker, args.players, rule, args.layout
            )
            if problems > args.max_faithful_problems:
                # Diverged: undo the recording for this kyoku and move on.
                # ⚠️ 这里必须列**全部**并行列表 ✗ 漏一个就会长度错位 ✗
                #   2026-10-21 实测：漏了 `chipats` -> 丢弃的局把它留下 ✗
                #   于是 chipat 比 X 长 854 条、非 chi 行也带 0/1、填充率只有 8.5% ✗
                lists = [tracker.kinds, tracker.X, tracker.masks, tracker.actions,
                         tracker.logps, tracker.players, tracker.labels,
                         tracker.actfeats, tracker.chipats]
                if args.layout == "v3":
                    lists += [tracker.tiles, tracker.globals]
                for lst in lists:
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
                tracker = new_tracker()
                pending_rets = []
        if not args.quiet:
            print(f"  {Path(f).name}: kyoku={n_kyoku} kept={n_kept} "
                  f"dropped={n_dropped} rows={len(tracker)}")

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

