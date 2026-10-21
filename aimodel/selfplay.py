"""Full-information self-play data generator.

Why this exists
---------------
Behaviour cloning learns from a *heuristic teacher*, so the ceiling is the teacher.
Self-play is the way past it, but plain self-play still only gives you the same
information a human game log gives you: your own hand and the public board.

The engine knows everyone's tiles. That is information no real game log can ever
contain, and it is exactly what a defence model needs: "if I throw this tile,
does it deal in, to whom, for how much?"

This module produces data ONLY -- it does not train anything.

What each recorded decision contains
------------------------------------
Input (what the seat could legitimately see at that moment):
    X             : the normal observation encoding (aimodel.encode.encode)
    actfeat       : the normal per-tile action features (aimodel.encode.action_features)
    kind / action : which decision it was, and which index was chosen

Labels (only obtainable by peeking, hence self-play):
    chosen_deal_in / chosen_loser / chosen_han / chosen_points
                  : did the *chosen* discard actually deal in, to whom, for how much
    danger_all    : (34,) for every tile in hand -- would discarding it deal in
    danger_han    : (34,) best han the tile would give away (0 if none)
    danger_pts    : (34,) points it would give away (0 if none)
    opp_tenpai    : (players,) is that opponent tenpai right now
    opp_waits     : (players, 34) tiles that complete that opponent's hand
    opp_wait_han  : (players, 34) han of that completion (0 when not a winning tile)
    hand_all      : (players, 34) every seat's concealed hand (the peek)
    shanten_all   : (players,) shanten per seat
    result_*      : the final outcome of that hand (winner / loser / han / points)

Usage
-----
    python -m aimodel.selfplay --players 3 --matches 200 --jobs 6 --out data/sp3
    python -m aimodel.selfplay --players 4 --matches 200 --jobs 6 --out data/sp4
    python -m aimodel.selfplay --stats --out data/sp3          # report only

Each worker writes its own shard ``<out>/shard_<tag>.npz``; nothing is ever
appended to a shared file, so workers never contend on I/O.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import sanma as S, variants
from .agents import Agent, HeuristicAgent, MLAgent, RandomAgent
from .encode import (
    HEAD_SIZE, action_features, action_index, action_mask, encode, layout_for,
)
from .engine import Decision, Game, View, dora_from_index
from .fastshanten import shanten
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"

#: decision kinds that are recorded (discard is the one the danger model needs)
DISCARD = "discard"


# ---------------------------------------------------------------------------
# peeking: wait enumeration and deal-in evaluation
# ---------------------------------------------------------------------------


def _dora_count_for(game: Game, p: int) -> int:
    """How many dora seat ``p`` is holding, using the game's real indicators."""
    pl = game.players[p]
    inds = [game._indicator(i) for i in range(game.dora_slots)]
    types = [dora_from_index(i, game.rule) for i in inds]
    tiles = pl.dora_tiles()
    return sum(tiles.count(d) for d in types)


def win_context(game: Game, p: int, tile: int, tsumo: bool) -> S.WinContext:
    """The same context the engine itself would use when scoring seat ``p``."""
    pl = game.players[p]
    ura = 0
    if pl.riichi:
        ura_types = [
            dora_from_index(game._indicator(i, ura=True), game.rule)
            for i in range(game.dora_slots)
        ]
        ura = sum(pl.dora_tiles().count(d) for d in ura_types)
    return S.WinContext(
        win_tile=tile,
        is_tsumo=tsumo,
        round_wind=game.round_wind,
        seat_wind=game.seat_wind(p),
        is_riichi=pl.riichi,
        is_double_riichi=pl.double_riichi,
        is_ippatsu=pl.ippatsu and pl.riichi,
        is_haitei=(not tsumo and game.wall_left() == 0),
        dora=_dora_count_for(game, p),
        ura=ura,
        aka=pl.aka_count(),
        kita=pl.kita,
        rule=game.rule,
    )


def deal_in_info(game: Game, from_p: int, tile: int) -> tuple[int, int, int]:
    """If ``from_p`` discards ``tile``, what does it give away?

    Returns ``(loser_seat, han, points)``; ``(-1, 0, 0)`` when nobody can ron it.
    Multi-ron is possible, so the first (closest seat) winner is reported.
    """
    n = game.n
    best = (-1, 0, 0)
    for k in range(1, n):
        q = (from_p + k) % n
        pl = game.players[q]
        need = 4 - len(pl.melds)
        if need < 0:
            continue
        cand = list(pl.hand)
        # pl.hand is a 34-length COUNT array per tile type, so the tile total is
        # sum(cand), not len(cand) -- len() is always 34 and made every guard
        # below fail, which silently produced an all-zero deal-in label.
        if sum(cand) != need * 3 + 1:
            continue
        cand[tile] += 1
        if not S.is_winning_shape(cand, len(pl.melds)):
            continue
        agari = S.evaluate_hand(cand, pl.melds, win_context(game, q, tile, tsumo=False))
        if agari is None:
            continue  # shape is there but no yaku (e.g. open hand with nothing)
        pay = S.settle(
            agari, is_dealer=(q == game.dealer), is_tsumo=False, winner=q,
            loser=from_p, rule=game.rule, dealer_seat=game.dealer, n_players=n,
        )
        pts = int(pay.deltas[from_p])
        if best[0] < 0 or pts < best[2]:
            best = (q, int(agari.total_han), pts)
    return best


def wait_info(game: Game, q: int) -> tuple[np.ndarray, np.ndarray]:
    """``(waits, han)`` for seat ``q``: which tiles complete the hand, at what han.

    ``waits[t]`` is 1 when drawing/discarding ``t`` would win for ``q``; ``han[t]``
    is the resulting han (0 when ``waits[t] == 0``).
    """
    pl = game.players[q]
    waits = np.zeros(34, dtype=np.int8)
    han = np.zeros(34, dtype=np.int16)
    need = 4 - len(pl.melds)
    # sum(), not len(): pl.hand is a 34-slot count array (see deal_in_info)
    if need < 0 or sum(pl.hand) != need * 3 + 1:
        return waits, han
    for t in range(34):
        if pl.hand[t] >= 4:
            continue
        cand = list(pl.hand)
        cand[t] += 1
        if not S.is_winning_shape(cand, len(pl.melds)):
            continue
        agari = S.evaluate_hand(cand, pl.melds, win_context(game, q, t, tsumo=False))
        if agari is None:
            continue
        waits[t] = 1
        han[t] = int(agari.total_han)
    return waits, han


# ---------------------------------------------------------------------------
# recording agent
# ---------------------------------------------------------------------------


class FullInfoAgent(Agent):
    """Plays like ``inner`` and records full-information labels at every decision.

    ``game`` must be attached after the Game is constructed (the Game needs the
    agents first), so it is a plain attribute set by the worker.
    """

    def __init__(self, inner: Agent, rec: dict, watch: bool = True) -> None:
        self.inner = inner
        self.rec = rec
        self.watch = watch
        self.game: Game | None = None
        self.name = getattr(inner, "name", "agent")

    # -- recording helpers --------------------------------------------------
    def _note(self, key: str, value) -> None:
        self.rec.setdefault(key, []).append(value)

    def decide(self, decision: Decision, view: View) -> int:
        choice = int(self.inner.decide(decision, view))
        if not self.watch or self.game is None:
            return choice
        g = self.game
        me = view.player
        kind = decision.kind
        if kind not in HEAD_SIZE:
            return choice
        head = action_index(decision, choice)

        # inputs: exactly what this seat may see
        _enc_view = getattr(self.inner, "encode_view", None)
        self._note("X", _enc_view(view) if _enc_view is not None else encode(view))
        self._note("kind", kind)
        self._note("action", head)
        self._note("turn", view.turn)

        n = g.n
        hand_all = np.zeros((n, 34), dtype=np.int8)
        shanten_all = np.zeros(n, dtype=np.int8)
        for q in range(n):
            pl = g.players[q]
            hand_all[q, :] = np.asarray(pl.hand, dtype=np.int8)
            shanten_all[q] = shanten(pl.hand, len(pl.melds))
        self._note("hand_all", hand_all)
        self._note("shanten_all", shanten_all)

        if kind == DISCARD:
            _af_of = getattr(self.inner, "action_features_of", None)
            af = _af_of(view) if _af_of is not None else action_features(view)
            self._note("actfeat", af)
            # per-tile danger for everything in hand
            d_all = np.zeros(34, dtype=np.int8)
            d_han = np.zeros(34, dtype=np.int16)
            d_pts = np.zeros(34, dtype=np.int32)
            for t in decision.options:
                t = int(t)
                q, hn, pts = deal_in_info(g, me, t)
                if q >= 0:
                    d_all[t] = 1
                    d_han[t] = hn
                    d_pts[t] = pts
            self._note("danger_all", d_all)
            self._note("danger_han", d_han)
            self._note("danger_pts", d_pts)
            # what the chosen discard actually gives away
            chosen_tile = int(decision.options[choice])
            q, hn, pts = deal_in_info(g, me, chosen_tile)
            self._note("chosen_loser", q)
            self._note("chosen_deal_in", 1 if q >= 0 else 0)
            self._note("chosen_han", hn)
            self._note("chosen_pts", pts)
            self._note("chosen_tile", chosen_tile)
        else:
            self._note("actfeat", np.zeros((34, _action_dim()), dtype=np.float32))
            self._note("danger_all", np.zeros(34, dtype=np.int8))
            self._note("danger_han", np.zeros(34, dtype=np.int16))
            self._note("danger_pts", np.zeros(34, dtype=np.int32))
            self._note("chosen_loser", -1)
            self._note("chosen_deal_in", 0)
            self._note("chosen_han", 0)
            self._note("chosen_pts", 0)
            self._note("chosen_tile", -1)

        # opponents' readiness, from the peek
        waits = np.zeros((n, 34), dtype=np.int8)
        whan = np.zeros((n, 34), dtype=np.int16)
        tenpai = np.zeros(n, dtype=np.int8)
        for q in range(n):
            if q == me:
                continue
            w, h = wait_info(g, q)
            waits[q] = w
            whan[q] = h
            tenpai[q] = 1 if (w.any() or shanten(g.players[q].hand, len(g.players[q].melds)) == 0) else 0
        self._note("opp_waits", waits)
        self._note("opp_wait_han", whan)
        self._note("opp_tenpai", tenpai)
        return choice


_ACTION_DIM_CACHE: list[int] = []


def _action_dim() -> int:
    """``encode.ACTION_DIM`` without a circular-looking import at module top."""
    from .encode import ACTION_DIM

    return ACTION_DIM


# ---------------------------------------------------------------------------
# opponent pool
# ---------------------------------------------------------------------------


def load_pool(players: int, version: str | None, limit: int = 0) -> list[tuple[str, Net]]:
    """Every already-trained model of the same player count, any version."""
    out: list[tuple[str, Net]] = []
    seen: set[str] = set()
    for ver in (None, "v1.3", "v1.2", "v1.1", "v1"):
        try:
            specs = variants.all_specs(ver)
        except Exception:
            continue
        for spec in specs:
            if spec.players != players or spec.name in seen:
                continue
            if not spec.path.exists():
                continue
            try:
                out.append((spec.name, Net.load(spec.path)))
                seen.add(spec.name)
            except Exception:
                continue
    if limit:
        out = out[:limit]
    return out


def make_agent(
    seat: int, seed: int, players: int, pool: list[tuple[str, Net]],
    pool_ratio: float, teacher_ratio: float, record: bool,
) -> Agent:
    """One seat: recorded net / pool model / teacher, by the given ratios."""
    rng = np.random.default_rng(seed * 7919 + seat)
    u = float(rng.random())
    if u < teacher_ratio:
        return HeuristicAgent(seed * 31 + seat)
    if pool and u < teacher_ratio + pool_ratio:
        name, net = pool[int(rng.integers(len(pool)))]
        return MLAgent(net, seed=seed * 31 + seat, greedy=False)
    return RandomAgent(seed * 31 + seat)  # replaced by caller when recording


# ---------------------------------------------------------------------------
# one worker
# ---------------------------------------------------------------------------


def run_worker(args_tuple) -> dict:
    (out_dir, tag, players, n_matches, seed0, pool_ratio, teacher_ratio,
     record_all, max_dec) = args_tuple
    # one BLAS thread per worker: many workers, tiny matrices
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    pool = load_pool(players, None)
    rule = S.rule_for(players)
    rec: dict = {}
    outcomes: list[list[tuple]] = []   # per-match list of hand outcomes
    n_hands = 0
    results = {"tsumo": 0, "ron": 0, "draw": 0, "abort": 0, "kyuushu": 0}
    t0 = time.perf_counter()

    for m in range(n_matches):
        seed = seed0 + m
        agents = []
        for seat in range(players):
            base = make_agent(seat, seed, players, pool, pool_ratio, teacher_ratio, True)
            agents.append(FullInfoAgent(base, rec, watch=True))
        g = Game(agents, seed=seed, rule=rule)
        for a in agents:
            a.game = g  # late binding: the game exists now
        res = g.play()
        n_hands += len(res.hands)

        per_hand = []
        for h in res.hands:
            per_hand.append((
                h.kind,
                int(h.winner),
                int(h.loser),
                int(h.agari.total_han) if h.agari is not None else 0,
                int(h.deltas[h.winner]) if h.winner >= 0 and h.deltas else 0,
            ))
            results[h.kind] = results.get(h.kind, 0) + 1
        outcomes.append(per_hand)

    # ---- pack ----
    kinds = rec.get("kind", [])
    n_dec = len(kinds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def stack(key, dtype=None, shape=None):
        vals = rec.get(key, [])
        if not vals:
            return np.zeros((0,) + (shape or ()), dtype=dtype or np.float32)
        arr = np.asarray(vals)
        return arr.astype(dtype) if dtype is not None else arr

    # --- derive the danger-model training set from the discard decisions ---
    disc_idx = np.asarray([i for i, k in enumerate(kinds) if k == DISCARD], dtype=np.int64)
    if len(disc_idx):
        af_all = stack("actfeat", np.float32)
        af = af_all[disc_idx] if af_all.size else np.zeros((0, 34, _action_dim()), np.float32)
        tile = stack("chosen_tile", np.int16)[disc_idx]
        onehot = np.zeros((len(disc_idx), 34), dtype=np.int8)
        valid = (tile >= 0) & (tile < 34)
        onehot[np.arange(len(disc_idx))[valid], tile[valid]] = 1
        # rows aligned with `af`: the features of the tile that was actually thrown
        chosen_feat = af[np.arange(len(disc_idx)), np.clip(tile, 0, 33)]
        dmem = {
            "d_actfeat": af,
            "d_tile_onehot": onehot,
            "d_chosen_feat": chosen_feat,
            "d_deal_in": stack("chosen_deal_in", np.int8)[disc_idx],
            "d_han": stack("chosen_han", np.int16)[disc_idx],
            "d_pts": stack("chosen_pts", np.int32)[disc_idx],
            "d_loser": stack("chosen_loser", np.int8)[disc_idx],
            "d_danger_all": stack("danger_all", np.int8)[disc_idx],
            "d_danger_han": stack("danger_han", np.int16)[disc_idx],
            "d_danger_pts": stack("danger_pts", np.int32)[disc_idx],
            "d_shanten_all": stack("shanten_all", np.int8)[disc_idx],
            "d_hand_all": stack("hand_all", np.int8)[disc_idx],
            "d_opp_tenpai": stack("opp_tenpai", np.int8)[disc_idx],
            "d_opp_waits": stack("opp_waits", np.int8)[disc_idx],
            "d_opp_wait_han": stack("opp_wait_han", np.int16)[disc_idx],
            "d_turn": stack("turn", np.int16)[disc_idx],
        }
    else:
        dmem = {}

    data = {
        "X": stack("X", np.float32),
        # NOTE: no mask array here on purpose. action_mask() is sized per decision
        # kind (discard 34, call 3, kan/kita/riichi 2), so stacking it across
        # decisions raises "inhomogeneous shape". The danger model does not need
        # it, and the normal training path keeps its own masks in memory.
        "kind": np.asarray(kinds, dtype="U8"),
        "action": stack("action", np.int16),
        "turn": stack("turn", np.int16),
        "hand_all": stack("hand_all", np.int8),
        "shanten_all": stack("shanten_all", np.int8),
        "opp_waits": stack("opp_waits", np.int8),
        "opp_wait_han": stack("opp_wait_han", np.int16),
        "opp_tenpai": stack("opp_tenpai", np.int8),
        "outcomes": np.frombuffer(json.dumps(outcomes).encode(), dtype=np.uint8),
        "meta": np.frombuffer(json.dumps({
            "players": players,
            "matches": n_matches,
            "hands": n_hands,
            "decisions": n_dec,
            "discards": int(len(disc_idx)),
            "seed0": seed0,
            "pool": [n for n, _ in pool],
            "pool_ratio": pool_ratio,
            "teacher_ratio": teacher_ratio,
            "results": results,
            "seconds": round(time.perf_counter() - t0, 1),
        }).encode(), dtype=np.uint8),
    }
    data.update(dmem)
    path = out / f"shard_{tag}.npz"
    np.savez_compressed(path, **data)
    deal_in = int(dmem["d_deal_in"].sum()) if dmem else 0
    return {
        "tag": tag, "path": str(path), "matches": n_matches, "hands": n_hands,
        "decisions": n_dec, "discards": int(len(disc_idx)), "deal_in": deal_in,
        "bytes": path.stat().st_size if path.exists() else 0,
        "seconds": round(time.perf_counter() - t0, 1), "pool": [n for n, _ in pool],
        "results": results,
    }


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def show_stats(out_dir: str) -> int:
    """Report what the shards actually contain.

    The two numbers that decide whether a danger model is trainable at all:
      * deal-in samples       -- how many positive examples exist
      * NO-RIICHI decisions that are still dangerous -- the hand-written
        tile_danger() has no defence at all when nobody has declared riichi,
        so this is the slice a learned model has to beat it on.
    """
    out = Path(out_dir)
    shards = sorted(out.glob("shard_*.npz"))
    if not shards:
        print(f"no shards in {out}")
        return 1
    tot_dec = tot_disc = tot_deal = tot_hands = tot_matches = 0
    tot_bytes = 0
    tenpai_pos = 0
    noriichi_tot = noriichi_danger = 0
    loss_hist: dict[int, int] = {}
    pts_sum = 0
    han_sum = 0
    for f in shards:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        npl = int(meta["players"])
        tot_dec += int(meta["decisions"])
        disc = int(meta.get("discards", 0))
        tot_disc += disc
        tot_hands += int(meta["hands"])
        tot_matches += int(meta["matches"])
        tot_bytes += f.stat().st_size
        if disc and "d_deal_in" in z.files:
            di = z["d_deal_in"]
            tot_deal += int(di.sum())
            pts_sum += int(z["d_pts"].sum())
            han_sum += int(z["d_han"].sum())
            for v in z["d_loser"][di == 1]:
                loss_hist[int(v)] = loss_hist.get(int(v), 0) + 1
            opp = z["d_opp_tenpai"]
            tenpai_pos += int((opp.sum(axis=1) > 0).sum())
            # riichi flags live in X (the observation block), so read them there.
            # X has one row per DECISION; the d_* arrays have one row per DISCARD,
            # so index X through the discard positions, not directly.
            O, _ = layout_for(npl)
            kinds = z["kind"]
            disc_pos = np.nonzero(kinds == DISCARD)[0]
            x = z["X"][disc_pos]
            riichi_block = x[:, O["riichi"]:O["riichi"] + npl]
            nori = riichi_block.sum(axis=1) == 0
            noriichi_tot += int(nori.sum())
            if nori.any():
                d_all = z["d_danger_all"][nori]
                noriichi_danger += int((d_all.max(axis=1) > 0).sum())
        z.close()
    print(f"\n{'=' * 66}")
    print(f"  selfplay stats  --  {out}")
    print(f"{'=' * 66}")
    print(f"  shards              {len(shards)}")
    print(f"  matches             {tot_matches:,}")
    print(f"  hands               {tot_hands:,}")
    print(f"  decisions           {tot_dec:,}")
    print(f"  discard decisions   {tot_disc:,}  ({tot_disc / max(tot_dec,1):.1%} of all)")
    print(f"  deal-in samples     {tot_deal:,}  ({tot_deal / max(tot_disc,1):.3%} of discards)")
    if tot_deal:
        print(f"    avg han given away{tot_deal and han_sum / tot_deal:>8.2f}")
        print(f"    avg points lost   {pts_sum / tot_deal:>8.0f}  (negative = the loser pays)")
    print(f"  opp-tenpai samples  {tenpai_pos:,}  ({tenpai_pos / max(tot_disc,1):.1%} of discards)")
    print(f"  NO-RIICHI discards  {noriichi_tot:,}  of which still give away a win: "
          f"{noriichi_danger:,} ({noriichi_danger / max(noriichi_tot,1):.2%})")
    print(f"  on-disk             {tot_bytes / 1e6:,.1f} MB")
    if loss_hist:
        print(f"  deal-in to seat     " + "  ".join(f"{k}:{v}" for k, v in sorted(loss_hist.items())))
    print(f"{'=' * 66}\n")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="full-information self-play data generator")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--matches", type=int, default=200, help="matches PER WORKER")
    ap.add_argument("--jobs", type=int, default=6, help="worker processes")
    ap.add_argument("--out", type=str, default="data/selfplay")
    ap.add_argument("--pool-ratio", type=float, default=0.35)
    ap.add_argument("--teacher-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="report on existing shards")
    ap.add_argument("--max-decisions", type=int, default=0, help="0 = no cap")
    ap.add_argument("--tag", type=str, default="", help="shard name suffix")
    args = ap.parse_args(argv)

    if args.stats:
        return show_stats(args.out)

    stamp = args.tag or f"p{args.players}_{int(time.time())}"
    work = []
    for i in range(args.jobs):
        work.append((
            args.out, f"{stamp}_{i:03d}", args.players, args.matches,
            args.seed + i * 1_000_003, args.pool_ratio, args.teacher_ratio,
            True, args.max_decisions,
        ))

    print(f"selfplay: players={args.players} workers={args.jobs} "
          f"matches/worker={args.matches} out={args.out}")
    t0 = time.perf_counter()
    ctx = get_context("spawn")  # spawn: safe on Windows and reproducible
    with ctx.Pool(args.jobs) as pool:
        results = []
        for r in pool.imap_unordered(run_worker, work):
            results.append(r)
            print(f"  [{len(results)}/{len(work)}] {r['tag']}  hands={r['hands']:,} "
                  f"decisions={r['decisions']:,} deal_in={r['deal_in']:,} "
                  f"{r['bytes']/1e6:.0f}MB {r['seconds']:.0f}s")
    dt = time.perf_counter() - t0
    dec = sum(r["decisions"] for r in results)
    hands = sum(r["hands"] for r in results)
    matches = sum(r["matches"] for r in results)
    bytes_ = sum(r["bytes"] for r in results)
    print(f"\ntotal: {matches:,} matches  {hands:,} hands  {dec:,} decisions  "
          f"{bytes_/1e6:,.1f} MB  in {dt/60:.1f} min")
    print(f"rate: {dec/max(dt,1):,.0f} decisions/s  "
          f"{hands/max(dt,1)*3600:,.0f} hands/hour")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
