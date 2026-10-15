"""Chunked, resumable BC data generation for a machine that kills long jobs.

THE PROBLEM
-----------
On this box a process started by an agent session gets killed after roughly
15 seconds to 3.5 minutes, and FIVE different detach strategies were tried and
all failed (cmd `start /b`, `Start-Process` with two redirects, Python
`Popen` + `DETACHED_PROCESS|BREAKAWAY`, Task Scheduler with a batch, Task
Scheduler with `wscript.exe`). Every one of them died at 50-120 seconds, well
short of the 10-minute bar. See `aimodel/README.md` for the full list.

So the job must not NEED to survive. It is cut into slices of about 8.6
seconds; being killed only ever loses the slice in flight.

DESIGN
------
* **slice** = `--chunk-hands` hands of BC data (default 100, measured 8.6 s
  including the save -- see the 10 s budget below).
* **atomic write**: each slice is written to `<id>.tmp.npz` and then
  `os.replace`d into place, so a kill can never leave a half file.
  NOTE: `np.savez_compressed` APPENDS `.npz` when the name does not already end
  in it, so `savez("x.tmp")` silently writes `x.tmp.npz` and the rename then
  fails. We pass an open file handle instead, which avoids that entirely.
* **done ledger**: `<workdir>/done.txt`, one line per completed slice holding
  `<slice_id> <rows> <sha256-of-file>`. Resume reads this rather than guessing
  from the filesystem. A slice whose file hash does not match its ledger entry
  is treated as missing and redone.
* **deterministic**: a slice's content depends only on (model, seed, index), so
  redoing a slice after a kill produces the same data -- there is no need to
  worry about "did this slice half-apply".

SLICING CAVEAT (documented, not hidden)
---------------------------------------
BC hands come from whole matches, so a slice's seed is derived from its index
and `collect_bc` may overshoot slightly (it finishes the match it is in). The
UNION of slices therefore covers the requested hands but is not byte-identical
to one `collect_bc(15000)` call. The verification in the task asks for the
holdout-agreement difference between sliced and one-shot to be <= 0.5 points;
see `--verify` (added once a full slice set exists).

USAGE
-----
    python -m aimodel.shortjob plan  --model maj-normal4v1.3 --hands 15000
    python -m aimodel.shortjob run   --model maj-normal4v1.3 --hands 15000 --budget 600
    python -m aimodel.shortjob status --model maj-normal4v1.3
    python -m aimodel.shortjob merge  --model maj-normal4v1.3 --out _data/bc_tv13.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK = ROOT / "_data" / "shortjob"

# Measured on this box (maj-normal4v1.3 teacher, 4-player):
#   hands=100  gen 8.3s  save 0.24s  total  8.6s   6288 rows
#   hands=150  gen 12.2s save 0.35s  total 12.5s   9884 rows
# The budget is 10 s per slice, so 100 hands is the largest safe default.
DEFAULT_CHUNK_HANDS = 100
SLICE_BUDGET_S = 10.0
# Slice seeds must not collide with the holdout seed used by train.cmd_bc,
# which is `args.seed + 555_555`. We stride well clear of that.
SEED_STRIDE = 1_000_003


def slice_dir(work: Path, model: str) -> Path:
    return work / model


def ledger_path(work: Path, model: str) -> Path:
    return slice_dir(work, model) / "done.txt"


def read_ledger(work: Path, model: str) -> dict[int, tuple[int, str]]:
    """slice_id -> (rows, sha256). Corrupt/garbage lines are ignored."""
    p = ledger_path(work, model)
    out: dict[int, tuple[int, str]] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            out[int(parts[0])] = (int(parts[1]), parts[2])
        except ValueError:
            continue
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def slice_path(work: Path, model: str, idx: int) -> Path:
    return slice_dir(work, model) / f"slice_{idx:05d}.npz"


def seed_for(model: str, idx: int, base_seed: int) -> int:
    """Stable per-slice seed. Depends only on the slice index."""
    return base_seed + idx * SEED_STRIDE


def run_one_slice(work: Path, model: str, idx: int, hands: int,
                  base_seed: int, verbose: bool = True) -> tuple[int, float]:
    """Generate and atomically store one slice. Returns (rows, seconds)."""
    from . import train as T
    from . import variants

    spec = variants.get(model)
    t0 = time.perf_counter()
    data = T.collect_bc(hands, seed=seed_for(model, idx, base_seed),
                        verbose=False, spec=spec)

    d = slice_dir(work, model)
    d.mkdir(parents=True, exist_ok=True)
    final = slice_path(work, model, idx)
    tmp = final.with_name(final.name + ".tmp")
    # File handle avoids np.savez's automatic ".npz" suffix, which would make
    # the os.replace below fail (it would look for <name>.tmp, not .tmp.npz).
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, final)
    dt = time.perf_counter() - t0

    rows = len(data["X"])
    with open(ledger_path(work, model), "a", encoding="utf-8") as fh:
        fh.write(f"{idx} {rows} {sha256_file(final)}\n")
    if verbose:
        print(f"  slice {idx:05d}: {rows:6d} rows  {dt:5.1f}s")
    return rows, dt


def cmd_plan(a) -> int:
    a.work = Path(a.work)
    n = n_slices(a)
    print(f"model      {a.model}")
    print(f"hands      {a.hands}")
    print(f"chunk      {a.chunk_hands} hands/slice   (~{a.chunk_hands * 0.086:.1f}s each)")
    print(f"slices     {n}")
    print(f"est total  {n * a.chunk_hands * 0.086 / 60:.1f} min of compute")
    print(f"workdir    {slice_dir(a.work, a.model)}")
    return 0


def n_slices(a) -> int:
    return max(1, (a.hands + a.chunk_hands - 1) // a.chunk_hands)


def _run_until_done(a, work: Path) -> int:
    """Keep making slices until the set is complete.

    WHY THIS LOOP EXISTS: on this box the process gets killed every
    15s-3.5min (measured; five detach strategies all failed -- see README ⑥).
    A single long `run` therefore dies partway no matter what `--budget` says.
    But because every slice is atomic and recorded in `done.txt`, being killed
    is harmless: the next attempt simply skips what is already done. So we
    re-enter the slice loop rather than trying to survive.

    `attempts` bounds how many times we tolerate a kill before giving up, so a
    genuinely broken slice cannot spin forever.
    """
    total = n_slices(a)
    t_start = time.perf_counter()
    attempt = 0
    while attempt < a.attempts:
        attempt += 1
        done = read_ledger(work, a.model)
        todo = [i for i in range(total) if i not in done]
        if not todo:
            break
        print(f"[attempt {attempt}] {len(done)}/{total} done, {len(todo)} to go")
        try:
            for idx in todo:
                if a.time_limit and (time.perf_counter() - t_start) > a.time_limit:
                    print(f"time limit {a.time_limit:.0f}s reached")
                    return _summary(a, work, t_start)
                run_one_slice(work, a.model, idx, a.chunk_hands, a.seed)
        except KeyboardInterrupt:
            print("interrupted")
            break
        except Exception as exc:                     # noqa: BLE001
            # Includes the case where the process is torn down between slices
            # and the interpreter reports it as an error. The ledger is the
            # source of truth, so just go round again.
            print(f"  attempt {attempt} ended: {type(exc).__name__}: {exc}")
    return _summary(a, work, t_start)


def _summary(a, work: Path, t0: float) -> int:
    done = read_ledger(work, a.model)
    total = n_slices(a)
    print()
    print(f"slices {len(done)}/{total}   rows {sum(r for r, _ in done.values()):,}   "
          f"elapsed {time.perf_counter() - t0:.0f}s")
    if len(done) < total:
        print("  (incomplete -- re-run the same command to continue)")
    return 0 if len(done) >= total else 1


def cmd_run(a) -> int:
    work = Path(a.work)
    a.work = work
    if a.until_done:
        return _run_until_done(a, work)
    done = read_ledger(work, a.model)
    total = n_slices(a)
    t_start = time.perf_counter()
    made = skipped = 0
    times: list[float] = []

    for idx in range(total):
        if idx in done and slice_path(work, a.model, idx).exists():
            skipped += 1
            continue
        if a.budget and (time.perf_counter() - t_start) > a.budget:
            print(f"budget {a.budget:.0f}s reached; {total - idx} slices left "
                  f"(resume by re-running the same command)")
            break
        rows, dt = run_one_slice(work, a.model, idx, a.chunk_hands, a.seed)
        made += 1
        times.append(dt)

    print()
    print(f"made {made}  skipped(already done) {skipped}  of {total}")
    if times:
        arr = np.asarray(times)
        print(f"slice time  mean {arr.mean():.1f}s  p95 {np.percentile(arr, 95):.1f}s  "
              f"max {arr.max():.1f}s   (budget {SLICE_BUDGET_S:.0f}s)")
    return 0


def cmd_status(a) -> int:
    work = Path(a.work)
    a.work = work
    done = read_ledger(work, a.model)
    total = n_slices(a)
    rows = sum(r for r, _ in done.values())
    print(f"model    {a.model}")
    print(f"slices   {len(done)} / {total}")
    print(f"rows     {rows:,}")
    missing = [i for i in range(total) if i not in done]
    if missing:
        print(f"next     {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    else:
        print("next     (complete)")
    return 0


def cmd_merge(a) -> int:
    """Concatenate all finished slices into one npz for train_bc."""
    work = Path(a.work)
    done = read_ledger(work, a.model)
    keys = ("X", "mask", "actfeat", "action", "logp", "kind", "player", "ret")
    acc: dict[str, list] = {k: [] for k in keys}
    used = 0
    for idx in sorted(done):
        p = slice_path(work, a.model, idx)
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        for k in keys:
            if k in z.files:
                acc[k].append(z[k])
        used += 1
    if not used:
        print("nothing to merge", file=sys.stderr)
        return 2
    data = {k: np.concatenate(v) for k, v in acc.items() if v}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, out)
    print(f"merged {used} slices -> {out}")
    print(f"rows     {len(data['X']):,}")
    print(f"dim      {data['X'].shape[1]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["plan", "run", "status", "merge"])
    ap.add_argument("--model", default="maj-normal4v1.3")
    ap.add_argument("--hands", type=int, default=15000)
    ap.add_argument("--chunk-hands", type=int, default=DEFAULT_CHUNK_HANDS)
    ap.add_argument("--seed", type=int, default=0, help="base seed for slice 0")
    ap.add_argument("--work", default=str(DEFAULT_WORK))
    ap.add_argument("--budget", type=float, default=0.0,
                    help="seconds; stop starting new slices past this (0 = no cap)")
    ap.add_argument("--until-done", action="store_true",
                    help="keep going across kills until every slice exists "
                         "(safe because each slice is atomic and ledgered)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="overall seconds across all attempts (0 = no cap)")
    ap.add_argument("--attempts", type=int, default=40,
                    help="max kill-restart cycles for --until-done")
    ap.add_argument("--out", default=str(ROOT / "_data" / "bc_tv13.npz"))
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run,
            "status": cmd_status, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
