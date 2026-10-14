"""Train the danger model on SELF-PLAY data and compare with the logs baseline.

Motivation
----------
The original objective's third priority says the key validation is that the
danger score must be meaningful *when nobody has declared riichi*. In the human
Tenhou logs that slice is starving: only **316** positives (945 kyoku). The
self-play generator produces **27,224** such samples from 12,820 hands -- 86x
more -- because it records every seat's hand, every opponent's wait and whether
the game was actually lost on that discard.

This script pools the self-play shards and runs the SAME procedure as
`train_danger_baseline.py` (60/20/20, early stopping on validation only, test
touched once) so the two numbers are directly comparable:

    logs      : overall 0.6996 | NO-RIICHI 0.6838 | WITH-RIICHI 0.5872
    self-play : this run

Usage:
    python -m aimodel.train_danger_selfplay <shard_dir_glob> [--epochs N]
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from .encode import layout_for
from .train_danger_baseline import auc, standardise


def load_pool(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Pool every self-play shard into (danger_X, dealt_in_label)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shards matched {pattern}")
    xs, ys = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        # d_deal_in[p] == 1 means the discard at that decision actually lost the hand
        xs.append(z["X"].astype(np.float32))
        dealt = z["d_deal_in"].astype(np.int8)
        # X covers every decision; d_* only the discard decisions. Align by
        # taking the discard-decision rows out of X.
        kind = z["kind"]
        disc = np.flatnonzero(kind == "discard")
        if len(disc) != len(dealt):
            # fall back: match on the recorded count
            disc = np.arange(len(dealt))
        xs[-1] = z["X"][disc].astype(np.float32)
        ys.append(dealt.astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="danger model on self-play data")
    ap.add_argument("pattern", help="glob for self-play shards")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="_data/danger_model.npz",
                    help="where to write the trained model for --danger-model")
    args = ap.parse_args(argv)

    X, y = load_pool(args.pattern)
    print(f"self-play pool: {X.shape[0]:,} rows x {X.shape[1]} dims, "
          f"{int(y.sum()):,} positives ({y.mean() * 100:.3f}%)")
    print("  (logs baseline for comparison: 42,522 rows, 1.230% positives)")

    off, _ = layout_for(4)
    ri = X[:, off["riichi"]:off["riichi"] + 4].sum(axis=1)
    no_riichi = ri == 0
    print(f"  NO-RIICHI rows: {int(no_riichi.sum()):,} "
          f"({no_riichi.mean()*100:.1f}%), of which positive {int(y[no_riichi].sum()):,}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_tr, n_va = int(len(X) * 0.6), int(len(X) * 0.2)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    Xtr, Xva, Xte = standardise(X[tr], X[va], X[te])
    # keep the training statistics so inference can reproduce the same scaling
    Xtr_mu = X[tr].mean(axis=0, keepdims=True)
    Xtr_sd = X[tr].std(axis=0, keepdims=True)
    Xtr_sd[Xtr_sd < 1e-6] = 1.0
    ytr, yva, yte = y[tr], y[va], y[te]
    print(f"  split: train {len(tr):,} (pos {int(ytr.sum())})  "
          f"val {len(va):,} (pos {int(yva.sum())})  test {len(te):,} (pos {int(yte.sum())})")

    w_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)
    w = np.where(ytr > 0, w_pos, 1.0).astype(np.float32)

    wvec = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    best = (-1.0, -1, wvec.copy(), 0.0)
    for ep in range(args.epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ wvec + b, -30, 30)))
        g = (p - ytr) * w
        wvec -= args.lr * (Xtr.T @ g / len(ytr) + 1e-4 * wvec)
        b -= args.lr * float(g.mean())
        a = auc(yva, 1.0 / (1.0 + np.exp(-np.clip(Xva @ wvec + b, -30, 30))))
        if a > best[0]:
            best = (a, ep, wvec.copy(), b)
        if ep % 25 == 0:
            print(f"  epoch {ep:3d}  val AUC = {a:.4f}")
    val_auc, best_ep, wvec, b = best
    print(f"  selected epoch {best_ep} by validation (val AUC {val_auc:.4f})")

    st = 1.0 / (1.0 + np.exp(-np.clip(Xte @ wvec + b, -30, 30)))
    nr = no_riichi[te]
    print()
    print("=== RESULTS (test set, model chosen on validation) ===")
    print(f"  overall      AUC = {auc(yte, st):.4f}   (n={len(te):,}, pos={int(yte.sum())})")
    print(f"  NO-RIICHI    AUC = {auc(yte[nr], st[nr]):.4f}"
          f"   (n={int(nr.sum()):,}, pos={int(yte[nr].sum())})   <- the key slice")
    print(f"  WITH-RIICHI  AUC = {auc(yte[~nr], st[~nr]):.4f}"
          f"   (n={int((~nr).sum()):,}, pos={int(yte[~nr].sum())})")
    print()
    print("  logs baseline : overall 0.6996 | NO-RIICHI 0.6838 | WITH-RIICHI 0.5872")

    # Persist for --danger-model. Store the standardisation alongside the weights
    # so inference does not depend on the training data still being on disk.
    out = args.save
    np.savez_compressed(
        out,
        w=wvec.astype(np.float32),
        b=np.float32(b),
        mu=Xtr_mu.astype(np.float32),
        sd=Xtr_sd.astype(np.float32),
        players=np.int32(4),
        dim=np.int32(X.shape[1]),
        auc_overall=np.float32(auc(yte, st)),
        auc_no_riichi=np.float32(auc(yte[nr], st[nr])),
        val_auc=np.float32(val_auc),
        selected_epoch=np.int32(best_ep),
    )
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
