"""Baseline "will this discard deal in?" classifier, trained on Tenhou logs.

Purpose
-------
`encode.tile_danger()` is a hand-written constant model (middles 0.62, terminals
0.22, honour tiles by remaining count, suji 0.65x). Its documented blind spot is
that it has no defence at all when nobody has declared riichi. This trains a
model from the extracted logs to see whether a learned danger score can do
better on exactly that slice.

Data comes from `aimodel/logdata.py` over `GL1.xlsm` (real Tenhou 4-player
hands), filtered to faithful replays by `aimodel/filter_faithful.py`.

This is deliberately a LOGISTIC REGRESSION, not the full network: it is trained
with plain numpy, it cannot overfit 42k x 560 easily, and it gives a clean
baseline number. A fancier model is only worth building if this one shows real
signal on the no-riichi slice.

Usage
-----
    python aimodel/train_danger_baseline.py <npz> [--epochs N] [--lr X]
"""

from __future__ import annotations

import argparse

import numpy as np

from .encode import layout_for


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney), tie-aware."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return [(a - mu) / sd for a in (train, *others)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="danger-model baseline")
    ap.add_argument("npz", help="npz from aimodel.logdata")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    z = np.load(args.npz, allow_pickle=True)
    X = z["danger_X"].astype(np.float32)
    y = z["danger_dealt"].astype(np.float32)
    print(f"data: {X.shape[0]:,} rows, {X.shape[1]} dims, "
          f"{int(y.sum()):,} positives ({y.mean() * 100:.2f}%)")

    off, _ = layout_for(4)
    ri = X[:, off["riichi"]:off["riichi"] + 4].sum(axis=1)
    no_riichi = ri == 0

    # Three-way split. The positive rate is only 1.23% (~521 positives total), so
    # a weighted logistic on this data overfits within a few passes -- measured:
    # test AUC peaks at epoch 0 and then falls monotonically. Picking the best
    # epoch by looking at TEST would leak, so model selection happens on VAL and
    # TEST is touched exactly once, at the end.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_tr = int(len(X) * 0.6)
    n_va = int(len(X) * 0.2)
    tr = perm[:n_tr]
    va = perm[n_tr:n_tr + n_va]
    te = perm[n_tr + n_va:]

    Xtr, Xva, Xte = standardise(X[tr], X[va], X[te])
    ytr, yva, yte = y[tr], y[va], y[te]

    # class weights: positives are ~1.2%, so weight them up
    w_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)
    w = np.where(ytr > 0, w_pos, 1.0).astype(np.float32)
    print(f"  split: train {len(tr):,} (pos {int(ytr.sum())})  "
          f"val {len(va):,} (pos {int(yva.sum())})  test {len(te):,} (pos {int(yte.sum())})")

    wvec = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    best = (-1.0, 0, wvec.copy(), b)          # (val AUC, epoch, weights, bias)
    for ep in range(args.epochs):
        p = 1.0 / (1.0 + np.exp(-(Xtr @ wvec + b)))
        g = (p - ytr) * w
        wvec -= args.lr * (Xtr.T @ g / len(ytr) + 1e-4 * wvec)
        b -= args.lr * float(g.mean())

        # early stopping / model selection on VALIDATION only
        sv = 1.0 / (1.0 + np.exp(-(Xva @ wvec + b)))
        a = auc(yva, sv)
        if a > best[0]:
            best = (a, ep, wvec.copy(), b)
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}  val AUC = {a:.4f}"
                  + ("   <- best so far" if ep == best[1] else ""))

    val_auc, best_ep, wvec, b = best
    print(f"  selected epoch {best_ep} by validation (val AUC {val_auc:.4f})")

    # TEST is evaluated once, with the val-selected model
    s_te = 1.0 / (1.0 + np.exp(-(Xte @ wvec + b)))
    no_ri_te = no_riichi[te]
    print()
    print("=== RESULTS (test set, model chosen on validation) ===")
    print(f"  overall      AUC = {auc(yte, s_te):.4f}   (n={len(yte):,}, pos={int(yte.sum())})")
    if no_ri_te.any():
        print(f"  NO-RIICHI    AUC = {auc(yte[no_ri_te], s_te[no_ri_te]):.4f}"
              f"   (n={int(no_ri_te.sum()):,}, pos={int(yte[no_ri_te].sum())})"
              "   <- the hand-written model's blind spot")
    with_ri = ~no_ri_te
    if with_ri.any():
        print(f"  WITH-RIICHI  AUC = {auc(yte[with_ri], s_te[with_ri]):.4f}"
              f"   (n={int(with_ri.sum()):,}, pos={int(yte[with_ri].sum())})")
    print("  (AUC 0.5 = no information)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
