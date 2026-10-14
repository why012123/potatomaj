"""Danger-model, non-linear, using the per-tile action features at the thrown tile.

The logistic baseline (`train_danger_baseline.py`) reads only `danger_X` (the
observation) and reaches test AUC 0.6996 overall / 0.6838 on the NO-RIICHI slice
but only 0.5872 when someone HAS declared riichi. That weakness is expected: a
linear model on the observation cannot express "this specific tile, given who is
riichi and what their pond looks like" -- it needs the interaction.

`logdata.py` already stores that interaction: `danger_A` is the (34, ACTION_DIM)
per-tile action-feature block, which includes the hand-written `tile_danger`
value, genbutsu flags, suji, ukeire and shanten. Here we feed the ROW FOR THE
TILE THAT WAS ACTUALLY THROWN (`danger_tile` indexes it) into the project's own
`Net`, so the model sees both the board and the tile's own danger features.

Method (the same discipline as the baseline, learned the hard way):
  * 60/20/20 split, model selection on VALIDATION only, TEST touched once;
  * report the NO-RIICHI slice separately -- that is the whole point.

Usage:
    python -m aimodel.train_danger_net <npz> [--epochs N] [--hidden 64,64]
"""

from __future__ import annotations

import argparse

import numpy as np

from .encode import layout_for
from .net import softmax_rows


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class DangerNet:
    """Tiny 2-layer MLP with one sigmoid output. Plain numpy."""

    def __init__(self, dim: int, hidden: tuple[int, ...], seed: int = 0,
                 lr: float = 1e-2) -> None:
        rng = np.random.default_rng(seed)
        sizes = [dim, *hidden]
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])
                             ).astype(np.float32) for i in range(len(hidden))]
        self.b = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(hidden))]
        self.Wo = (rng.normal(0, np.sqrt(2.0 / sizes[-1]), (sizes[-1], 1)) * 0.1
                   ).astype(np.float32)
        self.bo = np.zeros(1, dtype=np.float32)
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in self._params()}
        self.v = {k: np.zeros_like(v) for k, v in self._params()}
        self.t = 0

    def _params(self):
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            yield f"W{i}", w
            yield f"b{i}", b
        yield "Wo", self.Wo
        yield "bo", self.bo

    def forward(self, X: np.ndarray):
        acts = []
        h = X
        for w, b in zip(self.W, self.b):
            z = h @ w + b
            h = np.maximum(z, 0.0)
            acts.append((h, z))
        logit = (h @ self.Wo + self.bo).ravel()
        return logit, acts

    def step(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        logit, acts = self.forward(X)
        p = 1.0 / (1.0 + np.exp(-logit))
        g = (p - y) * w / len(y)
        grads = {}
        gWo = acts[-1][0].T @ g[:, None]
        gbo = np.array([g.sum()], dtype=np.float32)
        gh = g[:, None] @ self.Wo.T
        for i in range(len(self.W) - 1, -1, -1):
            h, z = acts[i]
            gz = gh * (z > 0)
            grads[f"W{i}"] = (acts[i - 1][0] if i > 0 else X).T @ gz
            grads[f"b{i}"] = gz.sum(axis=0)
            if i > 0:
                gh = gz @ self.W[i].T
        grads["Wo"], grads["bo"] = gWo, gbo
        self.t += 1
        for k, p_ in self._params():
            gk = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk * gk
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            p_ -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean((p - y) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        logit, _ = self.forward(X)
        return 1.0 / (1.0 + np.exp(-logit))


def build_features(z) -> tuple[np.ndarray, np.ndarray]:
    """(observation, action-features-of-the-thrown-tile) -> one design matrix."""
    X = z["danger_X"].astype(np.float32)
    A = z["danger_A"].astype(np.float32)
    tile = z["danger_tile"].astype(np.int64)
    tile = np.clip(tile, 0, 33)
    rows = np.arange(len(X))
    chosen = A[rows, tile]                       # (n, ACTION_DIM)
    onehot = np.zeros((len(X), 34), dtype=np.float32)
    onehot[rows, tile] = 1.0
    return np.concatenate([X, chosen, onehot], axis=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="danger model (MLP)")
    ap.add_argument("npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=str, default="64,64")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    z = np.load(args.npz, allow_pickle=True)
    F = build_features(z)
    y = z["danger_dealt"].astype(np.float32)
    hidden = tuple(int(x) for x in args.hidden.split(","))
    print(f"features {F.shape}  positives {int(y.sum())} ({y.mean()*100:.2f}%)  hidden {hidden}")

    off, _ = layout_for(4)
    ri = z["danger_X"][:, off["riichi"]:off["riichi"] + 4].sum(axis=1)
    no_riichi = ri == 0

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(F))
    n_tr, n_va = int(len(F) * 0.6), int(len(F) * 0.2)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    mu, sd = F[tr].mean(0, keepdims=True), F[tr].std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    Fn = (F - mu) / sd

    ytr = y[tr]
    w_pos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)
    wgt = np.where(ytr > 0, w_pos, 1.0).astype(np.float32)

    net = DangerNet(Fn.shape[1], hidden, seed=args.seed, lr=args.lr)
    best = (-1.0, None, None, 0)
    for ep in range(args.epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(tr), args.batch):
            idx = tr[order[s:s + args.batch]]
            net.step(Fn[idx], y[idx], wgt[order[s:s + args.batch]])
        a = auc(y[va], net.predict(Fn[va]))
        if a > best[0]:
            snap = ([w.copy() for w in net.W], [b.copy() for b in net.b],
                    net.Wo.copy(), net.bo.copy())
            best = (a, ep, snap, 0)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val AUC = {a:.4f}")

    val_auc, best_ep, snap, _ = best
    net.W, net.b, net.Wo, net.bo = snap
    print(f"  selected epoch {best_ep} by validation (val AUC {val_auc:.4f})")

    st = net.predict(Fn[te])
    nr = no_riichi[te]
    print()
    print("=== RESULTS (test set, model chosen on validation) ===")
    print(f"  overall      AUC = {auc(y[te], st):.4f}   (n={len(te):,}, pos={int(y[te].sum())})")
    print(f"  NO-RIICHI    AUC = {auc(y[te][nr], st[nr]):.4f}"
          f"   (n={int(nr.sum()):,}, pos={int(y[te][nr].sum())})   <- blind spot")
    print(f"  WITH-RIICHI  AUC = {auc(y[te][~nr], st[~nr]):.4f}"
          f"   (n={int((~nr).sum()):,}, pos={int(y[te][~nr].sum())})")
    print("  baseline (logistic, observation only): overall 0.6996 / no-riichi 0.6838 / with-riichi 0.5872")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
