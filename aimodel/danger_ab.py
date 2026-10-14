"""危险的「手写版 vs 学出来的」离线 A/B。

为什么是离线对比、而不是 `ab.py` 各 300 场
--------------------------------------------
`ab.py` 比的是**两份策略权重**（`--a/--b`）。危险度不是一个策略，它是喂给教师与
网络的一路特征；要让 300 场对局反映出危险度的差别，得先用新危险度**重训一版策略**
再对局 —— 那是几小时的事，而且中间任何一处不同都会污染结论，**问不出「哪个危险度更准」**。

而这个问题可以**直接在已有数据上量**：

* 手写版的分数**已经在数据里** —— `ACTION_BLOCKS` 里有 `("danger", 1)`，
  所以 `sp_big` shard 的 `d_actfeat[i, tile, ACTION_OFFSET["danger"]]` 就是它；
* 学出来的分数由 `danger_model.npz` 给出；
* 标签是 `d_deal_in`（这一手是不是真的放铳了）。

于是同一批样本上比 AUC 即可 —— **口径完全一致，不掺策略强弱**。

⚠️ 这不是 `ab.py` 那种对局级 A/B，**不能替代它**；它回答的是更前置的问题：
「学出来的危险度到底比手写版准多少」。若这里都没有优势，跑 300 场也没有意义。

用法::

    python -m aimodel.danger_ab <shard_glob> --model <npz> [--limit N]
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from .encode import ACTION_OFFSET
from .train_danger_baseline import auc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="danger score offline A/B")
    ap.add_argument("pattern", help="glob for self-play shards")
    ap.add_argument("--model", default="_data/danger_model.npz")
    ap.add_argument("--limit", type=int, default=None, help="max rows to score")
    ap.add_argument("--holdout", type=float, default=0.6,
                    help="只在这个比例的后段上比较 —— 模型训练时用的是前 60%%，"
                         "全量比较会让学出来的模型占便宜（它见过那些样本）")
    args = ap.parse_args(argv)

    m = np.load(args.model, allow_pickle=True)
    w, b = m["w"].astype(np.float32), float(m["b"])
    mu, sd = m["mu"][0].astype(np.float32), m["sd"][0].astype(np.float32)
    model_dim, model_players = int(m["dim"]), int(m["players"])
    if model_players != 4:
        raise SystemExit(f"模型是 {model_players} 人的，本脚本按四麻比较")

    hand_s, learn_s, ys = [], [], []
    for f in sorted(glob.glob(args.pattern)):
        z = np.load(f, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = z["d_deal_in"].astype(np.int8)
        kind = z["kind"]
        disc = np.flatnonzero(kind == "discard")
        if len(disc) != len(y):
            continue
        Xd = X[disc]
        if Xd.shape[1] != model_dim:
            continue
        # 手写版：数据里已经算好的那一列
        A = z["d_actfeat"].astype(np.float32)          # (n, 34, ACTION_DIM)
        tile = z["d_tile_onehot"].astype(np.int8).argmax(axis=1)
        hs = A[np.arange(len(tile)), tile, ACTION_OFFSET["danger"]]
        # 学出来的
        xs = (Xd - mu) / sd
        zz = np.clip(xs @ w + b, -30, 30)
        ls = 1.0 / (1.0 + np.exp(-zz))
        hand_s.append(hs.astype(np.float32))
        learn_s.append(ls.astype(np.float32))
        ys.append(y)
        if args.limit and sum(len(v) for v in ys) >= args.limit:
            break

    H = np.concatenate(hand_s)
    L = np.concatenate(learn_s)
    Y = np.concatenate(ys).astype(np.float32)
    if args.limit:
        H, L, Y = H[:args.limit], L[:args.limit], Y[:args.limit]

    print(f"rows {len(Y):,}   positives {int(Y.sum()):,} ({Y.mean()*100:.3f}%)")
    print()
    print("=== AUC on the SAME samples (higher = better danger ranking) ===")
    a_hand = auc(Y, H)
    a_learn = auc(Y, L)
    print(f"  [all rows]   hand-written {a_hand:.4f}   learned {a_learn:.4f}"
          f"   diff {a_learn - a_hand:+.4f}")
    print("               (the learned model saw the first 60% of these during"
          " training, so this favours it)")

    # 公平比较：只在模型没见过的后段上算。注意模型训练时是按
    # train(60)/val(20)/test(20) 切的，这里取后 40% 就避开了它的训练集。
    cut = int(len(Y) * args.holdout)
    if 0 < cut < len(Y):
        Yh, Hh, Lh = Y[cut:], H[cut:], L[cut:]
        ah, al = auc(Yh, Hh), auc(Yh, Lh)
        print()
        print(f"=== FAIR: held-out rows only (last {1 - args.holdout:.0%}) ===")
        print(f"  rows {len(Yh):,}   positives {int(Yh.sum()):,} ({Yh.mean()*100:.3f}%)")
        print(f"  hand-written tile_danger : {ah:.4f}")
        print(f"  learned model            : {al:.4f}")
        print(f"  difference               : {al - ah:+.4f}")
    else:
        al, ah = a_learn, a_hand

    print()
    print("  注：本批自我对弈中处于立直状态的决策极少，所以上面基本等同于")
    print("      「没人立直」那一片；牌谱那边该片的正样本只有 316 条，不适合做此对比。")
    print()
    if al > ah:
        print(f"  结论：学出来的危险度**更准**（+{al - ah:.4f} AUC，留出集）。")
        print("        但这只是**排序质量**；要主张「换成它后打得更强」，仍需")
        print("        用新危险度重训一版策略再跑 ab.py 配对对局。")
    else:
        print(f"  结论：学出来的危险度**没有更准**（{al - ah:+.4f}），"
              "        不该切换默认。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
