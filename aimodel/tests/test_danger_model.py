"""危险度模型接入的自检（``--danger-model`` 那条路）。

锁住三件容易**静默**出错的事 —— 它们都不会抛异常，只会让结果悄悄变错：

1. **`danger_score` 必须与训练时的输入布局一致。** 这一点是构造性成立的：
   `selfplay.py` 记的 `X`、`logdata.py` 抽的 `danger_X`、以及这里现算的，
   调用的都是同一个 `encode.encode(view)`。本测试把这条**钉死**——
   将来谁改了其中一处，这里会立刻红。
2. **人数 / 维度不匹配时必须退回手写路径**，绝不能把 560 维的权重乘到
   486 维向量上（不报错、结果全错）。
3. **没载入模型时行为完全不变**（旧的手写 `tile_danger` 路径仍是默认）。
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from aimodel import encode, sanma as S  # noqa: E402
from aimodel.engine import Game, View  # noqa: E402


def _live_view(players: int, seed: int = 0) -> View:
    """造一个**结构合法**的 View（`View` 的字段全部必填，没有默认值）。

    不去 spy `encode`：`RandomAgent` 根本不调它，spy 抓不到东西（我试过，
    抓到 0 个 —— 那次「抓到真实 View」的检查红了一次，就是因为这个）。
    这里直接把牌摆好，保证 `encode(view)` 能跑通即可；
    本测试要验的是**人数/维度守卫**，不是盘面本身。
    """
    rule = S.rule_for(players)
    n = players
    hand = [0] * 34
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        hand[t] += 1
    return View(
        player=0,
        hand=hand,
        drawn=hand.index(1) if 1 in hand else 0,
        melds=[],
        hand_red=[0] * 34,
        discards=[[0] * 34 for _ in range(n)],
        melds_all=[[] for _ in range(n)],
        kita=[0] * n,          # 每家一个计数，长度必须是人数（不是 34）
        scores=[25000] * n,
        dora_indicators=[0],
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=70,
        turn=1,
        riichi=[False] * n,
        ippatsu=[False] * n,
        is_dealer=True,
        last_discard=None,
        rule=rule,
    )


def _dummy_model(dim: int, players: int) -> str:
    """写一个合法的空模型（权重全 0），只为走通接入逻辑。"""
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez_compressed(
        path,
        w=np.zeros(dim, dtype=np.float32),
        b=np.float32(0.0),
        mu=np.zeros((1, dim), dtype=np.float32),
        sd=np.ones((1, dim), dtype=np.float32),
        dim=np.int32(dim),
        players=np.int32(players),
        auc_overall=np.float32(0.0),
        auc_no_riichi=np.float32(0.0),
        val_auc=np.float32(0.0),
        selected_epoch=np.int32(0),
    )
    return path


def main() -> int:
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # --- 1. 布局：三处用的是同一个 encode -----------------------------
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = open(os.path.join(src_root, "selfplay.py"), encoding="utf-8").read()
    ld = open(os.path.join(src_root, "logdata.py"), encoding="utf-8").read()
    check("selfplay 记 X 用的是 encode(view)", 'encode(view)' in sp)
    check("logdata 抽 danger_X 用的是 encode(view)", 'encode(view)' in ld)

    # --- 2. 人数/维度校验 --------------------------------------------
    O4, D4 = encode.layout_for(4)
    O3, D3 = encode.layout_for(3)
    # 2026-10-21：末尾追加 match_state 17 / called_tile 34 / last_discard 34
    # -> 3p 571、4p 645。这里**不写死新 DIM**，改成跟随布局；
    # 同时钉住「只追加、老块一个都没挪」：老最后一块 tenpai 的 offset 不变。
    check("四麻布局与 feature_blocks 求和一致",
          D4 == sum(w for _, w in encode.feature_blocks(4)), f"D4={D4}")
    check("三麻布局与 feature_blocks 求和一致",
          D3 == sum(w for _, w in encode.feature_blocks(3)), f"D3={D3}")
    check("四麻老块 offset 未变（tenpai 仍是 559）", O4.get("tenpai") == 559,
          f"tenpai@{O4.get('tenpai')}")
    check("三麻老块 offset 未变（tenpai 仍是 485）", O3.get("tenpai") == 485,
          f"tenpai@{O3.get('tenpai')}")

    path4 = _dummy_model(D4, 4)
    try:
        encode.load_danger_model(path4)
        check("载入四麻模型成功", encode.danger_model_loaded())

        view4 = _live_view(4)
        if view4 is None:
            check("抓到真实四麻 View", False, "play_hand 没有产生 encode 调用")
        else:
            check("抓到真实四麻 View", True,
                  f"hand={sum(view4.hand)}张 melds={len(view4.melds)}")
            # 人数对、维度对 → 应当用模型（权重全 0 → 0.5）
            d = encode.danger_score(view4, next(i for i, c in enumerate(view4.hand) if c))
            check("四麻局面用模型（全 0 权重 → 0.5）", abs(d - 0.5) < 1e-6, f"got {d}")

        view3 = _live_view(3)
        if view3 is not None:
            t = next(i for i, c in enumerate(view3.hand) if c)
            d3 = encode.danger_score(view3, t)
            hand = encode.tile_danger(view3, t)
            check("三麻局面退回手写 tile_danger", abs(d3 - hand) < 1e-9,
                  f"model={d3:.4f} hand={hand:.4f}")
    finally:
        os.unlink(path4)

    # --- 3. 卸载模型后行为回到手写 -----------------------------------
    encode._DANGER_MODEL = None
    check("卸载后 danger_model_loaded() 为 False", not encode.danger_model_loaded())
    v = _live_view(4)
    if v is not None:
        t = next(i for i, c in enumerate(v.hand) if c)
        check("没模型时 danger_score == tile_danger",
              abs(encode.danger_score(v, t) - encode.tile_danger(v, t)) < 1e-12)

    print()
    if fails:
        print(f"❌ {len(fails)} 项失败：" + "、".join(fails))
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
