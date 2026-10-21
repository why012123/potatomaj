#!/usr/bin/env python
"""``tenhou_bc --layout {v2,v3}`` 的单元测试（dot V1 路线书第 ③ 步）。

钉死四件事：

1. **默认 v2、v2 走旧列**：``build_arg_parser()`` 不带参数时 ``layout=="v2"``；
   v2 的 tracker 就是原来的 ``agents.Tracker``，``arrays()`` 的列（X / mask /
   actfeat / action / logp / kind / player / chipat）一个不多一个不少。
2. **v3 列**：``tiles (n,34,20)`` + ``globals (n,21)``，**没有 X**；其余列与 v2
   逐位相同；``tiles[0] == encode_tile(view)[0]``。
3. **加载器门禁**：``validate_dataset`` / ``load_shard`` 能自动判 v2/v3；
   把 v3 的 F 从 20 改成 19（或 34 改成 35、globals 改短）必须 ``ValueError``。
4. ``--shard`` 的数据形状：v3 tracker 的 ``arrays()`` 已经是
   ``(N,34,20)``，分片只是把这个 N 切开（真实分片跑见交付报告）。

跑法::

    .venv/Scripts/python.exe aimodel/tests/test_tenhou_bc_layout.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import Tracker as BaseTracker  # noqa: E402
from aimodel.encode import HEAD_SIZE, action_mask, encode  # noqa: E402
from aimodel.encode_tile import F as TILE_F  # noqa: E402
from aimodel.encode_tile import G as GLOBAL_G  # noqa: E402
from aimodel.encode_tile import encode_tile  # noqa: E402
from aimodel.engine import Decision, View  # noqa: E402
from aimodel.tenhou_bc import (  # noqa: E402
    LAYOUTS,
    V3Tracker,
    _record_decision,
    build_arg_parser,
    load_shard,
    make_tracker,
    validate_dataset,
)

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


def expect_value_error(fn, label: str) -> bool:
    try:
        fn()
    except ValueError as e:
        print(f"  ✅ {label}: ValueError（{str(e)[:70]}…）")
        return True
    except Exception as e:  # noqa: BLE001
        check(False, f"{label}: 抛了 {type(e).__name__} 而不是 ValueError: {e}")
        return False
    check(False, f"{label}: **没有报错**（负向验证失败）")
    return False


def counts(tiles: list[int]) -> list[int]:
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    return c


def make_view() -> View:
    """一个固定的四麻中盘 View（和 test_encode_tile 的 4p_mid 同构）。"""
    rule = S.rule_for(4)
    return View(
        player=1,
        hand=counts([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 13]),
        drawn=13,
        melds=[],
        hand_red=[0] * 34,
        discards=[[31, 31, 4], [13, 33], [22, 22, 27], [30, 5, 6]],
        melds_all=[
            [S.Meld("pon", 31, from_player=1)],
            [],
            [S.Meld("chi", 9, from_player=3, called=10)],
            [S.Meld("pon", 28, from_player=0)],
        ],
        kita=[0, 0, 0, 0],
        scores=[25000, 32000, 21000, 22000],
        dora_indicators=[8, 30],
        round_wind=S.EAST,
        seat_wind=S.SOUTH,
        wall_left=40,
        turn=7,
        riichi=[False, False, True, True],
        ippatsu=[False, False, False, False],
        is_dealer=False,
        last_discard=(2, 27),
        rule=rule,
        furiten=True,
        kyoku=2,
        honba=1,
        kyotaku=1,
        dealer_repeat=True,
        hands_left=3,
        ranks=(3, 1, 4, 2),
        tsumogiri=(
            (False, False, True),
            (False, True),
            (True, False, False),
            (False, False, False),
        ),
        called_tile=None,
    )


def decision() -> Decision:
    # options 顺序故意不是升序：action_index 必须把它映射成牌型 13
    return Decision("discard", [13, 0, 1], tile=None)


def record_one(layout: str, tracker=None) -> tuple[dict, View, Decision]:
    view = make_view()
    dec = decision()
    if tracker is None:
        tracker = make_tracker(layout)
    _record_decision(tracker, layout, dec, view, 0)
    data = tracker.arrays()
    # flush() 才会加 ret；单测这里补一列零，模拟真实分片
    n = len(data["X"]) if "X" in data else len(data["tiles"])
    data["ret"] = np.zeros(n, dtype=np.float32)
    return data, view, dec


def main() -> int:
    print("=" * 76)
    print("tenhou_bc layout 单元测试（v2 默认 / v3 per-tile）")
    print("=" * 76)

    # ------------------------------------------------------------------
    print("\n== 1. CLI：默认 v2、可选 v3、v4 必须被 argparse 拒 ==")
    check(LAYOUTS == ("v2", "v3"), f"LAYOUTS 应为 ('v2','v3')，实际 {LAYOUTS}")
    parser = build_arg_parser()
    check(parser.parse_args([]).layout == "v2",
          f"不带 --layout 时必须默认 v2，实际 {parser.parse_args([]).layout}")
    check(parser.parse_args(["--layout", "v3"]).layout == "v3",
          "--layout v3 没被接受")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--layout", "v4"])
            check(False, "--layout v4 应该被 argparse 拒绝")
        except SystemExit:
            print("  ✅ --layout v4 被 argparse 拒绝（choices 生效）")
    check(isinstance(make_tracker("v2"), BaseTracker) and type(make_tracker("v2")) is BaseTracker,
          "v2 必须返回原来的 agents.Tracker（不是子类）")
    check(isinstance(make_tracker("v3"), V3Tracker), "v3 必须返回 V3Tracker")
    expect_value_error(lambda: make_tracker("v9"), "未知 layout")

    # ------------------------------------------------------------------
    print("\n== 2. v2：列与旧行为一致（X 必须等于 encode(view)）==")
    v2, view, dec = record_one("v2")
    check(len(v2["X"]) == 1, f"v2 行数 {len(v2['X'])} != 1")
    check(tuple(v2["X"].shape) == (1, len(encode(view))),
          f"v2 X 形状 {v2['X'].shape} != (1,{len(encode(view))})")
    check(v2["X"].dtype == np.float32, f"v2 X dtype {v2['X'].dtype} != float32")
    check(np.array_equal(v2["X"][0], encode(view)),
          "v2 的 X 与 encode(view) 不是逐位相同")
    for k in ("mask", "actfeat", "action", "logp", "kind", "player", "chipat"):
        check(k in v2, f"v2 缺少旧列 {k}")
    check("tiles" not in v2 and "globals" not in v2, "v2 不应出现 tiles/globals")
    check(np.array_equal(v2["mask"][0], action_mask(dec)),
          "v2 mask 与 action_mask 不一致")
    check(int(v2["action"][0]) == 13, f"v2 action 应为 13，实际 {v2['action'][0]}")
    check(v2["kind"][0] == "discard" and int(v2["player"][0]) == 1,
          "v2 kind/player 错")
    check(int(v2["chipat"][0]) == -1, "v2 chipat 默认应为 -1")
    print("  ✅ v2 列、X、mask、action、kind、player、chipat 全部对")

    # ------------------------------------------------------------------
    print("\n== 3. v3：tiles/globals + 其余列与 v2 逐位相同、没有 X ==")
    v3, _, _ = record_one("v3")
    check("X" not in v3, "v3 不应包含 X（避免同一份信息存两遍）")
    check(tuple(v3["tiles"].shape) == (1, 34, TILE_F),
          f"v3 tiles 形状 {v3['tiles'].shape} != (1,34,{TILE_F})")
    check(tuple(v3["globals"].shape) == (1, GLOBAL_G),
          f"v3 globals 形状 {v3['globals'].shape} != (1,{GLOBAL_G})")
    check(v3["tiles"].dtype == np.float32 and v3["globals"].dtype == np.float32,
          "v3 tiles/globals dtype 应为 float32")
    exp_tiles, exp_glob = encode_tile(view)
    check(np.array_equal(v3["tiles"][0], exp_tiles),
          "v3 tiles[0] 与 encode_tile(view)[0] 不是逐位相同")
    check(np.array_equal(v3["globals"][0], exp_glob),
          "v3 globals[0] 与 encode_tile(view)[1] 不是逐位相同")
    for k in ("mask", "actfeat", "action", "logp", "kind", "player", "chipat", "ret"):
        check(k in v3, f"v3 缺少列 {k}")
        check(np.array_equal(np.asarray(v3[k]), np.asarray(v2[k])),
              f"v3 的 {k} 与 v2 不一致（其余列必须保持不变）")
    print("  ✅ v3 tiles/globals 与 encode_tile 逐位一致；其余列与 v2 相同")

    # -- 两行 / N 行的 stacking（--shard 写盘前就是这个形状）------------
    t3 = make_tracker("v3")
    for _ in range(3):
        _record_decision(t3, "v3", decision(), make_view(), 0)
    d3 = t3.arrays()
    check(tuple(d3["tiles"].shape) == (3, 34, TILE_F),
          f"多行 v3 tiles 形状 {d3['tiles'].shape} != (3,34,{TILE_F})")
    check(tuple(d3["globals"].shape) == (3, GLOBAL_G),
          f"多行 v3 globals 形状 {d3['globals'].shape} != (3,{GLOBAL_G})")
    print(f"  ✅ N=3 时 tiles{d3['tiles'].shape} / globals{d3['globals'].shape}（--shard 的基础）")

    # ------------------------------------------------------------------
    print("\n== 4. 加载器：自动判 layout + F=19 必须 ValueError ==")
    check(validate_dataset(v2) == "v2", "validate_dataset(v2) 应返回 'v2'")
    check(validate_dataset(v3) == "v3", "validate_dataset(v3) 应返回 'v3'")
    expect_value_error(lambda: validate_dataset(v2, expect_layout="v3"),
                       "把 v2 当 v3 加载")
    expect_value_error(lambda: validate_dataset(v3, expect_layout="v2"),
                       "把 v3 当 v2 加载")

    bad = dict(v3)
    bad["tiles"] = v3["tiles"][:, :, :-1]        # F: 20 -> 19（正是用户要求的负向）
    expect_value_error(lambda: validate_dataset(bad), "v3 tiles F=19")
    bad2 = dict(v3)
    bad2["tiles"] = v3["tiles"][:, :-1, :]        # 34 -> 33
    expect_value_error(lambda: validate_dataset(bad2), "v3 tiles 牌位=33")
    bad3 = dict(v3)
    bad3["tiles"] = v3["tiles"][:, :, :]          # 保持 20
    bad3["globals"] = v3["globals"][:, :-1]       # G: 21 -> 20
    expect_value_error(lambda: validate_dataset(bad3), "v3 globals G=20")
    bad4 = dict(v3)
    bad4["X"] = np.zeros((1, 645), dtype=np.float32)
    expect_value_error(lambda: validate_dataset(bad4), "v3 同时带 X")
    bad5 = dict(v2)
    bad5["X"] = np.zeros((2, 645), dtype=np.float32)   # 行数不一致
    expect_value_error(lambda: validate_dataset(bad5), "v2 列长度不一致")

    # 真实写盘再读：坏 F=19 的 npz 必须被 load_shard 拒
    with tempfile.TemporaryDirectory() as td:
        good_path = Path(td) / "v3_good.npz"
        np.savez_compressed(good_path, **v3)
        layout, got = load_shard(good_path)
        check(layout == "v3", f"load_shard(v3) layout={layout}")
        check(np.array_equal(got["tiles"], v3["tiles"]), "load_shard 读回的 tiles 不一致")

        bad_path = Path(td) / "v3_bad_f19.npz"
        np.savez_compressed(bad_path, **bad)
        expect_value_error(lambda: load_shard(bad_path), "load_shard 读到 F=19 的 npz")

        v2_path = Path(td) / "v2_good.npz"
        np.savez_compressed(v2_path, **v2)
        layout2, got2 = load_shard(v2_path)
        check(layout2 == "v2", f"load_shard(v2) layout={layout2}")
        check(np.array_equal(got2["X"], v2["X"]), "load_shard 读回的 X 不一致")

    print("\n" + "=" * 76)
    if FAILED:
        print(f"❌ {len(FAILED)} 条失败")
        for m in FAILED:
            print(f"  - {m.splitlines()[0]}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
