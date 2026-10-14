"""牌谱管线：MJAI 读写、重放对拍、数据抽取。

这一套的分量在于**它证明了三件事**：

1. 导出的牌谱能还原成牌山、重放回引擎，结果（和牌者 / 和牌张 / 番符 / 役种 /
   点数收支）与牌谱逐项一致 —— 也就是说，等真实牌谱到手，「拿牌谱验规则」
   这条链路是通的
2. 从牌谱抽出来的训练数据和现有网络**特征维度一致**，能直接进训练管线
3. 那些容易搞错的地方（拔北的岭上补牌、吃的 pai 语义、立直后自动摸切）
   都有回归用例钉住

自检里抓出来过的 5 个真 bug，就是这套用例的来源：

* 牌山没补齐到整副张数 → 王牌错位、发牌全乱
* 拔北（不只是杠）也从牌山尾部补牌 → 漏了它，第一个拔北之后全错位
* 立直后引擎自动摸切、不问 agent → 靠提问推进队列会让牌谱队列越积越多
* 副露 / 荣和只看动作种类、不核对是哪张牌 → 引擎递什么牌都照单全收
* 吃的 ``pai`` 是「被吃的那张」，不是顺子最小的那张
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aimodel import encode, engine as E, logcheck as LC, logdata as LD, mjai  # noqa: E402
from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402


# ---------------------------------------------------------------------------
# ① 牌名转换
# ---------------------------------------------------------------------------
def test_tile_names():
    bad = []
    for t in range(34):
        for red in (False, True):
            if red and (t % 9 != 4 or t >= 27):
                continue
            name = mjai.tile_to_mjai(t, red)
            t2, r2 = mjai.tile_from_mjai(name)
            if t2 != t or r2 != red:
                bad.append((t, red, name, t2, r2))
    assert not bad, f"牌名往返不一致：{bad[:5]}"
    assert mjai.tile_to_mjai(4, True) == "0m"
    assert mjai.tile_to_mjai(S.NORTH) == "N"
    assert mjai.tile_to_mjai(S.CHUN) == "C"
    print("  ✅ 34 种牌 × 赤 5 的牌名往返一致")


# ---------------------------------------------------------------------------
# ② 重放对拍（核心）
# ---------------------------------------------------------------------------
def _export(seed: int, n: int) -> list[dict]:
    acts: list[dict] = []
    game = E.Game([HeuristicAgent(seed * n + i) for i in range(n)],
                  seed=seed, rule=S.rule_for(n))
    game.on_action = acts.append
    game.play()
    return mjai.export(acts, players=n)


def test_replay_roundtrip_3p(games: int = 4):
    reports = []
    for seed in range(games):
        for kyoku in mjai.iter_kyoku(_export(seed, 3)):
            reports.append(LC.check_kyoku(kyoku, n=3))
    bad = [r for r in reports if not r.ok]
    assert not bad, f"{len(bad)}/{len(reports)} 局对拍不一致，例如：{bad[0].line()}"
    print(f"  ✅ 三麻：{len(reports)} 局自己导出的牌谱，重放后逐项一致")


def test_replay_roundtrip_4p(games: int = 4):
    """四麻要单独测：吃、王牌 14 张、三色同顺都只在四麻出现。"""
    reports = []
    for seed in range(games):
        for kyoku in mjai.iter_kyoku(_export(seed, 4)):
            reports.append(LC.check_kyoku(kyoku, n=4))
    bad = [r for r in reports if not r.ok]
    assert not bad, f"{len(bad)}/{len(reports)} 局对拍不一致，例如：{bad[0].line()}"
    print(f"  ✅ 四麻：{len(reports)} 局自己导出的牌谱，重放后逐项一致")


def test_wall_reconstruction():
    """牌山必须补齐到整副张数 —— 少一张王牌就会整体错位。"""
    for n, total in ((3, 108), (4, 136)):
        kyoku = list(mjai.iter_kyoku(_export(0, n)))[0]
        wall = LC.kyoku_to_wall(kyoku, n, S.rule_for(n))
        assert len(wall) == total, f"{n} 麻牌山 {len(wall)} 张，应为 {total}"
        # 发牌区：按座位依次发 13 张，要和牌谱的起手对得上
        hs = S.rule_for(n).hand_size
        for p in range(n):
            got = sorted(t for t, _ in wall[p * hs:(p + 1) * hs])
            want = sorted(mjai.tile_from_mjai(x)[0] for x in kyoku[0]["tehais"][p])
            assert got == want, f"P{p} 发牌区与牌谱起手不符"
    print("  ✅ 牌山补齐到整副张数，发牌区与牌谱起手一致")


# ---------------------------------------------------------------------------
# ③ 数据抽取
# ---------------------------------------------------------------------------
def test_extract_shapes():
    n = 3
    samples = LD.Samples()
    for seed in range(2):
        for kyoku in mjai.iter_kyoku(_export(seed, n)):
            LD.extract_kyoku(kyoku, n=n, samples=samples)
    data = samples.arrays()
    dim = encode.layout_for(n)[1]
    assert data["X"].shape[1] == dim, f"特征维度 {data['X'].shape[1]} ≠ 网络要求 {dim}"
    assert len(data["X"]) == len(data["action"]) == len(data["kind"])
    assert data["X"].shape[0] > 100, "抽出来的样本太少，像是没跑通"
    if len(data["danger_tile"]):
        assert data["danger_X"].shape[1] == dim
        assert len(data["danger_dealt"]) == len(data["danger_tile"])
        assert data["danger_dealt"].sum() <= len(data["danger_tile"])
    print(f"  ✅ 抽出 {len(data['X'])} 条决策样本、{len(data['danger_tile'])} 条危险度样本"
          f"（维度 {dim}）")


def test_pipeline_matches_training():
    """抽出来的动作下标必须落在对应决策头的合法范围内。"""
    samples = LD.Samples()
    for kyoku in mjai.iter_kyoku(_export(1, 3)):
        LD.extract_kyoku(kyoku, n=3, samples=samples)
    data = samples.arrays()
    for i, kind in enumerate(data["kind"]):
        head = encode.HEAD_SIZE[kind]
        assert 0 <= data["action"][i] < head, f"{kind} 的动作下标 {data['action'][i]} 越界"
    print("  ✅ 动作下标全部落在对应决策头的范围内")


def main() -> None:
    print("【牌谱管线】")
    test_tile_names()
    test_wall_reconstruction()
    test_replay_roundtrip_3p()
    test_replay_roundtrip_4p()
    test_extract_shapes()
    test_pipeline_matches_training()
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
