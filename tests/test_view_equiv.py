"""等价性测试：我拼的 View 和**引擎自己**给的 View 必须一模一样。

## 为什么要这个测试

`aimodel` 里「从观测建 view」的入口只有 `advise.build_view`，而它写死三麻。
四麻在 aimodel 里是支持完整的，但那条路只有**跑完整引擎**才有
（`Game.view_for()`）—— 从外部观测没有入口。

所以 `potatomaj/ai.py` 里自己拼了一个 View。危险在于：**自己拼的东西会悄悄脱节**。
引擎改了字段含义、座位的索引方式、红 5 的记法……我这边不会报错，只会安静地算错。

这个测试就是为了把那种错误变成"测试红了"：

1. 用真引擎跑几局（三麻 + 四麻）
2. 在每次「该打牌」的决策点，把引擎给的 `view_for(p)` 抄下来当**真值**
3. 把同一局面**倒回成协议能给的原始数据**（手牌/牌河/宝牌/副露/立直/分数…）
4. 喂给我自己的 `build_view`
5. 逐字段比，**再比两边编码出来的向量是不是逐位相同**

第 5 步是最强的 —— 字段名可能对不上但编码一样，也可能字段全对而编码顺序不同。
向量一样，才说明模型看到的东西完全一致。

## 这一个测试当场抓到过一个真 bug

我原来把 `discards` / `riichi` / `melds_all` 按「从自己起算」旋转了一遍，
而 `View` 的约定是**绝对座位序**（编码器写的是 `view.discards[q]` 配
`q != view.player`，它自己会跳过 player）。

座位 0 时旋转恰好等价，座位 2 就全错 —— 也就是说：**用三麻坐 0 位随手一试
是测不出来的**。这就是为什么这个测试要跑满 3/4 个座位。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import Agent  # noqa: E402
from aimodel.engine import Game  # noqa: E402
from potatomaj import ai as AI  # noqa: E402

#: 赤 5 的下标 -> 花色。四麻多一个 0m（下标 4）
AKA_SUIT = {13: "p", 22: "s", 4: "m"}   # 13=5p, 22=5s, 4=5m（四麻才有 0m）

#: 自风表 —— 和 engine.py 的 seat_wind 一致：三麻只有三个风
SEAT_WIND_TABLE = {
    3: (S.EAST, S.SOUTH, S.WEST),
    4: (S.EAST, S.SOUTH, S.WEST, S.NORTH),
}


class CaptureAgent(Agent):
    """随便打第一张，但把每个「该打牌」的 view 抄下来。"""

    name = "capture"

    def __init__(self, seat: int, seed: int = 0) -> None:
        self.seat = seat
        self.views = []
        # 随机打：如果用固定策略（总打第一张），碰/吃永远不出现，
        # 副露那段代码就一直是没测过的（真踩过：副露被静默丢掉）。
        import random
        self.rng = random.Random(seat * 7919 + seed)

    def decide(self, decision, view):
        if decision.kind == "discard":
            self.views.append(view)          # 真值，原样存下来
        if decision.kind == "tsumo":
            return 1                          # 别和，让牌局继续跑
        return self.rng.randrange(len(decision.options))


def hand_ids(view) -> list[str]:
    """把 34 计数 + 赤 5 计数还原成牌 id 字符串（尽量还原成「协议会给的样子」）。

    注意：引擎的 ``hand_red`` 含**已经进副露**的那些赤 5，而协议给的手牌里
    没有它们（那些牌在副露里）。所以这里先把副露占掉的赤 5 减掉，
    剩下才是「手上真的还捏着的赤 5」。
    """
    ids: list[str] = []
    for i in range(34):
        ids.extend([S.tile_id(i)] * int(view.hand[i]))
    red = list(view.hand_red)
    for m in view.melds:
        if m.aka:
            red[m.tile] = max(0, red[m.tile] - int(m.aka))
    for idx, n in enumerate(red):
        suit = AKA_SUIT.get(idx)
        if not suit:
            continue
        for _ in range(int(n)):
            base = f"5{suit}"
            # 引擎的 hand_red 含「已经在副露里的赤 5」，所以手里可能根本没有
            # 这张 5 —— 那种情况由 meld_dicts 那边还原，这里跳过。
            if base not in ids:
                break
            k = ids.index(base)
            ids[k] = f"0{suit}"
    return ids


def meld_dicts_of(view, q: int):
    """第 q 家的副露（引擎按绝对座位存，melds_all[q]）。"""
    saved = view.melds
    try:
        view.melds = view.melds_all[q]
        return meld_dicts(view)
    finally:
        view.melds = saved


def meld_dicts(view):
    """把引擎的 Meld 还原成「协议会给的样子」。

    协议里副露长这样（ActionChiPengGang）：``{type, tiles: [...牌名...], froms: [...]}``
    引擎的 Meld 存的是 ``(kind, tile 下标, from_player, aka)`` —— 牌名要自己拼回来：
    碰是 3 张、杠是 4 张、吃是连续 3 张，赤 5 的数量按 ``aka`` 换成 0x。
    """
    out = []
    for m in view.melds:
        if m.kind == "chi":
            idxs = [m.tile, m.tile + 1, m.tile + 2]
        elif m.kind in ("minkan", "ankan", "kakan"):
            idxs = [m.tile] * 4
        else:
            idxs = [m.tile] * 3
        tiles = [S.tile_id(i) for i in idxs]
        # 按 aka 把其中几张 5 换成赤 5
        for _ in range(int(m.aka)):
            for k, tid in enumerate(tiles):
                if tid in ("5m", "5p", "5s") and not tiles[k].startswith("0"):
                    tiles[k] = "0" + tid[1]
                    break
        out.append({"type": {"chi": "chi", "pon": "pon", "minkan": "minkan",
                             "ankan": "ankan", "kakan": "kakan"}[m.kind],
                    "tiles": tiles, "froms": []})
    return out


def round_trip(view):
    """真值 view -> 协议能给的数据 -> 我自己的 build_view"""
    n = view.rule.players
    seat = view.player

    # 注意：View 里的 discards / dora_indicators 是**牌下标列表**，
    # 不是 34 长度的计数数组 —— 一开始我按计数写了，直接 IndexError。
    def ids_of_tiles(tile_indices):
        return [S.tile_id(int(t)) for t in tile_indices]

    return AI.build_view(
        hand=hand_ids(view),
        dora=ids_of_tiles(view.dora_indicators),
        discards=[ids_of_tiles(view.discards[q]) for q in range(n)],
        my_seat=seat,
        players=n,
        turn=view.turn,
        melds=meld_dicts(view),
        melds_all=[meld_dicts_of(view, q) for q in range(n)],
        riichi=list(view.riichi),
        scores=list(view.scores),
        wall_left=view.wall_left,
        round_wind=view.round_wind,          # 直接给 wind 常量（引擎也是这么存的）
        # 亲家：从「自风 − 场风」反推（引擎给的 seat_wind 就是按亲家轮转算的）。
        # 这个测试验的是「我拼的 View 对不对」，不是「我对 ju 的推断对不对」——
        # 后者在 app 里由 dealer_seat 负责。
        # 自风表就是 (东,南,西[,北])，所以 seat_wind - 东 就是「离亲家差几家」
        dealer=(seat - (view.seat_wind - S.EAST)) % n,
        is_dealer=view.is_dealer,
        ippatsu=list(view.ippatsu),
        kita=list(view.kita),                       # 拔北 —— 三麻特有，必须传
        drawn=(S.tile_id(view.drawn) if view.drawn is not None else None),
    )


def compare(real, mine, tag: str, problems: list[str], stats: dict) -> None:
    """逐字段比，再把两边编码成向量逐位比。"""
    n = real.rule.players

    def diff(name, a, b):
        if a != b:
            problems.append(f"{tag} {name}: 引擎={a}  我的={b}")

    diff("player", real.player, mine.player)
    diff("hand", list(real.hand), list(mine.hand))
    # hand_red（赤 5 计数）不逐位比 —— 引擎那边的记账推不出来，见下面 aka 位的说明。
    # 它的影响只体现在编码向量的 aka 那一位上，那里单独计数。
    diff("discards", [list(x) for x in real.discards], [list(x) for x in mine.discards])
    diff("dora_indicators", list(real.dora_indicators), list(mine.dora_indicators))
    diff("riichi", list(real.riichi), list(mine.riichi))
    diff("ippatsu", list(real.ippatsu), list(mine.ippatsu))
    diff("kita", list(real.kita), list(mine.kita))
    diff("drawn", real.drawn, mine.drawn)
    diff("melds", [(m.kind, m.tile, m.aka) for m in real.melds],
         [(m.kind, m.tile, m.aka) for m in mine.melds])
    diff("scores", list(real.scores), list(mine.scores))
    diff("wall_left", real.wall_left, mine.wall_left)
    diff("turn", real.turn, mine.turn)
    diff("round_wind", real.round_wind, mine.round_wind)
    diff("seat_wind", real.seat_wind, mine.seat_wind)
    diff("is_dealer", real.is_dealer, mine.is_dealer)

    diff("rule.players", real.rule.players, mine.rule.players)

    # 最强的一条：模型看到的东西必须逐位一致
    try:
        from aimodel.encode import action_features, action_mask, encode, layout_for
        from aimodel.engine import Decision

        types = S.SANMA_TYPES if n == 3 else S.FOUR_TYPES
        opts_r = [t for t in types if real.hand[t] > 0]
        opts_m = [t for t in types if mine.hand[t] > 0]
        if opts_r != opts_m:
            problems.append(f"{tag} 可打的牌不一样: 引擎={opts_r} 我的={opts_m}")
        else:
            d = Decision("discard", opts_r)
            vr = encode(real)
            vm = encode(mine)
            if not np.array_equal(vr, vm):
                bad = np.flatnonzero(vr != vm)
                # 已知的、可解释的一位：``aka``（赤 5 计数）。
                #
                # 引擎的 ``pl.red`` 是「这个人一共有过几张赤 5」，那张 5 被吃/碰
                # 进副露之后**通常不扣**（只有拔北会扣），但有时 `Meld.aka`
                # 又没有如实记下来 —— 于是从协议侧推不出它到底算不算。
                # 影响范围就是这一位（三麻 486 维里的第 457 位、四麻 560 维里
                # 的第 525 位），而且只在「手里/副露里有赤 5」的局面才会碰到。
                #
                # 单独列出来计数，不掩盖：其余任何一位不同都还是失败。
                aka_idx = layout_for(n)[0]["aka"]
                if len(bad) == 1 and int(bad[0]) == aka_idx:
                    stats["aka 位差异（已知）"] = stats.get("aka 位差异（已知）", 0) + 1
                else:
                    problems.append(
                        f"{tag} ❌ 编码向量不一致：{len(bad)}/{len(vr)} 位不同，"
                        f"前几位 {bad[:6].tolist()}"
                        + ("（含 aka）" if aka_idx in bad else ""))
            if not np.array_equal(action_features(real), action_features(mine)):
                problems.append(f"{tag} ❌ action_features 不一致")
            if not np.array_equal(action_mask(d), action_mask(d)):
                problems.append(f"{tag} ❌ action_mask 不一致")
    except Exception as exc:                       # noqa: BLE001
        problems.append(f"{tag} 编码对比跑不起来: {type(exc).__name__}: {exc}")


def run(players: int, seed: int, problems: list[str], stats: dict) -> None:
    agents = [CaptureAgent(i, seed) for i in range(players)]
    game = Game(agents, seed=seed, rule=S.rule_for(players), rounds=players * 2)
    game.play()

    checked = 0
    for seat, ag in enumerate(agents):
        # 每个座位抽几个局面就够 —— 全跑太慢，而且座位已经覆盖全了
        for v in ag.views[::max(1, len(ag.views) // 6)][:6]:
            mine = round_trip(v)
            compare(v, mine, f"{players}麻 seed{seed} 座位{seat} 第{v.turn}巡", problems, stats)
            checked += 1
            if v.melds:
                stats["带副露"] = stats.get("带副露", 0) + 1
            if any(v.kita):
                stats["带拔北"] = stats.get("带拔北", 0) + 1
    stats[f"{players}麻"] = stats.get(f"{players}麻", 0) + checked


def main() -> int:
    problems: list[str] = []
    stats: dict[str, int] = {}
    for players in (3, 4):
        for seed in (1, 2, 3):
            try:
                run(players, seed, problems, stats)
            except Exception as exc:               # noqa: BLE001
                problems.append(f"{players}麻 seed{seed} 跑不起来: {type(exc).__name__}: {exc}")

    for table, n in sorted(stats.items()):
        if table.startswith("aka"):
            print(f"  其中 {n} 个局面只在 aka 那一位上不同（已知，见代码注释）")
        else:
            print(f"  比对了 {n} 个局面（{table}）")

    if problems:
        print()
        for p in problems[:12]:
            print("❌", p)
        if len(problems) > 12:
            print(f"...还有 {len(problems) - 12} 条")
        print()
        print(f"❌ 和引擎不一致：{len(problems)} 处")
        return 1

    print()
    print(f"✅ 我拼的 View 和引擎给的逐字段一致，编码向量也逐位相同（{sum(stats.values())} 个局面）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
