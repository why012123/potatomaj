"""拿一个**独立的第三方算番库**当参照，和我们自己的引擎对拍。

## 为什么要这个

2026-10-21 用户用一张截图发现我们的引擎把**顺子手判成対々和** ✗（多吃 2 番）。
顺着查又发现第二处：**吃的牌被加上了刻子的符** ✗（多算符）。
两个都是「把吃当成刻子」这一类 ✗ 而 `test_yaku.py` 的 34 个役种**只测了无副露的手** ✗
—— 所以这类 bug 完全测不出来 ✓

靠人再写几个用例不够 ✗ 必须**大批量和独立实现**对拍 ✓
参照实现：`MahjongRepository/mahjong`（MIT，Python，业界常用）✓

## 用法

    # 先把参照库装到隔离目录（不污染项目 venv）
    .venv/bin/python -m pip install --target /tmp/mjlib mahjong

    # 随机对拍 2000 手
    .venv/bin/python tools/compare_scoring.py --n 2000

    # 只对拍「有副露」的手（我们出错的那一类）
    .venv/bin/python tools/compare_scoring.py --n 2000 --melds-only

## 比什么

* 和不和（两边都得认为这和牌成立）✓
* **番数** ✓
* **符数** ✓
* （役种名两边命名体系不同 ✗ 先不逐条映射 ✓ 番/符对上就说明大体一致 ✓）

## 注意

* 只用四麻 ✓（参照库只有四麻 ✗）
* 不用赤 5 ✓ 不带宝牌 ✓ —— 先把「役/符」这一层对齐 ✓ 免得混入别的差异 ✗
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.fastshanten import shanten  # noqa: E402


def load_reference(lib_dir: str):
    """把第三方库从隔离目录里导进来（不动项目 venv ✓）。"""
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    try:
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
        from mahjong.meld import Meld as RMeld
        from mahjong.tile import TilesConverter
    except Exception as e:                        # noqa: BLE001
        raise SystemExit(
            f"❌ 导不进来第三方库（{type(e).__name__}: {e}）\n"
            f"   先装：.venv/bin/python -m pip install --target {lib_dir} mahjong"
        )
    return HandCalculator, HandConfig, OptionalRules, RMeld, TilesConverter


def counts_to_136(counts, TilesConverter=None) -> list[int]:
    """34 长度计数 -> 136 张的牌 id 列表（每张牌型最多 4 张）。"""
    out: list[int] = []
    for t in range(34):
        for k in range(counts[t]):
            out.append(t * 4 + k)
    return out


def our_meld_to_ref(kind: str, tile: int, RMeld, opened_from: int = 1):
    """我们的 Meld(kind, tile) -> 参照库的 Meld。"""
    if kind == "chi":
        ids = [tile * 4, (tile + 1) * 4, (tile + 2) * 4]
        mtype = RMeld.CHI
        opened = True
    elif kind == "pon":
        ids = [tile * 4 + i for i in range(3)]
        mtype = RMeld.PON
        opened = True
    elif kind == "minkan":
        ids = [tile * 4 + i for i in range(4)]
        mtype = RMeld.KAN
        opened = True
    elif kind == "ankan":
        ids = [tile * 4 + i for i in range(4)]
        mtype = RMeld.KAN        # 参照库没有独立 ankan 常量：KAN + opened=False
        opened = False
    elif kind == "kakan":
        ids = [tile * 4 + i for i in range(4)]
        mtype = RMeld.SHOUMINKAN
        opened = True
    else:
        raise ValueError(f"未知副露类型 {kind}")
    if opened:
        return RMeld(mtype, ids, opened=True, called_tile=ids[-1],
                     who=0, from_who=opened_from)
    return RMeld(mtype, ids, opened=False, who=0)


def our_kind(kind: str) -> str:
    """生成器用的是 "tri"/"seq"，我们引擎/参照库要的是 "pon"/"chi"。"""
    return {"tri": "pon", "seq": "chi"}[kind]


def random_complete_hand(rng: random.Random, want_melds: int):
    """随机造一手**已经和了**的牌（可以带 want_melds 组副露）。

    做法：先随机挑 4 组面子（刻子或顺子）+ 1 对，再检查张数不超过 4 张 ✓
    """
    meld_kinds: list[tuple[str, int]] = []
    n_meld_needed = 4
    while True:
        counts = [0] * 34
        meld_kinds = []
        ok = True
        for _ in range(n_meld_needed):
            if rng.random() < 0.45:                     # 刻子
                t = rng.randrange(34)
                need = [t] * 3
                kind, base = "tri", t
            else:                                        # 顺子
                suit = rng.randrange(3)
                start = rng.randrange(7)
                base = suit * 9 + start
                need = [base, base + 1, base + 2]
                kind = "seq"
            for t in need:
                counts[t] += 1
            meld_kinds.append((kind, base))
        pair = rng.randrange(34)
        counts[pair] += 2
        if max(counts) <= 4 and sum(counts) == 14:
            break
        # 极少见：5 张以上 -> 重来
    # 把其中 want_melds 组「面子」变成副露（优先挑刻子/顺子都行）
    #   注意：变成副露要从暗牌里扣掉，且副露本身也算在和牌形里
    rng.shuffle(meld_kinds)
    called = meld_kinds[:want_melds]
    concealed = [0] * 34
    for i, (kind, base) in enumerate(meld_kinds):
        if i < want_melds:
            continue
        if kind == "tri":
            concealed[base] += 3
        else:
            for t in (base, base + 1, base + 2):
                concealed[t] += 1
    concealed[pair] += 2
    return concealed, called, pair


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第三方算番库 vs 我们的引擎")
    ap.add_argument("--n", type=int, default=500, help="对拍多少手")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib-dir", default="/tmp/mjlib")
    ap.add_argument("--melds-only", action="store_true", help="只测有副露的手")
    ap.add_argument("--show", type=int, default=8, help="最多打印几组差异")
    args = ap.parse_args(argv)

    HandCalculator, HandConfig, OptionalRules, RMeld, TilesConverter = (
        load_reference(args.lib_dir)
    )
    calc = HandCalculator()
    rules = OptionalRules(has_aka_dora=False, has_open_tanyao=True,
                          # 我们（雀魂规则）四暗刻单骑/国士十三面 算**双倍**役满 ✓
                          # 参照库默认关掉 -> 不比的话会误报「番不同」✗（server 2026-10-21 指出 ✓）
                          has_double_yakuman=True, kazoe_limit=0)
    rng = random.Random(args.seed)
    rule4 = S.rule_for(4)

    stats = {"n": 0, "both_win": 0, "han_mismatch": 0, "fu_mismatch": 0,
             "ours_only": 0, "ref_only": 0}
    shown = 0

    while stats["n"] < args.n:
        want = rng.choice([1, 2, 3] if args.melds_only else [0, 0, 1, 2, 3])
        concealed, called, pair = random_complete_hand(rng, want)
        if shanten(concealed, len(called)) != -1 and sum(concealed) != (4 - len(called)) * 3 + 2:
            continue
        stats["n"] += 1

        # 和了牌：从暗牌里挑一张
        cands = [t for t in range(34) if concealed[t] > 0]
        if not cands:
            continue
        win = rng.choice(cands)

        # 我们的引擎
        try:
            our = S.evaluate_hand(
                concealed,
                [S.Meld(our_kind(k), base, 1) for k, base in called],
                S.WinContext(win_tile=win, is_tsumo=False, round_wind=S.EAST,
                             seat_wind=S.SOUTH, rule=rule4),
            )
        except Exception as e:                       # noqa: BLE001
            print(f"  ⚠️ 我们引擎抛异常: {type(e).__name__}: {e}")
            continue

        # 参照库
        # ⚠️ 它的 `tiles` 是**整手 14 张**（含副露那几张）✗ 不是只传暗牌 ✓
        #    我第一次只传了暗牌 -> 它一律报 hand_not_winning ✗（不是它错，是我错 ✓）
        counter = [0] * 34
        def take(t: int, n: int) -> list[int]:
            ids = []
            for _ in range(n):
                ids.append(t * 4 + counter[t])
                counter[t] += 1
            return ids

        ref_melds = []
        meld_ids: list[int] = []
        for k, b in called:
            if k == "tri":
                ids = take(b, 3)
                ref_melds.append(RMeld(RMeld.PON, ids, opened=True, called_tile=ids[-1],
                                       who=0, from_who=1))
            else:
                ids = take(b, 1) + take(b + 1, 1) + take(b + 2, 1)
                ref_melds.append(RMeld(RMeld.CHI, ids, opened=True, called_tile=ids[-1],
                                       who=0, from_who=1))
            meld_ids.extend(ids)
        concealed_ids = [i for t in range(34) for i in take(t, concealed[t])]
        tiles136 = meld_ids + concealed_ids
        # ⚠️ 和了牌**只能从暗牌里取** —— 原来从含副露的 tiles136 里取第一张，
        #    可能指到副露里那张；副露的牌不可能被和成，参照库会算出完全不同的结果。
        #    （server 2026-10-21 指出 ✓ 对的 ✓）
        wt = next(x for x in concealed_ids if x // 4 == win)
        try:
            ref = calc.estimate_hand_value(
                tiles136, wt, melds=ref_melds or None,
                dora_indicators=None,
                config=HandConfig(is_tsumo=False, is_riichi=False,
                                  player_wind=S.SOUTH, round_wind=S.EAST, options=rules),
            )
        except Exception as e:                       # noqa: BLE001
            print(f"  ⚠️ 参照库抛异常: {type(e).__name__}: {e}")
            continue

        if our is None and ref.error:
            continue                                  # 两边都说不成和 ✓
        if our is None and not ref.error:
            stats["ref_only"] += 1
            if shown < args.show:
                print(f"  🔴 只有参照库认为成和: 暗牌{concealed} 副露{called} 和{win} "
                      f"-> ref {ref.han}番{ref.fu}符")
                shown += 1
            continue
        if our is not None and ref.error:
            stats["ours_only"] += 1
            if shown < args.show:
                print(f"  🔴 只有我们认为成和: 暗牌{concealed} 副露{called} 和{win} "
                      f"-> ours {our.han}番{our.fu}符 | 参照报错: **{ref.error}**")
                shown += 1
            continue

        stats["both_win"] += 1
        # ⚠️ **役满要按倍数比，不能比 han/fu** ✗
        #    我们的 `sanma.Agari` 对役满返回 `han=0, fu=0`，倍数放在 `yakuman` 字段里
        #    （`base_points` 也走 `if yakuman: return 8000*mult`，所以**点数是对的** ✓）
        #    而参照库对 1 倍役满给 `han=13`、2 倍给 26 ✓
        #    -> 直接比 han/fu 会把**每一手役满**都记成「番不同 + 符不同」✗ 这是假阳性
        #       （2026-10-21 实测：400 手里 2~5 次，正好是那 1% 的量级）
        if our.yakuman > 0:
            han_ok = (our.yakuman * 13) == ref.han
            fu_ok = True          # 役满的点数只由倍数决定，fu 不参与
            stats["yakuman"] = stats.get("yakuman", 0) + 1
        else:
            han_ok = our.han == ref.han
            fu_ok = our.fu == ref.fu
        if not han_ok:
            stats["han_mismatch"] += 1
        if not fu_ok:
            stats["fu_mismatch"] += 1
        if (not han_ok or not fu_ok) and shown < args.show:
            yau = [k for k, _ in our.yaku]
            print(f"  🔴 暗牌{concealed} 副露{called} 和{win}")
            print(f"      我们 {our.han}番{our.fu}符 yakuman={our.yakuman} {yau}")
            print(f"      参照 {ref.han}番{ref.fu}符 {[y.name for y in (ref.yaku or [])]}")
            shown += 1

    n = max(1, stats["both_win"])
    print()
    print(f"  对拍 {stats['n']} 手，两边都成和 {stats['both_win']} 手")
    print(f"    番不同 {stats['han_mismatch']} ({stats['han_mismatch']/n:.1%})"
          f"   符不同 {stats['fu_mismatch']} ({stats['fu_mismatch']/n:.1%})")
    print(f"    只有我们成和 {stats['ours_only']}   只有参照成和 {stats['ref_only']}")
    return 0 if stats["han_mismatch"] == 0 and stats["fu_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
