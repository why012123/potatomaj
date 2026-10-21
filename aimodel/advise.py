"""把「识图」和「AI 决策」接起来：给一张雀魂三麻截图，告诉你怎么打。

外层工程 :mod:`potatomaj` 已经能把截图读成牌（手牌 / 三家牌河 / 宝牌指示牌），
这里接着往下走一步 —— 用训练好的神经网络给建议。

用法::

    # 直接给截图
    python -m aimodel.advise ../samples/sanma5.webp

    # 用已经识别好的 JSON
    python -m aimodel.advise --json 识别结果.json

    # 手写牌型（不依赖截图，方便试）
    # ⚠️ 要 **14 张**才出「打哪张」的建议 ✓
    #    13 张是**等摸牌**的形态 ✗ 那种局面给不出弃牌建议 ✓ 只会列进张 ✓
    #    （我用 13 张当例子试过 ✓ 结果只打印牌效、一条建议都没有 ✓ 已改成 14 张 ✓）
    python -m aimodel.advise --hand "1m 1m 2p 3p 8p 8p 9p 0s 6s 6s 8s 1z 6z 6z"

说明：截图里读不到的场况（点数、巡目、谁立直了）只能按默认值假设，
所以建议只当作「牌效参考」，不能代替读牌。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import sanma as S
from .agents import HeuristicAgent
from .encode import action_mask
from .feature_dispatch import action_features_for, encode_for
from .engine import Decision, View
from .fastshanten import shanten, ukeire, useful_tiles
from .net import Net

WEIGHTS = Path(__file__).resolve().parent / "weights"
#: 摸牌区总张数（三麻 51 张）
WALL_TOTAL = 51


#: 赤 5 的牌型下标
_AKA_INDEX = {"p": 13, "s": 22}


def counts_from_ids(ids: list[str]) -> tuple[list[int], list[int]]:
    """牌 id 列表 -> (34 计数, 赤 5 计数)。识别不出来的牌（``?``）会被忽略。"""
    counts = [0] * 34
    red = [0] * 34
    for tid in ids:
        if not tid or tid == "?":
            continue
        for i, c in enumerate(S.parse_hand(tid)):
            counts[i] += c
        if tid.startswith("0") and tid[1] in _AKA_INDEX:
            red[_AKA_INDEX[tid[1]]] += 1
    return counts, red


def build_view(
    hand: list[str],
    ponds: dict[str, list[str]],
    dora_indicators: list[str],
    *,
    turn: int = 6,
    scores: list[int] | None = None,
    riichi: list[bool] | None = None,
) -> tuple[View, list[str]]:
    """把识别结果拼成一个「自家视角」的观测。返回 (view, 警告列表)。"""
    warns: list[str] = []
    hc, red = counts_from_ids(hand)
    n_hand = sum(hc)
    if n_hand not in (13, 14):
        warns.append(f"手牌读到 {n_hand} 张（正常是 13 或 14 张），建议结果仅供参考")

    self_pond = counts_from_ids(ponds.get("self", []))[0]
    opp1 = counts_from_ids(ponds.get("top", []))[0]
    opp2 = counts_from_ids(ponds.get("left", []))[0]

    def pond_list(counts: list[int]) -> list[int]:
        return [i for i in range(34) for _ in range(counts[i])]

    dora_idx = []
    for d in dora_indicators:
        c = S.parse_hand(d)
        for i, v in enumerate(c):
            if v:
                dora_idx.append(i)
    if not dora_idx:
        warns.append("没读到宝牌指示牌，宝牌按 0 张算")

    consumed = n_hand + sum(sum(x) for x in (self_pond, opp1, opp2))
    wall_left = max(1, WALL_TOTAL - max(0, turn - 1) * 3)

    view = View(
        player=0,
        hand=hc,
        drawn=None,
        melds=[],
        hand_red=red,
        discards=[pond_list(self_pond), pond_list(opp1), pond_list(opp2)],
        melds_all=[[], [], []],
        kita=[0, 0, 0],
        scores=list(scores) if scores else [35000, 35000, 35000],
        dora_indicators=dora_idx,
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=wall_left,
        turn=turn,
        riichi=list(riichi) if riichi else [False, False, False],
        ippatsu=[False, False, False],
        is_dealer=False,
        last_discard=None,
        rule=S.DEFAULT_RULE,
    )
    return view, warns


def suggest(view: View, net: Net | None, topk: int = 3) -> list[tuple[str, float]]:
    """给 14 张的手牌排序建议（牌名, 概率）。"""
    options = [t for t in S.SANMA_TYPES if view.hand[t] > 0]
    decision = Decision("discard", options)
    mask = action_mask(decision)
    if net is None:
        return []
    _dim = int(getattr(net, "dim", 0) or 0)
    p = net.policy(encode_for(_dim, view.rule.players, view), "discard", mask,
                   action_features_for(_dim, view.rule.players, view))
    order = np.argsort(-p)[:topk]
    return [(S.tile_cn(int(t)), float(p[int(t)])) for t in order if p[int(t)] > 1e-6]


def report(view: View, net: Net | None, unknown: int = 0) -> str:
    n = view.hand_size
    n_melds = view.meld_count()
    lines: list[str] = []
    lines.append(f"手牌({n}): " + " ".join(S.tile_cn(i) for i in range(34) for _ in range(view.hand[i])))
    lines.append(f"宝牌指示: " + (" ".join(S.tile_cn(d) for d in view.dora_indicators) or "无"))
    for p, tag in enumerate(("自己", "下家", "上家")):
        pond = " ".join(S.tile_cn(t) for t in view.discards[p]) or "（空）"
        lines.append(f"牌河·{tag}: {pond}")
    if unknown:
        lines.append(f"⚠️ 有 {unknown} 张没认出来")

    sh = shanten(view.hand, n_melds)
    if sh < 0:
        lines.append("✅ 已经和牌")
        return "\n".join(lines)

    if n == 14:
        lines.append(f"向听 {sh}")
        if net is not None:
            lines.append("")
            lines.append("🤖 神经网络建议（越靠前越好）:")
            for name, p in suggest(view, net):
                lines.append(f"   打 {name}   概率 {p:.1%}")
        # 教师建议作对照
        teacher = HeuristicAgent(0)
        opts = [t for t in S.SANMA_TYPES if view.hand[t] > 0]
        ti = teacher.decide(Decision("discard", opts), view)
        lines.append(f"📐 启发式教师建议: 打 {S.tile_cn(opts[ti])}")
        lines.append("")
        best = suggest(view, net, 1)
        if best:
            pick = next(t for t in S.SANMA_TYPES if S.tile_cn(t) == best[0][0])
            rest = _drop(view.hand, pick)
            after = shanten(rest, n_melds)
            ups = ukeire(rest, _visible(view), n_melds)
            wait = "".join(f"{S.tile_cn(t)}×{c} " for t, c in sorted(ups.items()))
            lines.append(
                f"打完 {best[0][0]} 之后：{after} 向听"
                + (f"，进张 {wait}" if ups else "（无进张）")
            )
    else:
        ups = ukeire(view.hand, _visible(view), n_melds)
        wait = "".join(f"{S.tile_cn(t)}×{c} " for t, c in sorted(ups.items()))
        lines.append(f"向听 {sh}" + (f"，进张 {wait}" if ups else ""))
        # ⚠️ 这里原来**什么都不说** ✗ -> 用户拿着 13 张试 ✓ 只看到牌效 ✓
        #    会以为「建议功能没做」✗（我自己就这么误会了一次 ✓ 见模块 docstring）
        #    -> 明确告诉他这是等摸牌的形态 ✓ 以及要怎么看建议 ✓
        if n < 14:
            lines.append(
                f"（{n} 张 = 等摸牌的形态，没有「打哪张」可建议；"
                f"想看出牌建议请给 14 张）"
            )
    return "\n".join(lines)


def _drop(counts: list[int], tile: int) -> list[int]:
    c = list(counts)
    c[tile] -= 1
    return c


def _visible(view: View) -> list[int]:
    from .encode import visible_counts

    return visible_counts(view)



def default_weights() -> str:
    """默认权重：**走注册表取最新的三麻模型** ✓

    ⚠️ 以前写死 `weights/policy.npz` ✗ 而那个文件实测是 `maj-normalv1.npz` 的
    逐字节副本（**v1**，连 v1.1 都不是）✗ 用它会给出一个过时模型的建议，
    而且从界面上完全看不出来 ✓
    """
    try:
        from . import variants

        sanma = [s for s in variants.all_specs() if s.players == 3 and s.path.exists()]
        if sanma:
            # 版本号大的优先；同版本按名字定序，保证可复现
            best = max(sanma, key=lambda s: (s.version, s.name))
            return str(best.path)
    except Exception:                                   # noqa: BLE001
        pass
    return str(WEIGHTS / "policy.npz")

def load_from_image(path: Path) -> dict:
    """调用外层识图模块，拿回结构化结果。

    ⚠️ **这条路已经废弃了** ✗ —— 截图识别的整条链路被有意删掉，
    改成了协议解码（`electron/lib/actions.js`，见 `potatomaj/__main__.py` 的说明）✓
    原因是协议解码比截图识别可靠得多 ✓

    这里保留它只是为了**兼容老用法**，并且给一个清楚的报错 ✗
    而不是抛一个看不出所以然的 `ImportError` / `FileNotFoundError` ✓

    现在推荐的三条输入路径：
      ``--hand "123m456p77z5s"``   手写牌型（最常用）
      ``--json x.json``            别处解码好的结构化结果
      （或直接调 :func:`build_view` / :func:`suggest` 自己拼 View）
    """
    sys.path.insert(0, str(WEIGHTS.parents[1]))
    try:
        from potatomaj.classify import TemplateLibrary
        from potatomaj.recognize import recognize
    except ImportError as e:
        raise SystemExit(
            f"❌ 截图识别链路不可用（{e}）\n"
            "   这条路已经被有意废弃 —— 现在走协议解码 ✓\n"
            "   请改用：--hand \"牌型\"  或  --json <文件>\n"
            "   （原因见 potatomaj/__main__.py 的说明）"
        ) from None

    templates = WEIGHTS.parents[1] / "templates"
    if not templates.is_dir():
        raise SystemExit(
            f"❌ 模板库不存在：{templates}\n"
            "   截图识别那条路已废弃，模板库也没跟着留 ✗\n"
            "   请改用：--hand \"牌型\"  或  --json <文件>"
        )
    lib = TemplateLibrary(str(templates))
    if len(lib) == 0:
        raise SystemExit(f"❌ 模板库为空：{templates}（请改用 --hand / --json）")
    return recognize(path, lib).to_dict()


def expand_chunk(chunk: str) -> list[str]:
    """``"123m"`` -> ``["1m","2m","3m"]``；``"5p"`` 原样返回。"""
    if len(chunk) > 2 and chunk[-1] in "mpsz":
        return [d + chunk[-1] for d in chunk[:-1]]
    return [chunk] if chunk else []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="三麻出牌建议（识图 + 神经网络）")
    ap.add_argument("image", nargs="?", help="雀魂三麻截图")
    ap.add_argument("--json", dest="json_path", help="已经识别好的 JSON 文件")
    ap.add_argument("--hand", help='手写牌型，如 "123m456p77z"')
    ap.add_argument("--pond", action="append", default=[], help="牌河（可重复，最多 3 组）")
    ap.add_argument("--dora", action="append", default=[], help="宝牌指示牌（可重复）")
    ap.add_argument("--turn", type=int, default=6, help="假设的巡目")
    ap.add_argument("--weights", type=str, default=default_weights())
    args = ap.parse_args(argv)

    net = Net.load(args.weights) if Path(args.weights).exists() else None
    if net is None:
        print(f"⚠️ 找不到权重 {args.weights}，只给牌效（不给神经网络建议）")

    if args.hand:
        # 手写模式：把 "123m456p77z" 展开成牌 id 列表
        ids = [t for chunk in args.hand.split() for t in expand_chunk(chunk)]
        ponds = {"self": [], "top": [], "left": []}
        for i, spec in enumerate(args.pond[:3]):
            key = ("self", "top", "left")[i]
            ponds[key] = [t for chunk in spec.split() for t in expand_chunk(chunk)]
        dora = [t for spec in args.dora for t in expand_chunk(spec)]
        payload = {"hand": ids, "ponds": ponds, "dora_indicators": dora, "uncertain": 0}
    elif args.json_path:
        payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    elif args.image:
        payload = load_from_image(Path(args.image))
    else:
        ap.print_help()
        return

    view, warns = build_view(
        payload.get("hand", []),
        payload.get("ponds", {}),
        payload.get("dora_indicators", []),
        turn=args.turn,
    )
    print(report(view, net, unknown=int(payload.get("uncertain", 0) or 0)))
    for w in warns:
        print("⚠️ " + w)


if __name__ == "__main__":
    main()
