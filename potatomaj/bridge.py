"""给 Electron 外壳用的「大脑」服务：stdin/stdout 上一行一个 JSON。

    → {"id":1,"cmd":"analyze","hand":[...],"dora":[...],"discards":[...]}
    ← {"id":1,"ok":true,"shanten":2,"text":"2 向听，进张：...","options":[...]}

只看牌、算牌效和危险度。**牌从哪来不是这里的事** —— 全部由 Node 侧的协议解码提供
（见 electron/lib/actions.js）。这个进程里没有一行识图代码。

为什么是分开的两个进程：算牌效要 numpy（还有 aimodel 的神经网络），
协议解码在 Node 侧更顺；两边各干各擅长的，用一个薄协议对接。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from potatomaj import ai as AI                          # noqa: E402
from potatomaj.danger import describe as describe_danger, recommend, visible_counts  # noqa: E402
from potatomaj.shanten import (                         # noqa: E402
    analyze as analyze_hand, discard_options, melds_from_hand_size,
)
from potatomaj.tiles import sort_tiles                  # noqa: E402

def cmd_analyze(req: dict) -> dict:
    """手牌 + 宝牌 + 四家牌河 -> 向听/进张 + **安全度建议**。

    有了四家牌河（协议解码，实测零错），就能算「打哪张最安全」，
    再和「打了不掉向听」交叉，给出既安全又不损牌效的一手。
    """
    hand = list(req.get("hand") or [])
    dora = list(req.get("dora") or [])
    discards = [list(r) for r in (req.get("discards") or [])]
    my_seat = int(req.get("my_seat") or 0)
    if not hand:
        return {"ok": False, "error": "没有手牌"}

    # 碰/吃/杠出来的牌不在手牌行里，所以手牌行会比正常短 3 的倍数。
    # 副露的**具体牌**现在读不到，但向听只需要知道**有几个**副露。
    melds = req.get("melds")
    if melds is None:
        melds = melds_from_hand_size(len(hand) + (0 if req.get("is_my_turn") is False else 0))
    melds = int(melds)

    sanma = bool(req.get("sanma", False))
    out = {"ok": True, "melds": melds, **analyze_hand(hand, sanma=sanma, melds=melds)}
    if dora:
        out["dora"] = sort_tiles(dora)

    turn = max((len(r) for r in discards), default=0)
    riichi = req.get("riichi") or []
    # 几麻由上层认出来（三麻没有 2m~8m）—— 不写死，否则四麻会少算一家
    if req.get("three_player") is not None:
        sanma = bool(req.get("three_player"))
    seen = visible_counts([], discards, dora) if discards else {}

    danger: dict = {}
    if discards and any(discards):
        # riichi 是协议解出来的**确定信息**（见 electron/lib/gamestate.js）——
        # 有了它，对听牌那家的危险度就不再是估算
        d = recommend(hand, discards, my_seat, dora, sanma=sanma, melds=melds,
                      seats=3 if sanma else 4, riichi=riichi)
        danger = {r["tile"]: r for r in d["tiles"]}
        out["danger"] = {
            "turn": d["turn"],
            "riichi": [i for i, v in enumerate(riichi) if v],
            "text": describe_danger(d),
            "recommend": d.get("recommend"),
            "tiles": [{"tile": r["tile"], "danger": r["danger"], "why": r["why"]}
                      for r in d["tiles"]],
        }

    # ---- 每种牌打掉之后的「收益」 ----
    # 主判据是 aimodel 那个神经网络（三麻专用，适用性闸门见 potatomaj.ai）；
    # 用不上时退回纯牌效的进张枚数 —— 两条路都给出一份能横向比较的数
    # 立直中：手牌是**锁死**的 —— 摸到什么就必须打什么，没有「该打哪张」这个问题。
    # 所以不要给一个会被当成建议的排序，直接说清楚，同时仍然给出安全度信息。
    riichi_list = req.get("riichi") or []
    i_am_riichi = (0 <= int(my_seat) < len(riichi_list)) and bool(riichi_list[int(my_seat)])

    options = discard_options(hand, sanma=sanma, melds=melds, seen=seen)
    # 网络：按牌桌挑模型（三麻 maj-flashv1.1 / 四麻 maj-flash4v1.1，可在 build/models.json 里改）。
    # 副露的**具体牌**协议给得到，所以带副露也照样喂进去。
    probs, note = AI.rank(
        hand, dora, discards, my_seat,
        turn=turn,
        players=3 if sanma else 4,
        melds=req.get("meld_list") or [],
        riichi=req.get("riichi") or [],
        scores=req.get("scores") or [],
        wall_left=req.get("left_tile_count"),
        round_wind=req.get("round_wind"),
        kita=req.get("kita") or [],
        drawn=req.get("drawn"),
        ippatsu=req.get("ippatsu") or [],
        dealer=req.get("dealer"),
    )
    note = (f"{AI.model_name(3 if sanma else 4)}"
            + (f"：{note}" if note else ""))

    top = None
    if options:
        for o in options:
            o["score"] = round(probs.get(o["tile"], 0.0) * 100) if probs else None
            r = danger.get(o["tile"])
            o["danger"] = r["danger"] if r else None
            o["why"] = r["why"] if r else ""
        if probs:
            options.sort(key=lambda r: (-(r["score"] or 0), r["loss"], -r["ukeire"], r["tile"]))
        top = options[0]
        out["options"] = options
        out["mode"] = "model" if probs else "ukeire"
        if probs and AI.model_name(3 if sanma else 4) == AI.TEACHER:
            out["mode_note"] = "启发式教师（牌效 + 防守 + 弃和，aimodel 的 HeuristicAgent）"
        elif probs:
            out["mode_note"] = f"神经网络 {AI.model_name(3 if sanma else 4)}"
        else:
            out["mode_note"] = note or "纯牌效：进张枚数"

    # ---- 最后决定「那句话说什么」----
    #
    # 顺序有讲究：牌子（options）照常给，但**文案**在立直时要改掉。
    # 之前这里写反了 —— 先写了立直文案，后面又被「建议打 X」覆盖回去，
    # 结果界面在立直中还在推荐一张你根本不能打的牌（立直后是强制摸切）。
    if options and top:
        out["best"] = top["tile"]
        out["best_percent"] = top["score"]
        if i_am_riichi:
            out["text"] = "立直中 —— 摸到什么就必须打什么（强制摸切），这里只看危险度"
            out["mode_note"] = "立直锁牌：没有「该打哪张」这个选择"
            out["riichi_locked"] = True
        elif probs:
            tail = f"进张 {top['ukeire']} 枚" if top["ukeire"] else "无进张"
            is_teacher = AI.model_name(3 if sanma else 4) == AI.TEACHER
            # 教师的分数是「向听×1e6 + 进张×权重 − 危险×权重」归一化到 0~1 的排序值，
            # **不是概率** —— 所以不能写「模型 X%」，免得跟神经网络的数放在一起比
            label = (f"教师评分 {top['score']}" if is_teacher else f"模型 {top['score']}%")
            out["text"] = (f"{out.get('shanten')} 向听　建议打 {top['tile']}"
                           f"（{label}）　{tail}")
            out["score_is_probability"] = not is_teacher
        elif out.get("danger"):
            out["text"] = describe_danger({
                "turn": turn, "tiles": list(danger.values()),
                "recommend": {"tile": top["tile"], "why": top.get("why") or "",
                              "danger": top.get("danger")},
            })
    return out


def _call_kinds(ops: list[dict]) -> dict[str, list[list[str]]]:
    """把协议里的 ``OptionalOperationList`` 变成 ``{chi/pon/kan/ron: [组合]}``。

    ``kind`` 是 electron/lib/actions.js 按**实测**的 type 编号标好的
    （2=吃 3=碰 5=大明槓 9=荣和）；万一那边只给了 type，这里也认。
    组合字符串形如 ``"7s|8s"``（吃，两张手牌）/ ``"1z|1z"``（碰）/
    ``"4p|4p|4p"``（槓）。
    """
    out: dict[str, list[list[str]]] = {}
    for o in ops or []:
        if not isinstance(o, dict):
            continue
        kind = str(o.get("kind") or "")
        if kind not in ("chi", "pon", "kan", "ron"):
            kind = {2: "chi", 3: "pon", 5: "kan", 9: "ron"}.get(
                int(o.get("type") or 0), "")
        if not kind:
            continue
        combos = [[t for t in str(c).split("|") if t]
                  for c in (o.get("combination") or [])]
        # 形状校验（防御）：type 编号是实测出来的，万一以后服务器换了编号，
        # 组合的形状还能兜底 —— 吃是两张**不同**的牌，碰是两张相同，
        # 大明槓是三张相同，荣和没有组合。对不上就不当成那个动作。
        if kind == "chi":
            combos = [c for c in combos if len(c) == 2 and c[0] != c[1]]
        elif kind == "pon":
            combos = [c for c in combos if len(c) == 2 and c[0] == c[1]]
        elif kind == "kan":
            combos = [c for c in combos if len(c) == 3 and c[0] == c[1] == c[2]]
        elif kind == "ron":
            combos = []
        if kind == "ron":
            out.setdefault("ron", [])
        elif combos:
            out.setdefault(kind, []).extend(combos)
    return out


def cmd_analyze_call(req: dict) -> dict:
    """「要不要鸣牌」：吃 / 碰 / 杠 / 荣和。

    ## 为什么单独一条命令

    出牌建议那条路（``analyze``）只在「我摸完 14 张」时有意义。鸣牌发生在
    **别人打牌之后、我还是 13 张**的时候，选项是 ``pass`` / ``chi`` /
    ``pon`` / ``kan`` —— 引擎里这是 ``call`` / ``chi`` 两种决策，和 ``discard``
    完全不同的动作空间。以前 HUD 根本没有这条路，所以在鸣牌提示里什么都不说。

    数据来源是协议里的 ``ActionDiscardTile.operation``（见 electron/lib/actions.js
    的 ``operationListAt``）—— **不是**从手牌猜「能不能碰」。猜的话立直后不能鸣、
    振听、已经过牌、食替这些情况全会错。
    """
    from aimodel import sanma as S
    from aimodel.agents import HeuristicAgent, MLAgent
    from aimodel.engine import Decision
    from aimodel.fastshanten import shanten

    hand = list(req.get("hand") or [])
    melds = list(req.get("meld_list") or [])
    meld_n = len(melds)
    if not hand:
        return {"ok": False, "error": "没有手牌"}
    # 鸣牌窗口里手牌必须是 13 张等效（10 张 + 1 副露、7 张 + 2 副露…）。
    # 对不上说明前面已经失准 —— 这时候给建议只会误导，明确说不判断。
    if len(hand) + 3 * meld_n != 13:
        return {"ok": True, "decision": "unknown",
                "text": f"⚠️ 鸣牌：手牌 {len(hand)} 张 + {meld_n} 副露"
                        f"（等效 {len(hand) + 3 * meld_n}），不是 13 张，先不判断",
                "mode": "call", "hand_ok": False}
    # 三麻不可能有 2m~8m（含赤 5m 的 0m）。出现了说明牌桌类型认错/状态错，
    # 别拿三麻模型硬算 —— 和出牌建议那边的 model_ok 闸门同一个道理。
    if bool(req.get("sanma", req.get("three_player", False))) and any(
        str(t).endswith("m") and str(t)[0] not in "19" for t in hand
    ):
        return {"ok": True, "decision": "unknown",
                "text": "⚠️ 三麻手牌里出现 2m~8m（牌桌类型可能认错了），先不判断",
                "mode": "call", "hand_ok": False}

    kinds = _call_kinds(req.get("ops") or [])
    if not kinds:
        return {"ok": True, "decision": "unknown", "text": "（没有可鸣的牌）",
                "mode": "call", "hand_ok": True}

    # 荣和：能荣就和。操作列表说合法就是合法（振听不会给这个操作）。
    if "ron" in kinds:
        return {"ok": True, "decision": "ron", "text": "🀄 鸣牌建议：荣和",
                "mode": "call", "hand_ok": True, "kinds": sorted(kinds)}

    call_tile = req.get("call_tile")
    try:
        tile = AI.tile_index(str(call_tile)) if call_tile else None
    except ValueError:
        tile = None

    # 引擎的优先级：先问碰/槓，没人要才轮到吃。这里保持一致。
    can_pon = bool(kinds.get("pon"))
    can_kan = bool(kinds.get("kan"))
    can_chi = bool(kinds.get("chi"))
    if can_pon or can_kan:
        options = ["pass"] + (["pon"] if can_pon else []) + (["kan"] if can_kan else [])
        decision = Decision("call", options, tile=tile)
    elif can_chi:
        decision = Decision("chi", ["pass", "chi"], tile=tile)
    else:
        return {"ok": True, "decision": "unknown", "text": "（没有可鸣的牌）",
                "mode": "call", "hand_ok": True}

    players = 3 if bool(req.get("three_player", req.get("sanma", False))) else 4
    try:
        view = AI.build_view(
            hand, list(req.get("dora") or []),
            [list(r) for r in (req.get("discards") or [])],
            int(req.get("my_seat") or 0),
            players,
            turn=max((len(r) for r in (req.get("discards") or [])), default=1),
            melds=melds,
            riichi=req.get("riichi") or [],
            scores=req.get("scores") or [],
            wall_left=req.get("left_tile_count"),
            round_wind=req.get("round_wind"),
            kita=req.get("kita") or [],
            drawn=None,                     # 鸣牌时没有「刚摸到的那张」
            melds_all=req.get("melds_all"),
            ippatsu=req.get("ippatsu") or [],
            dealer=req.get("dealer"),
        )
    except Exception as exc:                # noqa: BLE001
        return {"ok": False, "error": f"拼观测失败：{exc!r}"}

    # 用和出牌建议同一套模型；网络没这个头（旧四麻权重缺 chi）就自动交给教师。
    model = AI.model_name(players)
    # 兜底教师的口味要跟着模型走：flash 的吃牌阈值是 5 向听、mountain 是 1。
    # 写死 normal 会让「flash 模型缺 chi 头」时自动变得比它本来的风格保守。
    style = next((s for s in ("normal", "mountain", "flash", "huge")
                  if s in model), "normal")
    agent = None
    who = model
    net = None
    if model != AI.TEACHER:
        net = AI.net(players)
        if net is not None:
            try:
                agent = MLAgent(net, seed=0, greedy=True, style=style)
            except Exception:               # noqa: BLE001
                agent = None
    if agent is None:
        agent = HeuristicAgent(0, style=style)
        who = "教师"
    elif net is not None and (
        decision.kind not in getattr(net, "kinds", [])
        or decision.kind in getattr(net, "untrained_heads", ())
    ):
        # 网络缺这个头（典型：老四麻权重没有 chi）-> MLAgent 内部会走教师。
        # 界面上要说清楚，别让人以为这是模型自己的判断。
        who = f"教师（{model} 缺 {decision.kind} 头）"
    try:
        choice = int(agent.decide(decision, view))
    except Exception as exc:                # noqa: BLE001
        return {"ok": False, "error": f"决策失败：{exc!r}"}
    choice = choice if 0 <= choice < len(decision.options) else 0
    label = str(decision.options[choice])

    before = shanten(view.hand, view.meld_count())
    cn = {"pass": "过（不鸣）", "pon": "碰", "kan": "杠", "chi": "吃"}.get(label, label)
    text = f"🀄 鸣牌建议：{cn}"
    detail = f"（向听 {before}"
    if label == "pon" and tile is not None:
        rest = list(view.hand)
        rest[tile] -= 2
        detail += f"→{shanten(rest, view.meld_count() + 1)}"
    elif label == "kan":
        detail += "；槓后要岭上补牌"
    elif label == "chi":
        # 吃的具体是哪两张：按向听最好的那一组显示（引擎也是这么挑的）
        best, best_s = None, 99
        for pair in kinds.get("chi", []):
            if len(pair) != 2:
                continue
            c = list(view.hand)
            ok = True
            for t in pair:
                try:
                    i = AI.tile_index(t)
                except ValueError:
                    ok = False
                    break
                if c[i] <= 0:
                    ok = False
                    break
                c[i] -= 1
            if not ok:
                continue
            s = shanten(c, view.meld_count() + 1)
            if s < best_s:
                best, best_s = pair, s
        if best:
            detail += f"→{best_s}，吃 {''.join(S.tile_cn(AI.tile_index(t)) for t in best)}"
    detail += "）"
    text += detail
    return {
        "ok": True, "decision": label, "text": text, "mode": "call",
        "mode_note": f"{who}，向听 {before}", "hand_ok": True,
        "options": [{"index": i, "label": str(o)} for i, o in enumerate(decision.options)],
    }


def cmd_list_models(req: dict) -> dict:
    """列出可用的模型（界面上的「模型」菜单用这个）。"""
    out = {}
    for players in (3, 4):
        items = []
        for name in sorted(p.stem for p in AI.WEIGHTS_DIR.glob("*.npz")):
            # 三麻的不带 4，四麻的带 4；「bcbase / prev / _v1」这类中间产物不列
            if name.endswith("_bcbase") or name.endswith("_prev") or name == "policy":
                continue
            if name.startswith("policy_"):
                continue
            has4 = "4v" in name
            if (players == 4) != has4:
                continue
            items.append(name)
        if AI.TEACHER not in items:
            items.insert(0, AI.TEACHER)      # 启发式教师：不依赖权重，永远可选
        out[str(players)] = {"current": AI.model_name(players), "models": items}
    return {"ok": True, "tables": out, "config": str(AI.config_path())}


def cmd_set_model(req: dict) -> dict:
    """换模型：写进 build/models.json，并让缓存失效（下一次分析就用新的）。"""
    players = 3 if req.get("three_player", True) else 4
    name = str(req.get("model") or "").strip()
    if not name:
        return {"ok": False, "error": "没给模型名"}
    # teacher 是 aimodel 里的启发式教师，没有权重文件，要单独放行
    if name != AI.TEACHER and not (AI.WEIGHTS_DIR / f"{name}.npz").exists():
        return {"ok": False, "error": f"没有这个模型：{name}.npz"}
    cfg = {}
    try:
        cfg = json.loads(AI.config_path().read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg[str(players)] = name
    AI.config_path().parent.mkdir(parents=True, exist_ok=True)
    AI.config_path().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    AI._user_config.cache_clear()          # 让下一次分析读到新配置
    AI._net_for.cache_clear()
    return {"ok": True, "players": players, "model": AI.model_name(players),
            "weights": ("（启发式，无权重文件）" if AI.model_name(players) == AI.TEACHER
                        else AI.model_path(players).name)}


COMMANDS = {"analyze": cmd_analyze, "analyze-call": cmd_analyze_call,
            "ping": lambda r: {"ok": True, "pong": True},
            "list-models": cmd_list_models, "set-model": cmd_set_model}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id")
        cmd = req.get("cmd")
        handler = COMMANDS.get(cmd)
        if handler is None:
            resp = {"ok": False, "error": f"未知指令: {cmd}"}
        else:
            try:
                resp = handler(req)
            except Exception:
                resp = {"ok": False, "error": traceback.format_exc(limit=3)}
        resp["id"] = rid
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
