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


COMMANDS = {"analyze": cmd_analyze, "ping": lambda r: {"ok": True, "pong": True},
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
