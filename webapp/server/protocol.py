"""协议层：把引擎状态转成**某个座位能看到的** JSON。

这是整个服务端最关键的一道闸门 —— 别人的手牌在这里就**根本不会被写进**
字典里，所以不管客户端怎么改，它都拿不到。前端「不显示」不算防护，
「不下发」才算。

快照结构（版本 ``v`` 固定为 1）::

    {
      "t": "state", "v": 1, "seq": 41,
      "phase": "playing",              # lobby|playing|hand_end|finished
      "room": {"id","name","players","mode","seats","ai_delay","rules",
               "next_acks","dealt","record"},
      "viewer": {"seat","spectator","reveal","name"},
      "round": {"hand_no","dealer","round_wind","turn","wall","dora",
                "ura","sticks","rounds","max_hands","scores_before"},
      "seats": [ {seat,name,kind,ai,online,host,score,score_delta,hand_count,
                  hand,drawn,melds,pond,pond_red,riichi,double_riichi,
                  riichi_tile,riichi_ippatsu,ippatsu,kita,seat_wind,is_dealer,
                  rel,menzen,connected,ai_degraded,no_record,declared_rank,
                  main_mode} ],
      "actions": {"kind","options","deadline","ms_left"} | null,
      "last": {...},                   # 最近一个动作，前端拿来做动画
      "result": {...} | null,          # 一局 / 一场的结果
      "log": ["..."], "chat": [...],
      "spectator_ok": bool, "finished": bool, "server_time": 1729.0,
    }

座位视野规则
------------

* 自己：``hand`` + ``drawn`` 全给
* 别家：``hand`` **永远是 null**，只给 ``hand_count``
* 观战者：只有纯 AI 房（``spectator_ok``）才能 ``reveal`` 全亮；
  有真人在座时服务端直接拒绝，返回的 ``hand`` 依旧是 null
"""

from __future__ import annotations

import time
from typing import Any

from aimodel import sanma as S
from aimodel.engine import Game, View

from . import tiles as T

PROTOCOL_VERSION = 1

KIND_CN = {
    "discard": "出牌",
    "tsumo": "自摸",
    "ron": "荣和",
    "riichi": "立直",
    "call": "鸣牌",
    "chi": "吃",
    "kita": "拔北",
    "kan": "杠",
    "kyuushu": "九种九牌",
}

OPTION_CN = {
    "yes": "要",
    "no": "不要",
    "pass": "过",
    "pon": "碰",
    "kan": "杠",
    "chi": "吃",
}

#: ★ 光有上面那张表不够：`yes/no` 是**通用**选项字，拔北 / 立直 / 和了 都用它，
#: 译成「要 / 不要」谁看得懂？（用户原话：「只显示 yes/no 干嘛？」）
#: 所以按**决策类型**再给一层标签：同一个 "yes"，在拔北里是「拔北」，
#: 在立直里是「立直」，在荣和里是「荣和」。
OPTION_CN_BY_KIND = {
    "kita": {"yes": "拔北", "no": "不拔"},
    "riichi": {"yes": "立直", "no": "不立直"},
    "ron": {"yes": "荣和", "no": "过"},
    "tsumo": {"yes": "自摸", "no": "过"},
    "kyuushu": {"yes": "九种九牌·流局", "no": "继续打"},
    "kan": {"pass": "不杠", "kan": "杠", "yes": "杠", "no": "不杠"},
    "call": {"pass": "过", "pon": "碰", "kan": "杠", "chi": "吃"},
}


def option_cn(kind: str, option: object) -> str:
    """一个选项按钮上该写什么字（先按决策类型查，再退回通用表）。

    吃的选项长这样：``chi:21:22``（引擎用**牌下标**编码「吞哪两张」）。
    这里翻成「吃 4s5s」—— 手里 12345s 遇到 3s 时，按钮就能一眼分清是吃
    12s 还是吃 45s（以前只有一个「吃」，等于引擎替玩家做了决定）。
    """
    key = str(option)
    if key.startswith("chi:"):
        try:
            _, a, b = key.split(":")
            return "吃 " + T.index_to_id(int(a)) + T.index_to_id(int(b))
        except (ValueError, IndexError):
            return "吃"
    return (OPTION_CN_BY_KIND.get(kind) or {}).get(key) or OPTION_CN.get(key) or key

WIND_CN = {27: "东", 28: "南", 29: "西", 30: "北"}
ROUND_WIND_CN = {27: "东", 28: "南"}

#: 座位相对自己的方位（逆时针：下家 -> 右，对家 -> 上，上家 -> 左）
REL_CN = ("自己", "下家", "对家", "上家")


def _wind_cn(idx: int) -> str:
    return WIND_CN.get(idx, "?")


def round_wind_cn(idx: int) -> str:
    return ROUND_WIND_CN.get(idx, "?")


def rel_label(viewer: int, seat: int, n: int) -> str:
    """``seat`` 相对 ``viewer`` 是下家 / 对家 / 上家。"""
    if viewer < 0 or viewer == seat:
        return "自己"
    d = (seat - viewer) % n
    if n == 4:
        return ("自己", "下家", "对家", "上家")[d]
    return ("自己", "下家", "上家")[d]


# ---------------------------------------------------------------------------
# 座位
# ---------------------------------------------------------------------------


def seat_view(
    game: Game,
    seat: int,
    *,
    viewer: int,
    reveal: bool,
    info: dict,
    scores_before: list[int],
    pond_red: list[int] | None = None,
) -> dict:
    """一个座位的公开信息 + （仅自己 / 全亮观战时）手牌。

    ``game.players`` 只在 ``Game._setup()`` 之后才存在，所以调用方必须
    先确认这一局已经发过牌（见 :func:`webapp.server.table.Room.dealt`）。

    ``pond_red``：牌河里每张牌是不是赤 5。引擎的 ``Player`` 没有这个字段
    （它是牌型下标的历史，赤 5 身份丢了），由房间在
    ``discard`` 事件里从 ``pl.red[tile]`` 现读现记（见 ``Room._on_ui``）。
    """
    players = game.players
    pl = players[seat]
    n = len(players)
    mine = seat == viewer
    shown = mine or (reveal and viewer < 0)
    red = list(pond_red) if pond_red is not None else []

    return {
        "seat": seat,
        "name": info.get("name") or f"P{seat}",
        "kind": info.get("kind", "human"),
        "ai": info.get("ai"),
        "online": bool(info.get("online", True)),
        "host": bool(info.get("host")),
        # 局间「已确认下一局」—— 客户端据此显示「还等谁确认」
        "next_ack": bool(info.get("next_ack")),
        "score": game.scores[seat],
        "score_delta": game.scores[seat] - scores_before[seat],
        "hand_count": pl.hand_size,
        "hand": T.hand_list(pl.hand, pl.red) if shown else None,
        "drawn": T.index_to_id(pl.drawn) if (shown and pl.drawn is not None) else None,
        "melds": [T.meld_json(m, viewer) for m in pl.melds],
        "pond": [T.index_to_id(t) for t in pl.pond],
        "pond_red": red,
        "riichi": pl.riichi,
        "double_riichi": pl.double_riichi,
        "riichi_tile": (
            T.index_to_id(pl.riichi_tile) if pl.riichi_tile is not None else None
        ),
        "riichi_ippatsu": bool(pl.ippatsu and pl.riichi),
        "ippatsu": bool(pl.ippatsu),
        "kita": pl.kita,
        "seat_wind": _wind_cn(game.seat_wind(seat)),
        "is_dealer": seat == game.dealer,
        "rel": rel_label(viewer, seat, n),
        "menzen": pl.menzen,
        "connected": bool(info.get("connected", True)),
        "ai_degraded": bool(info.get("ai_degraded")),
        "ai_degrade_reason": info.get("ai_degrade_reason", ""),
        "no_record": bool(info.get("no_record")),
        "declared_rank": info.get("declared_rank", ""),
        "main_mode": info.get("main_mode", ""),
    }


# ---------------------------------------------------------------------------
# 可选动作（**只有服务端能决定**）
# ---------------------------------------------------------------------------


def action_view(pending: Any, viewer: int, now: float) -> dict | None:
    """把一次待决策转成前端能用的按钮列表。

    不是自己的决策就给 ``null``（真人房）；观战者能看到内容但没按钮。

    ``pending.deadline is None`` = **这一手不计时**（房里只有一个真人，
    服务端会一直等）—— 这时把 ``deadline``/``ms_left`` 都给 null，
    前端据此不画思考条、不显示倒计时。
    """
    if pending is None:
        return None
    if pending.deadline is None:
        clock = {"deadline": None, "ms_left": None, "no_timeout": True}
    else:
        clock = {
            "deadline": round(pending.deadline, 3),
            "ms_left": max(0, int((pending.deadline - now) * 1000)),
        }
    if pending.seat != viewer:
        if viewer >= 0:
            return None
        base = {
            "kind": pending.kind,
            "seat": pending.seat,
            "kind_cn": KIND_CN.get(pending.kind, pending.kind),
            "options": [],
            "waiting": True,
        }
        base.update(clock)
        return base

    opts: list[dict] = []
    if pending.kind == "discard":
        for i, t in enumerate(pending.options):
            tid = T.index_to_id(int(t))
            opts.append({
                "index": i,
                "tile": tid,
                "cn": T.tile_cn(int(t)),
                "riichi_ok": i in pending.riichi_ok,
                "tsumogiri": bool(pending.drawn is not None
                                  and int(t) == pending.drawn),
            })
    else:
        for i, o in enumerate(pending.options):
            opts.append({
                "index": i,
                "label": str(o),
                "cn": option_cn(pending.kind, o),
            })

    mine = {
        "kind": pending.kind,
        "seat": pending.seat,
        "kind_cn": KIND_CN.get(pending.kind, pending.kind),
        "options": opts,
        "tile": T.index_to_id(int(pending.tile)) if pending.tile is not None else None,
        "tile_cn": T.tile_cn(int(pending.tile)) if pending.tile is not None else None,
        "waiting": False,
        "timeout": None if pending.timeout is None else round(pending.timeout, 1),
        "fallback": pending.fallback,
    }
    mine.update(clock)
    return mine


# ---------------------------------------------------------------------------
# 结算
# ---------------------------------------------------------------------------


def hand_result_view(game: Game) -> dict | None:
    """最近一局的完整结算（牌型 / 役种 / 番符 / 收支）。

    结算页要的东西全在这里：真役和宝牌**分两组**给，放铳者单独标出来。
    """
    if not game.hands:
        return None
    h = game.hands[-1]
    base: dict = {
        "hand_no": len(game.hands),
        "kind": h.kind,
        "turns": h.turns,
        "note": h.note,
        "deltas": list(h.deltas),
        "scores": list(game.scores),
        "sticks": game.sticks,
        "winner": h.winner,
        "loser": h.loser,
        "winners": list(h.winners or ()),
        "label": {
            "tsumo": "自摸", "ron": "荣和", "draw": "流局",
            "kyuushu": "九种九牌", "abort": "途中流局",
        }.get(h.kind, h.kind),
        "agari": [],
    }
    if h.kind in ("draw", "kyuushu", "abort"):
        base["tenpai"] = list(h.tenpai)
        return base

    winners = list(h.winners or (h.winner,))
    agaris = list(h.agaris or ((h.agari,) if h.agari else ()))
    for i, q in enumerate(winners):
        a = agaris[i] if i < len(agaris) else None
        pl = game.players[q]
        win_tile = None
        # 引擎在 _mark_agari 里记了「和的是哪张」，回放 / 结算要标红它
        if q == h.winner and game.last_agari and game.last_agari.get("tile", -1) >= 0:
            win_tile = T.index_to_id(int(game.last_agari["tile"]))
        elif pl.drawn is not None and h.kind == "tsumo":
            win_tile = T.index_to_id(pl.drawn)
        # ★ 这里**绝不能 continue**：这一手明明是荣和/自摸，只要 agari 明细缺一条
        #   （双响、winners 与 agaris 长度对不上等），老代码就把整条跳过 ——
        #   客户端拿到 agari=[] 就会把赢牌显示成「流局」（用户看到的那个 bug）。
        #   明细缺了也要出一条：座位、手牌、和了牌都在，役种给空数组即可。
        base["agari"].append({
            "seat": q,
            "tsumo": h.kind == "tsumo",
            "hand": T.hand_list(pl.hand, pl.red),
            "melds": [T.meld_json(m, -1) for m in pl.melds],
            "win_tile": win_tile,
            "yaku": ([{"key": k, "name": S.YAKU_CN.get(k, k), "han": v}
                      for k, v in a.yaku] if a is not None else []),
            "dora": ([{"key": k, "name": S.YAKU_CN.get(k, k), "han": v}
                      for k, v in a.dora_detail.items() if v] if a is not None else []),
            "han": getattr(a, "han", 0),
            "dora_han": getattr(a, "dora_han", 0),
            "total_han": getattr(a, "total_han", 0),
            "fu": getattr(a, "fu", 0),
            "yakuman": bool(getattr(a, "yakuman", False)),
            "describe": a.describe() if a is not None else "",
            "gain": h.deltas[q] if q < len(h.deltas) else 0,
        })
    return base


def match_result_view(result: Any) -> dict | None:
    if result is None:
        return None
    return {
        "scores": list(result.scores),
        "ranks": list(result.ranks),
        "deltas": list(result.deltas),
        "start_scores": list(result.start_scores),
        "rank_points": [round(x, 1) for x in result.rank_points],
        "coins": [round(x, 1) for x in result.coins],
        "seed": result.seed,
        "stats": [s.summary() for s in result.stats],
        "hands": [
            {
                "kind": h.kind,
                "text": h.describe(),
                "deltas": list(h.deltas),
                "turns": h.turns,
                "note": h.note,
            }
            for h in result.hands
        ],
    }


# ---------------------------------------------------------------------------
# 整张快照
# ---------------------------------------------------------------------------


def _ura_visible(game: Game) -> bool:
    """里宝牌什么时候能看：有人立直和牌之后。"""
    if not game.hands:
        return False
    last = game.hands[-1]
    if last.kind not in ("tsumo", "ron"):
        return False
    for a in (last.agaris or ((last.agari,) if last.agari else ())):
        if a is not None and getattr(a, "dora_detail", None) and "ura" in a.dora_detail:
            return True
    return False


def build_state(
    game: Game | None,
    *,
    viewer: int,
    reveal: bool,
    room: dict,
    seats: list[dict],
    phase: str,
    seq: int,
    spectator_ok: bool,
    scores_before: list[int],
    sticks_before: int,
    ready: bool = False,
    pending: Any = None,
    last_event: dict | None = None,
    finished: bool = False,
    result: dict | None = None,
    log: list[str] | None = None,
    chat: list[dict] | None = None,
    spectator_count: int = 0,
    now: float | None = None,
) -> dict:
    """拼一帧完整快照。

    ``ready`` 表示「这一局已经发牌了」—— ``Game`` 的 ``players`` / ``turn``
    这些都是 ``_setup()`` 里才建的，引擎线程刚起来的那一小段时间里
    去读它们会把整个事件循环回调炸掉（然后客户端永远等不到下一帧）。
    """
    now = time.time() if now is None else now
    out: dict = {
        "t": "state",
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "phase": phase,
        "now": round(now, 3),
        "room": room,
        "viewer": {
            "seat": viewer,
            "spectator": viewer < 0,
            "reveal": bool(reveal and spectator_ok),
            "name": (seats[viewer]["name"] if 0 <= viewer < len(seats) else "观战"),
        },
        "seats": seats,
        "actions": action_view(pending, viewer, now) if game else None,
        "last": last_event,
        "result": result,
        "log": (log or [])[-60:],
        "chat": (chat or [])[-40:],
        "spectator_ok": spectator_ok,
        "spectator_count": spectator_count,
        "finished": finished,
        "server_time": round(now, 3),
    }
    if game is None or not ready:
        out["round"] = None
        return out

    out["round"] = {
        # ★ 「第几局」不能用累计手数：连庄也 +1，会打出「东五局」这种东西。
        #   局数 = 庄家轮转次数 + 1（引擎的 `seat_rounds`，老引擎没有就退回手数），
        #   连庄次数 = `honba`（本场）。
        "hand_no": min(game.rounds, max(1, getattr(game, "seat_rounds", len(game.hands))
                                       + (0 if phase in ("hand_end", "finished") else 1))),
        "hands_played": len(game.hands),
        "honba": getattr(game, "honba", 0),
        "dealer": game.dealer,
        "round_wind": round_wind_cn(game.round_wind),
        "round_wind_idx": game.round_wind,
        "turn": game.turn + 1,
        "wall": game.wall_left(),
        "dora": [T.index_to_id(game._indicator(i)) for i in range(game.dora_slots)],
        "ura": [T.index_to_id(game._indicator(i, ura=True))
                for i in range(game.dora_slots if _ura_visible(game) else 0)],
        "sticks": game.sticks,
        "rounds": game.rounds,
        "max_hands": game.max_hands,
        "scores_before": list(scores_before),
        "sticks_before": sticks_before,
    }
    return out


def view_of(game: Game, seat: int) -> View:
    """给 agent / 提示用的座位视野（引擎原生对象）。"""
    return game.view_for(seat)
