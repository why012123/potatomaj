"""牌谱对拍：把**真实牌谱**喂回引擎，看它算出来的是不是和牌谱记录一致。

这是抓规则 bug 最狠的手段。之前项目里那个「混一色被判成清一色、副露白多算
3 番」的错，就是靠人眼永远看不出来、写用例也想不到 —— 但拿真实对局一跑就会
当场露馅：牌谱写的是 5 番 8000 点，引擎算出来 8 番 12000 点。

## 怎么做到「重放」

引擎本来是随机发牌的，这里给它**灌一副指定牌山**（``Game.rigged_wall``），
按牌谱还原：四家起手 → 摸牌顺序 → 岭上牌 → 王牌（宝牌指示）。然后让
:class:`LogAgent` 照着牌谱出牌，跑完整局，最后比对结果。

## 用法::

    python -m aimodel.logcheck game.json          # 对拍一个牌谱
    python -m aimodel.logcheck logs/ --limit 500  # 批量对拍一个目录
    python -m aimodel.logcheck --selftest         # 自检：自己导出再自己对拍
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import mjai, sanma as S
from .agents import Agent
from .engine import Decision, Game, View


# ---------------------------------------------------------------------------
# 把牌谱还原成引擎能吃的牌山
# ---------------------------------------------------------------------------
def kyoku_to_wall(kyoku: list[dict], n: int, rule: S.Rule) -> list[tuple[int, int]]:
    """一局 MJAI 事件 -> 引擎的牌山。

    排布顺序（和引擎内部一致）::

        [P0 的 13 张][P1 的 13 张]…  ← 按座位依次发牌
        [正常摸牌，按牌谱顺序]
        [岭上牌，倒序]               ← 引擎从牌山尾部取
        [王牌 dead_wall 张]           ← dead[0] 是第一个宝牌指示牌
    """
    start = kyoku[0]
    hands: list[list[tuple[int, int]]] = []
    for tehais in start["tehais"]:
        hands.append([mjai.tile_from_mjai(x) for x in tehais])

    front: list[tuple[int, int]] = []
    back: list[tuple[int, int]] = []
    prev_kan = False
    for e in kyoku[1:]:
        t = e.get("type")
        # 杠和**拔北**都从牌山尾部补牌 —— 三麻里拔北就是「北的杠」，
        # 漏掉它会让第一个拔北之后的牌山整体错位（自检里踩过）。
        if t in ("kan", "ankan", "kakan", "kita"):
            prev_kan = True
        elif t == "tsumo":
            tile = mjai.tile_from_mjai(e["pai"])
            (back if prev_kan else front).append(tile)
            prev_kan = False

    wall: list[tuple[int, int]] = []
    for h in hands[:n]:
        wall.extend(h)
    wall.extend(front)

    # ⚠️ 必须补齐到整副牌的张数：引擎是 `live = wall[:-dead_wall]`，
    # 少一张王牌就会整体错位、发牌全乱（自检里踩过这个坑）。
    total = len(S.tile_types(n)) * 4
    live_len = total - rule.dead_wall
    pad = live_len - len(wall) - len(back)
    if pad < 0:
        raise ValueError(f"牌谱里的摸牌数超出牌山容量（多 {(-pad)} 张）")
    # 垫牌放中间：岭上牌必须紧贴王牌，因为引擎从牌山尾部取岭上
    wall.extend([(0, 0)] * pad)
    wall.extend(reversed(back))

    # 王牌：dead[0]、dead[2]、dead[4]… 是宝牌指示牌，其余位置填无关牌
    dead: list[tuple[int, int]] = [(0, 0)] * rule.dead_wall
    indicators = [mjai.tile_from_mjai(e["dora_marker"])
                  for e in kyoku if e.get("type") == "dora"]
    for i, ind in enumerate(indicators[: max(1, rule.dead_wall // 2)]):
        if i * 2 < len(dead):
            dead[i * 2] = ind
    wall.extend(dead)
    return wall


# ---------------------------------------------------------------------------
# 照牌谱出牌的 agent
# ---------------------------------------------------------------------------
@dataclass
class Mismatch:
    seat: int
    kind: str
    expected: str
    got: str

    def __str__(self) -> str:
        return f"P{self.seat} {self.kind}: 牌谱写的是 {self.expected}，引擎这边 {self.got}"


class LogAgent(Agent):
    """照着牌谱出牌。

    **队列由引擎的动作流推进，不是由提问推进** —— 这一点很关键：
    立直之后引擎会自动摸切、*根本不问 agent*（这是对的设计），如果靠
    ``decide()`` 来消费队列，那些牌就会一直堆在队首，等该荣和的时候队首
    还压着两张打牌 → 答「不荣」→ 引擎判定见逃振听 → 整局跑歪。
    （这个坑在自检里花了三轮才挖出来。）

    所以：``decide()`` 只**看**队首来决定选哪个选项，真正把队列往前推的是
    :meth:`on_action`（由引擎的 ``on_action`` 钩子逐个派发）。
    """

    def __init__(self, seat: int, script: list[tuple],
                 problems: list[Mismatch] | None = None) -> None:
        self.seat = seat
        self.script = list(script)
        self.problems = problems if problems is not None else []

    # -- 由引擎动作流推进 ------------------------------------------------
    def on_action(self, a: dict) -> None:
        who = a.get("player", a.get("winner"))
        if who != self.seat:
            return
        k = a["kind"]
        if k == "discard":
            got = self._pop("discard")
            if got is None:
                self.problems.append(Mismatch(
                    self.seat, k, S.tile_cn(a["tile"]), "牌谱没记这一步"))
            elif got[2] != a["tile"]:
                self.problems.append(Mismatch(
                    self.seat, k, S.tile_cn(got[2]), S.tile_cn(a["tile"])))
        elif k == "riichi":
            self._pop("riichi")
        elif k == "kita":
            self._pop("kita")
        elif k == "win":
            if not self._pop("win"):
                self.problems.append(Mismatch(self.seat, k, "没有和牌", "引擎和了"))
        elif k == "call":
            got = self._pop("call")
            if got is None or got[1] != a["call"]:
                self.problems.append(Mismatch(
                    self.seat, k, str(got[1]) if got else "没记这一步", a["call"]))

    def _pop(self, kind: str):
        if self.script and self.script[0][0] == kind:
            return self.script.pop(0)
        return None

    def _peek(self, *kinds: str):
        if self.script and self.script[0][0] in kinds:
            return self.script[0]
        return None

    # -- 回答引擎的提问（只看，不推进队列）-------------------------------
    def decide(self, decision: Decision, view: View) -> int:
        k = decision.kind
        opts = list(decision.options)

        def idx(name: str) -> int:
            return opts.index(name) if name in opts else 0

        if k == "discard":
            got = self._peek("discard")
            if got is None:
                self.problems.append(Mismatch(
                    self.seat, k, "打牌", "牌谱已经没动作了"))
                return 0
            tile = got[2]
            if tile not in opts:
                self.problems.append(Mismatch(
                    self.seat, k, S.tile_cn(tile), "手里没有这张牌"))
                return 0
            return opts.index(tile)

        if k == "riichi":
            return idx("yes") if self._peek("riichi") else idx("no")

        # ⚠️ 副露和荣和都必须核对**牌是不是同一张** —— 只看动作种类的话，
        # 引擎随便递一张牌过来 agent 都会照单全收，对拍就废了（自检里踩过）。
        # tile 为 None 时（引擎没告诉我们是哪张）退回只比动作种类。
        want = decision.tile

        def tile_ok(got) -> bool:
            return want is None or got[2] == want

        if k == "call":
            got = self._peek("call")
            if got and got[1] in ("pon", "kan") and tile_ok(got):
                return idx(got[1])
            return idx("pass")

        if k == "chi":
            got = self._peek("call")
            if got and got[1] == "chi" and tile_ok(got):
                return idx("chi")
            return idx("pass")

        if k == "kan":
            got = self._peek("call")
            if got and got[1] in ("ankan", "kakan", "kan") and tile_ok(got):
                return idx("kan")
            return idx("pass")

        if k == "kita":
            return idx("yes") if self._peek("kita") else idx("no")

        if k == "ron":
            got = self._peek("win")
            return idx("yes") if got and tile_ok(got) else idx("no")

        if k == "tsumo":
            return idx("yes") if self._peek("win") else idx("no")

        return 0 if k != "kyuushu" else idx("no")


def build_scripts(kyoku: list[dict], n: int) -> list[list[tuple]]:
    """按座位把牌谱拆成「该谁做什么」的队列。

    ⚠️ 每一项都是**统一的三元组** ``(种类, 子类, 牌)``，没有的就填 ``None``。
    之前混着用 2 元组和 3 元组，结果核对牌面时下标错位（自检里踩过）。
    """
    scripts: list[list[tuple]] = [[] for _ in range(n)]
    for e in kyoku[1:]:
        t = e.get("type")
        if t == "dahai":
            scripts[e["actor"]].append(
                ("discard", None, mjai.tile_from_mjai(e["pai"])[0]))
        elif t == "reach":
            scripts[e["actor"]].append(("riichi", None, None))
        elif t == "kita":
            scripts[e["actor"]].append(("kita", None, None))
        elif t in ("chi", "pon", "kan"):
            scripts[e["actor"]].append(
                ("call", t, mjai.tile_from_mjai(e["pai"])[0]))
        elif t == "ankan":
            scripts[e["actor"]].append(
                ("call", "ankan", mjai.tile_from_mjai(e["consumed"][0])[0]))
        elif t == "kakan":
            scripts[e["actor"]].append(
                ("call", "kakan",
                 mjai.tile_from_mjai(e.get("pai") or e["consumed"][0])[0]))
        elif t == "hora":
            scripts[e["actor"]].append(
                ("win", None, mjai.tile_from_mjai(e["pai"])[0]))
    return scripts


# ---------------------------------------------------------------------------
# 对拍
# ---------------------------------------------------------------------------
@dataclass
class KyokuReport:
    hand_no: int
    ok: bool
    notes: list[str] = field(default_factory=list)
    mismatches: list[Mismatch] = field(default_factory=list)

    def line(self) -> str:
        flag = "✅" if self.ok else "❌"
        body = "；".join(self.notes) if self.notes else "一致"
        extra = f"　（{len(self.mismatches)} 处出牌不符）" if self.mismatches else ""
        return f"{flag} 第 {self.hand_no} 局：{body}{extra}"


def check_kyoku(kyoku: list[dict], n: int = 3, rule: S.Rule | None = None,
                verbose: bool = False) -> KyokuReport:
    """把一局牌谱喂回引擎，比对结果。"""
    rule = rule or S.rule_for(n)
    start = kyoku[0]
    dealer = int(start.get("oya", 0))
    scripts = build_scripts(kyoku, n)
    agents = [LogAgent(i, scripts[i]) for i in range(n)]

    game = Game(agents, seed=0, rule=rule)
    game.rigged_wall = kyoku_to_wall(kyoku, n, rule)
    if not verbose:
        game.on_event = None

    def dispatch(a: dict) -> None:
        for ag in agents:
            ag.on_action(a)

    game.on_action = dispatch
    res = game.play_hand(dealer)

    rep = KyokuReport(hand_no=int(start.get("kyoku", 1)), ok=True)
    for a in agents:
        rep.mismatches.extend(a.problems)

    hora = mjai.hora_of(kyoku)
    if hora is None:
        if res.kind not in ("draw", "kyuushu", "abort"):
            rep.ok = False
            rep.notes.append(f"牌谱是流局，引擎判成 {res.kind}")
            return rep
        # 流局也要逐项比 —— 不检的话「听牌判定 / 不聴罰符」这类错永远抓不到
        ryu = mjai.ryukyoku_of(kyoku)
        if ryu is not None:
            if ryu.get("tenpai") is not None and res.tenpai is not None:
                want = [bool(x) for x in ryu["tenpai"]]
                got = [bool(x) for x in res.tenpai]
                if want != got:
                    rep.ok = False
                    rep.notes.append(f"听牌判定不符：牌谱 {want}，引擎 {got}")
            if ryu.get("deltas"):
                want_d = [int(x) for x in ryu["deltas"]]
                got_d = [int(x) for x in res.deltas]
                if want_d != got_d:
                    rep.ok = False
                    rep.notes.append(f"流局收支不符：牌谱 {want_d}，引擎 {got_d}")
        return rep

    # ── 和牌：比对谁和、和哪张、番符、点数 ──
    if res.kind in ("draw", "kyuushu", "abort"):
        rep.ok = False
        rep.notes.append(f"牌谱是和牌（P{hora['actor']}），引擎判成{res.kind}")
        return rep
    if res.winner != hora["actor"]:
        rep.ok = False
        rep.notes.append(f"和牌者不符：牌谱 P{hora['actor']}，引擎 P{res.winner}")
    want_tile, _ = mjai.tile_from_mjai(hora["pai"])
    got_tile = game.last_agari["tile"] if game.last_agari else None
    if got_tile is not None and got_tile != want_tile:
        rep.ok = False
        rep.notes.append(
            f"和牌张不符：牌谱 {S.tile_cn(want_tile)}，引擎 {S.tile_cn(got_tile)}")

    agari = res.agari
    if agari is not None:
        if "han" in hora and agari.total_han != hora["han"]:
            rep.ok = False
            rep.notes.append(f"番数不符：牌谱 {hora['han']} 番，引擎 {agari.total_han} 番")
        if "fu" in hora and agari.fu != hora["fu"]:
            rep.ok = False
            rep.notes.append(f"符数不符：牌谱 {hora['fu']} 符，引擎 {agari.fu} 符")
        if "yaku" in hora:
            want = Counter(y.get("name") if isinstance(y, dict) else str(y)
                           for y in hora["yaku"])
            got = Counter(agari.yaku_cn(include_dora=True))
            if want != got:
                rep.ok = False
                rep.notes.append(f"役种不符：牌谱 {sorted(want)}，引擎 {sorted(got)}")
    if "deltas" in hora and list(res.deltas) != list(hora["deltas"]):
        rep.ok = False
        rep.notes.append(f"点数收支不符：牌谱 {list(hora['deltas'])}，引擎 {list(res.deltas)}")
    return rep


def check_file(path: str | Path, n: int = 3, limit: int | None = None) -> list[KyokuReport]:
    events = mjai.read(path)
    reports = []
    for i, kyoku in enumerate(mjai.iter_kyoku(events)):
        if limit and i >= limit:
            break
        reports.append(check_kyoku(kyoku, n=n))
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _summary(reports: list[KyokuReport], title: str = "") -> None:
    if title:
        print(f"\n═══ {title} ═══")
    ok = sum(1 for r in reports if r.ok)
    print(f"  {ok}/{len(reports)} 局与牌谱完全一致")
    problems = Counter()
    for r in reports:
        for note in r.notes:
            problems[note.split("：")[0]] += 1
    for kind, cnt in problems.most_common(8):
        print(f"    · {kind} × {cnt}")


def selftest(n_games: int = 5, players: int = 3) -> None:
    """自检：用自己的引擎导出牌谱，再自己对拍 —— 应该 100% 一致。

    这条通了，说明「导出 → 还原牌山 → 重放 → 比对」整条链是对的，
    等真实牌谱到手就能直接用。
    """
    from .agents import HeuristicAgent

    all_reports: list[KyokuReport] = []
    for seed in range(n_games):
        acts: list[dict] = []
        game = Game([HeuristicAgent(seed * players + i) for i in range(players)],
                    seed=seed, rule=S.rule_for(players))
        game.on_action = acts.append
        game.play()
        events = mjai.export(acts, players=players)
        for kyoku in mjai.iter_kyoku(events):
            all_reports.append(check_kyoku(kyoku, n=players))
    _summary(all_reports, f"自检：{n_games} 场自己导出的牌谱，重放对拍")
    bad = [r for r in all_reports if not r.ok]
    for r in bad[:6]:
        print("   ", r.line())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="牌谱对拍（真实牌谱喂回引擎，比结果）")
    ap.add_argument("path", nargs="?",
                    help="MJAI 牌谱文件或目录。**一个文件可以放任意多局** —— "
                         "按事件流里的 start_kyoku 自动切分，不需要一局一个文件")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--limit", type=int, default=None, help="最多对拍几局")
    ap.add_argument("--selftest", action="store_true", help="自检：自己导出再自己对拍")
    ap.add_argument("--games", type=int, default=5, help="自检跑几场")
    args = ap.parse_args(argv)

    if args.selftest or not args.path:
        selftest(args.games, args.players)
        return

    p = Path(args.path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        print(f"没找到牌谱：{args.path}")
        return
    reports: list[KyokuReport] = []
    total_found = 0
    for f in files:
        events = mjai.read(f)
        ks = list(mjai.iter_kyoku(events))
        total_found += len(ks)
        take = len(ks) if not args.limit else min(args.limit, len(ks))
        # ★ 必须把「切出几局」和「比了几局」都打出来 —— 不然「8/8 一致」会被
        #   误读成「全部都对」，而实际上可能只切出了 8 局（踩过这个坑）
        print(f"  {f.name}：{len(events)} 条事件 → 切出 {len(ks)} 局，本次对拍 {take} 局")
        if len(ks) == 0 and len(events) > 0:
            print("    ⚠️ 一条都没切出来 —— 检查牌谱里有没有 start_kyoku 事件")
        for ky in ks[:take]:
            reports.append(check_kyoku(ky, n=args.players))
    print()
    print(f"  合计：切出 {total_found} 局，本次对拍 {len(reports)} 局")
    _summary(reports, "对拍结果（累计）")
    bad = [r for r in reports if not r.ok]
    if bad:
        print("\n不一致的局（前 10）：")
        for r in bad[:10]:
            print("  ", r.line())


if __name__ == "__main__":
    main()
