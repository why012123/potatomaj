"""危险牌估计：基于**已解码的四家牌河**判断「打哪张最安全」。

只用得到三样东西，而且都是实测 100% 可靠的：
  * 我自己的手牌（识图）
  * 四家的牌河（协议解码，含张数和顺序）
  * 宝牌指示牌（识图）

三个判据，按可靠度从高到低：

1. **现物** —— 某家已经打过的牌，对他 100% 安全（同巡振听 / 舍张振听，
   他不可能和这张）。这是最硬的判据，不依赖任何推断。

2. **筋** —— 某家打过 4 万，那 1 万和 7 万就「过他筋」：两面听 23 万 / 56 万
   都会以 4 万为和牌张，既然 4 万安全，1/7 万的和牌可能就大幅下降。
   注意筋**不是绝对安全**（还有单骑、嵌张、双碰），只是概率明显降低。

3. **张数** —— 某张牌已现越多，别人听它的可能越低；4 张全现则相关听牌不成立。

局限（诚实说明）：**读不到立直/副露状态**（那部分协议还没破解），
所以这里给的是「相对安全度排序」，不是绝对保证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tiles import ALL_TILE_IDS, parse_tile, sort_tiles

#: 筋的对应关系：打过 x 会让 x±3 变安全（数牌限定）
def _suji_partners(tid: str) -> list[str]:
    t = parse_tile(tid)
    if t.suit == "z":
        return []
    out = []
    for d in (t.num - 3, t.num + 3):
        if 1 <= d <= 9:
            out.append(f"{d}{t.suit}")
    return out


def _index(tid: str) -> int:
    return ALL_TILE_IDS.index(tid)


@dataclass
class TileThreat:
    tile: str
    danger: float                     # 0 = 完全安全，越大越危险
    reasons: list[str] = field(default_factory=list)


def visible_counts(
    hand: list[str],
    discards: list[list[str]],
    dora: list[str] | None = None,
) -> dict[str, int]:
    """每种牌已经「看得见」几张（我的手牌 + 四家牌河 + 宝牌指示牌）。"""
    counts = {t: 0 for t in ALL_TILE_IDS}
    for t in list(hand) + [d for row in discards for d in row] + list(dora or []):
        if t in counts:
            counts[t] += 1
    return counts


def seat_danger(
    seat: int,
    tile: str,
    discards: list[list[str]],
    visible: dict[str, int],
    turn: int,
    is_riichi: bool = False,
) -> TileThreat:
    """我打 ``tile`` 时，``seat`` 家有多危险。

    ``is_riichi`` 是**确定信息**（协议解出来的），不是估计：这家已经宣告立直，
    也就是说他**一定听牌了**。所以对他的危险度不再是「回合越晚越可能听牌」那种
    猜测，而是实打实的：

    * 他打过的牌（现物）对他**绝对安全** —— 这一点没立直时也成立
    * 其余牌一律高危，不再按巡目打折
    * 筋、可见张数仍然起调节作用（筋确实更安全，剩得少也确实更难听）
    """
    row = discards[seat] if seat < len(discards) else []
    reasons: list[str] = []

    # 1) 现物：他打过这张 -> 对他绝对安全
    if tile in row:
        return TileThreat(tile, 0.0, [f"{seat}家现物"])

    if is_riichi:
        # 立直那家确定听牌 —— 基础危险度直接拉满，不再看巡目
        base = 1.0
        reasons.append(f"{seat}家立直")
    else:
        # 基础危险度：回合越晚，别人越可能已经听牌
        base = min(1.0, 0.25 + 0.075 * turn)

    # 2) 筋：他打过 tile±3 中的一张
    hit = [p for p in _suji_partners(tile) if p in row]
    if hit:
        base *= 0.45
        reasons.append(f"筋（{hit[0]} 已现）")

    # 3) 张数：这张牌已经现了几张
    seen = visible.get(tile, 0)
    left = 4 - seen
    if left <= 0:
        return TileThreat(tile, 0.0, reasons + ["四张全现"])
    base *= 0.55 + 0.15 * left          # 剩得越多，别人越可能听它
    if seen >= 3:
        reasons.append(f"已现 {seen} 张")

    return TileThreat(tile, round(base, 3), reasons or ["无筋且未见"])


def analyze(
    hand: list[str],
    discards: list[list[str]],
    my_seat: int = 0,
    dora: list[str] | None = None,
    seats: int | None = None,
    riichi: list[bool] | None = None,
) -> dict:
    """给我手里每张牌算「打出去的危险度」，从最安全到最危险排序。

    ``seats`` 是这一桌**实际有几家**。三麻是 3，四麻是 4。

    为什么必须显式传：``discards`` 是固定 4 行的数组，三麻那一桌永远有一行是空的。
    不传的话那个**不存在的第四家会被当成真人** —— 它没有牌河，所以现物、筋全不成立，
    但基础危险度照样算，而总体危险度取「最危险那一家」，于是它经常压过所有人。
    实测：三麻局面下每一张牌都被判成危险度 >= 0.5，界面上的红圈警示**永远亮着**，
    等于没有信息。
    """
    n = int(seats) if seats else len(discards)
    visible = visible_counts(hand, discards[:n], dora)
    turn = max((len(r) for r in discards[:n]), default=0)
    others = [s for s in range(n) if s != my_seat]

    riichi = list(riichi or [])
    riichi_seats = [s for s in others if s < len(riichi) and riichi[s]]
    results = []
    for tile in sort_tiles(set(hand)):
        threats = [(s, seat_danger(s, tile, discards, visible, turn,
                                  is_riichi=(s < len(riichi) and bool(riichi[s]))))
                   for s in others]
        # 整体危险度 = 对**最危险那一家**的危险度（木桶原理）
        worst_seat, worst = max(threats, key=lambda x: x[1].danger) if threats else (None, None)
        all_genbutsu = bool(threats) and all(t.danger <= 0 for _, t in threats)
        if all_genbutsu:
            why = "全场现物（每家都打过）"
            if riichi_seats:
                why += "，对立直那家也安全"
        else:
            why = "；".join(f"{s}家：" + "、".join(t.reasons) for s, t in threats if t.danger > 0)
        results.append({
            "tile": tile,
            "danger": 0.0 if all_genbutsu else round(worst.danger, 3),
            "all_safe": all_genbutsu,
            "worst_seat": worst_seat if not all_genbutsu else None,
            "why": why,
            "per_seat": [{"seat": s, "danger": t.danger, "reasons": t.reasons} for s, t in threats],
        })

    results.sort(key=lambda r: (r["danger"], r["tile"]))
    return {
        "turn": turn,
        "visible": {t: c for t, c in visible.items() if c},
        "tiles": results,
        "safest": [r["tile"] for r in results[:3]],
        "riskiest": [r["tile"] for r in results[-3:]][::-1],
    }


def recommend(
    hand: list[str],
    discards: list[list[str]],
    my_seat: int = 0,
    dora: list[str] | None = None,
    sanma: bool = True,
    melds: int = 0,
    seats: int | None = None,
    riichi: list[bool] | None = None,
) -> dict:
    """最有用的那一层：**在「不损牌效」的打牌里挑最安全的**。

    14 张等效（刚摸完）时，先算出哪些牌打了不掉向听，再在这些候选里按安全度
    排序；13 张等效时没有打牌决策，就给出「最安全的待打牌」供参考。

    ``melds`` 是副露数 —— 有副露时手牌行本来就短 3 的倍数，判断标准要按
    「等效张数」``len(hand) + 3 * melds`` 算，不能直接看手上有几张。
    """
    from .shanten import best_discards, shanten

    # 三麻一桌 3 家 —— 不传的话那个空的第四行会被当成真人（见 analyze 的说明）
    ana = analyze(hand, discards, my_seat, dora,
                  seats=seats if seats else (3 if sanma else 4),
                  riichi=riichi)
    by_tile = {r["tile"]: r for r in ana["tiles"]}
    effective = len(hand) + 3 * melds

    if effective == 14:
        keep = best_discards(hand, sanma, melds=melds)   # 打了不掉向听的那些
        cands = [by_tile[t] for t in keep if t in by_tile]
        if cands:
            cands.sort(key=lambda r: (r["danger"], r["tile"]))
            best = cands[0]
            ana["recommend"] = {
                "tile": best["tile"],
                "danger": best["danger"],
                "why": best["why"],
                "candidates": [c["tile"] for c in cands],
                "shanten": shanten(hand, melds=melds),
                "note": "在「打了不掉向听」的候选里挑的最安全一张",
            }
        else:
            ana["recommend"] = None
    else:
        safe = [r for r in ana["tiles"] if r["danger"] <= 0.15]
        ana["recommend"] = {
            "tile": safe[0]["tile"] if safe else None,
            "danger": safe[0]["danger"] if safe else None,
            "why": safe[0]["why"] if safe else "",
            "candidates": [r["tile"] for r in safe],
            "shanten": shanten(hand, melds=melds),
            "note": f"等效 {effective} 张，先记着这些安全牌",
        }
    return ana


def describe(analysis: dict) -> str:
    """一句话建议。"""
    rows = analysis["tiles"]
    if not rows:
        return "没有手牌"
    rec = analysis.get("recommend")
    safe_n = sum(1 for r in rows if r["danger"] <= 0.15)
    head = f"巡目 {analysis['turn']}"
    if rec and rec.get("tile"):
        return f"{head}　建议打 {rec['tile']}（{rec['why']}）　安全牌 {safe_n} 张"
    worst = rows[-1]
    return f"{head}　⚠ 手里没有安全牌，最危险的是 {worst['tile']}（{worst['why']}）"
