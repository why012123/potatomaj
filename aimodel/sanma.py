"""三麻（三人麻雀）规则内核：牌型表示、役种判定、符与点数计算。

牌型沿用外层工程的 34 种表示（``potatomaj.tiles.TILE_IDS``）::

    0..8   1m..9m      9..17  1p..9p
    18..26 1s..9s      27..33 1z..7z（东 南 西 北 白 发 中）

三麻只保留 **27 种牌型**（万子只剩 1m / 9m），每种 4 张，共 **108 张**。

实现的三麻规则（默认对齐雀魂三麻，全部可通过 :class:`Rule` 调整）：

* **108 张牌**：去掉 2m~8m，保留 1m/9m、全部筒索、全部字牌
* **不能吃（チー）**，只能碰 / 杠；可以**拔北**（抜きドラ，每张算 1 宝牌）
* **北不是役牌**，但可以留在手里当普通牌用（对子 / 刻子）
* **ツモ損なし**：子家自摸由另外两家各付 2× 基本点，总额与四麻一致
* 数え役満 13 翻起；不聴罰符共 3000 点
* 三色同順在数值上不可能（万子没有 2m~8m），但 **三色同刻是可能的**
* 一気通貫只能靠筒 / 索完成；国士无双仍是 13 种幺九

役满：国士无双（十三面双倍）、四暗刻（单骑双倍）、大三元、小/大四喜、
字一色、绿一色、清老头、九莲宝灯（纯正双倍）、四杠子、天和、地和。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from potatomaj.tiles import TILE_IDS, parse_tile

N_TYPES = 34

#: 三麻实际存在的 27 种牌型（万子只有 1m=0 / 9m=8）
SANMA_TYPES: tuple[int, ...] = (0, 8) + tuple(range(9, 34))

#: 四麻：34 种全都要
FOUR_TYPES: tuple[int, ...] = tuple(range(34))


def tile_types(players: int = 3) -> tuple[int, ...]:
    """这一桌实际用到的牌型。三麻 27 种（108 张），四麻 34 种（136 张）。"""
    return SANMA_TYPES if players == 3 else FOUR_TYPES


def aka_types(players: int = 3) -> tuple[int, ...]:
    """赤 5 用哪几种牌：三麻赤 5p/5s，四麻再加赤 5m。"""
    return AKA_TYPES if players == 3 else (4,) + AKA_TYPES

#: 字牌下标
EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
HAKU, HATSU, CHUN = 31, 32, 33

#: 三元牌
DRAGONS: tuple[int, ...] = (HAKU, HATSU, CHUN)

#: 风牌
WINDS: tuple[int, ...] = (EAST, SOUTH, WEST, NORTH)

#: 幺九牌（1/9 数牌 + 全部字牌），三麻里仍是 13 种
YAOCHU: tuple[int, ...] = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)

#: 赤 5：三麻只有赤 5p / 赤 5s
AKA_TYPES: tuple[int, ...] = (13, 22)

#: 绿一色用到的牌型（2s 3s 4s 6s 8s 發）
GREEN_TYPES: frozenset[int] = frozenset({19, 20, 21, 23, 25, HATSU})

#: 役名 -> 中文（日志 / 报告用）
YAKU_CN: dict[str, str] = {
    "menzen_tsumo": "门前清自摸和",
    "riichi": "立直",
    "double_riichi": "两立直",
    "ippatsu": "一发",
    "pinfu": "平和",
    "tanyao": "断幺九",
    "iipeikou": "一盃口",
    "ryanpeikou": "二盃口",
    "haitei": "海底摸月",
    "houtei": "河底捞鱼",
    "rinshan": "岭上开花",
    "chankan": "抢杠",
    "yakuhai_haku": "役牌·白",
    "yakuhai_hatsu": "役牌·发",
    "yakuhai_chun": "役牌·中",
    "yakuhai_round": "场风",
    "yakuhai_seat": "自风",
    "sanshoku": "三色同顺",
    "sanshoku_doukou": "三色同刻",
    "sankantsu": "三杠子",
    "toitoi": "对对和",
    "sanankou": "三暗刻",
    "shousangen": "小三元",
    "honroutou": "混老头",
    "honchantai": "混全带幺九",
    "junchantai": "纯全带幺九",
    "ittsuu": "一气通贯",
    "chiitoi": "七对子",
    "honitsu": "混一色",
    "chinitsu": "清一色",
    "dora": "宝牌",
    "ura": "里宝牌",
    "aka": "赤宝牌",
    "kita": "拔北宝牌",
    "kokushi": "国士无双",
    "kokushi13": "国士无双十三面",
    "suuankou": "四暗刻",
    "suuankou_tanki": "四暗刻单骑",
    "daisangen": "大三元",
    "shousuushii": "小四喜",
    "daisuushii": "大四喜",
    "tsuuiisou": "字一色",
    "ryuuiisou": "绿一色",
    "chinroutou": "清老头",
    "chuuren": "九莲宝灯",
    "chuuren_pure": "纯正九莲宝灯",
    "suukantsu": "四杠子",
    "tenhou": "天和",
    "chiihou": "地和",
}

#: 役满名
YAKUMAN_NAMES: frozenset[str] = frozenset(
    {
        "kokushi", "kokushi13", "suuankou", "suuankou_tanki", "daisangen",
        "shousuushii", "daisuushii", "tsuuiisou", "ryuuiisou", "chinroutou",
        "chuuren", "chuuren_pure", "suukantsu", "tenhou", "chiihou",
    }
)

#: 副露降番（喰い下がり）：役 -> (门清翻, 副露翻)
KUISAGARI: dict[str, tuple[int, int]] = {
    "sanshoku": (2, 1),
    "honchantai": (2, 1),
    "junchantai": (3, 2),
    "ittsuu": (2, 1),
    "honitsu": (3, 2),
    "chinitsu": (6, 5),
}


# ---------------------------------------------------------------------------
# 规则开关
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """一套麻将规则。默认值对齐雀魂三麻；四麻用 :func:`rule_for` 取。"""

    #: 人数：3 = 三麻，4 = 四麻
    players: int = 3
    #: ツモ損：子家自摸时，亲家付 2a、另一子家付 a（总额 3a）。
    #: 雀魂三麻官方 FAQ 明确采用「ツモ損」，天凤三麻同样是ツモ損。
    tsumo_loss: bool = True
    #: 数え役满的翻数门槛
    kazoe_han: int = 13
    #: 不聴罰符总额（由不聴者支付、聴牌者均分）——三麻是 2000，不是四麻的 3000
    noten_penalty: int = 2000
    #: 双响是否有效（False = 头跳）。雀魂 / 天凤三麻均为双荣和
    double_ron: bool = True
    #: 是否使用赤 5（三麻：赤 5p/5s；四麻：赤 5m/5p/5s）
    aka: bool = True
    #: 拔北是否算宝牌
    kita_dora: bool = True
    #: 北是否算役牌（雀魂三麻：不算）
    north_yakuhai: bool = False
    #: 连风牌（场风 == 自风）的雀头符
    renpu_fu: int = 4
    #: 起始点数（雀魂三麻段位战：35000 起、40000 返）
    start_score: int = 35000
    #: 返し点
    return_score: int = 40000
    #: 顺位点（ウマ）：三麻东风场是 1位+15 / 2位±0 / 3位-15（单位：千点）
    uma: tuple[int, ...] = (15, 0, -15)
    #: 额外段位分：三麻 1位+30、2位+0，3位的扣分随当前段位变化（这里取 -30 作默认）
    rank_bonus: tuple[int, ...] = (30, 0, -30)
    #: 入场料（铜币，只用于展示）
    entry_fee: int = 2500
    #: 场代（铜币，只用于展示）
    table_fee: int = 500
    #: 兑换比例：1000 点 = 50 铜币（只用于展示）
    points_per_coin: int = 50
    #: 王牌张数：三麻是 18 张 = 8 张岭上（杠 4 + 拔北 4）+ 5 宝牌 + 5 里宝
    dead_wall: int = 18
    #: 宝牌指示牌槽位上限（只有杠会翻新指示牌，拔北不翻）
    dora_slots: int = 5
    #: 拔北是否翻开新的宝牌指示牌（雀魂 / 天凤：不翻）
    kita_flips_dora: bool = False
    #: 万子只有 1m/9m，所以指示牌 1m 的宝牌是 9m、9m 的宝牌是 1m
    man_dora_cycle: bool = True
    #: 四槓散了：不同玩家合计开出第 4 个杠时途中流局
    abort_four_kans: bool = True
    #: 东风战打满（オーラス）后，一位未满返点就南入 —— 雀魂官方规则，默认开
    sudden_death: bool = True
    #: 供托（立直棒）
    riichi_stick: int = 1000
    #: 箱割れ：低于此点数立即结束对局
    bust_score: int = 0
    #: 是否允许九种九牌途中流局
    kyuushu_kyuuhai: bool = True
    #: 开局手牌张数
    hand_size: int = 13


DEFAULT_RULE = Rule()


# ---------------------------------------------------------------------------
# 显示 / 解析辅助
# ---------------------------------------------------------------------------


def ranks_of(scores: Sequence[int]) -> list[int]:
    """点数 -> 顺位（0 = 一位）。同分时按座位顺序，与雀魂一致。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def rank_points(scores: Sequence[int], rule: Rule = DEFAULT_RULE) -> list[float]:
    """雀魂段位场的最终得分（单位：千点）::

        (最终点数 - 返し点) / 1000  +  顺位点(uma)  +  额外段位分

    这就是「段位战」真正在算的数：东风场只有三局，名次比零碎点数重要得多
    （三麻 1 位 +15+30、3 位 -15-30，一来一回 90 点，相当于 9 万分），
    所以训练和评测都以它为准，而不是原始得失点。
    """
    ranks = ranks_of(scores)
    return [
        (scores[i] - rule.return_score) / 1000.0
        + rule.uma[ranks[i]]
        + rule.rank_bonus[ranks[i]]
        for i in range(len(scores))
    ]


def coin_delta(scores: Sequence[int], rule: Rule = DEFAULT_RULE) -> list[float]:
    """铜币收支（只用于展示）：点数按比例兑换，再扣入场料与场代。"""
    return [
        scores[i] / 1000.0 * rule.points_per_coin - rule.entry_fee - rule.table_fee
        for i in range(len(scores))
    ]


def tile_id(index: int) -> str:
    """牌型下标 -> ``"1m"`` / ``"5p"`` / ``"1z"``。"""
    if 0 <= index < N_TYPES:
        return TILE_IDS[index]
    raise IndexError(f"牌型下标越界: {index}")


def tile_cn(index: int) -> str:
    """牌型下标 -> 中文牌名（``0`` -> 一萬，``27`` -> 东）。"""
    return str(parse_tile(TILE_IDS[index]))


def hand_str(counts: Sequence[int]) -> str:
    """计数数组 -> ``"123m456p77z"`` 紧凑记法。"""
    out: list[str] = []
    for suit, base, width in (("m", 0, 9), ("p", 9, 9), ("s", 18, 9), ("z", 27, 7)):
        digits = "".join(str(n + 1) * counts[base + n] for n in range(width))
        if digits:
            out.append(digits + suit)
    return "".join(out)


def parse_hand(text: str) -> list[int]:
    """``"123m456p77z"`` -> 34 长度计数数组（``0m/0p/0s`` 并入普通 5）。"""
    counts = [0] * N_TYPES
    digits: list[str] = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        elif ch in "mpsz":
            base = {"m": 0, "p": 9, "s": 18, "z": 27}[ch]
            for d in digits:
                n = int(d)
                counts[base + (5 if n == 0 else n) - 1] += 1
            digits = []
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"无法解析的字符: {ch!r}（在 {text!r} 中）")
    if digits:
        raise ValueError(f"结尾的数字没有花色: {text!r}")
    return counts


# ---------------------------------------------------------------------------
# 副露
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Meld:
    """一组副露。三麻没有吃，只有碰 / 杠；四麻有吃。"""

    kind: str  # "pon" | "minkan" | "ankan" | "kakan" | "chi"
    tile: int  # 牌型下标（吃的时候是顺子里最小的那张）
    from_player: int = -1  # 副露来源座位；-1 表示自己（暗杠 / 加杠）
    aka: int = 0  # 这组副露里含几张赤 5
    #: 吃的时候「被吃的那张」（来自哪张打牌）。顺子最小的那张在 ``tile``，
    #: 但被吃的不一定是它 —— 导出 MJAI 时 ``pai`` 要的是**被吃的这张**。
    called: int = -1

    @property
    def is_kan(self) -> bool:
        return self.kind in ("minkan", "ankan", "kakan")

    @property
    def is_chi(self) -> bool:
        return self.kind == "chi"

    @property
    def is_concealed(self) -> bool:
        """暗杠不破坏门清。"""
        return self.kind == "ankan"

    @property
    def tiles(self) -> tuple[int, ...]:
        if self.is_chi:
            return (self.tile, self.tile + 1, self.tile + 2)
        return (self.tile,) * (4 if self.is_kan else 3)

    def __str__(self) -> str:  # pragma: no cover - 展示用
        if self.is_chi:
            return "吃" + "".join(tile_cn(t) for t in self.tiles)
        mark = {"pon": "碰", "minkan": "明杠", "ankan": "暗杠", "kakan": "加杠"}[self.kind]
        return f"{mark}{tile_cn(self.tile)}"


# ---------------------------------------------------------------------------
# 和牌上下文与结果
# ---------------------------------------------------------------------------


@dataclass
class WinContext:
    """和牌时的全部上下文。"""

    win_tile: int
    is_tsumo: bool
    round_wind: int = EAST
    seat_wind: int = EAST
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chiihou: bool = False
    dora: int = 0
    ura: int = 0
    aka: int = 0
    kita: int = 0
    rule: Rule = DEFAULT_RULE


@dataclass
class Agari:
    """一次和牌的判定结果。"""

    han: int  # 不含宝牌的翻数
    fu: int
    yaku: list[tuple[str, int]]  # (役名 key, 翻数) —— 只含真役
    dora_han: int
    yakuman: int  # 役满倍数，0 表示普通和牌
    dora_detail: dict[str, int] = field(default_factory=dict)

    @property
    def total_han(self) -> int:
        return self.han + self.dora_han

    def yaku_cn(self, include_dora: bool = True) -> list[str]:
        out = [YAKU_CN.get(k, k) + (str(v) if v > 1 else "") for k, v in self.yaku]
        if include_dora:
            for k, v in self.dora_detail.items():
                if v:
                    out.append(f"{YAKU_CN.get(k, k)}{v if v > 1 else ''}")
        return out

    def describe(self) -> str:
        if self.yakuman:
            return f"{self.yakuman}倍役满 " + " ".join(self.yaku_cn())
        return f"{self.total_han}翻{self.fu}符 " + " ".join(self.yaku_cn())


# ---------------------------------------------------------------------------
# 手牌拆解
# ---------------------------------------------------------------------------

#: 一个暗面子：("tri", 牌型) 或 ("seq", 起始牌型)
MeldShape = tuple[str, int]


def _can_seq(i: int) -> bool:
    return i < 27 and (i % 9) <= 6


def _decompose(
    counts: list[int], need_melds: int, need_pair: int, start: int = 0
) -> Iterator[tuple[list[MeldShape], int | None]]:
    """枚举「need_melds 个面子 + need_pair 个雀头」的所有拆法。

    会就地修改 ``counts`` 再还原，属内部函数。
    """
    if need_melds * 3 + need_pair * 2 > sum(counts):
        return
    i = start
    while i < N_TYPES and counts[i] == 0:
        i += 1
    if i >= N_TYPES:
        if need_melds == 0 and need_pair == 0:
            yield [], None
        return

    if need_melds > 0:
        if counts[i] >= 3:
            counts[i] -= 3
            for melds, pair in _decompose(counts, need_melds - 1, need_pair, i):
                yield [("tri", i), *melds], pair
            counts[i] += 3
        if _can_seq(i) and counts[i + 1] > 0 and counts[i + 2] > 0:
            counts[i] -= 1
            counts[i + 1] -= 1
            counts[i + 2] -= 1
            for melds, pair in _decompose(counts, need_melds - 1, need_pair, i):
                yield [("seq", i), *melds], pair
            counts[i] += 1
            counts[i + 1] += 1
            counts[i + 2] += 1

    if need_pair == 1 and counts[i] >= 2:
        counts[i] -= 2
        for melds, _ in _decompose(counts, need_melds, 0, i):
            yield melds, i
        counts[i] += 2

    counts[i] -= 1  # 这张牌不参与任何面子
    for melds, pair in _decompose(counts, need_melds, need_pair, i):
        yield melds, pair
    counts[i] += 1


# ---------------------------------------------------------------------------
# 和牌判定
# ---------------------------------------------------------------------------


def dora_from_index(i: int, rule: Rule = DEFAULT_RULE) -> int:
    """宝牌指示牌下标 -> 宝牌下标。

    数牌 8->9->1 循环；字牌 北(4z)->东(1z)、中(7z)->白(5z)。
    三麻没有 2m~8m，所以 ``1m`` 的宝牌是 ``9m``、``9m`` 的宝牌是 ``1m``。
    """
    if i >= 27:
        if i == NORTH:
            return EAST
        if i == CHUN:
            return HAKU
        return i + 1
    if i in (0, 8) and rule.man_dora_cycle and rule.players == 3:
        return 8 if i == 0 else 0
    base = i // 9 * 9
    n = i % 9
    return base + (0 if n == 8 else n + 1)


def is_winning_shape(counts: Sequence[int], n_melds: int) -> bool:
    """给定暗牌计数与「已有副露数」，判断是否和牌形。

    有 ``n_melds`` 组副露时，暗牌 = ``(4 - n_melds)`` 组面子 + 1 个雀头。
    """
    need = 4 - n_melds
    if need < 0 or sum(counts) != need * 3 + 2:
        return False
    for melds, pair in _decompose(list(counts), need, 1):
        if pair is not None and len(melds) == need:
            return True
    return False


def evaluate_hand(
    counts: Sequence[int], melds: Sequence[Meld], ctx: WinContext
) -> Agari | None:
    """判定一副和牌，返回翻/符/役种；无役或不成和牌形返回 ``None``。

    ``counts`` 是 **暗牌** 的 34 长度计数数组（已经包含和了牌）。
    """
    if len(melds) > 4:
        return None
    need = 4 - len(melds)
    if sum(counts) != need * 3 + 2:
        return None

    rule = ctx.rule
    menzen = all(m.is_concealed for m in melds)
    # 役牌 = 三元牌 + 场风 + 自风。三麻没有北家，北自然就不是役牌；
    # 四麻北家手里北就是自风 —— 不需要单独的开关。
    wind_yakuhai = {ctx.round_wind, ctx.seat_wind}

    cands: list[Agari] = []
    if menzen:
        cands.extend(_score_chiitoi(counts, ctx))
        cands.extend(_score_kokushi(counts, ctx))

    for shape_melds, pair in _decompose(list(counts), need, 1):
        if pair is None or len(shape_melds) != need:
            continue
        cands.extend(
            _score_normal(counts, melds, shape_melds, pair, menzen, wind_yakuhai, ctx)
        )

    if not cands:
        return None
    return max(
        cands,
        key=lambda a: (
            a.yakuman,
            base_points(a.total_han, a.fu, a.yakuman, rule),
            a.total_han,
            a.fu,
        ),
    )


# ---------------------------------------------------------------------------
# 各形态的役种判定
# ---------------------------------------------------------------------------


def _has_real_yaku(yaku: dict[str, int]) -> bool:
    """除宝牌外至少要有一个真役（拔北宝牌本身不是役）。"""
    return any(k not in ("dora", "ura", "aka", "kita") for k in yaku)


def _dora_agari(ctx: WinContext) -> tuple[int, dict[str, int]]:
    detail: dict[str, int] = {}
    if ctx.dora:
        detail["dora"] = ctx.dora
    if ctx.ura:
        detail["ura"] = ctx.ura
    if ctx.aka:
        detail["aka"] = ctx.aka
    if ctx.kita and ctx.rule.kita_dora:
        detail["kita"] = ctx.kita
    return sum(detail.values()), detail


def _situational_yaku(yaku: dict[str, int], ctx: WinContext) -> None:
    """与手牌形状无关的役。"""
    if ctx.is_tenhou:
        yaku["tenhou"] = 1
    if ctx.is_chiihou:
        yaku["chiihou"] = 1
    if ctx.is_double_riichi:
        yaku["double_riichi"] = 1
    elif ctx.is_riichi:
        yaku["riichi"] = 1
    if ctx.is_ippatsu:
        yaku["ippatsu"] = 1
    if ctx.is_rinshan:
        yaku["rinshan"] = 1
    if ctx.is_chankan:
        yaku["chankan"] = 1
    if ctx.is_haitei and ctx.is_tsumo:
        yaku["haitei"] = 1
    if ctx.is_houtei and not ctx.is_tsumo:
        yaku["houtei"] = 1


def _score_chiitoi(counts: Sequence[int], ctx: WinContext) -> list[Agari]:
    if any(c not in (0, 1, 2) for c in counts):
        return []
    if sum(1 for c in counts if c == 2) != 7:
        return []
    yaku: dict[str, int] = {"chiitoi": 2}
    if all(counts[i] == 0 for i in range(N_TYPES) if i not in YAOCHU):
        yaku["honroutou"] = 2
    if ctx.is_tsumo:
        yaku["menzen_tsumo"] = 1
    _situational_yaku(yaku, ctx)
    dora_han, detail = _dora_agari(ctx)
    return [Agari(sum(yaku.values()), 25, sorted(yaku.items()), dora_han, 0, detail)]


def _score_kokushi(counts: Sequence[int], ctx: WinContext) -> list[Agari]:
    kinds = sum(1 for i in YAOCHU if counts[i] >= 1)
    if kinds != 13 or sum(counts[i] for i in YAOCHU) != 14:
        return []
    pair = next(i for i in YAOCHU if counts[i] >= 2)
    thirteen = pair == ctx.win_tile
    name = "kokushi13" if thirteen else "kokushi"
    mult = 2 if thirteen else 1
    return [Agari(0, 0, [(name, mult)], 0, mult, {})]


def wait_types(shape_melds: list[MeldShape], pair: int, win: int) -> list[str]:
    """和牌张在这副牌里的**全部**听形解释（高点法要用）。

    ⚠️ 这里的「全部」是关键 —— 老代码取**第一个**含和牌张的顺子就返回，
    于是 `1m2m3m4m5m6m1s2s3s3s4s5s9s9s` 和 3s 这种牌：
    先撞上顺子 `123s`（start=0、offset=2）→ 判成**边张** ✗
    可同一张 3s 也能补 `345s` → 是**两面** ✓ 结果平和被丢掉、
    符数多算 2 符，荣和时甚至会**直接拒和**（实测复现）。

    返回去重后的候选，至少一个。调用方应**逐个算分取最高**（高点法）。
    """
    cands: list[str] = []
    if pair == win and not any(k == "tri" and t == win for k, t in shape_melds):
        cands.append("tanki")
    for kind, t in shape_melds:
        if kind != "seq" or t // 9 != win // 9:
            continue
        start = t % 9
        offset = (win % 9) - start
        if offset == 1:
            cands.append("kanchan")
        elif offset == 0:
            cands.append("ryanmen" if start <= 5 else "penchan")   # 89 等 7
        elif offset == 2:
            cands.append("ryanmen" if start >= 1 else "penchan")   # 12 等 3
    for kind, t in shape_melds:
        if kind == "tri" and t == win:
            cands.append("shanpon")
    if not cands:
        cands.append("tanki")
    return list(dict.fromkeys(cands))


def _wait_type(shape_melds: list[MeldShape], pair: int, win: int) -> str:
    """单个听形（保留给不需要高点法的调用方）。多种解释时**优先两面**。"""
    cands = wait_types(shape_melds, pair, win)
    for prefer in ("ryanmen", "kanchan", "penchan", "shanpon", "tanki"):
        if prefer in cands:
            return prefer
    return cands[0]


def _meld_has_yaochu(kind: str, tile: int) -> bool:
    if kind == "tri":
        return tile in YAOCHU
    return (tile % 9) in (0, 6)  # 以 1 或 7 起头的顺子含 1/9


def _score_normal(
    counts: Sequence[int],
    melds: Sequence[Meld],
    shape_melds: list[MeldShape],
    pair: int,
    menzen: bool,
    wind_yakuhai: set[int],
    ctx: WinContext,
) -> list[Agari]:
    """按**高点法**返回所有听形解释下的结果 —— 调用方取最高分。

    同一副牌 + 同一张和牌，听形可能有多种解释（两面 / 边张 / 嵌张 / 双碰 / 单骑），
    各自的役和符不同，规则上取**得点最高**的那种。
    """
    out: list[Agari] = []
    for wt in wait_types(shape_melds, pair, ctx.win_tile):
        out.extend(
            _score_normal_one(counts, melds, shape_melds, pair, menzen,
                              wind_yakuhai, ctx, wt)
        )
    return out


def _score_normal_one(
    counts: Sequence[int],
    melds: Sequence[Meld],
    shape_melds: list[MeldShape],
    pair: int,
    menzen: bool,
    wind_yakuhai: set[int],
    ctx: WinContext,
    wait_type: str,
) -> list[Agari]:
    win = ctx.win_tile
    shapes: list[MeldShape] = list(shape_melds) + [
        ("seq", m.tile) if m.is_chi else ("tri", m.tile) for m in melds
    ]

    # 暗刻数（含暗杠）；荣和时被和了牌补成的那组刻子不算暗刻
    concealed_tri = sum(1 for k, _ in shape_melds if k == "tri")
    concealed_tri += sum(1 for m in melds if m.is_concealed)
    if not ctx.is_tsumo and any(k == "tri" and t == win for k, t in shape_melds):
        concealed_tri -= 1

    # ⚠️ 顺子必须从 **shapes**（暗牌 + 副露）里数，不能只看 shape_melds（暗牌）✗
    # 原来写成 shape_melds，导致**「吃」副露被当成刻子** ✗：
    #   n_seq 少算一个 -> n_tri = len(shapes) - n_seq 多算一个
    #   -> 任何**有吃**的手都会被判成「四组全是刻子」-> 白送 対々和(2 番) ✗
    # 实测（2026-10-21，用户截图那手）：碰白+碰發+吃345m+111p+55s 荣1p
    #   改前 4 番 40 符，役 = toitoi + 白 + 發（顺子手被判対々和 ✗）
    # 同一个 seqs 还被三色同顺 / 一气通贯 / 混老头分支用 ✓ 一处错会连累多个役 ✗
    seqs = [t for k, t in shapes if k == "seq"]
    n_seq = len(seqs)
    n_tri = len(shapes) - n_seq

    # ---- 役满优先 ----
    yakuman = _check_yakuman(
        counts, melds, shapes, pair, concealed_tri, wait_type, ctx
    )
    if yakuman:
        name, mult = yakuman
        return [Agari(0, 0, [(name, mult)], 0, mult, {})]

    yaku: dict[str, int] = {}
    _situational_yaku(yaku, ctx)
    if ctx.is_tsumo and menzen and not (ctx.is_tenhou or ctx.is_chiihou):
        yaku["menzen_tsumo"] = 1

    # 平和：门清 + 全顺子 + 雀头非役牌 + 两面听
    if (
        menzen
        and n_seq == 4
        and wait_type == "ryanmen"
        and pair not in DRAGONS
        and pair not in wind_yakuhai
    ):
        yaku["pinfu"] = 1

    # 断幺九
    if all(counts[i] == 0 for i in YAOCHU) and all(
        not _meld_has_yaochu(k, t) for k, t in shapes
    ):
        yaku["tanyao"] = 1

    # 役牌（连风牌时场风 + 自风都算，共 2 翻）
    dragon_key = {HAKU: "yakuhai_haku", HATSU: "yakuhai_hatsu", CHUN: "yakuhai_chun"}
    for k, t in shapes:
        if k != "tri":
            continue
        if t in DRAGONS:
            yaku[dragon_key[t]] = 1
        elif t in wind_yakuhai:
            if t == ctx.round_wind:
                yaku["yakuhai_round"] = 1
            if t == ctx.seat_wind:
                yaku["yakuhai_seat"] = 1

    # 一盃口 / 二盃口（门清）
    if menzen:
        dup = 0
        seen: set[int] = set()
        for s in seqs:
            if s in seen:
                dup += 1
            seen.add(s)
        if dup >= 2:
            yaku["ryanpeikou"] = 3
        elif dup == 1:
            yaku["iipeikou"] = 1

    # 三色同顺：三麻万子凑不出顺子，所以这条永远不会触发；四麻是常规役
    for _start in range(7):
        if all(
            any(k == "seq" and t == _start + 9 * suit for k, t in shapes)
            for suit in range(3)
        ):
            yaku["sanshoku"] = 2
            break

    # 三色同刻（三麻也可能：111m 111p 111s）
    tri_tiles = [t for k, t in shapes if k == "tri"]
    for n in range(9):
        if all((n + 9 * s) in tri_tiles for s in range(3)):
            yaku["sanshoku_doukou"] = 2

    # 一气通贯（三麻里只能靠筒 / 索）
    for suit in range(3):
        starts = {t % 9 for t in seqs if t // 9 == suit}
        if {0, 3, 6} <= starts:
            yaku["ittsuu"] = 2  # 门清 2 翻，副露由 KUISAGARI 降为 1 翻

    if n_tri == 4:
        yaku["toitoi"] = 2
    if concealed_tri == 3:
        yaku["sanankou"] = 2
    if sum(1 for m in melds if m.is_kan) == 3:
        yaku["sankantsu"] = 2
    if sum(1 for t in tri_tiles if t in DRAGONS) == 2 and pair in DRAGONS:
        yaku["shousangen"] = 2

    # 全帯幺 / 混老头 / 混一色 / 清一色
    all_yaochu = all(_meld_has_yaochu(k, t) for k, t in shapes) and pair in YAOCHU
    if all_yaochu:
        if n_seq == 0:
            yaku["honroutou"] = 2
        else:
            has_honor = pair >= 27 or any(t >= 27 for _, t in shapes)
            yaku["honchantai" if has_honor else "junchantai"] = 2

    # 花色下标：万 0 / 筒 1 / 索 2 / **字 3**（不是 27！这里踩过一次坑：
    # 写成 `27 in suits_used` 永远为假，导致「一门 + 字牌」全被判成清一色）
    suits_used = {t // 9 for _, t in shapes} | {pair // 9}
    has_honor_suit = 3 in suits_used
    if len({s for s in suits_used if s < 3}) <= 1:
        yaku["honitsu" if has_honor_suit else "chinitsu"] = 3 if has_honor_suit else 6

    # 食い下がり
    if not menzen:
        for name, (_closed, open_han) in KUISAGARI.items():
            if name in yaku:
                yaku[name] = open_han

    if not _has_real_yaku(yaku):
        return []

    fu = _compute_fu(
        menzen=menzen, tsumo=ctx.is_tsumo, pinfu="pinfu" in yaku,
        melds=melds, shape_melds=shape_melds, pair=pair, wait_type=wait_type, ctx=ctx,
    )
    dora_han, detail = _dora_agari(ctx)
    return [Agari(sum(yaku.values()), fu, sorted(yaku.items()), dora_han, 0, detail)]


def _check_yakuman(
    counts: Sequence[int],
    melds: Sequence[Meld],
    shapes: list[MeldShape],
    pair: int,
    concealed_tri: int,
    wait_type: str,
    ctx: WinContext,
) -> tuple[str, int] | None:
    if ctx.is_tenhou:
        return "tenhou", 1
    if ctx.is_chiihou:
        return "chiihou", 1

    tri_tiles = [t for k, t in shapes if k == "tri"]

    if concealed_tri == 4 and all(k == "tri" for k, _ in shapes):
        return ("suuankou_tanki", 2) if wait_type == "tanki" else ("suuankou", 1)
    if sum(1 for t in tri_tiles if t in DRAGONS) == 3:
        return "daisangen", 1
    wind_tri = sum(1 for t in tri_tiles if t in WINDS)
    if wind_tri == 4:
        return "daisuushii", 2
    if wind_tri == 3 and pair in WINDS:
        return "shousuushii", 1
    # 字一色 / 清老头 / 绿一色 看的是「整手牌用到哪些牌型」，
    # 不能只看顺子的起手牌（678s 里藏着 7s）。
    used_types = {i for i in range(N_TYPES) if counts[i] > 0} | {m.tile for m in melds}
    if all(t >= 27 for t in used_types):
        return "tsuuiisou", 1
    terminals = (0, 8, 9, 17, 18, 26)
    if all(t in terminals for t in used_types):
        return "chinroutou", 1
    if all(t in GREEN_TYPES for t in used_types):
        return "ryuuiisou", 1
    if sum(1 for m in melds if m.is_kan) == 4:
        return "suukantsu", 1

    # 九莲宝灯（门清 + 单色；三麻的万子凑不出，只可能筒 / 索）
    if not melds:
        base = [3, 1, 1, 1, 1, 1, 1, 1, 3]
        for suit in (1, 2):
            block = list(counts[suit * 9 : suit * 9 + 9])
            if sum(block) != 14 or any(block[i] < base[i] for i in range(9)):
                continue
            pre = list(block)
            pre[ctx.win_tile % 9] -= 1
            return ("chuuren_pure", 2) if pre == base else ("chuuren", 1)
    return None


def _compute_fu(
    *,
    menzen: bool,
    tsumo: bool,
    pinfu: bool,
    melds: Sequence[Meld],
    shape_melds: list[MeldShape],
    pair: int,
    wait_type: str,
    ctx: WinContext,
) -> int:
    rule = ctx.rule
    if pinfu and tsumo:
        return 20
    fu = 20
    if menzen and not tsumo:
        fu += 10
    if tsumo and not pinfu:
        fu += 2

    for kind, tile in shape_melds:
        if kind != "tri":
            continue
        concealed = not (not tsumo and tile == ctx.win_tile)
        fu += _triplet_fu(tile, concealed, kan=False)
    for m in melds:
        # ⚠️ **顺子（吃）没有符** ✗ —— 原来不判 kind，把吃也当成刻子加符 ✗
        # 实测（2026-10-21）：111s(幺九暗刻 8) + 自風雀头(2) + 底 20 = 30 符，
        #   吃 234s 被加了 2 符 -> 32 -> 切上 40 ✗ 正确应当是 30 ✓
        if m.is_chi:
            continue
        fu += _triplet_fu(m.tile, m.is_concealed, kan=m.is_kan)

    if pair in DRAGONS:
        fu += 2
    if pair == ctx.round_wind and pair == ctx.seat_wind:
        fu += rule.renpu_fu
    elif pair in (ctx.round_wind, ctx.seat_wind):
        fu += 2
    # 北当雀头的符：只有它确实是场风 / 自风时才算（四麻北家）
    if wait_type in ("kanchan", "penchan", "tanki"):
        fu += 2

    if fu == 20 and not menzen:
        return 30  # 喰い平和形
    return -(-fu // 10) * 10


def _triplet_fu(tile: int, concealed: bool, kan: bool) -> int:
    yaochu = tile in YAOCHU
    if kan:
        if yaochu:
            return 32 if concealed else 16
        return 16 if concealed else 8
    if yaochu:
        return 8 if concealed else 4
    return 4 if concealed else 2


# ---------------------------------------------------------------------------
# 点数计算
# ---------------------------------------------------------------------------


def base_points(han: int, fu: int, yakuman: int = 0, rule: Rule = DEFAULT_RULE) -> int:
    """基本点。"""
    if yakuman:
        return 8000 * yakuman
    if han >= rule.kazoe_han:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    if han >= 5:
        return 2000
    return min(fu * (2 ** (2 + han)), 2000)


def _ceil100(x: float) -> int:
    return int(-(-x // 100) * 100)


@dataclass
class Payment:
    """一次和牌的点数收付。``deltas[p]`` 是各座位点数变化（不含供托）。"""

    total: int  # 和了者净得（自摸时为两家支付之和）
    deltas: tuple[int, int, int]
    detail: str


def settle(
    agari: Agari,
    *,
    is_dealer: bool,
    is_tsumo: bool,
    winner: int,
    loser: int | None,
    rule: Rule = DEFAULT_RULE,
    dealer_seat: int | None = None,
    n_players: int | None = None,
) -> Payment:
    """把翻/符换算成各家点数收付。

    自摸两种桌子是同一套算法：**亲家付 2a、其余子家各付 a**。区别只在人数 ——
    三麻少一个付款人，总额天然少一份（这就是「ツモ損」）；四麻人齐，自然满额。

    三麻「丸取」（``tsumo_loss=False``）是另一种口子：亲家自摸时两家各付 3a，
    让总额跟四麻看齐。
    """
    base = base_points(agari.total_han, agari.fu, agari.yakuman, rule)
    n = n_players or rule.players
    deltas = [0] * n
    if is_tsumo:
        if dealer_seat is None:
            dealer_seat = winner if is_dealer else next(p for p in range(n) if p != winner)
        if is_dealer:
            each = _ceil100(base * (3 if (n == 3 and not rule.tsumo_loss) else 2))
            for p in range(n):
                if p != winner:
                    deltas[p] -= each
                    deltas[winner] += each
            detail = f"{each} all"
        else:
            small = _ceil100(base)
            big = _ceil100(base * 2)
            for p in range(n):
                if p == winner:
                    continue
                pay = big if p == dealer_seat else small
                deltas[p] -= pay
                deltas[winner] += pay
            detail = f"{small}/{big}"
    else:
        amount = _ceil100(base * (6 if is_dealer else 4))
        deltas[winner] += amount
        assert loser is not None
        deltas[loser] -= amount
        detail = str(amount)
    return Payment(deltas[winner], tuple(deltas), detail)


def rule_for(players: int = 3) -> Rule:
    """按人数取一套「雀魂段位战」的默认规则。"""
    if players == 3:
        return Rule()
    return Rule(
        players=4,
        tsumo_loss=False,          # 四麻人齐，不存在自摸损
        noten_penalty=3000,        # 四麻不聴罰符 3000（三麻 2000）
        double_ron=True,
        kita_dora=False,           # 四麻没有拔北
        man_dora_cycle=False,      # 四麻万子是正常循环
        dead_wall=14,              # 四麻王牌 14 张
        start_score=25000,         # 四麻段位战 25000 起 / 30000 返
        return_score=30000,
        uma=(15, 5, -5, -15),
        rank_bonus=(20, 10, 0, -20),
        sudden_death=True,
    )


__all__ = [
    "N_TYPES", "SANMA_TYPES", "YAOCHU", "AKA_TYPES", "DRAGONS", "WINDS",
    "EAST", "SOUTH", "WEST", "NORTH", "HAKU", "HATSU", "CHUN", "YAKU_CN",
    "Rule", "DEFAULT_RULE", "Meld", "WinContext", "Agari", "Payment",
    "tile_id", "tile_cn", "hand_str", "parse_hand",
    "is_winning_shape", "evaluate_hand", "base_points", "settle", "dora_from_index",
    "tile_types", "aka_types", "FOUR_TYPES", "rule_for",
    "ranks_of", "rank_points", "coin_delta",
    "YAKUMAN_NAMES", "KUISAGARI",
]
