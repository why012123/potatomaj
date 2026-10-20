"""把 ``aimodel``（同事训练的网络）接进「该打哪张」这一步。

这一层只做三件事：**选模型、拼观测、翻译牌名**。不含任何算法。

## 模型怎么选

``aimodel/weights/`` 下有 8 个模型 = 4 种风格 × 三麻/四麻：

======================  ======  ================================
模型                    牌桌    风格（训练时的得分条件）
======================  ======  ================================
``maj-normalv1.1``      三麻    段位分（标准）
``maj-mountainv1.1``    三麻    段位分 − 3.0×放铳点数（防守）
``maj-flashv1.1``       三麻    段位分 + 4.0×和了局数（速攻）
``maj-hugev1.1``        三麻    段位分 + 1.5×累计番数（大牌）
``maj-normal4v1.1``     四麻    段位分
``maj-mountain4v1.1``   四麻    段位分 − 3.0×放铳
``maj-flash4v1.1``      四麻    段位分 + 4.0×和了次数
``maj-huge4v1.1``       四麻    段位分 + 1.5×番数
======================  ======  ================================

**默认按牌桌自动挑**（三麻用 ``maj-flashv1.1``、四麻用 ``maj-flash4v1.1``），
想换风格或换具体模型，改 ``build/models.json`` 就行 —— 见 :func:`config_path`。

## 和三麻专用时期的区别

以前这里有一道闸门：「手里出现 2m~8m 就不用网络」（因为那只有四麻才有）。
现在**四麻有自己的模型**，那道闸门换成了「按牌桌挑模型」。
副露也一样 —— 协议能把副露的**具体牌**读出来（``ActionChiPengGang`` 字段 3），
所以带副露的手牌也能喂进去，不再需要「有副露就退回纯牌效」。
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .tiles import parse_tile  # noqa: E402

WEIGHTS_DIR = ROOT / "aimodel" / "weights"

#: 每种牌桌的默认模型（不带版本号的写法指向最新版，见 aimodel/README.md）
DEFAULT_MODEL = {3: "maj-flashv1.1", 4: "maj-flash4v1.1"}

#: 八种风格 -> 文件名前缀（三麻）/ 完整名（四麻带 4）
STYLES = ("normal", "mountain", "flash", "huge")


def tile_index(tid: str) -> int:
    """牌名 -> 牌型下标。赤 5（0m/0p/0s）归到普通 5 上。

    aimodel 只提供 ``tile_id(下标 -> 名字)``，没有反函数，所以查 TILE_IDS。
    """
    from aimodel import sanma as S
    t = tid
    if t.startswith("0"):
        t = "5" + t[1:]
    return S.TILE_IDS.index(t)


def counts_from_ids(ids, players: int = 3):
    """牌名列表 -> ``(34 计数, 34 赤 5 计数)``。

    **没有用 ``advise.counts_from_ids``** —— 那个的赤 5 表是
    ``{"p": 13, "s": 22}``，只有三麻的两种，**四麻的赤 5 万（0m，下标 4）
    它根本不记**。表现是：四麻手里有 0m 时 hand_red 少一个，
    模型看到的「赤 5 有几张」就错了。

    这是等价性测试抓到的（四麻 seed2 座位1，编码向量差 1 位，
    位置 525 正好落在 aka 那块）。
    """
    from aimodel import sanma as S
    counts = [0] * 34
    red = [0] * 34
    for tid in ids or []:
        if not tid:
            continue
        try:
            base = tile_index(tid)
        except ValueError:
            continue
        counts[base] += 1
        if str(tid).startswith("0"):
            red[base] += 1
    return counts, red


def config_path() -> Path:
    """用户可改的配置文件位置（不存在就用默认）。"""
    return Path(os.environ.get("POTATOMAJ_MODELS") or (ROOT / "build" / "models.json"))


@lru_cache(maxsize=1)
def _user_config() -> dict:
    """读 ``build/models.json``。格式：

    .. code-block:: json

        {
          "3": "maj-flashv1.1",
          "4": "maj-flash4v1.1",
          "note": "三麻想换成标准型就写 maj-normalv1.1"
        }
    """
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_name(players: int = 3) -> str:
    """这个牌桌该用哪个模型。优先级：环境变量 > 配置文件 > 默认。"""
    env = os.environ.get("POTATOMAJ_MODEL")
    if env:
        return env
    cfg = _user_config()
    v = cfg.get(str(players))
    if isinstance(v, str) and v:
        return v
    return DEFAULT_MODEL.get(players, DEFAULT_MODEL[3])


def model_path(players: int = 3) -> Path:
    """模型文件路径。名字对不上时退回默认，并在说明里讲清楚。"""
    name = model_name(players)
    p = WEIGHTS_DIR / f"{name}.npz"
    if p.exists():
        return p
    fallback = WEIGHTS_DIR / f"{DEFAULT_MODEL.get(players, DEFAULT_MODEL[3])}.npz"
    if fallback.exists():
        return fallback
    return p


@lru_cache(maxsize=4)
def _net_for(path_str: str):
    """按路径加载并缓存。返回 None 表示没有网络 —— 这种情况必须能继续跑。"""
    try:
        from aimodel.net import Net
    except Exception:
        return None
    try:
        return Net.load(path_str)
    except Exception:
        return None


def net(players: int = 3):
    return _net_for(str(model_path(players)))


# ---------------------------------------------------------------------------
# 适用性判断
# ---------------------------------------------------------------------------

def model_ok(hand: tuple[str, ...] | list[str], players: int = 3,
             melds: int = 0) -> tuple[bool, str]:
    """这手牌能不能用网络。返回 ``(能不能用, 不能用的原因)``。"""
    if len(hand) + 3 * melds != 14:
        return False, f"手牌 {len(hand)} 张（+{melds} 副露），不是刚摸完的 14 张等效"
    for tid in hand:
        try:
            t = parse_tile(tid)
        except Exception:
            return False, f"手牌里有认不出来的 {tid!r}"
        if players == 3 and t.suit == "m" and t.num not in (1, 9):
            return False, f"手里有 {tid} —— 三麻没有 2m~8m，这个模型没见过"
    if model_name(players) == TEACHER:
        return True, ""            # 启发式教师不用权重文件
    if net(players) is None:
        return False, f"没找到 {model_name(players)}.npz"
    return True, ""


# ---------------------------------------------------------------------------
# 拼观测
# ---------------------------------------------------------------------------

def _meld_objects(melds: list[dict] | None, players: int, my_seat: int):
    """把协议解出来的副露翻成 ``aimodel`` 的 :class:`Meld`。

    协议给的 ``froms`` 是「每张牌来自哪一家」，用它算出 ``from_player``
    （相对我的座位；自己出的牌记 -1）。
    """
    from aimodel import sanma as S
    out = []
    n = players
    seat = my_seat if 0 <= my_seat < n else 0
    for m in (melds or []):
        # 拔北不是副露（它是「拿掉一张北 + 补摸一张」，手牌张数不变）。
        # 万一调用方把它的 type 写成 "kita"，这里直接跳过。
        if m.get("type") == "kita":
            continue
        tiles = [t for t in (m.get("tiles") or []) if t]
        if not tiles:
            continue
        # 杠分三种，不能都当大明槓 —— 暗杠不破坏门清，加杠还多一张。
        # 模型靠这个区分「他门清」和「他杠过」。
        kind = {"chi": "chi", "pon": "pon",
                "minkan": "minkan", "ankan": "ankan", "kakan": "kakan",
                "kan": "minkan", "kita": "kita"}.get(m.get("type"), "pon")
        # 牌名 -> 牌型下标。注意 aimodel 只提供 tile_id(下标->名字)，没有反函数，
        # 所以直接查 TILE_IDS。赤 5（0p/0s/0m）要先归到普通 5 上。
        #
        # **吃**：`S.Meld.tile` 约定是顺子里**最小**的那张（engine 的 `_do_chi`
        # 就是 `start = min(tile, a, b)`），而协议给的 `tiles` **不是升序** ——
        # 实测吃 6s 时它是 `["7s","8s","6s"]`。老代码直接拿 `tiles[0]` 当起点，
        # 于是这组吃被记成 7-8-9s（实际是 6-7-8s），模型看到的副露和可见牌全错。
        # 这是「吃完之后建议突然不对」的另一半原因（手牌跟踪那边的 bug 见
        # electron/lib/gamestate.js）。
        idxs = []
        for t in tiles:
            base = ("5" + t[1:]) if t.startswith("0") else t
            try:
                idxs.append(S.TILE_IDS.index(base))
            except ValueError:
                idxs = []
                break
        if not idxs:
            continue
        idx = min(idxs) if kind == "chi" else idxs[0]
        aka = sum(1 for t in tiles if t.startswith("0"))
        # 哪张是别家给的：froms 里有和副露者座位不同的那一张
        from_player = -1
        called = -1
        froms = m.get("froms") or []
        for i, f in enumerate(froms):
            if f != seat and i < len(tiles):
                from_player = (f - seat) % n
                if kind == "chi":
                    called = idxs[i]
                break
        out.append(S.Meld(kind=kind, tile=int(idx), from_player=from_player,
                          aka=int(aka), called=int(called)))
    return out


def rank(
    hand: list[str],
    dora: list[str] | None = None,
    discards: list[list[str]] | None = None,
    my_seat: int = 0,
    turn: int = 6,
    players: int = 3,
    melds: list[dict] | None = None,
    riichi: list[bool] | None = None,
    scores: list[int] | None = None,
    wall_left: int | None = None,
    round_wind: int | None = None,
    kita: list[int] | None = None,
    drawn: str | None = None,
    melds_all: list[list[dict]] | None = None,
    ippatsu: list[bool] | None = None,
    dealer: int | None = None,
) -> tuple[dict[str, float], str]:
    """用网络给「打哪张」排序。

    返回 ``({牌名: 概率}, 说明)``；用不了时返回 ``({}, 原因)``。
    概率是**同一手牌内部的相对好坏**（softmax，和恒为 1），可以直接横向比较。
    """
    from aimodel import sanma as S
    from aimodel.encode import action_features, action_mask, encode
    from aimodel.engine import Decision

    n = 3 if players == 3 else 4
    meld_n = len(melds or [])
    ok, why = model_ok(tuple(hand), players=players, melds=meld_n)
    if not ok:
        return {}, why

    view = build_view(
        hand=hand, dora=dora, discards=discards, my_seat=my_seat, players=n,
        turn=turn, melds=melds, riichi=riichi, scores=scores,
        wall_left=wall_left, round_wind=round_wind, kita=kita, drawn=drawn,
        melds_all=melds_all, ippatsu=ippatsu, dealer=dealer,
    )

    types = S.SANMA_TYPES if n == 3 else S.FOUR_TYPES
    options = [t for t in types if view.hand[t] > 0]
    if not options:
        return {}, "没有可打的候选牌"

    # 教师：不用神经网络，用 aimodel 里那套「牌效 + 防守 + 弃和」的启发式。
    # 分数不是概率（量纲完全不同），所以这里**归一化成一个排序信号**，
    # 只在同一手牌内部比较；界面上会标出来源是「教师」。
    if model_name(players) == TEACHER:
        raw = teacher_scores(view)
        if not raw:
            return {}, "教师没有给出候选"
        lo = min(raw.values())
        hi = max(raw.values())
        span = (hi - lo) or 1.0
        out = {}
        for t, v in raw.items():
            if v <= -1e17:
                continue                      # 掉向听的，不显示
            name = S.tile_id(int(t))
            out[name] = (v - lo) / span       # 0~1，越大越好
        return out, ""

    p = net(n).policy(
        encode(view), "discard", action_mask(Decision("discard", options)),
        action_features(view))

    # 网络按 34 种牌输出，而手里可能拿着赤 5（0m/0p/0s）—— 映射回手里实际那张
    have: dict[str, str] = {}
    for tid in hand:
        t = parse_tile(tid)
        base = f"{t.num}{t.suit}"
        if base not in have or tid.startswith("0"):
            have[base] = tid

    out: dict[str, float] = {}
    for i, prob in enumerate(p):
        if prob <= 0:
            continue
        base = S.tile_id(int(i))
        out[have.get(base, base)] = float(prob)
    return out, ""


def dealer_seat(players: int, round_no: int = 0) -> int:
    """这一局的亲家坐哪。

    协议里 ``ActionNewRound`` 的 ``ju`` 是「这一场里的第几局」（从 0 开始），
    亲家按局数轮转 —— 东 1 是起家、东 2 换下一家，以此类推。
    三麻只坐 3 个人，所以对 3 取模。

    这是个**推断**（协议没有直接给「谁是亲」）：它的依据是「局数轮转」这条规则。
    推断错了只会影响 ``is_dealer``（进而影响模型对亲子的判断），不会让牌读错。
    """
    n = 3 if players == 3 else 4
    return int(round_no) % n


def build_view(
    hand: list[str],
    dora: list[str] | None = None,
    discards: list[list[str]] | None = None,
    my_seat: int = 0,
    players: int = 3,
    turn: int = 6,
    melds: list[dict] | None = None,
    riichi: list[bool] | None = None,
    scores: list[int] | None = None,
    wall_left: int | None = None,
    round_wind: int | None = None,
    round_no: int = 0,
    kita: list[int] | None = None,
    drawn: str | None = None,
    is_dealer: bool | None = None,
    dealer: int | None = None,
    ippatsu: list[bool] | None = None,
    melds_all: list[list[dict]] | None = None,
):
    """把**协议解出来的牌局**拼成 ``aimodel`` 的 :class:`View`。

    ## 为什么这个函数在 aimodel 外面

    aimodel 里「从观测建 view」的入口只有 ``advise.build_view``，而它写死三麻
    （3 个牌河、3 个立直位、``DEFAULT_RULE``、``SANMA_TYPES``）。
    四麻在 aimodel 里是完整支持的，但那条路只有**跑完整引擎**才有
    （``Game.view_for()``），从外部观测没有入口。

    所以这里自己拼。为了不让它和引擎悄悄脱节，配了一个**等价性测试**：
    用真实引擎跑出局面、拿 ``view_for()`` 的真值逐字段比
    （``tests/test_view_equiv.py``）。动了这个函数就跑那个测试。

    ## 座位序：**绝对**，不是相对

    这一条踩过坑：``View`` 里的 ``discards`` / ``riichi`` / ``melds_all`` /
    ``scores`` / ``kita`` 全部是**按绝对座位号索引**的，而 ``player`` 说明
    「看的人是谁」。编码器写的是 ``view.discards[q]`` 配 ``q != view.player`` ——
    它自己会跳过 ``player``，**不需要**调用方先旋转。

    我之前按「从自己起算」旋转了一遍，座位 0 时恰好等价、座位 2 就全错。
    """
    from aimodel import sanma as S
    from aimodel.engine import View

    n = 3 if players == 3 else 4
    seat = my_seat if 0 <= my_seat < n else 0
    rule = S.rule_for(n)

    hc, red = counts_from_ids(hand, n)
    my_melds = _meld_objects(melds, n, seat)

    # 赤 5：**只把「吃」里、来自我手牌的那两张补回来**，而且要补到红 5
    # 真正的那一种牌上。
    #
    # 引擎的行为不一致（这是实测，不是猜的）：
    #   * `_do_chi`      **不扣** pl.red —— 手牌里的赤 5 进了顺子，pl.red 里还留着它；
    #   * `_do_pon` / 槓  会把 pl.red 扣到「手里还剩几张」为止 —— 赤 5 进了
    #                      碰/槓就不再算。
    # 模型是拿引擎的 view 训练的，所以这里必须照抄这个奇怪的口径。
    #
    # 两个坑：
    #   1. 老代码 `red[mm.tile] += aka` 把赤 5 的个数加到了**顺子起点**上。
    #      吃 4m5m6m（赤 5m 在中间）会记成 hand_red[4m]=1，而引擎是 hand_red[5m]=1。
    #   2. 被吃的那张如果是赤 5，它来自**别人**，本来就不在我的 pl.red 里 ——
    #      不能补。所以先用 froms 找出「别人给的那张」，只补另外两张。
    for m in (melds or []):
        if m.get("type") != "chi":
            continue
        tiles = list(m.get("tiles") or [])
        froms = m.get("froms") or []
        gave = -1
        if len(froms) == len(tiles):
            gave = next((i for i, f in enumerate(froms) if f != seat), -1)
        if gave < 0 and tiles:          # froms 缺失时和 gamestate 同口径：最后一张是被叫的
            gave = len(tiles) - 1
        for i, t in enumerate(tiles):
            if i == gave:
                continue
            if str(t).startswith("0"):
                try:
                    red[tile_index(t)] += 1
                except ValueError:
                    pass

    def pond_list(tiles):
        """牌河 -> 牌型下标列表，**保留出牌先后**。

        这里踩过坑：View.discards 是按打出的先后排的（引擎存的就是 pond 的顺序），
        不是按牌型排序。我原来用 counts_from_ids 再展开，等于按牌型排了序 ——
        编码向量会差两三位（模型看到的「他先打了什么」就变了）。
        等价性测试当场抓到了这个。
        """
        out = []
        for t in (tiles or []):
            try:
                out.append(tile_index(t))
            except ValueError:
                continue
        return out

    ponds = [pond_list(discards[i] if discards and i < len(discards) else []) for i in range(n)]

    dora_idx = []
    for d in (dora or []):
        c, _ = counts_from_ids([d])
        for i, v in enumerate(c):
            if v:
                dora_idx.append(i)

    # 副露：**别家的也要给**。只给自己的那份，模型会以为对家全是门清 ——
    # 编码向量里 opp*_melds / visible / meld_count 那几块就全错了。
    if melds_all is not None:
        melds_by_seat = [
            _meld_objects(melds_all[q] if q < len(melds_all) else [], n, q)
            for q in range(n)
        ]
    else:
        melds_by_seat = [[] for _ in range(n)]
        melds_by_seat[seat] = my_melds

    # 拔北：三麻特有，**必须从协议来**。写死 0 的话，模型看到的
    # 「北还剩几张、谁拔了几个」就全是错的 —— 等价性测试抓到过这个，
    # 差异正好落在 visible[北] 和 kita 那几位上。
    kita_list = list(kita or [])
    kita_list = (kita_list + [0] * n)[:n]
    kita_list = [int(x) for x in kita_list]

    # 别把巡目截断得太低 —— 引擎能跑到 19 巡以上（四麻流局前），
    # 我原来截在 18，编码向量里 turn 那一位就对不上了。
    turn_n = max(1, min(40, int(turn or 1)))
    winds = [S.EAST, S.SOUTH, S.WEST, S.NORTH]
    #: round_wind 传进来的是 wind 常量（27/28/29/30），也可能直接给 0/1/2/3
    rw_i = 0 if round_wind is None else (int(round_wind) - S.EAST
                                        if int(round_wind) >= S.EAST else int(round_wind))
    rw_i %= 4
    rw = winds[rw_i]
    # 亲家坐哪：优先用调用方给的（协议里由 ``ju`` 推），没给就按局数轮转推。
    dealer = int(dealer) if dealer is not None else dealer_seat(n, round_no)
    dealer = dealer % n
    # 自风 = 场风 + **我离亲家差几家**。
    #
    # 不能写成「场风 + 我的座位」—— 那只在东 1 局（亲家坐 0）成立。
    # 东 3 局时坐 0 位的人是西家，按座位算会算成东家。等价性测试抓到了这个
    # （引擎给的 seat_wind=WEST，我给的 EAST）。
    # 自风 = **离亲家差几家**，跟场风无关。
    #
    # 引擎里就是这么算的（engine.py seat_wind）：
    #     table = (EAST,SOUTH,WEST) if n==3 else (EAST,SOUTH,WEST,NORTH)
    #     return table[(p - dealer) % n]
    # 三麻只有三个风，北被跳过，所以不能按 4 取模。
    #
    # 我原来写成「场风 + 我的座位」，还写成对 4 取模 —— 两处都错。
    # 这是等价性测试抓到的第 8 个 bug。
    table = (S.EAST, S.SOUTH, S.WEST) if n == 3 else (S.EAST, S.SOUTH, S.WEST, S.NORTH)
    sw = table[(seat - dealer) % n]

    r_list = list(riichi or [])
    r_list = (r_list + [False] * n)[:n]
    score_list = list(scores or [])
    if len(score_list) != n:
        score_list = [rule.start_score] * n

    # 刚摸到的那张：协议给得到（ActionDealTile 字段 2）。
    # 它和手牌里那张是**同一张**（hand 里已经含它了），drawn 只是额外标一下
    # 「哪张是刚摸的」—— action_features 会用它区分摸切和手切，所以不能省。
    drawn_idx = None
    if drawn:
        try:
            drawn_idx = tile_index(drawn)
        except ValueError:
            drawn_idx = None

    return View(
        player=seat,                            # 「看的人」是我
        hand=hc,
        drawn=drawn_idx,
        melds=my_melds,
        hand_red=red,
        discards=ponds,                         # 绝对座位序
        melds_all=melds_by_seat,                # 绝对座位序
        kita=kita_list,
        scores=score_list,
        dora_indicators=dora_idx,
        round_wind=rw,
        seat_wind=sw,
        # 注意用 is not None：wall_left 真的会是 0（牌山摸完），
        # 写成 `if wall_left` 的话 0 被当成「没给」，会退回一个猜的值。
        wall_left=(int(wall_left) if wall_left is not None
                   else max(0, 55 - turn_n * n)),
        turn=turn_n,
        riichi=r_list,                          # 绝对座位序
        # 一发：立直之后、下一次打牌之前的那一圈。协议里没有直接字段，
        # 由调用方（gamestate）跟着立直和打牌推出来。写死 False 的话，
        # 模型看不到「某家一发中」——那正是最该防守的时候。
        ippatsu=((list(ippatsu or []) + [False] * n)[:n]),
        is_dealer=(bool(is_dealer) if is_dealer is not None else (seat == dealer)),
        last_discard=None,
        rule=rule,
    )


# ---------------------------------------------------------------------------
# 教师（不用神经网络的那条路）
# ---------------------------------------------------------------------------

#: 除了 8 个 npz 权重，还可以选这一个「模型」——
#: aimodel 里的启发式教师（牌效 + 基础防守 + 弃和）。
TEACHER = "teacher"


def teacher_scores(view, style: str = "normal") -> dict:
    """用 aimodel 的**启发式教师**给每张牌打分（越大越好）。

    返回值是 ``{牌型下标: 分数}``，分数只用于**同一手牌内部**排序。

    ## 为什么不把这套打分抄一遍

    抄一遍就会漂 —— 教师改了权重、加了新的考虑，我这边不会报错，
    只会安静地给出不一样的建议。所以这里**调用教师自己的方法**
    （``_visible`` / ``_opp_riichi`` / ``_pressure`` / ``_danger`` / ``_discard_cost``），
    只镜像那层循环结构。配了一个测试保证两边选出的牌一致：
    ``tests/test_teacher_equiv.py``。

    ## 分数不是概率

    教师的分数是「向听 × 1e6 + 进张 × 权重 − 危险 × 权重」这种量纲，
    跟神经网络的 softmax 概率（和为 1）**不是一回事**。
    界面按各自的口径显示，不要放在一起比。
    """
    from aimodel.agents import HeuristicAgent
    from aimodel.fastshanten import shanten, ukeire_count
    from aimodel import sanma as S

    agent = HeuristicAgent(0, style=style)
    n_melds = view.meld_count()
    hand = list(view.hand)
    sanma = view.rule.players == 3
    base_s = shanten(hand, n_melds)
    vis = agent._visible(view)
    opp_riichi = agent._opp_riichi(view)
    pressure = agent._pressure(view, opp_riichi)
    folding = agent.fold and bool(opp_riichi) and base_s >= agent.style.fold_shanten

    types = S.SANMA_TYPES if sanma else S.FOUR_TYPES
    options = [t for t in types if hand[t] > 0]

    cand = []
    for t in options:
        rest = list(hand)
        rest[t] -= 1
        cand.append((t, rest, shanten(rest, n_melds)))
    if not cand:
        return {}
    best_s = min(c[2] for c in cand)

    out: dict[int, float] = {}
    for t, rest, sh in cand:
        if folding:
            score = -agent._danger(view, t, opp_riichi) * 1_000_000 - sh * 1_000
            if sh == best_s:
                score += ukeire_count(rest, vis, n_melds, sanma=sanma)
        else:
            if sh > best_s:
                out[t] = -1e18            # 掉向听的直接压到底
                continue
            u = ukeire_count(rest, vis, n_melds, sanma=sanma)
            score = (-sh * 1_000_000 + u * agent.style.ukeire_weight
                     - agent._discard_cost(view, t) * agent.style.value_weight)
            if pressure > 0:
                weight = ((400.0 if sh <= 1 else 2500.0 if sh == 2 else 6000.0)
                          * pressure * agent.style.pressure_scale)
                score -= agent._danger(view, t, opp_riichi) * weight
        out[t] = score
    return out
