"""``StrongAgent`` 的回归测试。

要钉死的五件事（对应任务要求）：

a) **能跑**：四麻 / 三麻各打完整局，不抛异常
b) **无役不鸣**：构造「吃了就无役、不吃还有役」的局面，必须 pass
c) **有役该鸣**：构造「碰了明显更快、而且碰完有役」的局面，必须碰
d) **弃和时不会打危险牌**：他家立直 + 自己愚形低打点，必须挑现物/安全牌
e) **确定性**：同一局面调用两次、以及两个同种子 agent，结果必须相同

另外验证 ``strong:<变体>`` 的注册表，并打印单次 decide 的耗时（不设硬阈值，
避免慢机器误报；真正的性能数字写进报告）。

跑法：``.venv/bin/python aimodel/tests/test_strong.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402
from aimodel.engine import Decision, Game, View  # noqa: E402
from aimodel.fastshanten import shanten  # noqa: E402
from aimodel.strong import STRONG_VARIANTS, StrongAgent, make  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")
    else:
        print(f"  ✅ {msg}")


def make_view(
    players: int = 4,
    hand: str = "",
    *,
    player: int = 0,
    riichi: dict[int, bool] | None = None,
    discards: dict[int, list[int]] | None = None,
    melds: list[S.Meld] | None = None,
    turn: int = 6,
    wall_left: int = 50,
    scores: list[int] | None = None,
    dora: tuple[int, ...] = (0,),
) -> View:
    """造一个只填测试关心字段的 :class:`View`。"""
    rule = S.rule_for(players)
    h = S.parse_hand(hand)
    melds = list(melds or [])
    return View(
        player=player,
        hand=h,
        drawn=None,
        melds=list(melds),
        hand_red=[0] * 34,
        discards=[list((discards or {}).get(q, [])) for q in range(players)],
        melds_all=[list(melds) if q == player else [] for q in range(players)],
        kita=[0] * players,
        scores=scores or [rule.start_score] * players,
        dora_indicators=list(dora),
        round_wind=S.EAST,
        seat_wind=S.EAST,
        wall_left=wall_left,
        turn=turn,
        riichi=[bool((riichi or {}).get(q, False)) for q in range(players)],
        ippatsu=[False] * players,
        is_dealer=(player == 0),
        last_discard=None,
        rule=rule,
    )


class Timed:
    """包一层统计每个决策耗时（只用于打印，不影响决定）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.n = 0
        self.total = 0.0
        self.max = 0.0

    def decide(self, decision, view):
        t0 = time.perf_counter()
        out = self.inner.decide(decision, view)
        dt = time.perf_counter() - t0
        self.n += 1
        self.total += dt
        self.max = max(self.max, dt)
        return out


def main() -> int:
    print("─" * 74)
    print("① 四麻 / 三麻完整局（不崩）")
    print("─" * 74)
    for players in (4, 3):
        try:
            for seed in (0, 1):
                agents = [Timed(StrongAgent(seed * 10 + i)) for i in range(players)]
                res = Game(agents, seed=seed, rule=S.rule_for(players)).play()
                check(len(res.ranks) == players and res.hands,
                      f"{players} 麻 seed={seed}：完整局跑通（{len(res.hands)} 局）")
                n = sum(a.n for a in agents)
                avg = sum(a.total for a in agents) / max(1, n) * 1000
                mx = max(a.max for a in agents) * 1000
                print(f"     决策 {n:4d} 次，平均 {avg:5.2f} ms，最慢 {mx:6.1f} ms")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{players} 麻完整局抛异常：{type(exc).__name__}: {exc}")

    print()
    print("─" * 74)
    print("② 无役不鸣（硬规则）")
    print("─" * 74)
    hero = StrongAgent(0)
    teacher = HeuristicAgent(0)
    # 门清听牌：123m 456p 789s 22p 34s，吃 2s 会变成 234s 的副露。
    # 吃完之后：123m 456p 789s 22p + 234s（副露），
    # 没有断幺（1m/9s）、没有役牌、不是染手、没有全带/对对/一气/三色 -> 无役。
    v = make_view(4, "123m456p789s22p34s")
    chi = Decision("chi", ["pass", "chi"], tile=19)  # 2s
    pick = hero.decide(chi, v)
    check(pick == 0, f"无役可鸣时选择 pass（返回 {pick}，0=过）")
    # 负向对照：教师只看「向听 <= 3」就吃，所以它一定会吃。说明这个局面
    # 看起来确实诱人，不是「本来就没人会鸣」。
    check(teacher.decide(chi, v) == 1, "负向对照：教师会吃（说明局面足够诱人）")
    post = S.parse_hand("123m456p789s22p")
    info = hero._yaku_info(
        post, [S.Meld("chi", 19, -1, called=19)], v
    )
    check(not info.has_yaku,
          f"吃完后确实没有真役路线（route={info.routes}, guaranteed={info.guaranteed}）")

    print()
    print("─" * 74)
    print("③ 有役该鸣")
    print("─" * 74)
    # 1 向听：123m 456p 9p 11z 22z 78s。碰东（1z）后打 9p -> 听 6/9s，
    # 而且「东」是场风+自风，碰出来就是役牌（至少 1 番）。
    v2 = make_view(4, "123m456p9p11z22z78s")
    check(shanten(v2.hand, 0) == 1, "测试手牌确实是 1 向听（碰了能进听）")
    pon = Decision("call", ["pass", "pon"], tile=27)  # 东
    pick2 = hero.decide(pon, v2)
    check(pick2 == 1, f"有役且更快时选择碰（返回 {pick2}，1=碰）")
    post2 = list(v2.hand)
    post2[27] -= 2
    info2 = hero._yaku_info(post2, [S.Meld("pon", 27, -1)], v2)
    check(info2.has_yaku and info2.guaranteed >= 1,
          f"碰完有确定的役牌（guaranteed={info2.guaranteed}, route={info2.routes}）")

    print()
    print("─" * 74)
    print("④ 弃和时不会打危险牌")
    print("─" * 74)
    # 自己 19m 19p 19s 55p 5s 1234z：向听很高、无打点；
    # 下家立直，牌河里打过 5p（现物）。5s 是危险中张。
    v3 = make_view(
        4, "19m19p19s55p5s1234z",
        riichi={1: True}, discards={1: [13]},  # 5p 是现物
        turn=13, wall_left=20,
    )
    opts = [t for t in range(34) if v3.hand[t] > 0]
    pick3 = hero.decide(Decision("discard", opts), v3)
    chosen = opts[pick3]
    danger = hero._danger(v3, chosen, [1], hero._visible_true(v3))
    check(chosen != 22, f"没有打危险中张 5s（实际打 {S.tile_cn(chosen)}）")
    check(danger <= 0.05,
          f"选中的牌是安全的（{S.tile_cn(chosen)} 危险度 {danger:.2f}）")
    check(chosen in v3.discards[1],
          f"选中的牌是对手立直家的现物（{S.tile_cn(chosen)}）")

    print()
    print("─" * 74)
    print("⑤ 确定性")
    print("─" * 74)
    a = hero.decide(Decision("discard", opts), v3)
    b = hero.decide(Decision("discard", opts), v3)
    check(a == b, f"同一 agent、同一局面调用两次结果相同（{a} == {b}）")
    a2 = StrongAgent(0).decide(Decision("discard", opts), v3)
    b2 = StrongAgent(0).decide(Decision("discard", opts), v3)
    check(a2 == b2 == a, "两个同种子 agent 的结果也相同")
    # 鸣牌决策同样要确定
    c1 = hero.decide(chi, v)
    c2 = hero.decide(chi, v)
    check(c1 == c2 == 0, f"吃牌决策确定（{c1} == {c2}）")

    print()
    print("─" * 74)
    print("⑥ strong:<变体> 注册表")
    print("─" * 74)
    names = sorted(STRONG_VARIANTS)
    print(f"  可用变体：{names}")
    check("standard" in names and "attack" in names
          and "defense" in names and "value" in names,
          "至少注册 standard / attack / defense / value")
    try:
        a = make("strong", seed=0)
        check(isinstance(a, StrongAgent), "make('strong') 返回 StrongAgent")
        b = make("strong:attack", seed=0)
        check(b.variant == "attack", "make('strong:attack') 解析出 attack")
    except Exception as exc:  # noqa: BLE001
        check(False, f"变体构造失败：{type(exc).__name__}: {exc}")
    try:
        make("strong:this_does_not_exist")
        check(False, "未知变体应当报错")
    except SystemExit:
        check(True, "未知变体会报错（不是静默兜底）")

    print()
    print("─" * 74)
    print("⑦ 平台座位构造（webapp/server/agents.py）")
    print("─" * 74)
    try:
        from webapp.server import agents as AG

        seat_agent = AG.make_builtin_agent("strong", 0)
        check(isinstance(seat_agent, StrongAgent),
              f"平台 make_builtin_agent('strong') 返回 {type(seat_agent).__name__}")
        seat3 = AG.make_builtin_agent("strong3", 0)
        check(isinstance(seat3, StrongAgent), "三麻名 strong3 也能构造")
    except Exception as exc:  # noqa: BLE001
        check(False, f"平台座位构造失败：{type(exc).__name__}: {exc}")

    print()
    print("=" * 74)
    if FAILED:
        print(f"🔴 **失败 {len(FAILED)} 项**：")
        for f in FAILED:
            print(f"    · {f}")
        print("=" * 74)
        return 1
    print("✅ 全部通过")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
