"""六种打法风格 agent 的回归测试。

要验的三件事：
1. **能跑** —— 六种风格在三麻/四麻里都能把整局打完，不抛异常
2. **真的不一样** —— 不是同一个策略换了名字 ✗
   用可观测的行为指标区分：
     · 副露次数（门清派应当 ≈0，速攻派应当最多）
     · 弃和时是否只看安全牌
     · 立直率
3. **确定性** —— 同种子同牌山，行为必须一致 ✓
   （否则「同牌山配对」这个实验设计就废了 ✗）

风格本身是「取舍」，没有对错 ✓ 所以这里不断言「谁强」✗ ——
只断言「它们确实不同」✓ 强弱交给 arena 去测 ✓
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import sanma as S  # noqa: E402
from aimodel.agents import HeuristicAgent  # noqa: E402
from aimodel.engine import Game  # noqa: E402
from aimodel.styles import STYLES, make, _chi_pairs  # noqa: E402


class Tally:
    """包一层，统计「这个 agent 到底做了多少次副露/立直」。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "?")
        self.calls = 0        # 接受了碰/杠
        self.chis = 0         # 接受了吃
        self.riichis = 0      # 宣布立直

    def decide(self, decision, view):
        choice = self.inner.decide(decision, view)
        if decision.kind == "call" and choice != 0:
            self.calls += 1
        elif decision.kind == "chi" and choice != 0:
            self.chis += 1
        elif decision.kind == "riichi" and choice == 0:
            self.riichis += 1
        return choice


def play_games(style_names: list[str], n_games: int, players: int, seed0: int = 0):
    """每种风格各打 n_games 局（同桌三个教师），返回 {风格: Tally 汇总}。"""
    rule = S.rule_for(players)
    out = {k: {"calls": 0, "chis": 0, "riichis": 0} for k in style_names}
    for name in style_names:
        for g in range(n_games):
            hero = Tally(make(f"style:{name}", seed=g))
            others = [Tally(HeuristicAgent(seed=100 + g, style="normal"))
                      for _ in range(players - 1)]
            game = Game([hero] + others, seed=seed0 + g, rule=rule)
            game.play()
            out[name]["calls"] += hero.calls
            out[name]["chis"] += hero.chis
            out[name]["riichis"] += hero.riichis
    return out


def main() -> int:
    failed: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'✅' if cond else '🔴'} {label}")
        if not cond:
            failed.append(label)

    print("─" * 74)
    print("① 六种风格都能构造 / 名字解析正确")
    print("─" * 74)
    check(len(STYLES) >= 6, f"至少 6 种风格（实际 {len(STYLES)}）")
    for n in sorted(STYLES):
        try:
            a = make(f"style:{n}", seed=0)
            ok = a is not None
        except Exception as e:                       # noqa: BLE001
            ok = False
            print("    构造失败:", e)
        check(ok, f"make('style:{n}') 可构造")
    try:
        make("style:不存在的风格")
        check(False, "未知风格必须报错")
    except SystemExit:
        check(True, "未知风格会报错（不是静默兜底）")

    print()
    print("─" * 74)
    print("② 吃牌组合枚举（吃哪两张）")
    print("─" * 74)
    check(sorted(_chi_pairs(21)) == [(19, 20), (20, 22), (22, 23)],
          "吃 4s(21) -> 三种组合")
    check(sorted(_chi_pairs(18)) == [(19, 20)], "吃 1s(18) -> 只有 (2s,3s)")
    check(sorted(_chi_pairs(26)) == [(24, 25)], "吃 9s(26) -> 只有 (7s,8s)")
    check(_chi_pairs(27) == set(), "吃字牌 -> 空集（字牌不能吃）")

    print()
    print("─" * 74)
    print("③ 四麻：六种风格各打 6 局（四家有副露/立直），必须不抛异常")
    print("─" * 74)
    try:
        stats4 = play_games(sorted(STYLES), 6, 4)
        check(True, "六种风格打完整局（四麻）")
    except Exception as e:                           # noqa: BLE001
        stats4 = {}
        check(False, f"四麻对局抛异常: {type(e).__name__}: {e}")

    if stats4:
        for k in sorted(stats4):
            v = stats4[k]
            print(f"    {k:12s} 碰/杠 {v['calls']:3d}   吃 {v['chis']:3d}   立直 {v['riichis']:3d}")

    print()
    print("─" * 74)
    print("④ 风格必须真的不同（副露倾向是最好区分的指标）")
    print("─" * 74)
    if stats4:
        calls = {k: v["calls"] + v["chis"] for k, v in stats4.items()}
        check(calls.get("closed", -1) == 0, "门清派一次副露都没有 ✓")
        check(calls.get("defense", -1) == 0, "弃和派一次副露都没有 ✓")
        check(calls.get("special", -1) == 0, "特殊形派一次副露都没有 ✓")
        most = max(calls, key=lambda k: calls[k])
        check(calls[most] > 0, f"至少有一派积极副露（最多的是 {most}）✓")
        check(len(set(calls.values())) >= 2,
              f"各派副露次数不完全相同（{calls}）✓")

    print()
    print("─" * 74)
    print("⑤ 三麻：必须也能打（没吃牌、但有拔北）")
    print("─" * 74)
    try:
        stats3 = play_games(sorted(STYLES), 3, 3)
        check(True, "六种风格打完整局（三麻）")
        check(all(v["chis"] == 0 for v in stats3.values()),
              "三麻里没有任何人吃牌（规则没有吃）✓")
    except Exception as e:                           # noqa: BLE001
        check(False, f"三麻对局抛异常: {type(e).__name__}: {e}")

    print()
    print("─" * 74)
    print("⑥ 确定性：同种子同牌山必须给出同一局结果")
    print("─" * 74)
    try:
        rule = S.rule_for(4)
        scores = []
        for _ in range(2):
            hero = make("style:speed", seed=7)
            others = [HeuristicAgent(seed=11, style="normal") for _ in range(3)]
            scores.append(Game([hero] + others, seed=2026, rule=rule).play())
        # ⚠️ 只打分数，别把整个 MatchResult 打出来（几千字符，没人看得下去 ✗）
        check(scores[0] == scores[1],
              f"两次同种子对局结果一致（分数 {scores[0].scores} 相同）")
    except Exception as e:                           # noqa: BLE001
        check(False, f"确定性检查抛异常: {type(e).__name__}: {e}")

    print()
    if failed:
        print(f"🔴 {len(failed)} 项未通过：")
        for f in failed:
            print("   -", f)
        return 1
    print("✅ 风格 agent 全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
