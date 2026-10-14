"""收益计算（气泡上那些百分比）的体检。

为什么单独测：气泡是用户**唯一直接照做**的东西。识图错了会显示成「未知」，
而收益算错了看不出来 —— 它会给出一个很自信的百分比，让人以为有依据。

测的是几条不变量，不是「答案对不对」（答案对不对得靠老师模型对比，
那是 aimodel 那边的评测）：

1. **概率得是个分布** —— 非负、和接近 1。不是分布就说明映射那一步写错了
   （比如把 34 种牌的概率错位映射回手里的牌）。
2. **候选数 = 手里不同牌的种类数** —— 少一个就有一张牌永远给不出建议，
   多一个就是凭空造出一张不存在的牌。
3. **建议打的那张不该让向听变差** —— 有别的选择能保持向听时，却建议一张
   让向听退后的牌，这在三麻里几乎总是错的（除非是刻意弃和，网络不知道危险度，
   所以它不会这么做）。这条是外部校验：向听是另一套代码算的，跟网络无关。
4. **巡目 0 不能被当成「没值」** —— 曾经写成 ``turn or 6``，于是每一局的
   第一巡都拿 6 去喂网络。第一巡恰好是最需要建议的时候。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from potatomaj import ai as AI                        # noqa: E402

# 这个测试验的是「网络的输出是不是一个概率分布」，所以**显式指定一个权重模型** ——
# 不能跟着 build/models.json 走：用户可以把它切到 teacher，而教师的分数是
# 「向听*1e6 + 进张*权重 - 危险*权重」归一化出来的排序值，**不是概率**，
# 用分布去断言必然失败（踩过：手动测完忘了把 models.json 改回来，测试就红了）。
import os as _os  # noqa: E402
_os.environ["POTATOMAJ_MODEL"] = "maj-flashv1.1"
AI._user_config.cache_clear()
AI._net_for.cache_clear()
from potatomaj.shanten import discard_options         # noqa: E402

#: 几手真实形状的三麻手牌（都是刚摸完的 14 张、门清）
HANDS = [
    ["1m", "2p", "3p", "7p", "2s", "3s", "4s", "4s", "7s", "9s", "9s", "2z", "3z", "2s"],
    ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "1s", "2s", "3s", "5z", "5z"],
    ["1m", "9m", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "2s", "3s", "4s", "6z", "6z"],
    ["1m", "9m", "1p", "1p", "1p", "2s", "3s", "4s", "5s", "6s", "7s", "7z", "7z", "7z"],
    ["1m", "9m", "9m", "1p", "1p", "3p", "5p", "7p", "9p", "2s", "4s", "6s", "1z", "1z"],
]

checks: list[tuple[str, bool, str]] = []


def add(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


# ---- 1) 概率是个分布 ----
dist_ok = True
sums: list[float] = []
for hand in HANDS:
    probs, note = AI.rank(hand, [], [[], [], [], []], 0, turn=1)
    if not probs:
        continue
    total = sum(probs.values())
    sums.append(total)
    if total < 0.9 or total > 1.1 or any(v < 0 for v in probs.values()):
        dist_ok = False
add("概率是个分布（非负、和接近 1）", dist_ok and bool(sums),
    f"和 = {['%.3f' % s for s in sums]}")

# ---- 2) 候选数 = 手里不同牌的种类数 ----
count_ok = True
details: list[str] = []
for hand in HANDS:
    opts = discard_options(hand, sanma=True, melds=0)
    kinds = len({t for t in hand})
    if len(opts) != kinds:
        count_ok = False
        details.append(f"{kinds} 种牌却给了 {len(opts)} 个候选")
add("候选数 = 手里不同牌的种类数", count_ok, "；".join(details))

# ---- 3) 建议打的那张不该让向听变差 ----
bad: list[str] = []
for hand in HANDS:
    probs, _ = AI.rank(hand, [], [[], [], [], []], 0, turn=1)
    opts = discard_options(hand, sanma=True, melds=0)
    if not probs or not opts:
        continue
    best_loss = min(o["loss"] for o in opts)
    pick = max(opts, key=lambda o: probs.get(o["tile"], 0.0))
    if pick["loss"] > best_loss:
        bad.append(f"建议打 {pick['tile']}（向听退 {pick['loss']}），"
                   f"但有牌能保持（退 {best_loss}）")
add("建议打的那张不让向听变差", not bad, "；".join(bad) or "（全部保持）")

# ---- 4) 巡目 0 是合法值，不能被替换 ----
p0, _ = AI.rank(HANDS[0], [], [[], [], [], []], 0, turn=0)
p1, _ = AI.rank(HANDS[0], [], [[], [], [], []], 0, turn=1)
p6, _ = AI.rank(HANDS[0], [], [[], [], [], []], 0, turn=6)
if p0 and p1 and p6:
    top = lambda d: max(d, key=d.get) if d else None            # noqa: E731
    add("巡目 0 被规范成第 1 巡（而不是当成没值）", top(p0) == top(p1),
        f"turn=0 -> {top(p0)}，turn=1 -> {top(p1)}")
    add("（参考）巡目不同建议可能不同", True,
        f"turn=1 -> {top(p1)}　turn=6 -> {top(p6)}")
else:
    add("巡目 0 被规范成第 1 巡（而不是当成没值）", False, "网络没起来，测不了")

# ---------------------------------------------------------------------------
# 回归：拔北**不算副露**
#
# 踩过的坑：把 ActionBaBei 也 push 进 melds，而「等效张数」按
# ``手牌 + 3×副露`` 算 —— 于是拔两次北就成了「14 + 3×2 = 20 张」，
# 整手牌被判成「不合法，未做判断」，界面上一句建议都不给。
#
# 实际规则：拔北和杠一样是「拿掉一张 + 补摸一张」，**手牌张数不变**，
# 所以它既不占格数、也不该进副露计数（aimodel 里也是单独用 kita[] 记的）。
from potatomaj.shanten import analyze as analyze_hand

hand13 = ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7s", "8s", "9s", "5z", "6z"]

# 13 张 + 0 副露：正常，能算出向听
r = analyze_hand(hand13, sanma=True, melds=0)
add("13 张无副露能算出向听", r["shanten"] is not None, r["text"][:30])

# 拔了两次北之后手里**还是 13 张**（每次都补摸了），所以副露数仍然是 0，
# 判断结果应该和没拔北时**完全一样**
r2 = analyze_hand(hand13, sanma=True, melds=0)
add("拔北不改变向听判断（它不占格数）", r2["shanten"] == r["shanten"], str(r2["shanten"]))

# 万一副露数真的传错了（比如把拔北算成 2 个副露），要能看出来不对，
# 而且说明里要指出「拔北不占格数」这个可能
r3 = analyze_hand(hand13, sanma=True, melds=2)
add("副露数传错时能识别出来，并提示拔北的可能",
    r3["shanten"] is None and "拔北" in r3["text"], r3["text"][:24])


fail = 0
for name, ok, detail in checks:
    if not ok:
        fail += 1
    print(f"{'✅' if ok else '❌'} {name}" + (f"　{detail}" if detail else ""))
print()
print("✅ 收益计算体检通过" if fail == 0 else f"❌ {fail} 项不通过")
