"""八个模型 + 四种得分条件的测试。

要验的就两件事：

1. **注册表对**：八个模型都在，别名（``flash4`` / ``maj_flash`` …）能解析到正名，
   三麻四麻的人数、得分条件、教师风格一一对应。
2. **得分条件真的不一样**：同样一场牌，四种模式算出来的分数必须能区分开 ——
   放铳多的局在 ``mountain`` 下更惨、和得多的局在 ``flash`` 下更高、
   番数高的局在 ``huge`` 下更高。跑几场真对局对一遍。

跑法：``.venv/bin/python aimodel/tests/test_models.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import engine as E  # noqa: E402
from aimodel import rewards, sanma as S, variants  # noqa: E402
from aimodel.agents import STYLES, HeuristicAgent  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


print("== 注册表 ==")
# ⚠️ 别写死数量 ✗ 模型家族会继续增加（maj-pro 加进来时这条断言就误报过 ✓）
#    注册表里存的是**带版本号**的规范名（maj-normalv1.1）✓
#    所以按「8 个基础别名都能解析出来」断言结构，额外家族允许存在 ✓
_base = [f"maj-{cond}{suf}"
         for cond in ("normal", "mountain", "flash", "huge")
         for suf in ("", "4")]
_unresolved = []
for alias in _base:
    try:
        variants.resolve(alias)
    except KeyError:
        _unresolved.append(alias)
check(not _unresolved, f"这八个基础模型的名字解析不出来：{_unresolved}")
check(len(variants.MODELS) >= 8,
      f"至少要有 8 个基础模型，得到 {len(variants.MODELS)}")
_extra = sorted(n for n in variants.MODELS if "pro" in n)
print(f"  ✅ 8 个基础模型（四个条件 × 三/四麻）都能解析；"
      f"另有扩展家族：{_extra or '无'}")
# 正名带版本号（当前 v1.1），不带版本的写法都是别名
from aimodel import variants as _v  # noqa: E402

V = _v.VERSION
expect = {
    f"maj-normal{V}": (3, "normal"), f"maj-mountain{V}": (3, "mountain"),
    f"maj-flash{V}": (3, "flash"), f"maj-huge{V}": (3, "huge"),
    f"maj-normal4{V}": (4, "normal"), f"maj-mountain4{V}": (4, "mountain"),
    f"maj-flash4{V}": (4, "flash"), f"maj-huge4{V}": (4, "huge"),
}
for name, (n, mode) in expect.items():
    spec = variants.get(name)
    check(spec.players == n, f"{name} 人数应为 {n}，得到 {spec.players}")
    check(spec.reward == mode, f"{name} 得分条件应为 {mode}，得到 {spec.reward}")
    check(spec.style in STYLES, f"{name} 的教师风格 {spec.style} 不存在")

# 别名：用户说的是「原名称+4」，所以 flash4 这种写法一定要认
for alias, canonical in [
    ("flash4", f"maj-flash4{V}"), ("maj-flash4", f"maj-flash4{V}"),
    ("flash", f"maj-flash{V}"), ("maj_flash4", f"maj-flash4{V}"),
    ("MAJ-HUGE4", f"maj-huge4{V}"), ("mountain", f"maj-mountain{V}"),
    ("majflash4", f"maj-flash4{V}"), ("normal4", f"maj-normal4{V}"),
    (f"maj-flash4{V}", f"maj-flash4{V}"),          # 正名自己
]:
    try:
        got = variants.resolve(alias)
    except KeyError:
        got = None
    check(got == canonical, f"别名 {alias} 应解析为 {canonical}，得到 {got}")
try:
    variants.resolve("不存在的东西")
    check(False, "乱写的名字应该报错")
except KeyError:
    pass
print(f"  ✅ 8 个模型 + 别名解析（flash4 -> maj-flash4 等）")

print("\n== 四种得分条件的权重 ==")
check(set(rewards.MODES) == {"normal", "mountain", "flash", "huge"}, "四种模式")
check(rewards.WEIGHTS["mountain"]["deal_in"] >= 3.0, "mountain 必须重罚放铳")
check(rewards.WEIGHTS["flash"]["agari"] >= 4.0, "flash 必须重奖和牌次数")
check(rewards.WEIGHTS["huge"]["han"] >= 1.5, "huge 必须重奖番数")
check(rewards.WEIGHTS["normal"]["deal_in"] == 0, "normal 不该有额外惩罚")
for m in rewards.MODES:
    check(abs(rewards.norm(0.0, m)) < 1e-9, f"{m}: 0 应归一化成 0")
    check(rewards.norm(1e9, m) == 1.0 and rewards.norm(-1e9, m) == -1.0, f"{m}: 要剪到 ±1")
print("  ✅ 权重方向正确、归一化会剪顶")

print("\n== 真对局上跑一遍 ==")
# 用「爱放铳」和「爱和牌」两种极端，看看四种模式能不能把它们区分开
rows = []
for seed in range(24):
    agents = [HeuristicAgent(seed * 3 + i, style="flash") for i in range(3)]
    r = E.Game(agents, seed=seed, rule=S.DEFAULT_RULE).play()
    for p in range(3):
        parts = rewards.parts_of(r, p)
        rows.append({
            "parts": parts,
            **{m: rewards.reward(m, r, p) for m in rewards.MODES},
        })

check(len(rows) > 50, "样本量太少")
for m in rewards.MODES:
    vals = [x[m] for x in rows]
    check(all(abs(v) < 1e6 for v in vals), f"{m}: 出现离谱数值 {max(vals)}")
    check(max(vals) > min(vals), f"{m}: 所有局算出同一个分？")

# 1) 放铳越多的局，mountain 相对 normal 应该越吃亏
worst = max(rows, key=lambda x: x["parts"].deal_in_points)
if worst["parts"].deal_in_points > 0:
    gap = worst["normal"] - worst["mountain"]
    check(gap > 0, f"放铳 {worst['parts'].deal_in_points:.0f} 点的局，mountain 应比 normal 低")
    print(f"  ✅ 放铳 {worst['parts'].deal_in_points:.0f} 点的局："
          f"normal {worst['normal']:+.1f} vs mountain {worst['mountain']:+.1f}（差 {gap:.1f}）")

# 2) 和牌次数多的局，flash 应该占便宜
most = max(rows, key=lambda x: x["parts"].agari_count)
if most["parts"].agari_count > 0:
    gap = most["flash"] - most["normal"]
    check(gap > 0, "和了 1 局以上的局，flash 应比 normal 高")
    print(f"  ✅ 和了 {most['parts'].agari_count} 局的局："
          f"flash {most['flash']:+.1f} vs normal {most['normal']:+.1f}（+{gap:.1f}）")

# 3) 番数高的局，huge 应该占便宜
biggest = max(rows, key=lambda x: x["parts"].han_total)
if biggest["parts"].han_total > 0:
    gap = biggest["huge"] - biggest["normal"]
    check(gap > 0, "有番数的局，huge 应比 normal 高")
    print(f"  ✅ 累计 {biggest['parts'].han_total} 番的局："
          f"huge {biggest['huge']:+.1f} vs normal {biggest['normal']:+.1f}（+{gap:.1f}）")

# 4) 四种模式在同一局上不能算出完全一样的值（除非这局什么都没有）
distinct = sum(1 for x in rows if len({round(x[m], 3) for m in rewards.MODES}) > 1)
check(distinct > len(rows) * 0.5, f"四种模式应当经常给出不同的分（只有 {distinct}/{len(rows)} 不同）")
print(f"  ✅ {distinct}/{len(rows)} 局能区分出四种模式")

print("\n== 四麻的得分条件 ==")
r4 = S.rule_for(4)
rows4 = []
for seed in range(12):
    agents = [HeuristicAgent(seed * 4 + i, style="mountain") for i in range(4)]
    res = E.Game(agents, seed=seed, rule=r4).play()
    check(len(res.rank_points) == 4, "四麻应有 4 个段位分")
    for p in range(4):
        rows4.append(rewards.reward("normal", res, p))
check(len(rows4) > 20, "四麻样本量太少")
print(f"  ✅ 四麻 12 场跑通，normal 得分范围 {min(rows4):+.1f} ~ {max(rows4):+.1f}")

print("\n== 教师风格都还在 ==")
for name in ("normal", "mountain", "flash", "huge"):
    check(name in STYLES, f"缺教师风格 {name}")
check(STYLES["mountain"].fold_shanten < STYLES["flash"].fold_shanten, "防守应比速攻更早弃和")
check(STYLES["huge"].value_weight > STYLES["flash"].value_weight, "大牌应比速攻更看重打点")
check(STYLES["flash"].call_shanten >= STYLES["mountain"].call_shanten, "速攻应比防守更爱叫牌")
print("  ✅ 四套教师口味的倾向关系正确")

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项失败：")
    for f in FAILED[:15]:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过")
