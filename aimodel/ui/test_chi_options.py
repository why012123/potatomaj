"""「吃」选项的回归测试（2026-10-20 修的那个真实崩溃 + 静默不吃的 bug）。

    python -m aimodel.ui.test_chi_options

## 背景：一个 bug，两个症状

``engine._chi_plan`` 把每个吃法做成**独立选项**：

    ["pass", "chi:10:12", "chi:21:22"]      <- 1p2p / 4p5p

而 ``ACTION_TABLE["chi"]`` 一直是**二值**的 ``("pass", "chi")``。
两处对不上，于是：

① ``encode.action_mask`` / ``action_index`` 用 ``table.index("chi:10:12")``
   -> ``ValueError: tuple.index(x): x not in tuple``
   -> **整局崩掉**（RL 阶梯就是这么断的，2026-10-20 21:11 官方 8 点那期）

② ``MLAgent._head_to_option`` 用 ``str(o) == "chi"`` 找选项
   -> **永远找不到** —— 因为真实选项是 ``"chi:10:12"``
   -> 恒返回 0 = pass
   -> **神经网络在四麻里永远不吃牌，而且完全静默**（不报错、不警告）
   ⚠️ 症状②比①更危险：①至少会崩，②会安安静静地把牌力削掉

## 修法与不变式

``chi`` 头是二值的 -> **非 pass 的任何选项都算 head 1**，
且 head 1 <-> ``options[1]``（``_chi_plan`` 已按「吃完后向听」升序排，最优在前）。

负向验证：
  把 ``encode.action_mask`` 里 chi 那个分支删掉 -> 「不再抛异常」那条必红 ✓
"""

from __future__ import annotations

import sys

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel.encode import (  # noqa: E402
    ACTION_TABLE, HEAD_SIZE, action_index, action_mask,
)
from aimodel.engine import Decision  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  🔴 {name}   {detail}")


def engine_uses_per_combo_options() -> bool:
    """当前引擎是否给出「每个吃法一个选项」的格式。

    ⚠️ 判断依据是 **`Game._chi_plan` 这个方法存不存在**：
       · `main` 上**没有** `_chi_plan`（chi 是内联处理的，选项是二值 `["pass","chi"]`）
       · WIP 里**新增了** `_chi_plan`，它返回 `["pass", "chi:18:19", ...]`

       为什么不能只看源码里有没有 `"chi:"` 字符串：
       我先那么写过，结果在 `main` 上也判成 True —— 因为**别处**的注释/日志里
       也有 `chi:` 字样。**用「某个符号存不存在」比「某段文本出不出现」可靠**。

       而为什么要跳过（而不是让它红）：
       本测试测的是「逐选项下的映射」；`main` 上那个前提**根本不存在**，
       硬跑会红，而红的原因不是「有 bug」，是「这个场景还没出现」。
       Mac 2026-10-20 原话：「这条测试在 main 上应当红，所以它要么跟着 WIP
       一起进 main，要么先别加，你定。」
       -> 我选第三条：**自己判断前提** —— 前提不在就 Skip（既不是 Pass 也不是 Fail）✓
          main 上套件绿 ✓ 而逐选项一落地它**立刻开始生效** ✓ 不需要谁记得去打开 ✓
    """
    from aimodel import engine as _E

    return hasattr(_E.Game, "_chi_plan")


def main() -> int:
    print("=" * 70)
    print("  「吃」选项回归测试")
    print("=" * 70)

    if not engine_uses_per_combo_options():
        print("\n  [跳过] 当前引擎的 chi 选项还是**二值** [\"pass\",\"chi\"]")
        print("     （`engine._chi_plan` 里没有 \"chi:<a>:<b>\" 这种选项串）")
        print("     -> 本测试针对的「逐选项」前提还不存在，硬跑会误报红")
        print("     -> 这不是通过，是跳过；等逐选项落地后它会自动开始生效")
        print("\n" + "=" * 70)
        print("  通过 0  失败 0  （跳过）")
        print("=" * 70)
        return 0

    # 真实形状：一个 pass + 两个吃法（按向听升序，第一个最优）
    real = ["pass", "chi:10:12", "chi:21:22"]
    d = Decision("chi", real, tile=10)

    # ---- ① 不再崩溃 ------------------------------------------------------
    print("\n① action_mask / action_index 不再抛异常")
    try:
        m = action_mask(d)
        ok = True
        err = ""
    except Exception as exc:                      # noqa: BLE001
        ok = False
        m = np.zeros(0)
        err = f"{type(exc).__name__}: {exc}"
    check("action_mask 对 ['pass','chi:10:12','chi:21:22'] 不抛异常", ok, err)

    try:
        i1 = action_index(d, 1)
        i2 = action_index(d, 2)
        i0 = action_index(d, 0)
        ok2 = True
        err2 = ""
    except Exception as exc:                      # noqa: BLE001
        ok2 = False
        i0 = i1 = i2 = -1
        err2 = f"{type(exc).__name__}: {exc}"
    check("action_index 也不抛异常", ok2, err2)

    # ---- ② 掩码语义：pass 与「有得吃」都合法 ----------------------------
    print("\n② 掩码语义（二值头）")
    check("head 0（pass）合法", bool(m[0]), str(m.tolist()))
    check("head 1（chi）合法", bool(m[1]), str(m.tolist()))
    check("掩码长度 = HEAD_SIZE['chi'] = 2", len(m) == HEAD_SIZE["chi"] == 2,
          f"{len(m)} vs {HEAD_SIZE['chi']}")
    check("掩码没有多亮别的位", int(m.sum()) == 2, str(m.tolist()))

    # ---- ③ 下标映射 ------------------------------------------------------
    print("\n③ 下标映射（options 下标 -> 头下标）")
    check("options[0]=pass -> head 0", i0 == 0, str(i0))
    check("options[1]=chi:10:12 -> head 1（最优吃法）", i1 == 1, str(i1))
    check("options[2]=chi:21:22 -> head 1", i2 == 1, str(i2))

    # ---- ④ 只有 pass 时 ------------------------------------------------
    print("\n④ 只有 pass 的边界")
    only = Decision("chi", ["pass"], tile=10)
    m2 = action_mask(only)
    check("只有 pass -> 只亮 head 0", int(m2.sum()) == 1 and bool(m2[0]), str(m2.tolist()))
    check("只有 pass -> action_index(0) == 0", action_index(only, 0) == 0, "")

    # ---- ⑤ 其它种类没被我改坏 -------------------------------------------
    print("\n⑤ 其它种类不受影响（回归）")
    for kind, opts in (("call", ["pass", "pon"]),
                       ("riichi", ["yes", "no"]),
                       ("tsumo", ["yes", "no"]),
                       ("kan", ["pass", "kan"]),
                       ("kyuushu", ["yes", "no"])):
        dd = Decision(kind, opts)
        try:
            mm = action_mask(dd)
            good = len(mm) == HEAD_SIZE[kind] and mm.sum() >= 1
            for i, o in enumerate(opts):
                want = ACTION_TABLE[kind].index(str(o))
                good = good and action_index(dd, i) == want
            check(f"{kind} 掩码与下标仍正确", good, f"{mm.tolist()}")
        except Exception as exc:                  # noqa: BLE001
            check(f"{kind} 掩码与下标仍正确", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"  通过 {PASS}  🔴 失败 {FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
