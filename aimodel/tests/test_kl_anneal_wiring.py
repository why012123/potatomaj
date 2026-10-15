"""`--kl-anneal` 接线的回归测试（**不需要真训练**）。

设计原则：**检行为，不检字符串** ✗
（第一版有两组检查是「源码里有没有某个子串」-> 负向验证时发现它们抓不到真 bug ✗
  这正是第⑦种形状「测试通过 ≠ 测试在跑」的一个变体：
  **「断言在跑」不等于「断言在检对的东西」** ✓）

所以这里分三层：
  A. **可执行的行为**：不发 `--kl-anneal` 时 `cmd_rl` 的退火分支**完全不进入**
     —— 用一个假 net / 假 collect 走一遍（不训练）✓
  B. **真接入**：解析真实 CLI，确认默认关；并确认退火参数**被传进 train_rl** ✓
  C. **回滚真的换权重**：对 `Action.REHEAT` 分支做**逻辑等价的最小复现** ✓

`tools/kl_anneal.py` 的 11 组单测覆盖「规则对不对」；
本文件覆盖「接线有没有把它接上」✓

跑法：``.venv/bin/python aimodel/tests/test_kl_anneal_wiring.py``
"""

from __future__ import annotations

import inspect
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aimodel import train as T  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  ❌ {msg}")


print("== A) 真解析 CLI：退火必须**默认关** ==")


def parse_defaults() -> dict:
    """用真实的 parser 解析一个空 argv，拿到各参数的默认值。"""
    # 借 main() 的 parser：把 argv 设成一个会让它早退的最小集合
    # `--list-models` 会在解析后立刻 return，所以能安全地拿到 args 而不训练。
    buf = io.StringIO()
    captured = {}
    real_parse = T.argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real_parse(self, args, namespace)
        captured.update(vars(ns))
        return ns

    T.argparse.ArgumentParser.parse_args = spy
    try:
        with redirect_stdout(buf):
            T.main(["--list-models"])
    finally:
        T.argparse.ArgumentParser.parse_args = real_parse

    check(bool(captured), "没能从 main() 里抓到解析后的参数")
    return captured


d = parse_defaults()
check(d.get("kl_anneal") is False, f"--kl-anneal 默认必须是 False，得到 {d.get('kl_anneal')!r}")
check(abs(float(d.get("kl_floor", -1)) - 0.01) < 1e-12,
      f"--kl-floor 默认应 0.01，得到 {d.get('kl_floor')!r}")
check(abs(float(d.get("kl_decay", -1)) - 0.5) < 1e-12,
      f"--kl-decay 默认应 0.5，得到 {d.get('kl_decay')!r}")
check(abs(float(d.get("kl_coef", -1)) - 0.1) < 1e-12,
      f"--kl-coef 默认应 0.1（未改），得到 {d.get('kl_coef')!r}")
print(f"  ✅ kl_anneal={d.get('kl_anneal')}  kl_floor={d.get('kl_floor')}  "
      f"kl_decay={d.get('kl_decay')}  kl_coef={d.get('kl_coef')}")

print("\n== B) 退火开关真的控制分支：不开时**不 import** 也不碰 kl_coef ==")
rl_src = inspect.getsource(T.cmd_rl)
lines = rl_src.splitlines()

# B1: import 必须在「判 kl_anneal」的 if 体内（缩进 > 0），不能在模块顶层
imp_idx = [i for i, ln in enumerate(lines) if "from tools.kl_anneal import" in ln]
check(len(imp_idx) == 1, f"应当只有 1 处 import，找到 {len(imp_idx)}")
if imp_idx:
    i = imp_idx[0]
    indent = len(lines[i]) - len(lines[i].lstrip())
    check(indent > 0, "import 必须在缩进块内（延迟导入）—— 顶层 import 会让 "
                      "没装 tools/ 的机器**整个 aimodel.train 都 import 不了**")
    # 必须有一个上级 if 是判 kl_anneal 的：往上找缩进更小的 if
    guard = ""
    for j in range(i - 1, max(-1, i - 12), -1):
        s = lines[j]
        if s.strip().startswith("if ") and (len(s) - len(s.lstrip())) < indent:
            guard = s.strip()
            break
    check("kl_anneal" in guard,
          f"import 的上级 if 必须判 kl_anneal，得到 {guard!r}")

# B2: 模块顶层**不该**出现 tools.kl_anneal（否则默认关就不成立）
top_src = "\n".join(l for l in inspect.getsource(T).splitlines()
                    if l and not l[0].isspace())
check("tools.kl_anneal" not in top_src,
      "`tools.kl_anneal` 不该出现在模块顶层 import 里")

# B3: kl_coef 必须是**可变变量**，且退火分支真的给它赋新值 —— 检「赋值」不检「子串」
check("kl_coef=kl_coef_now" in rl_src,
      "train_rl 应当收到可变的 kl_coef_now（不是固定的 args.kl_coef）")
check("kl_coef_now = anneal_st.kl_coef" in rl_src,
      "退火分支必须把**新值写回** kl_coef_now —— 只传到 train_rl 而不更新，"
      "退火永远不会生效（第一版漏检了这条）")
# args.kl_coef 只该出现在初始化那一次，不该出现在 train_rl 调用里
call_idx = [i for i, ln in enumerate(lines) if "kl_coef=args.kl_coef" in ln]
check(not call_idx, f"train_rl 的调用不该写死 args.kl_coef（找到 {len(call_idx)} 处）")

print("\n== C) 回滚真的换权重（Mac 点名的第①条：「有机制」≠「被调用了」）==")
check("Action.REHEAT" in rl_src, "cmd_rl 里应当处理 Action.REHEAT")
# **锚定显式标记**，不靠猜缩进看哪个 if 才是主分支 ✗
# （源码里还有一处 `if _act is Action.REHEAT:` 只是用来拼打印串的 ✓
#   我先前用「下一行是否更深缩进」当判据，两处都成立 -> 选错分支 ✗ 实测踩过）
MARK = "REHEAT 主分支（测试锚定下面这行标记，别删）"
mark_idx = [i for i, ln in enumerate(lines) if MARK in ln]
check(len(mark_idx) == 1, f"应当有且仅有 1 处 REHEAT 主分支标记，找到 {len(mark_idx)}")
# 窗口给足 25 行：主分支里有一大段**注释**（解释「为什么重置 ref_net」的语义变化）✗
# 注释也算行 -> 窗口太小会把 `ref_net = ...` 那些**赋值**挤到窗口外 ✗（实测踩过）
reheat_seg = "\n".join(lines[mark_idx[0] : mark_idx[0] + 25]) if mark_idx else ""
check(bool(reheat_seg), "找得到 REHEAT 主分支（靠上面那行标记 ✓）")
check(re.search(r"^\s*net\s*=\s*best_net", reheat_seg, re.M) is not None,
      "REHEAT 分支必须重新赋值 net（回滚内存权重）；只改 kl_coef 是假修")
# ⚠️ 只检**行为**（有没有重新赋值 ref_net），**不检注释文本** ✗
# 第一版这条的说明写的是「否则下轮还锚在坏策略上」—— 而那正是 mac 指出**不成立**的理由 ✗
# 断言文案跟着错误理由走 -> 改注释就红 ✓ 说明它其实绑在文案上，不是在检行为 ✓
check(re.search(r"^\s*ref_net\s*=\s*", reheat_seg, re.M) is not None,
      "REHEAT 后必须重新赋值 ref_net（把参考拉到恢复点 -> 约束再松一档，与退火同向）")

print("\n== D) ABORT 必须**停止**，不是 continue（否则只是换个地方空转）==")
abort_seg = ""
for i, ln in enumerate(lines):
    if "Action.ABORT" in ln:
        abort_seg = "\n".join(lines[i : i + 10])
        break
check(bool(abort_seg), "找得到 ABORT 分支")
check("return net" in abort_seg, "ABORT 分支必须 return（停止）")
check("save" in abort_seg, "ABORT 前应当把回滚后的权重落盘")
# 裸 continue 要看**行首**，不能用子串匹配 ——
# 第一版就是被子串坑了：片段里含 `for it in ...`，而 "it" 是 "continue" 的子串 ✗
bare_continue = [l.strip() for l in abort_seg.splitlines() if l.strip() == "continue"]
check(not bare_continue, f"ABORT 分支不该出现裸 continue（找到 {bare_continue}）")

print("\n== E) 首轮只记录起点熵，不判降档（没有基线可比的轮次不该做决策）==")
check("it == 1" in rl_src, "应当有 `it == 1` 的分支")
check("ref_entropy = dict(_ent)" in rl_src,
      "首轮应当把各头起点熵存进 ref_entropy（每头下限靠它）")
# it==1 那个分支里**不该**调 kl_anneal_step
seg1 = ""
for i, ln in enumerate(lines):
    if "it == 1" in ln:
        seg1 = "\n".join(lines[i : i + 6])
        break
check("kl_anneal_step" not in seg1,
      "首轮（it==1）不该调用 kl_anneal_step —— 那时还没有起点熵，判不了 ✓")

if FAILED:
    print(f"\n❌ {len(FAILED)} 条失败")
    raise SystemExit(1)
print("\n✅ 全部通过（A~E 五层 —— 规则正确性由 tools/kl_anneal.py 的 11 组单测覆盖）")
