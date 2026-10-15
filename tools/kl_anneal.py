#!/usr/bin/env python
"""KL 退火：纯逻辑 + 可单测（**不接训练循环**）。

为什么需要它
============
RL 阶段用 KL 锚定把策略拉回「BC 克隆出来的参考策略」（`net.py:150 set_reference`）。
那是**防训崩**的设计（P1-2）✓ 但代价是**策略结构上飞不远** ✗
—— 所以「自我迭代超不过教师」有一条结构性原因就在这里。

退火的目的是**放开锚定**，但**不能一次放开**：要按模型实际状态逐步放。

为什么是「爬-守」而不是日程
==========================
* **固定日程**：和模型状态无关 ✗ —— 正是「带阈值的规则在阈值外静默失效」那种毛病
  （同类教训：天梯的 `--max-gap 200` 让两端静默冻结）
* **留出率**：RL 阶段**不模仿教师** ✗ 留出率在这里没有意义 ✗
* **逐头熵**：我们要放开锚定的**唯一理由是「让它能飞」**，
  而「能不能飞」的直接观测量就是**策略还有没有多样性** = 熵 ✓
  **KL 是结果、熵是余量** —— 要管的是**余量别耗尽** ✓

规则（优先级从上到下）
====================
1. **先判坍缩**：任一头熵 < 该头的下限 -> `REHEAT`（回滚 + 锚定回退一档）
2. **再判降档**：实测 KL < `kl_lo` **且**所有头熵都高于「下限 + 余量」-> `DESCEND`
3. **都不满足** -> `HOLD`（原地守着）

`kl_coef` 有 **下限 `kl_floor`，永不归零** —— 这是 P1-2 的精神：
留一个很小的锚 ≠ 没锚，它仍能挡住「一步崩掉」的梯度。

用法::

    from tools.kl_anneal import AnnealState, AnnealConfig, kl_anneal_step

    st = AnnealState(kl_coef=0.1, ref_entropy={"discard": 2.1, ...})
    st, action = kl_anneal_step(st, ent_by_kind={"discard": 2.0, ...}, measured_kl=0.3)
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping


class Action(str, Enum):
    """这一轮该做什么。"""

    HOLD = "hold"        # 守着（不降也不回退）
    DESCEND = "descend"  # 降一档锚定（放开一点）
    REHEAT = "reheat"    # 检测到坍缩 -> 回退一档并回滚权重


@dataclass(frozen=True)
class AnnealConfig:
    """退火超参。默认值见频道里那份设计。"""

    #: 起始锚定强度（与 `--kl-coef` 的默认值一致）
    kl_coef0: float = 0.1
    #: **下限，永不归零** —— P1-2 的防崩精神
    kl_floor: float = 0.01
    #: 每降一档乘多少
    decay: float = 0.5
    #: 「几乎没被惩罚」的阈值：实测 KL 低于它才考虑降档
    kl_lo: float = 0.05
    #: 各头熵下限 = 该头在 RL 起点的熵 × 这个比例
    floor_ratio: float = 0.5
    #: 但下限不低于这个绝对值（防「本来就接近确定」的头把下限压到不可能触发）
    floor_abs_min: float = 0.02
    #: 降档要求「熵高出下限」多少（= 起点熵 × 这个比例）
    margin_ratio: float = 0.10
    #: 连续触发几次坍缩之后才允许把锚定**翻倍**（避免震荡）
    reheat_double_after: int = 2


@dataclass
class AnnealState:
    """退火状态。**可序列化**（每轮存下来，便于复现与事后复盘）。"""

    kl_coef: float
    #: 各头在 RL **起点**的熵（用来定每头的下限）
    ref_entropy: dict[str, float] = field(default_factory=dict)
    #: 上一档的锚定值（`REHEAT` 时回退到这里，而不是盲目翻倍）
    prev_kl_coef: float | None = None
    #: 连续坍缩次数
    consecutive_collapse: int = 0
    #: 历史（每轮一条），便于画轨迹与事后审计
    history: list[dict] = field(default_factory=list)


def _floor_for(kind: str, ref_ent: float, cfg: AnnealConfig) -> float:
    """某个头的熵下限。"""
    return max(ref_ent * cfg.floor_ratio, cfg.floor_abs_min)


def _margin_for(kind: str, ref_ent: float, cfg: AnnealConfig) -> float:
    return ref_ent * cfg.margin_ratio


def collapsed_heads(
    state: AnnealState, ent_by_kind: Mapping[str, float], cfg: AnnealConfig | None = None
) -> list[str]:
    """哪些头的熵掉到下限以下了（空 = 没坍缩）。"""
    cfg = cfg or AnnealConfig()
    out = []
    for k, ref in state.ref_entropy.items():
        cur = ent_by_kind.get(k)
        if cur is None:
            continue  # 这一轮没这个头的数据 -> 不判它（宁可漏判也不误判）
        if cur < _floor_for(k, ref, cfg):
            out.append(k)
    return out


def headroom(
    state: AnnealState, ent_by_kind: Mapping[str, float], cfg: AnnealConfig | None = None
) -> dict[str, float]:
    """每个头「离下限还有多远」（负 = 已跌破）。用来做可读的日志。"""
    cfg = cfg or AnnealConfig()
    out = {}
    for k, ref in state.ref_entropy.items():
        cur = ent_by_kind.get(k)
        if cur is None:
            continue
        out[k] = cur - _floor_for(k, ref, cfg)
    return out


def kl_anneal_step(
    state: AnnealState,
    *,
    ent_by_kind: Mapping[str, float],
    measured_kl: float,
    cfg: AnnealConfig | None = None,
) -> tuple[AnnealState, Action]:
    """跑一轮退火决策。**纯函数**（不改传入的 state，返回新的）。

    :param ent_by_kind: 本轮各决策头的平均熵（`net.step()` 的 ``entropy_by_kind``）
    :param measured_kl: 本轮实测的 KL（`step()` 返回的 ``kl``）
    :returns: ``(新状态, 该做的动作)``
    """
    cfg = cfg or AnnealConfig()
    bad = collapsed_heads(state, ent_by_kind, cfg)

    # --- 1. 先判坍缩（优先于一切）---
    if bad:
        n = state.consecutive_collapse + 1
        if n >= cfg.reheat_double_after:
            # 连续坍缩 -> 允许翻倍（说明回退一档还不够）
            new_coef = min(state.kl_coef * 2.0, cfg.kl_coef0)
        else:
            # 首次坍缩 -> **回退一档**（不是翻倍，避免震荡）
            new_coef = state.prev_kl_coef if state.prev_kl_coef is not None else state.kl_coef * 2.0
            new_coef = min(new_coef, cfg.kl_coef0)
        new = replace(
            state,
            kl_coef=new_coef,
            prev_kl_coef=state.kl_coef,
            consecutive_collapse=n,
            history=state.history + [{
                "action": Action.REHEAT.value, "kl_coef": new_coef,
                "collapsed": bad, "measured_kl": measured_kl,
            }],
        )
        return new, Action.REHEAT

    # --- 2. 再判能不能降档 ---
    h = headroom(state, ent_by_kind, cfg)
    has_margin = bool(h) and all(
        v > _margin_for(k, state.ref_entropy[k], cfg) for k, v in h.items()
    )
    can_descend = (
        measured_kl < cfg.kl_lo
        and has_margin
        and state.kl_coef > cfg.kl_floor
    )
    if can_descend:
        new_coef = max(cfg.kl_floor, state.kl_coef * cfg.decay)
        new = replace(
            state,
            kl_coef=new_coef,
            prev_kl_coef=state.kl_coef,
            consecutive_collapse=0,
            history=state.history + [{
                "action": Action.DESCEND.value, "kl_coef": new_coef,
                "measured_kl": measured_kl, "headroom_min": min(h.values()),
            }],
        )
        return new, Action.DESCEND

    # --- 3. 守着 ---
    new = replace(
        state,
        consecutive_collapse=0,
        history=state.history + [{
            "action": Action.HOLD.value, "kl_coef": state.kl_coef,
            "measured_kl": measured_kl,
            "headroom_min": min(h.values()) if h else math.nan,
        }],
    )
    return new, Action.HOLD


# ---------------------------------------------------------------------------
# 单测（合成轨迹，不需要训练）
# ---------------------------------------------------------------------------

def _expect(cond: bool, msg: str, failed: list[str]) -> None:
    if not cond:
        failed.append(msg)
        print(f"  ❌ {msg}")


def selftest() -> int:
    print("== KL 退火逻辑单测（合成轨迹）==")
    failed: list[str] = []
    cfg = AnnealConfig()
    ref = {"discard": 2.0, "call": 1.0, "chi": 0.5}   # 起点熵
    # 下限 = max(ref*0.5, 0.02) -> discard 1.0 / call 0.5 / chi 0.25
    # 余量 = ref*0.1          -> discard 0.2 / call 0.1 / chi 0.05

    print("\n-- 1) 熵有余量 + KL 很小 -> 应当降档 --")
    st = AnnealState(kl_coef=0.1, ref_entropy=dict(ref))
    st, act = kl_anneal_step(st, ent_by_kind=dict(ref), measured_kl=0.01, cfg=cfg)
    _expect(act is Action.DESCEND, f"应当 DESCEND，得到 {act}", failed)
    _expect(abs(st.kl_coef - 0.05) < 1e-9, f"kl_coef 应 0.1->0.05，得到 {st.kl_coef}", failed)
    _expect(st.prev_kl_coef == 0.1, f"prev 应记为 0.1，得到 {st.prev_kl_coef}", failed)

    print("\n-- 2) KL 还大 -> 应当守着（不降）--")
    st2 = AnnealState(kl_coef=0.1, ref_entropy=dict(ref))
    st2, act2 = kl_anneal_step(st2, ent_by_kind=dict(ref), measured_kl=0.5, cfg=cfg)
    _expect(act2 is Action.HOLD, f"应当 HOLD，得到 {act2}", failed)
    _expect(st2.kl_coef == 0.1, "守着时 kl_coef 不该变", failed)

    print("\n-- 3) 熵掉到下限以下 -> 应当回退（不是继续降）--")
    st3 = AnnealState(kl_coef=0.05, ref_entropy=dict(ref), prev_kl_coef=0.1)
    low = {"discard": 0.9, "call": 1.0, "chi": 0.5}   # discard 0.9 < 下限 1.0
    st3, act3 = kl_anneal_step(st3, ent_by_kind=low, measured_kl=0.01, cfg=cfg)
    _expect(act3 is Action.REHEAT, f"应当 REHEAT，得到 {act3}", failed)
    _expect(abs(st3.kl_coef - 0.1) < 1e-9,
            f"首次坍缩应回退到 prev(0.1)，得到 {st3.kl_coef}", failed)

    print("\n-- 4) 连续坍缩 -> 才允许翻倍 --")
    st4 = AnnealState(kl_coef=0.02, ref_entropy=dict(ref), prev_kl_coef=0.05,
                      consecutive_collapse=1)
    st4, act4 = kl_anneal_step(st4, ent_by_kind=low, measured_kl=0.01, cfg=cfg)
    _expect(act4 is Action.REHEAT, "连续坍缩也应是 REHEAT", failed)
    _expect(abs(st4.kl_coef - 0.04) < 1e-9,
            f"连续坍缩应翻倍 0.02->0.04，得到 {st4.kl_coef}", failed)

    print("\n-- 5) 下限：永不归零 --")
    st5 = AnnealState(kl_coef=0.011, ref_entropy=dict(ref))
    for _ in range(20):
        st5, a5 = kl_anneal_step(st5, ent_by_kind=dict(ref), measured_kl=0.0, cfg=cfg)
    _expect(abs(st5.kl_coef - cfg.kl_floor) < 1e-9,
            f"应当停在下限 {cfg.kl_floor}，得到 {st5.kl_coef}", failed)
    _expect(st5.kl_coef > 0, "下限必须 > 0（P1-2 防崩精神）", failed)

    print("\n-- 6) 上限：翻倍不得越过起始值 --")
    st6 = AnnealState(kl_coef=0.09, ref_entropy=dict(ref), prev_kl_coef=0.09,
                      consecutive_collapse=5)
    st6, _ = kl_anneal_step(st6, ent_by_kind=low, measured_kl=0.0, cfg=cfg)
    _expect(st6.kl_coef <= cfg.kl_coef0,
            f"翻倍不该超过 kl_coef0={cfg.kl_coef0}，得到 {st6.kl_coef}", failed)

    print("\n-- 7) 缺某个头的数据 -> 不判它（宁可漏判不误判）--")
    st7 = AnnealState(kl_coef=0.1, ref_entropy=dict(ref))
    only = {"discard": 2.0}       # 只报了一个头，且它健康
    st7, act7 = kl_anneal_step(st7, ent_by_kind=only, measured_kl=0.0, cfg=cfg)
    _expect(act7 is Action.DESCEND,
            f"缺数据的头不参与判定 -> 应 DESCEND，得到 {act7}", failed)

    print("\n-- 8) 纯函数：不改传入的 state --")
    st8 = AnnealState(kl_coef=0.1, ref_entropy=dict(ref))
    before = st8.kl_coef
    _ = kl_anneal_step(st8, ent_by_kind=dict(ref), measured_kl=0.0, cfg=cfg)
    _expect(st8.kl_coef == before, "传入的 state 不该被修改（纯函数）", failed)

    print("\n-- 9) 历史可复盘 --")
    st9 = AnnealState(kl_coef=0.1, ref_entropy=dict(ref))
    for _ in range(3):
        st9, _ = kl_anneal_step(st9, ent_by_kind=dict(ref), measured_kl=0.0, cfg=cfg)
    _expect(len(st9.history) == 3, f"应记 3 条历史，得到 {len(st9.history)}", failed)
    _expect(all("action" in h and "kl_coef" in h for h in st9.history),
            "每条历史都该有 action 与 kl_coef", failed)

    print()
    if failed:
        print(f"❌ {len(failed)} 条失败")
        return 1
    print("✅ 全部通过（9 组）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KL 退火（纯逻辑 + 自检）")
    ap.add_argument("--selftest", action="store_true", help="跑合成轨迹单测")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
