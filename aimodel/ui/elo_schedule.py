"""官方 ELO 的档位调度：每天 8/11/14/17/20 点各跑一期。

## 为什么单独一个模块

「什么时候该开一期」是**纯逻辑**（给一个时间戳和一个已跑集合，答案是"跑/不跑"），
可以脱开 HTTP 服务和训练线程单独测 ✓ —— 所以它不在 :mod:`arena_server` 里
混着写 ✓（与 ``tools/kl_anneal.py`` 同一个思路）。

## 口径（用户 2026-10-20 定的）

    · 每天 **08:00 / 11:00 / 14:00 / 17:00 / 20:00** 各一期（每整三小时 ✓）
    · 每期 **4000 局**（实测 1.98 局/秒 -> 约 34 分钟 ✓ 每档有 3 小时可用 ✓）
    · 期号 = ``YYYY-MM-DDTHH``（用**档位的小时**，不是"现在几点" ✗）

## ⚠️ 几个必须处理的情况

1. **同一档只跑一次** ✗ —— 调度器每分钟检查一次 ✓ 靠「该档位已存在期存档」判重 ✓
   （判据是**落盘**的期存档 ✓ 不是内存里的一个集合 ✗ —— 服务重启后集合就没了 ✗）
2. **错过就补跑** ✓ —— 服务在 9 点才起来时，8 点那档还没跑就补 ✓
   但**只补当天的** ✗（昨天的 20 点档不补 ✓ 那没有意义）
3. **正在跑的时候不叠加** ✗ —— 一期没跑完，下一档即使到了也不开第二期 ✓
4. **档位窗口**：只在该小时的第 0 分钟之后触发 ✓（不是"整三小时内任意时刻" ✓
   否则 20:59 起来也会开一期，而那一期实际上要跑到 21:33 ✗）
   -> 用 ``GRACE_MIN`` 定义「错过多久还算补跑」✓
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: 官方档位（本地时间的小时）
SLOTS: tuple[int, ...] = (8, 11, 14, 17, 20)

#: 每期跑多少局
MATCHES_PER_SLOT = 4000

#: 错过多久之内还算「补跑」（分钟）。超过就不再补这一档。
#: 4000 局约 34 分钟，所以留 90 分钟足够；再晚了那一期会跨到下一个档位。
GRACE_MIN = 90


def slot_of(when: float) -> int | None:
    """``when`` 落在哪个档位的小时上；不在档位返回 ``None``。"""
    h = time.localtime(when).tm_hour
    return h if h in SLOTS else None


def season_id_for(when: float, slot: int) -> str:
    """该档位的期号：``YYYY-MM-DDTHH``（用**档位小时** ✓ 不是 now 的小时 ✗）。"""
    t = time.localtime(when)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{slot:02d}"


def due_slot(when: float, existing: set[str], *, busy: bool = False,
             grace_min: int = GRACE_MIN,
             slots: tuple[int, ...] = SLOTS) -> tuple[int, str] | None:
    """现在该跑哪一档？返回 ``(slot, season_id)`` 或 ``None``。

    ``existing``：**已经存在**的期号集合（从磁盘读 ✓ 服务重启也不会重跑 ✗）
    ``busy``：当前有期在跑 -> 一律返回 ``None`` ✓（绝不叠加 ✗）

    规则（按顺序）：
      1. ``busy`` -> 不跑 ✓
      2. 今天**每一个**已到的档位（``现在 - 档位起点 <= grace_min``）逐个检查：
         没有对应期号 -> 就是它 ✓
      3. 全都跑过 / 都过期了 -> 不跑 ✓
    """
    if busy:
        return None
    t = time.localtime(when)
    day_start = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    for slot in sorted(slots):
        slot_at = day_start + slot * 3600
        delta_min = (when - slot_at) / 60.0
        if delta_min < 0:
            continue          # 这一档还没到 ✓
        if delta_min > grace_min:
            continue          # 错过太久，不补了 ✓
        sid = season_id_for(when, slot)
        if sid in existing:
            continue          # 已经跑过 ✓
        return slot, sid
    return None


def next_slot_at(when: float, *, slots: tuple[int, ...] = SLOTS) -> float:
    """下一次该跑的时刻（给页面显示「下次 X 点」用）。"""
    t = time.localtime(when)
    day_start = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    for slot in sorted(slots):
        at = day_start + slot * 3600
        if at > when:
            return at
    # 今天全过了 -> 明天的第一档
    return day_start + 86400 + min(slots) * 3600


@dataclass
class SlotInfo:
    """给页面用的一行。"""

    slot: int
    season_id: str
    state: str        #: "done" / "due" / "upcoming" / "missed"
    at: float


def today_slots(when: float, existing: set[str], *,
                slots: tuple[int, ...] = SLOTS,
                grace_min: int = GRACE_MIN) -> list[SlotInfo]:
    """今天五个档位的状态（官方页面的时间轴）。"""
    t = time.localtime(when)
    day_start = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    out: list[SlotInfo] = []
    for slot in sorted(slots):
        at = day_start + slot * 3600
        sid = season_id_for(when, slot)
        if sid in existing:
            state = "done"
        elif when < at:
            state = "upcoming"
        elif (when - at) / 60.0 <= grace_min:
            state = "due"
        else:
            state = "missed"
        out.append(SlotInfo(slot=slot, season_id=sid, state=state, at=at))
    return out
