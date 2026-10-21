# -*- coding: utf-8 -*-
"""MJAI 牌名的**赤五两种写法**必须都认。

2026-10-21 实测的真事故：
    新转换器 `tools/tenhou_xml_to_mjai.py` 按**标准 MJAI** 把赤五写成 `5mr`，
    而 `aimodel/mjai.py:tile_from_mjai` 只认本项目的 `0m` ->
    抽取 5,717 局鳳凰数据时**在第一个含赤五的手牌上就退出** ✗

    ValueError: 认不出的牌名: '5mr'

所以这个测试锁住三件事：
    ① 标准写法 `5mr/5pr/5sr` 认得出，且认定是赤（red=True）
    ② 本项目写法 `0m/0p/0s` 照旧认得出
    ③ 【负向】真的认不出的牌名**仍然要抛**，不许因为「放宽」而变成静默接受
       也：【负向】`6mr` 这种非法赤后缀必须抛，不能把 6m 当赤五
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimodel import mjai as M  # noqa: E402

FAILS = []


def check(name, got, want):
    if got == want:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}: 得 {got!r} 期望 {want!r}")
        FAILS.append(name)


def raises(name, fn):
    try:
        got = fn()
    except ValueError:
        print(f"  ✅ {name}（确实抛了 ValueError）")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {name}: 抛了 {type(e).__name__} 而不是 ValueError")
        FAILS.append(name)
        return
    print(f"  ❌ {name}: 没抛，返回了 {got!r}")
    FAILS.append(name)


def main():
    print("▶ mjai 赤五两种写法")

    # 牌型下标：5m = 4, 5p = 13, 5s = 22（0-based，m/p/s 各 9 张）
    print("-- ① 标准写法 5mr/5pr/5sr --")
    for name, want in (("5mr", (4, True)), ("5pr", (13, True)), ("5sr", (22, True))):
        check(f"{name} -> {want}", M.tile_from_mjai(name), want)

    print("-- ② 本项目写法 0m/0p/0s 不能退化 --")
    for name, want in (("0m", (4, True)), ("0p", (13, True)), ("0s", (22, True))):
        check(f"{name} -> {want}", M.tile_from_mjai(name), want)

    print("-- ③ 普通牌不受影响 --")
    check("5m 不是赤", M.tile_from_mjai("5m"), (4, False))
    check("1m", M.tile_from_mjai("1m"), (0, False))
    check("9s", M.tile_from_mjai("9s"), (26, False))
    check("白 P", M.tile_from_mjai("P"), (M.S.HAKU, False))

    print("-- ④ 与 tile_to_mjai 往返 --")
    check("tile_to_mjai(4, red=True) == '0m'", M.tile_to_mjai(4, red=True), "0m")
    check("tile_to_mjai(4) == '5m'", M.tile_to_mjai(4), "5m")
    # 标准写法读进来，本项目写法写出去，牌型下标必须一致
    t_in, red_in = M.tile_from_mjai("5mr")
    t_out = M.tile_from_mjai(M.tile_to_mjai(t_in, red_in))[0]
    check("5mr -> 0m -> 同一个下标", t_out, 4)

    print("-- ⑤ 【负向】非法赤后缀必须抛，不许把 6m 当赤五 --")
    raises("6mr 抛错", lambda: M.tile_from_mjai("6mr"))
    raises("1mr 抛错", lambda: M.tile_from_mjai("1mr"))

    print("-- ⑥ 【负向】真认不出的仍然要抛（防「放宽」变成静默接受）--")
    raises("'X' 抛错", lambda: M.tile_from_mjai("X"))
    raises("'10m' 抛错", lambda: M.tile_from_mjai("10m"))
    raises("'m' 抛错", lambda: M.tile_from_mjai("m"))
    raises("'' 抛错", lambda: M.tile_from_mjai(""))
    raises("'5x' 抛错", lambda: M.tile_from_mjai("5x"))

    if FAILS:
        print(f"❌ mjai 赤五：{len(FAILS)} 项失败 -> {FAILS}")
        return 1
    print("✅ mjai 赤五两种写法全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
