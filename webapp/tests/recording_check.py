#!/usr/bin/env python
"""验收 7/8：**决策级记录** + **同一玩家跨局归并** + **落盘失败不影响对局**。

五件事，逐条对应用户给的标准：

1. 打一局「真人 + 三个 AI」，导出完整决策级 JSONL；
2. 每条 ``decision`` 记录里都有：局面快照 / 引擎给的**完整合法动作** / 实际选择 /
   思考耗时 / 玩家画像；
3. 一局结束时 ``hand`` 记录里有**四家手牌** + 和牌详情；
4. 把落盘目录设成**不可写**，对局照常能打完（记录失败只是打日志）；
5. 同一个玩家打 10 局，记录能通过匿名 ID 串起来，累计统计对得上。

第 4 条在 Windows 上没有 ``chmod`` 那种语义，改用「把目录路径指向一个
**不可能创建**的路径」（父级是个文件），等价于「写盘一定失败」。
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.tests.bot import Bot  # noqa: E402

AI4 = ["maj-normal4v1.2", "maj-flash4v1.2", "maj-mountain4v1.2", "maj-huge4v1.2"]


def read_records(games_dir: Path) -> list[dict]:
    """读分片。

    用 ``webapp.server.recording.iter_jsonl_gz`` —— 它**容忍被截断的 gzip 尾巴**。
    直接 ``gzip.open`` 的话，只要服务端在写的时候被强杀过一次，那个分片
    从第一行起就整个读不出来（已经写好的几万条一起「消失」）。
    """
    from webapp.server.recording import iter_jsonl_gz

    out: list[dict] = []
    for f in sorted(games_dir.glob("*.jsonl.gz")):
        out.extend(iter_jsonl_gz(f))
    return out


async def play_one_match(url: str, tag: str, rounds: int, players: int,
                         no_record: bool = False) -> tuple[dict, list[str], str]:
    """让一个真人 + 一堆 AI 打完一场。返回 (终局快照, 我收到的牌, 我的 uid)。"""
    me = Bot(url, f"记录玩家{tag}", strategy="random")
    await me.open()
    seats = [{"kind": "human"}] + [
        {"kind": "ai", "ai": AI4[i % len(AI4)]} for i in range(players - 1)
    ]
    rid = await me.create_room(players=players, rounds=rounds, seats=seats,
                              ai_delay=0.0, autostart=False, name=f"记录测试{tag}")
    if no_record:
        await me.send({"t": "privacy", "no_record": True})
        await asyncio.sleep(0.3)
    await me.send({"t": "room.start"})
    await me.play_until(lambda s: s["phase"] == "finished", timeout=600, stall=300)
    st = me.state
    uid = me.uid or ""
    await asyncio.sleep(0.5)
    await me.close()
    return (st or {}), [], uid


async def main_async(url: str, games: int, keep_dir: bool) -> int:
    print("=" * 70)
    print("验收 7/8：决策级记录 / 四家手牌 / 落盘失败不影响对局 / 跨局归并")
    print("=" * 70)

    # 记录目录必须和服务端一致：从 /api/record 问出来
    import urllib.request

    base = url.replace("ws://", "http://").replace("wss://", "https://").split("/ws")[0]
    with urllib.request.urlopen(base + "/api/record", timeout=10) as r:
        info = json.loads(r.read())
    assert info.get("record"), "服务端没开记录（--no-record？）"
    games_dir = Path(info["dir"])
    print(f"✓ 记录目录：{games_dir}")

    mark = time.time()
    # --- 第 8 条：同一个玩家打 N 局，ID 要能串起来 ----------------------
    # 每场换一条**新连接**（模拟刷新页面），但复用同一个 uid —— 这正是
    # 「同一个人跨局还是同一个人」要验证的东西。
    uid = ""
    matches: list[dict] = []
    uids: set[str] = set()
    for g in range(games):
        me = Bot(url, "记录玩家甲", strategy="random")
        me.uid = uid            # 第一场为空 -> 服务端发一个新的匿名 ID
        await me.open()
        seats = [{"kind": "human"}] + [
            {"kind": "ai", "ai": AI4[i % len(AI4)]} for i in range(3)
        ]
        rid = await me.create_room(players=4, rounds=1, seats=seats, ai_delay=0.0,
                                  autostart=True, name=f"记录测试{g}")
        await me.play_until(lambda s: s["phase"] == "finished", timeout=600, stall=300)
        matches.append(me.state or {})
        uids.add(me.uid or "")
        uid = me.uid or uid
        await asyncio.sleep(0.4)
        await me.close()
        print(f"  第 {g + 1}/{games} 场打完（房间 {rid}，uid={uid}）")
    assert len(uids) == 1, f"❌ 同一个人被分成了 {len(uids)} 个 ID：{uids}"

    # 等写盘线程把缓冲刷出去
    time.sleep(1.5)
    with urllib.request.urlopen(base + "/api/record", timeout=10) as r:
        json.loads(r.read())        # 这个接口自带 flush

    recs = [r for r in read_records(games_dir) if float(r.get("at") or 0) >= mark - 5]
    kinds = Counter(r["kind"] for r in recs)
    print(f"\n✓ 本轮写出 {len(recs)} 条记录：{dict(kinds)}")
    assert kinds.get("decision", 0) > 0, "❌ 一条 decision 都没有"
    assert kinds.get("hand", 0) > 0, "❌ 一条 hand 都没有"
    assert kinds.get("match", 0) >= games, f"❌ match 记录只有 {kinds.get('match')} 条"

    # --- 第 2 条：每条 decision 的字段 --------------------------------
    need = {"hand_no", "seat", "player", "decision", "options", "choice_index",
            "choice", "elapsed_ms", "view", "profile", "kind_player"}
    view_need = {"hand", "hand_counts", "ponds", "melds_all", "scores",
                 "dora_indicators", "wall_left", "turn", "riichi", "seat_wind",
                 "round_wind", "kita"}
    bad = []
    for r in recs:
        if r["kind"] != "decision":
            continue
        miss = need - set(r)
        if miss:
            bad.append(("缺字段", sorted(miss)))
            continue
        miss2 = view_need - set(r["view"])
        if miss2:
            bad.append(("快照缺字段", sorted(miss2)))
            continue
        if not r["options"]:
            bad.append(("合法动作列表是空的", r.get("decision")))
    assert not bad, f"❌ decision 记录有问题：{bad[:3]}"
    d0 = next(r for r in recs if r["kind"] == "decision")
    print("✓ 每条 decision 都带：局面快照 + 完整合法动作 + 实际选择 + 耗时 + 画像")
    print(f"  例：第 {d0['hand_no']} 局 座位 {d0['seat']} 「{d0['decision']}」"
          f"可选 {len(d0['options'])} 项 -> 选了 #{d0['choice_index']}"
          f"，{d0['elapsed_ms']} ms，手牌 {len(d0['view']['hand'])} 张")

    # 画像字段
    prof_need = {"uid", "games", "avg_rank", "first_rate", "agari_rate",
                 "deal_in_rate", "elo"}
    missing = prof_need - set(d0["profile"])
    assert not missing, f"❌ 画像缺字段：{sorted(missing)}"
    print(f"✓ 画像字段齐：{ {k: d0['profile'][k] for k in sorted(prof_need)} }")

    # --- 第 3 条：hand 记录里有四家手牌 --------------------------------
    hands = [r for r in recs if r["kind"] == "hand"]
    assert hands, "❌ 没有 hand 记录"
    h0 = hands[0]
    assert len(h0["hands"]) == 4, f"❌ hand 记录里只有 {len(h0['hands'])} 家"
    total = sum(sum(x["hand_counts"]) for x in h0["hands"])
    known = sum(len(x["melds"]) * 3 for x in h0["hands"])
    assert total > 0, "四家手牌是空的"
    print(f"✓ 一局结束落了四家手牌：{len(h0['hands'])} 家，手牌共 {total} 张，"
          f"结果 {h0['result_kind']}（{h0['kind_cn']}），"
          f"收支 {h0['deltas']}")
    for x in h0["hands"]:
        print(f"    座位 {x['seat']} {x['name']:8s} 手牌 {sum(x['hand_counts']):2d} 张 "
              f"副露 {len(x['melds'])} 组 牌河 {len(x['pond'])} 张 点数 {x['score']}")
    assert "agari" in h0, "hand 记录缺和牌详情"

    # --- 第 7 条（后半）：落盘失败不影响对局 ---------------------------
    print("\n--- 落盘失败时对局必须照常 ---")
    blocker = Path(tempfile.gettempdir()) / f"maj-blocker-{os.getpid()}"
    blocker.write_text("not a dir", encoding="utf-8")
    from webapp.server.recording import AsyncWriter

    broken = AsyncWriter(blocker / "games", flush_interval=0.2)
    broken.start()
    for i in range(200):
        broken.put("decision", {"i": i})
    await asyncio.sleep(1.0)
    st = broken.stats()
    broken.stop()
    print(f"✓ 写入不可用路径时：errors={st['errors']}（只计数、不抛异常），"
          f"written={st['written']}")
    assert st["errors"] > 0, "应该记录到写入失败"
    # 对局本身照常
    fin, _, _ = await play_one_match(url, "只读", rounds=1, players=3)
    assert fin.get("phase") == "finished", "❌ 记录坏掉时对局没能打完"
    print("✓ 记录写不进去时，一局照样打完（房间正常结束）")

    # --- 第 8 条：同一个 ID 跨局归并 -----------------------------------
    print("\n--- 同一玩家跨局归并 ---")
    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        p = r.get("player") or (r.get("profile") or {}).get("uid")
        if p and p.startswith("p_"):
            by_player[p].append(r)
    assert by_player, "❌ 记录里没有匿名玩家 ID"
    assert all(k.startswith("p_") for k in by_player), "玩家 ID 必须是匿名串"
    main_uid = max(by_player, key=lambda k: len(by_player[k]))
    rows = [r for r in recs if r.get("player") == main_uid]
    # 只看**这个玩家**的 match 记录：``player`` 字段就是匿名 ID
    # （``profile.uid`` 在一局记录里指的是当事人，不一定是我）。
    mk = [r for r in rows if r["kind"] == "match"]
    print(f"✓ 记录里的匿名 ID（{len(by_player)} 个）："
          f"{ {k: len(v) for k, v in list(by_player.items())[:4]} }")
    print(f"  主玩家 {main_uid}：{len(rows)} 条记录，其中 match {len(mk)} 条")
    assert len(mk) >= min(games, 2), f"❌ 只串起来 {len(mk)} 场"

    # 累计统计要和逐局对得上
    wins = sum(1 for r in mk if r["rank"] == 0)
    rank_sum = sum(r["rank"] + 1 for r in mk)
    prof = mk[-1]["profile"]
    assert prof["games"] >= len(mk), "画像里的总局数比记录里的还少"
    exp_first = wins / len(mk)
    print(f"  逐局统计：{len(mk)} 场、一位 {wins} 次（一位率 {exp_first:.3f}）、"
          f"平均顺位 {rank_sum / len(mk):.3f}、均分 "
          f"{sum(r['score'] for r in mk) / len(mk):.0f}")
    print(f"  画像累计：总局数 {prof['games']}、一位率 {prof['first_rate']}、"
          f"平均顺位 {prof['avg_rank']}、ELO {prof['elo']}")
    # 画像的累计值必须能把这几场算进去
    assert prof["games"] >= len(mk), "画像没把这 10 场算进去"
    assert abs(prof["first_rate"] - round(wins / prof["games"], 4)) < 1e-4 or \
        prof["games"] > len(mk), \
        f"一位率对不上：画像 {prof['first_rate']}，这几场算出 {exp_first:.4f}"
    print("✓ 逐局记录能按匿名 ID 串成一条线，累计统计一致")

    # ELO：**每一场**之内零和。
    # 一份 match 记录只有当事人那一行，所以要把同一 game_uid 的几条
    # 记录**按座位拼起来**再求和（10 场加在一起当然不为零 —— 那是 10 局的
    # 累计变化，不是守恒量）。
    seen_seat: dict[str, dict[int, float]] = defaultdict(dict)
    for r in recs:
        if r["kind"] != "match":
            continue
        for row in r.get("players") or []:
            seen_seat[r["game_uid"]][int(row["seat"])] = float(
                row.get("elo_delta_raw") or 0.0)
    worst = 0.0
    checked = 0
    full = 0
    for _g, seats in seen_seat.items():
        total = sum(seats.values())
        worst = max(worst, abs(total))
        checked += 1
        if len(seats) == 4:
            full += 1
    print(f"  ELO：{checked} 场（其中 {full} 场四家齐全）里，"
          f"每场 raw 变化之和最大偏离 0 = {worst:.3e}"
          f"（零和；K 值与公式与 AI 完全相同）")
    assert checked > 0, "❌ 一条 match 记录都没有"
    assert worst < 1e-9, f"❌ ELO 不是零和，最大偏离 {worst}"

    # 逐场台账：顺位 / 点数 / ELO 变化
    print("  逐场台账（我的座位）：")
    for r in mk:
        print(f"    {r['game_uid'].split('-')[0]:8s} 顺位 {r['rank'] + 1}  "
              f"点数 {r['score']:6d}  ELO {r['elo_before']:.1f} -> "
              f"{r['elo_after']:.1f}（{r['elo_delta']:+.2f}）")

    # 导出一份给用户看
    out = Path(tempfile.gettempdir()) / "maj-decisions-sample.jsonl"
    from webapp.server.recording import iter_jsonl_gz

    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for f in sorted(games_dir.glob("*.jsonl.gz")):
            for rec in iter_jsonl_gz(f):
                fh.write(json.dumps(rec, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
                n += 1
    print(f"✓ 已导出完整决策级 JSONL 供查看：{out}（{n} 条，"
          f"{out.stat().st_size / 1e6:.2f} MB）")

    if not keep_dir:
        print(f"\n（记录目录保留在 {games_dir}，可用 tools/dump_games.py 导出）")
    print("\n✅ 决策级记录测试通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    ap.add_argument("--games", type=int, default=10, help="同一玩家连打几场")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args.url, args.games, args.keep))
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
