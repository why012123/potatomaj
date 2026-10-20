"""ELO 混战池的网页界面：一个后台线程一直打，页面实时看天梯 + 图表。

跑法::

    python -m aimodel.ui.arena_server              # 默认 http://127.0.0.1:8770
    python -m aimodel.ui.arena_server --players 4 --port 8771

页面功能：按 ELO 排序的天梯、三条实时柱状图（ELO / 一位率 / 和了率）、
分数曲线、场次数量可调、开始/暂停/重置、K 值与分差上限可调。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..arena import Arena, STATE_FILE

#: 收藏功能的**口令**所在的环境变量名 ✓
#: 没设这个变量 -> 收藏功能关闭（返回 403 并说明原因 ✓ 不静默失败 ✗）
ADMIN_KEY_ENV = "MAJ_ELO_ADMIN_KEY"

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


def log(msg: str) -> None:
    """服务端自己的日志。**写 stderr** ✓ 不污染 ``/api/arena`` 的输出 ✓。

    存档失败之类的问题要看得见，但绝不能拖垮跑局 —— 所以调用处都包了 try。
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Session:
    """一个池子 + 一个一直在跑的后台线程。"""

    def __init__(self, players: int = 3, k: float = 24.0, max_gap: float = 200.0,
                 strict: bool = True, pace: float = 0.0, copies: int = 3,
                 divisor: float = 500.0,
                 models: list[str] | None = None) -> None:
        #: 只让这几个模型参赛（None = 全部）✓ 传给 Arena -> collect(include=...)
        self.models = models
        self.copies = copies
        self.divisor = divisor
        self.arena = Arena(players=players, k=k, max_gap=max_gap, strict=strict,
                           copies=copies, divisor=divisor, models=models)
        self.lock = threading.Lock()
        self.running = False
        self.target = 0          # 还要跑多少场（0 = 无限）
        self.done = 0
        self.pace = pace         # 每场之间停多少秒（0 = 全速）
        self.error = ""
        self.thread: threading.Thread | None = None
        self.t0 = time.time()
        #: **当期存档**（见 :mod:`aimodel.elo_store`）。每开跑一次就开一期 ✓
        #: 只记 `{seed, players, group, ranks, scores}` ≈ 233 B/局 ✓
        #: （arena 每局完全确定 ✓ 实测同 seed 两次跑动作逐个相同 ✓ 所以回放=重跑引擎 ✓）
        self.season = None
        #: 记录开关（关掉就不落盘；默认开 ✓ 233 B/局 不心疼 ✓）
        self.record = True

    # -- 控制 ---------------------------------------------------------------
    def start(self, matches: int = 100, pace: float | None = None,
              sid: str | None = None) -> None:
        """开跑。``sid`` 给了就用它当**期号**（官方档位要指定 ✓ 手动池让它自己生成 ✓）。"""
        with self.lock:
            self.target = matches
            self.done = 0
            if pace is not None:
                self.pace = pace
            self.error = ""
            if self.running:
                return
            # 开一期存档：手动池的期号带场次和一次性后缀，免得两次跑撞一起
            if self.record:
                try:
                    from aimodel import elo_store as E

                    if sid is None:
                        sid = (f"{E.season_id()}-{self.arena.players}p"
                               f"-{int(time.time()) % 100000}")
                    # ⚠️ **每期换一个牌山种子** ✗ —— 否则每期都打**同一副牌**
                    #    （用户 2026-10-20 指出：seed_base 恒为 0 ✗）
                    #    从**期号**派生：同一期恒定 ✓（补跑/续跑可复现）不同期不同 ✓
                    prev = E.load_season(sid)
                    if prev is not None and prev.seed_base:
                        seed = prev.seed_base      # 补跑/续跑：沿用该期种子
                    else:
                        seed = E.seed_from_sid(sid)
                    self.arena.seed = seed
                    # ⚠️ **续跑要把场次序号接上** ✗ 否则 seed_of(index) 会重复发出
                    #    同一批种子 -> 补跑出来的前几局和已有的**一模一样** ✗（重复数据）
                    if prev is not None:
                        try:
                            have = sum(1 for _ in E.iter_matches(prev))
                        except OSError:
                            have = 0
                        if have:
                            self.arena.n_matches = have
                            log(f"↩️ 续跑 {sid}：已有 {have} 局，场次序号从 {have} 接着数")
                    self.season = E.open_season(
                        sid, players=self.arena.players, copies=self.copies,
                        seed_base=seed, target=matches)
                    # ⚠️ **把该期的分数读回来** ✗ —— 否则每次重启 arena 的分数都归零 ✓
                    #    -> 期未跑完就重启 => 最终 ELO 表只反映**最后一次跑**的那段 ✗
                    #       （2026-10-20 实测：同一期重启三次，冠军变了三次 ✓ 那是假的 ✓）
                    st_path = self.season.dir / "arena.json"
                    if st_path.exists():
                        try:
                            loaded = Arena.load(st_path)
                            if loaded is not None and loaded.players == self.arena.players \
                                    and len(loaded.participants) == len(self.arena.participants):
                                self.arena = loaded
                                self.arena.seed = seed
                                log(f"↩️ 读回 {sid} 的分数状态："
                                    f"{self.arena.n_matches} 场、"
                                    f"{len(self.arena.participants)} 个参战者")
                            else:
                                log("⚠️ 存档的 arena 与当前阵容不符，忽略它（从零开始）")
                        except Exception as exc:  # noqa: BLE001
                            log(f"⚠️ 读回分数失败（从零开始）：{type(exc).__name__}: {exc}")
                except Exception as exc:  # noqa: BLE001 - 存档失败不能拖垮跑局
                    log(f"⚠️ 开期失败（不影响跑局）：{type(exc).__name__}: {exc}")
                    self.season = None
            self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False

    def reset(self) -> None:
        self.pause()
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        with self.lock:
            self.arena = Arena(players=self.arena.players, k=self.arena.k,
                               max_gap=self.arena.max_gap, strict=self.arena.strict,
                               copies=self.copies, divisor=self.divisor,
                               models=self.models)   # ← 重置也要保住阵容 ✗
            self.done = 0
            self.error = ""

    def _run(self) -> None:
        # ⚠️ 所有出口都**先跳出循环、再调 `_finish()`** ✗
        #    因为 `_finish()` 自己要拿 `self.lock` ✓ 而 `threading.Lock` 不可重入 ✗
        #    -> 在锁里调它就**死锁** ✓（第一版就写成了这样 ✓ 已改 ✓）
        while True:
            with self.lock:
                if not self.running:
                    break
                if self.target and self.done >= self.target:
                    self.running = False
                    break
                pace = self.pace
            try:
                res = self.arena.run_match()
            except Exception as exc:                    # noqa: BLE001
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.running = False
                break
            with self.lock:
                if res is None:
                    self.error = (f"池子里已经没有分差 ≤ {self.arena.max_gap:.0f} 的组合了"
                                  f"（存活 {len(self.arena.pool())} 个）")
                    self.running = False
                    break
                self.done += 1
                season = self.season
            # 逐局存档：只记 {seed, players, group, ranks, scores} ✓ ≈233 B ✓
            # 放在锁外面：写盘不该卡住 /api/arena 的读取 ✓
            if season is not None:
                try:
                    from aimodel import elo_store as E

                    E.append_match(season, res)
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self.error = f"存档失败：{type(exc).__name__}: {exc}"
            # 周期性存盘：崩 / 重启最多丢 200 局 ✓ 否则一期跑一半被重启就前功尽弃 ✗
            if season is not None and self.done and self.done % 200 == 0:
                try:
                    self.arena.save(season.dir / "arena.json")
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self.error = f"周期存盘失败：{type(exc).__name__}: {exc}"
            if pace > 0:
                time.sleep(pace)
        self._finish()

    def _finish(self) -> None:
        """一轮跑完：存该期的最终 ELO 表（「历史数据」不受 3 天限制 ✓）。"""
        with self.lock:
            season = self.season
            arena = self.arena
        if season is None:
            return
        try:
            from aimodel import elo_store as E

            season.finished_at = time.time()
            season.done = arena.n_matches
            table = arena.table()
            if table:
                # ⚠️ 冠军按**模型均分** ✓ 不是单个实例 ✗（用户 2026-10-20 要求）
                #    每个模型有 copies 份参赛者（默认 3），单个第一名噪声大 ✗
                #    两个都记：champion = 均分第一的模型 ✓ champion_inst = 单实例第一 ✓
                (season.champion, season.champion_avg,
                 season.champion_inst, season.champion_inst_score) = \
                    E.champion_by_model(table)
                season.champion_key = season.champion
            E.save_season(season)
            E.save_scores(season.sid, table, {"players": arena.players,
                                              "matches": arena.n_matches})
            try:
                arena.save(season.dir / "arena.json")   # 分数状态跟着期一起存 ✓
            except Exception as exc:  # noqa: BLE001
                log(f"⚠️ 存分数状态失败：{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            log(f"⚠️ 收尾存表失败：{type(exc).__name__}: {exc}")

    # -- 快照 ---------------------------------------------------------------
    def snapshot(self, top: int = 40) -> dict:
        with self.lock:
            a = self.arena
            rows = a.table()[:top]
            return {
                "players": a.players,
                "k": a.k,
                "max_gap": a.max_gap,
                "strict": a.strict,
                "running": self.running,
                "target": self.target,
                "done_total": a.n_matches,
                "done_run": self.done,
                "pace": self.pace,
                "seed": a.seed,
                "divisor": a.divisor,
                "error": self.error,
                "alive": sum(1 for p in a.participants if not p.at_floor),
                "total": len(a.participants),
                "elapsed": round(time.time() - self.t0, 1),
                "table": rows,
                "summary": self._summary(a),
                "log": a.log[-14:],
                "copies": self.copies,
            }

    @staticmethod
    def _summary(a: Arena) -> list[dict]:
        """按模型汇总：同一个模型的几个参赛者合起来看（均分 / 最好 / 最差）。"""
        groups: dict[str, list] = {}
        for p in a.participants:
            groups.setdefault(p.source or p.key, []).append(p)
        out = []
        for src, ps in groups.items():
            rated = [x for x in ps if x.games]
            if not rated:
                continue
            elos = [x.rating for x in ps]
            hands = sum(x.hands for x in ps)
            out.append({
                "模型": src,
                "份数": len(ps),
                "均分": round(sum(elos) / len(elos), 1),
                "最好": round(max(elos), 1),
                "最差": round(min(elos), 1),
                "极差": round(max(elos) - min(elos), 1),
                "场数": sum(x.games for x in ps),
                "一位率": round(sum(x.wins for x in ps) / max(1, sum(x.games for x in ps)), 4),
                "和了率": round(sum(x.agari for x in ps) / max(1, hands), 4),
                "放铳率": round(sum(x.deal_in for x in ps) / max(1, hands), 4),
                "版本": ps[0].version, "风格": ps[0].style,
                "保底": sum(1 for x in ps if x.at_floor and x.games),
            })
        out.sort(key=lambda r: -r["均分"])
        return out

    def settings(self, payload: dict) -> None:
        with self.lock:
            a = self.arena
            if "k" in payload:
                a.k = float(payload["k"])
            if "max_gap" in payload:
                a.max_gap = float(payload["max_gap"])
            if "strict" in payload:
                a.strict = bool(payload["strict"])
            if "pace" in payload:
                self.pace = float(payload["pace"])
            if "seed" in payload:
                a.seed = int(payload["seed"])
            if "divisor" in payload:
                a.divisor = float(payload["divisor"])


class Handler(BaseHTTPRequestHandler):
    session: Session
    quiet = True

    def log_message(self, fmt: str, *args) -> None:      # noqa: A003
        if not self.quiet:
            super().log_message(fmt, *args)

    def _json(self, data, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _page(self, name: str) -> None:
        """发一个静态页。缺文件就 404 ✓ 不抛异常 ✓。"""
        page = STATIC / name
        if not page.exists():
            self.send_error(404, f"缺少 {name}")
            return
        body = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                            # noqa: N802
        u = urlparse(self.path)
        # 门户 / 手动池 / 官方 / 回放 —— 四个页面各一个路径 ✓
        if u.path in ("/", "/index.html"):
            self._page("portal.html")
            return
        if u.path in ("/arena", "/arena.html"):
            self._page("arena.html")
            return
        if u.path in ("/official", "/official.html"):
            self._page("official.html")
            return
        if u.path in ("/replay", "/replay.html"):
            self._page("replay.html")
            return
        if u.path == "/api/arena":
            self._json(self.session.snapshot())
            return
        if u.path == "/api/elo/schedule":
            from aimodel import elo_store as E
            from aimodel.ui import elo_schedule as S

            existing = {s.sid for s in E.list_seasons()}
            now = time.time()
            self._json({
                "ok": True,
                "slots": [{"slot": r.slot, "season_id": r.season_id,
                           "state": r.state,
                           "at": time.strftime("%m-%d %H:%M", time.localtime(r.at))}
                          for r in S.today_slots(now, existing)],
                "matches_per_slot": S.MATCHES_PER_SLOT,
                "grace_min": S.GRACE_MIN,
                "next": time.strftime("%m-%d %H:%M", time.localtime(S.next_slot_at(now))),
                "running": self.session.running,
                "current_season": (self.session.season.sid
                                   if self.session.season else ""),
            })
            return
        if u.path == "/api/arena/player":
            q = parse_qs(u.query)
            key = (q.get("key") or [""])[0]
            with self.session.lock:
                d = self.session.arena.player_detail(key)
            if d is None:
                self._json({"ok": False, "error": f"没有这个参赛者: {key}"}, 404)
                return
            self._json(d)
            return
        # ---- ELO 期次 / 回放（见 aimodel/elo_store.py）----------------------
        if u.path == "/api/elo/seasons":
            from aimodel import elo_store as E

            se = E.list_seasons()
            rows = []
            for s in se:
                d = s.as_dict()
                # ⚠️ meta 里的 done 只在**收尾时**才写 ✗ -> 跑到一半时它是陈旧的 ✗
                #    权威口径是 matches.jsonl 的**实际行数** ✓（调度器也用这个 ✓）
                try:
                    real = sum(1 for _ in E.iter_matches(s))
                except OSError:
                    real = 0
                d["done"] = real
                d["has_replay"] = (s.dir / "matches.jsonl").exists() and not s.pruned
                # ⚠️ 有 ELO 表却没有冠军 —— 会真实发生 ✗ 两种情况：
                #    ① 那期跑在「冠军按均分算」这个逻辑之前（2026-10-20 之前）✗
                #    ② `_finish()` 只在**跑完**时才写 -> 中途重启就没有冠军 ✗
                #    -> 有表就**现算**一个 ✓ 保证列表和详情都不出现「有分无冠」
                if (not d.get("champion")) or ("#" in str(d.get("champion"))) \
                        or not d.get("champion_avg"):
                    sc = E.load_scores(s.sid)
                    if sc and sc.get("table"):
                        (d["champion"], d["champion_avg"],
                         d["champion_inst"], d["champion_inst_score"]) = \
                            E.champion_by_model(sc["table"])
                rows.append(d)
            self._json({"ok": True, "seasons": rows, "stats": E.stats()})
            return
        if u.path == "/api/elo/season":
            from aimodel import elo_store as E

            q = parse_qs(u.query)
            sid = (q.get("id") or [""])[0]
            s = E.load_season(sid)
            if s is None:
                self._json({"ok": False, "error": f"没有这一期: {sid}"}, 404)
                return
            # ⚠️ meta 里的 done 只在收尾时才写 ✗ -> 跑到一半是陈旧的（会是 0）✗
            #    与 /api/elo/seasons 用同一口径：**实际行数** ✓
            #    （这个不一致是收证据时抓出来的：列表页对了、详情页还是 0 ✗）
            q2 = parse_qs(u.query)
            try:
                offset = max(0, int((q2.get("offset") or ["0"])[0]))
            except ValueError:
                offset = 0
            try:
                limit = min(2000, max(1, int((q2.get("limit") or ["200"])[0])))
            except ValueError:
                limit = 200
            matches = []
            real = 0
            for i, m in enumerate(E.iter_matches(s)):
                real += 1
                if i < offset or i >= offset + limit:
                    continue          # 分页 ✓ 一期 4000 局也能逐段翻
                matches.append({"index": i, "seed": m.get("seed"),
                                "group": m.get("group"), "ranks": m.get("ranks"),
                                "scores": m.get("scores")})
            sd = s.as_dict()
            sd["done"] = real
            if (not sd.get("champion")) or ("#" in str(sd.get("champion"))) \
                    or not sd.get("champion_avg"):
                sc0 = E.load_scores(sid)
                if sc0 and sc0.get("table"):
                    (sd["champion"], sd["champion_avg"],
                     sd["champion_inst"], sd["champion_inst_score"]) = \
                        E.champion_by_model(sc0["table"])
            self._json({"ok": True, "season": sd, "matches": matches,
                        "offset": offset, "limit": limit, "total": real,
                        "scores": E.load_scores(sid)})
            return
        if u.path == "/api/elo/replay":
            from aimodel import elo_store as E

            q = parse_qs(u.query)
            sid = (q.get("season") or [""])[0]
            try:
                idx = int((q.get("index") or ["0"])[0])
            except ValueError:
                self._json({"ok": False, "error": "index 不是整数"}, 400)
                return
            s = E.load_season(sid)
            if s is None:
                self._json({"ok": False, "error": f"没有这一期: {sid}"}, 404)
                return
            m = E.get_match(s, idx)
            if m is None:
                self._json({"ok": False, "error": f"这一期没有第 {idx} 局"}, 404)
                return
            # **用同一个引擎重跑** ✓ 一局约 0.5 秒 ✓
            try:
                actions, res = E.replay(m)
            except Exception as exc:                     # noqa: BLE001
                self._json({"ok": False,
                            "error": f"回放失败：{type(exc).__name__}: {exc}"}, 500)
                return
            self._json({
                "ok": True, "season": sid, "index": idx, "seed": m.get("seed"),
                "players": m.get("players", 4), "group": m.get("group"),
                # 存档里的原始结果 + 刚重跑出来的结果（应当一致 ✓ 不一致就是复现性坏了 ✗）
                "archived": {"ranks": m.get("ranks"), "scores": m.get("scores")},
                "replayed": {"ranks": list(res.ranks), "scores": list(res.scores)},
                "match": (list(res.ranks) == list(m.get("ranks") or [])),
                "actions": actions,
            })
            return
        self.send_error(404)

    def do_POST(self) -> None:                           # noqa: N802
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except json.JSONDecodeError:
            self._json({"ok": False, "error": "JSON 解析失败"}, 400)
            return
        if u.path == "/api/arena/start":
            self.session.settings(payload)
            self.session.start(int(payload.get("matches", 100)),
                               float(payload["pace"]) if "pace" in payload else None)
            self._json({"ok": True})
            return
        if u.path == "/api/arena/pause":
            self.session.pause()
            self._json({"ok": True})
            return
        if u.path == "/api/arena/reset":
            self.session.reset()
            self._json({"ok": True})
            return
        if u.path == "/api/arena/settings":
            self.session.settings(payload)
            self._json({"ok": True, "settings": self.session.snapshot()})
            return
        if u.path == "/api/elo/favorite":
            self._favorite(payload)
            return
        self.send_error(404)

    def _favorite(self, payload: dict) -> None:
        """收藏/取消收藏某一期（**只有管理员能改** ✓ 口令见 :data:ADMIN_KEY_ENV）。

        ⚠️ 为什么要有收藏：3 天前未收藏的期会被 `elo_store.prune`
           **精简成只留冠军名** ✗ 收藏就是「这一期我要永久留着」的标记 ✓
        """
        import os

        want_key = os.environ.get(ADMIN_KEY_ENV, "")
        if not want_key:
            self._json({"ok": False,
                        "error": f"服务端没设 {ADMIN_KEY_ENV}，收藏功能关闭"
                                 f"（重启时带上这个环境变量即可开启）"}, 403)
            return
        if str(payload.get("key") or "") != want_key:
            self._json({"ok": False, "error": "口令不对"}, 403)
            return
        sid = str(payload.get("sid") or "").strip()
        if not sid:
            self._json({"ok": False, "error": "缺 sid"}, 400)
            return
        from aimodel import elo_store as E

        s = E.load_season(sid)
        if s is None:
            self._json({"ok": False, "error": f"没有这一期: {sid}"}, 404)
            return
        s.favorite = bool(payload.get("favorite", True))
        E.save_season(s)
        self._json({"ok": True, "sid": sid, "favorite": s.favorite})


def _startup_audit() -> None:
    """启动自检：权重加载不了要**在启动时**就喊出来，别等玩家点了才发现。"""
    try:
        from ..variants import audit_weights

        r = audit_weights(verbose=True)
        if r["failed"]:
            print("⚠️ 有模型加载失败 —— 这些模型在界面上会点不动，先修权重再开服务")
    except Exception as e:                          # noqa: BLE001
        print(f"（权重自检跳过：{type(e).__name__}: {e}）")


def start_official(session: "Session", *, interval: float = 45.0,
                   matches: int | None = None, prune_every: float = 3600.0) -> threading.Thread:
    """官方档位调度：后台线程，每 ``interval`` 秒检查一次该不该开一期。

    **判重靠落盘的期号** ✓（不是内存集合 ✗ —— 服务重启后集合就没了 ✗）
    **不叠加** ✓（一期没跑完，下一档到了也不开第二期 ✓）
    **错过太久不补** ✓（见 :data:`aimodel.ui.elo_schedule.GRACE_MIN`）
    **顺带每小时清理一次** ✓（3 天前未收藏的期 -> 只留冠军名 ✓）

    ⚠️ 这是**用户明确要的功能** ✓ 不是「自动重启/看门狗」✗
       （后者是被禁止的 ✗ 见 Mac 那条「不留任何定时器」的纪律 ✓）
    """
    from aimodel import elo_store as E
    from aimodel.ui import elo_schedule as S

    target = matches or S.MATCHES_PER_SLOT

    def loop() -> None:
        last_prune = 0.0
        while True:
            time.sleep(interval)
            try:
                if session.running:
                    continue
                # ⚠️ 判重只看**跑满**的期（done >= target）✗ 不能只看"期号存在" ✗
                #    否则：某一期跑到一半被重启打断 -> 期号存在 -> 调度器认为已完成
                #    -> **那一期永远补不上**（用户 2026-10-20 的 8 点档就是这样丢的）
                existing = set()
                for s in E.list_seasons():
                    want = s.target or target
                    if s.pruned or (s.done and s.done >= want):
                        existing.add(s.sid)
                due = S.due_slot(time.time(), existing, busy=False)
                if due is not None:
                    slot, sid = due
                    prev = E.load_season(sid)
                    have = 0
                    if prev is not None:
                        try:
                            have = sum(1 for _ in E.iter_matches(prev))
                        except OSError:
                            have = 0
                    todo = max(1, target - have)
                    log(f"🕗 官方档位 {slot:02d}:00 -> 开一期 {sid}"
                        f"（已有 {have} 局，本次再跑 {todo} 局）")
                    session.start(todo, sid=sid)
                if time.time() - last_prune > prune_every:
                    last_prune = time.time()
                    gone = E.prune()
                    if gone:
                        log(f"🧹 清理了 {len(gone)} 期：{', '.join(gone)}")
            except Exception as exc:                 # noqa: BLE001
                log(f"⚠️ 调度器异常（继续跑）：{type(exc).__name__}: {exc}")

    th = threading.Thread(target=loop, daemon=True, name="elo-official")
    th.start()
    return th


def serve(players: int = 3, port: int = 8770, host: str = "127.0.0.1",
          copies: int = 3, divisor: float = 500.0,
          models: list[str] | None = None,
          loose: bool = False, official: bool = False) -> None:
    """``loose=True`` → ``Arena(strict=False)``：分差不够就放宽 ✓

    为什么该开：`strict` 下只让分差 ≤ ``max_gap`` 的配对 ✓
    评分散开之后**两端都找不到对手** ✗ → 他们的分数**不再更新** ✓
    → 早先一波运气好的被**永久冻在第一** ✓
    （实测：同一批模型跑两轮，排名完全洗牌 ✗
      第 1 名 264 场、第 15 名 162 场，而中间的第 8 名有 567 场 ✓）
    """
    _startup_audit()
    session = Session(players=players, copies=copies, divisor=divisor,
                      models=models, strict=not loose)
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"🏟  ELO 混战池：http://{host}:{port}")
    print(f"     {players} 麻 · {len(session.arena.participants)} 个参战者"
          f"（每个模型 {copies} 个，各记各的分）· 初始 500 分 · 分差上限 200 · 0 分保底不出局")
    if official:
        from aimodel.ui import elo_schedule as S
        start_official(session)
        nxt = time.strftime("%m-%d %H:%M", time.localtime(S.next_slot_at(time.time())))
        print(f"     🕗 官方调度已启动：每天 {"/".join(f"{h:02d}" for h in S.SLOTS)} 点各一期"
              f"（每期 {S.MATCHES_PER_SLOT} 局）· 下次 {nxt}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收工")
        session.pause()
        session.arena.save()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ELO 混战池网页版")
    ap.add_argument("--players", type=int, default=3, choices=[3, 4])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--copies", type=int, default=3, help="每个模型派几个参赛者")
    ap.add_argument("--loose", action="store_true",
                    help="分差不够就放宽（修掉「两端静默冻结」✓ 推荐开）")
    ap.add_argument("--models", nargs="*", default=None,
                    help="只让这几个模型参赛（默认全部）✓ 要写带版本号的正名，"
                         "例：--models maj-normal4v1.2 maj-pro4v1.1")
    ap.add_argument("--divisor", type=float, default=500.0,
                    help="点数段位分的除数（0 = 关掉点数项，只按名次算）")
    ap.add_argument("--official", action="store_true",
                    help="启用官方档位调度：每天 8/11/14/17/20 点各跑一期 4000 局")
    args = ap.parse_args(argv)
    serve(args.players, args.port, args.host, args.copies, args.divisor,
          models=args.models, loose=args.loose, official=args.official)


if __name__ == "__main__":
    main()
