#!/usr/bin/env python
"""ELO 平台与对战平台的**可用性自检**（只读，不改任何东西）。

    python tools/elo_healthcheck.py            # 人看
    python tools/elo_healthcheck.py --json     # 给定时任务/机器看

## 为什么要有这个（2026-10-20 的两次教训）

① **公网 404 挂了一小时没人发现** ✗
   我先前报过「HTTP 200」✓ 但那是**一小时前**的事 ✓ 之后隧道崩了 ✓ 而我的值守
   **只查频道、不查公网** ✗ -> Mac 反而先发现了 ✗
   ⇒ 对**用户可见的入口**必须**周期性主动探测** ✓ 不能靠「上次通了」✗

② **「进程名对上了」不等于「就是这个进程」** ✗
   机器上有 4 个 `frpc.exe` ✓ 全是别人商业穿透工具的 ✗
   而真正把 `maj.xjtuwhy.com` 指到 8790 的那个（`-c deploy/frpc-maj.toml`）**根本没跑** ✗
   ⇒ 判断依据必须是**命令行里的配置文件** ✓ 不是进程名 ✓

③ **名册一致性**（Mac 要求）✗
   `/api/rooms` 里每个 AI 座位的 `ai` 字段，都应该是 `/api/models` 里
   **存在且 usable** 的模型 ✓ 否则那一局会静默走教师兜底 ✓
   （静默降级比报错更难发现 —— 形状⑳/㉑ 那一族）

## 退出码

0 = 全部通过；1 = 有任一检查失败（给 cron/定时任务判断用）
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 公网入口（用户真正在用的那个面 ✓ 不是 127.0.0.1 ✗）
PUBLIC = "https://maj.xjtuwhy.com"

#: 本机三个服务的端口
LOCAL = {"webapp": 8790, "elo4p": 8964, "elo3p": 8965}

#: 隧道配置（判断 frpc 是不是「这条隧道」的唯一依据 ✓）
FRPC_CONF = "frpc-maj.toml"

#: 内置基线：它们**没有权重文件**，不该出现在 /api/models 里 ✓
#: （`arena.loose_participants` 里叫 teacher4#1 / random4#1 ✓
#:  房间座位回收时 `rooms.py:195` 会把 AI 默认设成 "teacher" ✓）
BASELINE_PREFIXES = ("teacher", "random")


def _get(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "maj-healthcheck"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400]
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(f"{type(e).__name__}: {e}") from e


def check_public(retries: int = 3) -> dict:
    """公网入口必须 200，且模型数对得上。

    ⚠️ **要重试** ✗ —— 2026-10-20 实测：一次 TimeoutError 之后连拉三次全成功
       （2.7s / 1.6s / 6.7s）-> 那条单次超时是**瞬时抖动**，不是故障。
       把抖动报成故障，会让人开始忽略这个检查 —— 那才是最坏的结果。
    -> 三次全失败才判红；中间成功一次即可（并写明第几次才成 ✓）
    """
    out = {"name": "public", "ok": False, "detail": "", "tries": []}
    last = ""
    for i in range(retries):
        try:
            st, body = _get(f"{PUBLIC}/api/models")
            if st != 200:
                last = f"/api/models -> HTTP {st}"
                out["tries"].append(last)
                continue
            ms = json.loads(body).get("models") or []
            out["tries"].append(f"HTTP 200, {len(ms)} 个模型")
            if ms:
                out["models"] = len(ms)
                out["ok"] = True
                out["detail"] = (f"HTTP 200, {len(ms)} 个模型"
                                 + ("" if i == 0
                                    else f"（第 {i + 1} 次才成，前 {i} 次失败）"))
                return out
            last = "模型数为 0"
        except Exception as e:                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            out["tries"].append(last)
    out["detail"] = f"{retries} 次都失败：{last}"
    return out


def check_tunnel() -> dict:
    """`frpc -c deploy/frpc-maj.toml` 这个**特定**进程在不在。

    ⚠️ 只认命令行里的配置文件 ✗ 不认进程名 ✓
       （机器上有 4 个别人的 frpc.exe，按名字数会误判成「在跑」✗）
    """
    out = {"name": "tunnel", "ok": False, "detail": ""}
    try:
        import subprocess
        # 用 PowerShell 的 CIM 查询（Windows）；非 Windows 直接跳过
        if sys.platform != "win32":
            out["ok"] = True
            out["detail"] = "非 Windows，跳过"
            return out
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='frpc.exe'\" | "
            "Where-Object { $_.CommandLine -like '*" + FRPC_CONF + "*' } | "
            "ForEach-Object { $_.ProcessId }"
        )
        # ⚠️ 必须给**全路径** ✗ —— 直接写 "powershell.exe" 会 FileNotFoundError
        #    （这个 Python 进程的 PATH 里不一定有 System32 ✓ 与调用方的 shell 无关 ✗）
        ps_exe = None
        for cand in (r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                     r"C:\Windows\System32\WindowsPowerShell\v1.0\pwsh.exe"):
            if Path(cand).exists():
                ps_exe = cand
                break
        if ps_exe is None:
            out["detail"] = "找不到 powershell.exe，跳过隧道检查"
            out["ok"] = True          # 查不了不算失败，但要**说明** ✓
            return out
        r = subprocess.run([ps_exe, "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=30)
        pids = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        out["pids"] = pids
        out["ok"] = bool(pids)
        out["detail"] = (f"在跑 PID {','.join(pids)}" if pids
                         else f"**没有**带 {FRPC_CONF} 的 frpc 进程")
    except Exception as e:                       # noqa: BLE001
        out["detail"] = f"{type(e).__name__}: {e}"
    return out


def check_local_ports() -> dict:
    out = {"name": "local_ports", "ok": True, "detail": "", "ports": {}}
    for label, port in LOCAL.items():
        try:
            st, _ = _get(f"http://127.0.0.1:{port}/", timeout=8)
            # 8790 的 / 是页面；8964/8965 的 / 也是页面 -> 都该 200
            ok = st == 200
        except Exception:                        # noqa: BLE001
            ok = False
        out["ports"][label] = ok
        if not ok:
            out["ok"] = False
    out["detail"] = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in out["ports"].items())
    return out


def check_roster() -> dict:
    """`/api/rooms` 里每个 AI 座位的模型，都必须在 `/api/models` 里且 usable。

    这是 Mac 2026-10-20 要求的检查：名册里若写了不可用的模型，
    那一局会**静默走教师兜底** ✗（不报错、不警告）✓
    """
    out = {"name": "roster", "ok": False, "detail": "", "problems": []}
    try:
        st, body = _get("http://127.0.0.1:8790/api/models", timeout=10)
        if st != 200:
            out["detail"] = f"/api/models -> HTTP {st}"
            return out
        models = {m.get("name"): m for m in (json.loads(body).get("models") or [])}

        st2, body2 = _get("http://127.0.0.1:8790/api/rooms", timeout=10)
        if st2 != 200:
            out["detail"] = f"/api/rooms -> HTTP {st2}"
            return out
        rooms = json.loads(body2).get("rooms") or []
        n_seats = 0
        baselines = 0
        for rm in rooms:
            for s in rm.get("seats") or []:
                if s.get("kind") != "ai":
                    continue
                n_seats += 1
                ai = s.get("ai")
                if not ai:
                    continue
                # ⚠️ 内置基线（teacher / random）**本来就没有权重文件** ✗
                #    座位回收时 `rooms.py:195` 会把 AI 默认设成 "teacher" ✓
                #    -> 那不是「模型不可用」，而是「这个位子现在由基线在打」✓
                #    第一版没排除它 -> 报了个假阳性 ✗（这个工具自己抓出来的 ✓）
                if str(ai).startswith(BASELINE_PREFIXES):
                    baselines += 1
                    continue
                m = models.get(ai)
                if m is None:
                    out["problems"].append(f"房间 {rm.get('id')} 座位 {s.get('seat')}: "
                                           f"模型 {ai} 不在可选列表里")
                elif not m.get("usable"):
                    miss = m.get("missing_heads") or []
                    out["problems"].append(f"房间 {rm.get('id')} 座位 {s.get('seat')}: "
                                           f"{ai} usable=False（缺头 {miss}）")
        out["ok"] = not out["problems"]
        out["rooms"] = len(rooms)
        out["ai_seats"] = n_seats
        out["baseline_seats"] = baselines
        out["detail"] = (f"{len(rooms)} 个房间 / {n_seats} 个 AI 座位"
                         f"（其中 {baselines} 个是内置基线 teacher/random ✓），"
                         + ("其余全部可用 ✓" if out["ok"]
                            else f"**{len(out['problems'])} 个问题** ✗"))
    except Exception as e:                       # noqa: BLE001
        out["detail"] = f"{type(e).__name__}: {e}"
    return out


def check_scheduler() -> dict:
    """官方档位调度：**该跑的那一期，是不是真的在跑**。

    为什么需要（2026-10-20 的现场）：
      官方 20:00 那期开了 4000 局的目标，但**第一局就崩**（chi 的 ValueError）
      -> ``running=False``、``done_run=0``、``error`` 有值
      -> 而其余五项检查**全部通过** ✗（公网 200、端口在、名册正常 ……）
      -> 于是**"平台看起来全好"** ✗ 但**那一期一局都没跑** ✗
    => 「服务活着」和「活干成了」是两件事 ✓ 这条检查专门看后者 ✓
    """
    out = {"name": "scheduler", "ok": False, "detail": "", "slots": []}
    try:
        sys.path.insert(0, str(ROOT))
        from aimodel import elo_store as E
        from aimodel.ui import elo_schedule as S

        # 该跑但没跑满的期（判重口径与调度器一致：done >= target 才算跑过 ✓）
        existing = set()
        for s in E.list_seasons():
            want = s.target or S.MATCHES_PER_SLOT
            if s.pruned or (s.done and s.done >= want):
                existing.add(s.sid)
        now = time.time()
        rows = S.today_slots(now, existing)
        out["slots"] = [{"slot": r.slot, "state": r.state, "sid": r.season_id}
                        for r in rows]
        due = [r for r in rows if r.state == "due"]

        # 天梯服务的实时状态
        st, body = _get("http://127.0.0.1:8964/api/arena", timeout=10)
        live = json.loads(body) if st == 200 else {}
        out["running"] = bool(live.get("running"))
        out["done_run"] = live.get("done_run")
        out["target"] = live.get("target")
        err = (live.get("error") or "").strip()

        if err:
            # **报错就是失败** ✓ 不管 due 不 due
            out["detail"] = f"天梯报错: {err[:90]}"
            return out
        if live.get("running"):
            out["ok"] = True
            out["detail"] = (f"在跑 {live.get('done_run')}/{live.get('target')} 局"
                             f"（今日待跑档位 {len(due)} 个）")
            return out
        if due:
            # 该跑却没在跑 -> 失败（这正是今天那个现场）
            out["detail"] = (f"**该跑却没在跑**：档位 "
                             f"{[r.slot for r in due]} 处于补跑窗口内，"
                             f"但 running=False")
            return out
        out["ok"] = True
        out["detail"] = ("空闲（当前没有处于补跑窗口的档位）"
                         f"· 今日档位 "
                         + ",".join(f"{r.slot:02d}:{r.state}" for r in rows))
    except Exception as e:                       # noqa: BLE001
        out["detail"] = f"{type(e).__name__}: {e}"
    return out


def check_elo_storage() -> dict:
    """ELO 存档的用量（对照 100 GB 配额）。"""
    out = {"name": "elo_storage", "ok": False, "detail": ""}
    try:
        sys.path.insert(0, str(ROOT))
        from aimodel import elo_store as E

        st = E.stats()
        out["ok"] = True
        out["detail"] = (f"{st['seasons_with_replay']} 期 / {st['matches']} 局 / "
                         f"{st['human']}（配额 {st['quota_human']}，"
                         f"已用 {st['used_pct']:.4f}%）")
    except Exception as e:                       # noqa: BLE001
        out["detail"] = f"{type(e).__name__}: {e}"
    return out


CHECKS = (check_public, check_tunnel, check_local_ports, check_roster,
          check_scheduler, check_elo_storage)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ELO / 对战平台可用性自检（只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    results = [fn() for fn in CHECKS]
    failed = [r for r in results if not r.get("ok")]

    if args.json:
        print(json.dumps({"ok": not failed, "checks": results}, ensure_ascii=False))
    else:
        print("=" * 70)
        print("  平台自检")
        print("=" * 70)
        for r in results:
            mark = "✅" if r.get("ok") else "🔴"
            print(f"  {mark} {r['name']:14} {r.get('detail', '')}")
            for p in r.get("problems") or []:
                print(f"       - {p}")
        print("=" * 70)
        print(f"  {'全部通过' if not failed else f'🔴 {len(failed)} 项失败'}")
        print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
