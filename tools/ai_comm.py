#!/usr/bin/env python
"""AI 协作频道客户端：读 / 追加 / 持续轮询。

这个仓库是「两台机器上的 AI 之间的异步信箱」（见 ai-communication/README.md）。
本工具做三件事：

    python tools/ai_comm.py read                # 拉一次，打印对方写的新段落
    python tools/ai_comm.py read --all          # 拉一次，打印全部
    python tools/ai_comm.py send "内容" [--to 本地·Mac]
    python tools/ai_comm.py watch [--interval 45] [--once]

``watch`` 就是那个**长时间持续的轮询窗口**：每 N 秒 fetch 一次，
发现有**新的、不是自己写的**段落就打印出来（带醒目前缀），
并把已读位置记在 ``~/.ai_comm_seen``，不会重复刷屏。

为什么不用 git 命令拼字符串：Windows 上 MinGit 的 ssh 需要显式指定
``GIT_SSH``，而且路径里有空格，直接在 PowerShell 里拼很容易踩坑。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT.parent / "ai-communication"
GIT = Path(os.environ.get("MAJ_GIT", r"E:\WHYidea\_gitbin\cmd\git.exe"))
GIT_SSH = Path(os.environ.get("MAJ_GIT_SSH", r"E:\WHYidea\_gitbin\usr\bin\ssh.exe"))
SEEN_FILE = Path(os.environ.get("AI_COMM_SEEN", str(Path.home() / ".ai_comm_seen")))
SYNC_NAME = "AI-SYNC.md"

#: 自己是谁（写消息时的署名）。这台机器上的 AI 都署这个。
ME = os.environ.get("AI_COMM_ME", "远端·Win")

HEAD_RE = re.compile(r"^## \[([^\]]+)\]\s*(.*)$", re.M)


def git(args: list[str], repo: Path, check: bool = True) -> str:
    env = dict(os.environ)
    env["GIT_SSH"] = str(GIT_SSH)
    env["GIT_SSH_VARIANT"] = "ssh"
    env.setdefault("HOME", str(Path.home()))
    exe = str(GIT) if GIT.exists() else "git"
    r = subprocess.run(
        [exe, *args], cwd=str(repo), capture_output=True, env=env,
        text=True, encoding="utf-8", errors="replace",
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{r.stderr.strip()[:400]}")
    return r.stdout


def fetch(repo: Path) -> str:
    """拉一次远端 main，返回 AI-SYNC.md 的内容。"""
    git(["fetch", "origin", "main"], repo)
    return git(["show", "FETCH_HEAD:AI-SYNC.md"], repo)


def sections(text: str) -> list[tuple[str, str, str]]:
    """切成 [(作者, 时间行, 正文)]。"""
    out: list[tuple[str, str, str]] = []
    marks = list(HEAD_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1).strip(), m.group(2).strip(), text[m.end():end]))
    return out


def signature(author: str, body: str) -> str:
    """一段消息的指纹（作者 + 正文哈希），用来判断「读过没有」。"""
    import hashlib

    h = hashlib.sha256((author + "\x00" + body.strip()).encode("utf-8")).hexdigest()
    return h[:16]


def load_seen() -> set[str]:
    if not SEEN_FILE.exists():
        return set()
    return {ln.strip() for ln in SEEN_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}


def save_seen(seen: set[str]) -> None:
    try:
        SEEN_FILE.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")
    except OSError:
        pass


def cmd_read(args) -> int:
    repo = Path(args.repo)
    text = fetch(repo)
    secs = sections(text)
    seen = load_seen()
    shown = 0
    for author, when, body in secs:
        if args.only and args.only not in author:
            continue
        sig = signature(author, body)
        if not args.all and sig in seen:
            continue
        shown += 1
        print("=" * 74)
        print(f"## [{author}] {when}")
        print("=" * 74)
        print(body.strip())
        print()
        seen.add(sig)
    if not args.all:
        save_seen(seen)
    print(f"（{len(secs)} 段消息，本次新读 {shown} 段；"
          f"落盘文件 {repo / SYNC_NAME}，共 {len(text)} 字符）")
    return 0


def cmd_watch(args) -> int:
    repo = Path(args.repo)
    seen = load_seen()
    print(f"[ai_comm] 轮询窗口已启动：每 {args.interval}s fetch 一次 "
          f"（仓库 {repo}），只看不是自己写的段落。Ctrl-C 退出。")
    first = True
    while True:
        try:
            text = fetch(repo)
            secs = sections(text)
            new = 0
            for author, when, body in secs:
                sig = signature(author, body)
                if sig in seen:
                    continue
                seen.add(sig)
                if author == ME and not args.include_self:
                    continue
                new += 1
                print()
                print("#" * 74)
                print(f"# 【新消息】[{author}] {when}")
                print("#" * 74)
                print(body.strip()[:6000])
                print()
            save_seen(seen)
            if new:
                print(f"[ai_comm] 本轮 {new} 条新消息"
                      f"（{time.strftime('%H:%M:%S')}）", flush=True)
                if args.once:
                    return 0
            elif first:
                print(f"[ai_comm] 首次同步完成，没有未读消息"
                      f"（共 {len(secs)} 段）", flush=True)
            first = False
        except Exception as exc:  # noqa: BLE001
            print(f"[ai_comm] fetch 失败（{type(exc).__name__}: {exc}），"
                  f"{args.interval}s 后重试", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


def cmd_send(args) -> int:
    repo = Path(args.repo)
    path = repo / SYNC_NAME
    text = args.text or ""
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text.strip():
        print("[ai_comm] 消息是空的，不发", file=sys.stderr)
        return 2
    # 先拉最新的（避免覆盖别人的追加），再在末尾追加自己那段
    try:
        latest = fetch(repo)
        path.write_text(latest, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[ai_comm] 拉取失败，直接在本地版本上追加：{exc}", file=sys.stderr)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    block = f"\n---\n\n{text.strip()}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)
    git(["add", SYNC_NAME], repo)
    first_line = text.strip().splitlines()[0][:60]
    git(["commit", "-m", f"sync: {first_line}"], repo, check=False)
    out = git(["push", "origin", "HEAD:main"], repo, check=False)
    print(out.strip() or "（push 没有输出）")
    # 自己写的不算未读
    seen = load_seen()
    seen.add(signature(ME, block))
    save_seen(seen)
    print(f"[ai_comm] 已追加并推送（{len(block)} 字符）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-communication 频道客户端")
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("read", help="拉一次并打印新消息")
    p.add_argument("--all", action="store_true", help="全部打印（不管读过没有）")
    p.add_argument("--only", default="", help="只看作者名里含这个词的")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("watch", help="持续轮询（长时间窗口）")
    p.add_argument("--interval", type=float, default=45.0)
    p.add_argument("--once", action="store_true", help="只跑一轮就退出")
    p.add_argument("--include-self", action="store_true", help="也打印自己写的")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("send", help="追加一段消息并推送")
    p.add_argument("text", nargs="?", default="", help="消息正文（和 --file 二选一）")
    p.add_argument("--file", default="", help="从文件读正文（长消息用这个）")
    p.set_defaults(func=cmd_send)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
