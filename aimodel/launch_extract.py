"""Launch the full Tenhou -> BC extraction DETACHED (survives the reaper).

Same reasoning as launch_v13.py: a child of an agent session is reaped after
~15s-3.5min, and the full 40k-kyoku extraction runs for about an hour.

Every shard is written atomically (tmp + os.replace) and carries a plain index
in its name, so if this dies partway the finished shards are still valid and
the run can be resumed by pointing --src at the remaining batches.

USAGE
    python -m aimodel.launch_extract              # full 40k
    python -m aimodel.launch_extract --limit 500  # quick slice
    python -m aimodel.launch_extract --status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT / "_data" / "tbc"
SITE = (
    r"C:\Users\whyderivation\.dsh\dsh-runtimes\dsh-primary-runtime"
    r"\dependencies\python\Lib\site-packages"
)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = SITE
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    e["OMP_NUM_THREADS"] = "4"
    e["OPENBLAS_NUM_THREADS"] = "4"
    return e


def cmd_start(a) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "extract.log"
    argv = [
        str(PY), "-u", "-m", "aimodel.tenhou_bc",
        f"--src={ROOT / '_data' / 'final40k'}",
        f"--out={ROOT / '_data' / 'tbc'}",
        "--players=4",
        "--shard=2000",
    ]
    if a.limit:
        argv.append(f"--limit={a.limit}")

    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 72 + "\n")
        fh.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("# " + " ".join(argv) + "\n")
        fh.write("=" * 72 + "\n")

    out = open(log, "a", encoding="utf-8")
    err = open(LOG_DIR / "extract.err.log", "a", encoding="utf-8")
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        try:
            p = subprocess.Popen(argv, cwd=str(ROOT), env=_env(),
                                 stdout=out, stderr=err,
                                 creationflags=flags | CREATE_BREAKAWAY_FROM_JOB)
        except OSError:
            p = subprocess.Popen(argv, cwd=str(ROOT), env=_env(),
                                 stdout=out, stderr=err, creationflags=flags)
    finally:
        out.close()
        err.close()

    print(f"launched Tenhou BC extraction, PID {p.pid}")
    print(f"log: {log}")
    time.sleep(25)
    if p.poll() is not None:
        print(f"FAILED: exited within 25s (code={p.returncode})")
        print("".join(open(log, encoding="utf-8", errors="replace")
                      .readlines()[-20:]))
        return 1
    print("OK: running detached (survived 25s). Shards land in _data/tbc as it goes.")
    return 0


def cmd_status(a) -> int:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*tenhou_bc*' } | "
          "ForEach-Object { '  PID {0}  WS {1:N0} MB' -f $_.ProcessId, "
          "($_.WorkingSet64/1MB) }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    print("extraction running:")
    print(r.stdout.strip() or "  (none)")

    shards = sorted(LOG_DIR.glob("tbc_*.npz"))
    total_mb = sum(s.stat().st_size for s in shards) / 1024 ** 2
    print(f"shards: {len(shards)}   {total_mb:.1f} MB")
    log = LOG_DIR / "extract.log"
    if log.exists():
        print("--- log tail ---")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            print("  " + line)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="launch Tenhou BC extraction detached")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    return cmd_status(a) if a.status else cmd_start(a)


if __name__ == "__main__":
    raise SystemExit(main())
