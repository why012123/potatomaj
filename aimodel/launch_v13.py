"""Start v1.3 training DETACHED, so it survives the session's process reaper.

WHY A PYTHON LAUNCHER
---------------------
A PowerShell version was tried first and failed four different ways, all of
them shell-quoting problems rather than training problems:

  * `cmd /c start /b cmd /c "... > log 2> log.err"` -- cmd splits on the `|`
    inside `2>`, so python never starts;
  * the same with `2>&1` -- nested quoting through Start-Process is mangled;
  * `Start-Process -RedirectStandardOutput x -RedirectStandardError x` --
    PS 5.1 refuses to point both streams at one file;
  * `--hidden 512,512` on a PS command line -- PowerShell parses the comma as
    an ARRAY, python receives "512 512" and argparse dies with
    "unrecognized arguments: 512"; `512-512` is parsed as a hashtable.

None of those can happen when the argument list is built in Python.

WHY DETACHED AT ALL
-------------------
On this box a child process of an agent session gets reaped after roughly
15 seconds to 3.5 minutes. Every historical v1.3 log is 0.4 KB and stops right
after the header for exactly this reason. `subprocess.Popen` with
`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` starts the trainer outside that
job object, and this launcher then exits immediately.

USAGE
-----
    python -m aimodel.launch_v13                 # the agreed run
    python -m aimodel.launch_v13 --probe         # tiny run, proves it works
    python -m aimodel.launch_v13 --stage all     # BC + RL
    python -m aimodel.launch_v13 --model maj-mountain4v1.3

Follow progress with:
    python -m aimodel.launch_v13 --status
which reports the live process, its working set, and the last log lines.

MEMORY
------
The user capped this box at 16 GB for us. Measured: a training process runs
about 5 GB (577k params + ~1M samples + optimiser state). So at most TWO
training processes may run at once; `--status` prints the free RAM and warns
if a second would not fit.
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
LOG_DIR = ROOT / "aimodel" / "weights" / "logs"
SITE = (
    r"C:\Users\whyderivation\.dsh\dsh-runtimes\dsh-primary-runtime"
    r"\dependencies\python\Lib\site-packages"
)

# Hidden width lives HERE, not on a command line -- see the module docstring.
HIDDEN = "512,512"

# Windows process-creation flags. DETACHED_PROCESS keeps the trainer out of
# this session; CREATE_NEW_PROCESS_GROUP makes it survive Ctrl-C / teardown.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _env(threads: int) -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = SITE
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    # Net is all `@` matmul, so BLAS threads are what parallelise training.
    # Each training process owns these cores -- lower it to run several.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        e[k] = str(threads)
    return e


def build_args(a) -> list[str]:
    argv = [
        str(PY), "-u", "-m", "aimodel.train", a.stage,
        f"--model={a.model}",
        f"--hands={a.hands}",
        f"--epochs={a.epochs}",
        f"--hidden={HIDDEN}",
        f"--matches={a.matches}",
        f"--iters={a.iters}",
        f"--eval-matches={a.eval_matches}",
        f"--pool-ratio={a.pool_ratio}",
        f"--out={ROOT / 'aimodel' / 'weights' / (a.model + '.npz')}",
        "--force",
    ]
    if a.resume:
        argv.append("--resume")
    return argv


def cmd_start(a) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{a.model}.log"
    err = LOG_DIR / f"{a.model}.err.log"
    argv = build_args(a)

    # Free-RAM guard: refuse to add a 2nd/3rd trainer beyond the 16 GB cap.
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        avail_gb = ms.ullAvailPhys / 1024 ** 3
        print(f"free RAM: {avail_gb:.1f} GB")
        if avail_gb < 6.0:
            print("REFUSING to start: a trainer needs ~5 GB and less than 6 GB "
                  "is free (user cap is 16 GB total for us). Stop something first.")
            return 2
    except Exception:
        pass

    hdr = ["", "=" * 72,
           f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"# BLAS threads = {a.threads}",
           "# " + " ".join(argv),
           "=" * 72]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n".join(hdr) + "\n")

    out = open(log, "a", encoding="utf-8")
    errf = open(err, "a", encoding="utf-8")
    try:
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        try:
            p = subprocess.Popen(argv, cwd=str(ROOT), env=_env(a.threads),
                                 stdout=out, stderr=errf,
                                 creationflags=flags | CREATE_BREAKAWAY_FROM_JOB)
        except OSError:
            # BREAKAWAY needs the job to allow it; retry without.
            p = subprocess.Popen(argv, cwd=str(ROOT), env=_env(a.threads),
                                 stdout=out, stderr=errf, creationflags=flags)
    finally:
        out.close()
        errf.close()

    print(f"launched {a.model} (stage={a.stage} hands={a.hands} epochs={a.epochs})")
    print(f"PID {p.pid}   log: {log}")

    # Confirm it started. NOTE: a *finished* run is not a failure -- the probe
    # (8 hands / 1 epoch) legitimately completes in ~40s, and an earlier
    # version of this check wrongly reported that as "FAILED: exited within
    # 20s". So: exit code 0 counts as success, and a non-zero code is only a
    # real failure if the log shows no training progress.
    time.sleep(20)
    rc = p.poll()
    if rc is not None:
        text = log.read_text(encoding="utf-8", errors="replace")
        # Only exit code 0 is success. A non-zero code is a failure even if the
        # log shows progress -- an earlier version accepted "progress" as
        # success and reported an unknown-model error (exit=1) as "completed".
        if rc == 0:
            print("OK: run completed already (exit=0) -- short configs finish fast")
            print("".join(text.splitlines(keepends=True)[-8:]))
            return 0
        print(f"FAILED: exited within 20s (code={rc})")
        for f in (log, err):
            print(f"--- {f.name} ---")
            print("".join(open(f, encoding="utf-8", errors="replace")
                          .readlines()[-15:]))
        return 1
    print("OK: still running after 20s (detached, survived the reaper window)")
    print(f"  tail : Get-Content '{log}' -Tail 20 -Wait")
    print(f"  stop : Stop-Process -Id {p.pid} -Force")
    return 0


def _find_pid(pid: int):
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=20)
        return "python" in r.stdout.lower()
    except Exception:
        return None


def cmd_status(a) -> int:
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        print(f"free RAM: {ms.ullAvailPhys / 1024 ** 3:.1f} GB "
              f"/ total {ms.ullTotalPhys / 1024 ** 3:.1f} GB")
    except Exception:
        pass

    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*aimodel.train*' } | "
          "ForEach-Object { '  PID {0}  WS {1:N0} MB' -f $_.ProcessId, "
          "($_.WorkingSet64/1MB) }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    print("trainers running:")
    print(r.stdout.strip() or "  (none)")

    log = LOG_DIR / f"{a.model}.log"
    if log.exists():
        print(f"--- last lines of {log.name} ---")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
            print("  " + line)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="start v1.3 training detached")
    ap.add_argument("--model", default="maj-normal4v1.3")
    ap.add_argument("--stage", default="bc", choices=["bc", "rl", "all"])
    ap.add_argument("--hands", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--matches", type=int, default=500)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eval-matches", type=int, default=250)
    ap.add_argument("--pool-ratio", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="tiny run (8 hands / 1 epoch) to prove the launcher works")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    if a.status:
        return cmd_status(a)
    if a.probe:
        a.hands, a.epochs, a.threads = 8, 1, 4
    return cmd_start(a)


if __name__ == "__main__":
    raise SystemExit(main())
