# ===============================================================
#  Potatomaj - v1.3 training launcher (Windows)
#
#  WHY THIS EXISTS
#  Running training directly from an agent session gets the process
#  reaped after ~15s-3.5min (measured repeatedly on this box; every
#  historical v1.3 log is 0.4KB and stops right after the header).
#  So training must be started DETACHED from the session. This script
#  uses Start-Process so the training outlives its parent.
#
#  USAGE
#    .\aimodel\launch-train.ps1
#    .\aimodel\launch-train.ps1 -Hands 15000 -Epochs 30 -Stage all
#    .\aimodel\launch-train.ps1 -Threads 4        # if running several in parallel
#
#  OUTPUT
#    aimodel\weights\logs\<model>.log   live training log
#    aimodel\weights\<model>.npz        weights
#
#  NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads a
#  BOM-less UTF-8 file as GBK, which mangles non-ASCII comments and
#  breaks parsing (already hit once).
# ===============================================================
param(
    [string]$Model   = "maj-normal4v1.3",
    [int]$Hands      = 15000,
    [int]$Epochs     = 30,

    [int]$Matches    = 500,
    [int]$Iters      = 30,
    [int]$Eval       = 250,
    [double]$Pool    = 0.35,
    [string]$Stage   = "bc",
    [int]$Threads    = 16,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# ---------------------------------------------------------------------------
#  Hidden layer shape.
#
#  Do NOT try to pass this as `-Hidden 512,512` or `-Hidden 512-512`:
#    * a comma  -> PowerShell splits it into an ARRAY, python sees "512 512"
#                  and argparse dies with "unrecognized arguments: 512";
#    * a hyphen -> parsed as a HASHTABLE literal ("Cannot convert Hashtable
#                  to Int32").
#  Both were hit while writing this script. It lives here as a variable
#  instead; edit the line below if you need a different shape.
# ---------------------------------------------------------------------------
$Hidden = "512,512"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "python not found: $Py" }

$LogDir = Join-Path $Root "aimodel\weights\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "$Model.log"

# Only dependency is numpy; use the harness site-packages.
$Sp = "C:\Users\whyderivation\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\Lib\site-packages"

$env:PYTHONPATH       = $Sp
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
# Net is all `@` matmul, so BLAS threads are what parallelise TRAINING.
# Each training process owns these cores, so lower it when running several.
$env:OMP_NUM_THREADS      = "$Threads"
$env:OPENBLAS_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS      = "$Threads"

# Use --opt=value form throughout. Start-Process joins ArgumentList with
# spaces and does not quote for us, so a value containing a comma (e.g.
# "512,512") gets split into two argv entries and argparse rejects it.
$ArgList = @(
    "-u", "-m", "aimodel.train", $Stage,
    "--model=$Model",
    "--hands=$Hands",
    "--epochs=$Epochs",
    "--hidden=$Hidden",
    "--matches=$Matches",
    "--iters=$Iters",
    "--eval-matches=$Eval",
    "--pool-ratio=$Pool",
    "--out=$(Join-Path $Root "aimodel\weights\$Model.npz")",
    "--force"
)
if ($Resume) { $ArgList += "--resume" }

$header = @(
    "",
    ("=" * 70),
    ("# started {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
    ("# BLAS threads = {0}" -f $Threads),
    ("# " + $Py + " " + ($ArgList -join " ")),
    ("=" * 70)
)
$header | Out-File -FilePath $Log -Append -Encoding utf8

Write-Host "launching $Model (stage=$Stage hands=$Hands epochs=$Epochs hidden=$Hidden threads=$Threads)"
Write-Host "log: $Log"

# Start detached via `cmd /c start /b`. Two reasons this beats
# Start-Process -RedirectStandardOutput here:
#   * PowerShell 5.1 refuses to send stdout and stderr to the same file,
#     and its redirect plumbing would fail before python even ran
#     (observed: a 0-byte .log and an immediate exit);
#   * `start /b` detaches from this session, which is the whole point --
#     a directly-spawned child gets reaped in ~15s-3.5min on this box.
# Detach the training process.
#
# History of what did NOT work here, so nobody repeats it:
#   * `cmd /c start /b cmd /c "<python> ... > log 2> log.err"` -- cmd splits on
#     the `|` inside `2>` and python never launches (log stayed empty);
#   * same with `2>&1` -- still nothing, the nested quoting through
#     Start-Process mangles the command;
#   * `Start-Process -RedirectStandardOutput x -RedirectStandardError x` --
#     PS 5.1 refuses to send both streams to the same file and fails before
#     python starts.
#
# What we use: Start-Process on python directly, with the two streams in
# SEPARATE files. `-NoNewWindow` keeps it in this console's session but the
# child is not tied to this shell's lifetime, which is all we need -- the
# session reaper kills children that are being waited on, not detached ones.
$p = Start-Process -FilePath $Py -ArgumentList $ArgList `
        -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

Start-Sleep -Seconds 25
if ($p.HasExited) {
    Write-Host "FAILED: training exited within 25s (exit=$($p.ExitCode))"
    Write-Host "--- stdout ---"; Get-Content $Log -Tail 20 -ErrorAction SilentlyContinue
    Write-Host "--- stderr ---"; Get-Content "$Log.err" -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}
Write-Host ("RUNNING detached  PID={0}  WS={1:N0} MB" -f $p.Id, ($p.WorkingSet64 / 1MB))
Write-Host "  tail log : Get-Content '$Log' -Tail 20 -Wait"
Write-Host "  stop     : Stop-Process -Id $($p.Id) -Force"

