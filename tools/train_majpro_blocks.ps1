# 把 maj-pro4v1.1 接着训到目标 epoch 数 —— 分块续跑，块间自动接上
#
# 为什么分块：这台机器会回收会话里的重计算进程（实测训练一开始就在 ~45 秒被杀）。
# 分块 = 每块训几个 epoch 就存盘，被杀也只损失当前这一块。
#
# 重要：`--resume` 只恢复**权重**，不恢复 epoch 计数（train_bc 总是从 ep=0 开始）。
# 所以 `--epochs=N` 的含义是「本块再训 N 轮」，不是「训到第 N 轮」。
# 因为训练没有 LR 衰减、l2 固定，分块跑和一次跑满在数学上等价。
#
# 用法（在项目根目录的 PowerShell 里）：
#     .\tools\train_majpro_blocks.ps1                          # 默认 6 块 x 5 轮
#     .\tools\train_majpro_blocks.ps1 -Blocks 2 -EpochsPerBlock 3
#
# 想推到留出率见顶就停：-Patience 2 或 3（连续 N 轮没进步即提前停）。

param(
    [string]$Model       = "maj-pro4v1.1",
    [string]$BcData      = "_data/bc_tv13.npz",
    [int]$Blocks         = 6,
    [int]$EpochsPerBlock = 5,
    [int]$Patience       = 0,
    [int]$Batch          = 256,
    [string]$Log         = "_runlogs/train_majpro_blocks.log"
)

$ErrorActionPreference = "Stop"

# ★ Windows PowerShell 5.1 默认按 ANSI 读子进程输出，而 python 输出 UTF-8
#   -> 中文全变乱码。必须显式把这三个编码都设成 UTF-8。
#   （和「.ps1 文件本身要带 BOM」是两件不同的事，两个都得做）
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "找不到 $py" }
if (-not (Test-Path (Join-Path $root $BcData))) { throw "找不到数据集 $BcData" }

$logPath = Join-Path $root $Log
New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $logPath -Value $line -Encoding utf8
}

Say "=== 分块续训开始：$Model  共 $Blocks 块 x 每块 $EpochsPerBlock 轮 ==="
Say "数据集 $BcData   patience=$Patience  batch=$Batch"

for ($i = 1; $i -le $Blocks; $i++) {
    Say "--- 第 $i/$Blocks 块：再训 $EpochsPerBlock 轮 ---"
    $t0 = Get-Date
    $trainArgs = @(
        "-u", "-m", "aimodel.train", "bc",
        "--model=$Model",
        "--bc-data=$BcData",
        "--resume",
        "--epochs=$EpochsPerBlock",
        "--patience=$Patience",
        "--batch=$Batch"
    )
    # 注意：不能用 `*>> $logPath` —— Windows PowerShell 5.1 会把原生命令输出
    # 按 UTF-16LE 追加进文件，日志直接变乱码。必须显式管道 + 指定编码。
    & $py @trainArgs 2>&1 | ForEach-Object { Add-Content -Path $logPath -Value $_ -Encoding utf8 }
    $code = $LASTEXITCODE
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Say "第 $i 块结束：exit=$code  耗时 ${dt}s"

    if ($code -ne 0) {
        Say "本块非零退出（很可能是被环境回收）。权重已在上一块存盘，"
        Say "重跑本脚本即可接着训 —— 已完成的 epoch 不会丢。"
        break
    }
}

Say "--- 收尾自检 ---"
& $py (Join-Path $root "tools\weight_check.py") $Model 2>&1 |
    ForEach-Object { Add-Content -Path $logPath -Value $_ -Encoding utf8 }
Get-Content $logPath -Tail 14 -Encoding utf8
Say "=== 全部结束，完整日志：$Log ==="
