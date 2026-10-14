# 土豆麻将 · 网页对战平台 —— 一键启停（Windows / 本机）
#
# 服务（python run.py）和 frp 隧道（frpc）是一套的：隧道断了外面就访问不到，
# 所以这个脚本**一起管**，并且带看门狗（崩溃 5 秒后自动重启，最多 20 次）。
#
#   .\deploy\maj.ps1 start      启动（服务 + 隧道）
#   .\deploy\maj.ps1 stop       停止
#   .\deploy\maj.ps1 restart    重启
#   .\deploy\maj.ps1 status     看状态（进程、端口、HTTP 健康、隧道）
#   .\deploy\maj.ps1 logs       跟着看日志
#
# 参数都可以用环境变量覆盖：MAJ_PORT / MAJ_HOST / MAJ_AI_DELAY / MAJ_AI_WORKERS
# 日志落在 <仓库>\logs\，PID 落在 <仓库>\run\。

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs', 'watch')]
    [string]$Action = 'status',

    [int]$Port = 8790,
    [string]$BindHost = '127.0.0.1',
    [double]$AiDelay = 0.6,
    [int]$AiWorkers = 4,
    [int]$MaxRestarts = 20,
    [switch]$NoTunnel
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RunDir = Join-Path $Root 'run'
$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $RunDir 'app.pid'
$TunnelPidFile = Join-Path $RunDir 'frpc.pid'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Frpc = 'E:\WHYidea\frp_0.61.1_windows_amd64\frpc.exe'
$FrpcConf = Join-Path $Root 'deploy\frpc-maj.toml'

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null

function Get-Alive([string]$PidPath) {
    if (-not (Test-Path $PidPath)) { return $null }
    $id = (Get-Content $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $id) { return $null }
    $p = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
    if ($p) { return $p }
    Remove-Item $PidPath -ErrorAction SilentlyContinue
    return $null
}

function Test-Port([int]$P) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $P)
        $c.Close()
        return $true
    } catch { return $false }
}

function Stop-All {
    foreach ($f in @($PidFile, $TunnelPidFile)) {
        $p = Get-Alive $f
        if ($p) {
            Write-Host "  停止 $($p.ProcessName) (PID $($p.Id))" -ForegroundColor Yellow
            # 连同子进程一起收掉（frpc / python 都可能带子进程）
            taskkill /PID $p.Id /T /F 2>$null | Out-Null
        }
        Remove-Item $f -ErrorAction SilentlyContinue
    }
    # 兜底：端口上还挂着的 python（比如上次手工起的）
    Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "$Root*" } |
        ForEach-Object {
            Write-Host "  清理残留 python (PID $($_.Id))" -ForegroundColor DarkYellow
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
}

function Start-App {
    if (-not (Test-Path $Python)) {
        throw "找不到 $Python —— 先在仓库根目录建虚拟环境：python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    }
    $args = @(
        'run.py', '--host', $BindHost, '--port', $Port,
        '--ai-delay', $AiDelay, '--ai-workers', $AiWorkers
    )
    $out = Join-Path $LogDir 'launch-server.log'
    $err = Join-Path $LogDir 'launch-server.err.log'
    $p = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
    $p.Id | Set-Content $PidFile
    Write-Host "  服务已启动 PID $($p.Id) -> http://$BindHost`:$Port" -ForegroundColor Green
    # 等端口起来（最多 20 秒）
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port $Port) { return $true }
        if ($p.HasExited) {
            Write-Host "  ❌ 启动失败，看 $err" -ForegroundColor Red
            Get-Content $err -Tail 20 -ErrorAction SilentlyContinue
            return $false
        }
    }
    Write-Host "  ⚠ 20 秒内端口还没通，但进程还在（可能在载模型）" -ForegroundColor Yellow
    return $true
}

function Start-Tunnel {
    if ($NoTunnel) { Write-Host "  （--NoTunnel，跳过隧道）" -ForegroundColor DarkGray; return }
    if (-not (Test-Path $Frpc)) {
        Write-Host "  ⚠ 找不到 frpc：$Frpc，跳过隧道" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $FrpcConf)) {
        Write-Host "  ⚠ 找不到 $FrpcConf，跳过隧道" -ForegroundColor Yellow
        return
    }
    $out = Join-Path $LogDir 'frpc.log'
    $p = Start-Process -FilePath $Frpc -ArgumentList @('-c', $FrpcConf) `
        -WorkingDirectory (Split-Path $Frpc) `
        -RedirectStandardOutput $out -RedirectStandardError $out `
        -PassThru -WindowStyle Hidden
    $p.Id | Set-Content $TunnelPidFile
    Write-Host "  隧道已启动 PID $($p.Id)（maj.xjtuwhy.com -> 127.0.0.1:$Port）" -ForegroundColor Green
}

function Show-Status {
    Write-Host "土豆麻将 · 平台状态" -ForegroundColor Cyan
    $app = Get-Alive $PidFile
    $tun = Get-Alive $TunnelPidFile
    Write-Host ("  服务进程   " + $(if ($app) { "运行中 PID $($app.Id)" } else { "未运行" }))
    Write-Host ("  端口 $Port  " + $(if (Test-Port $Port) { "通" } else { "不通" }))
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
        Write-Host ("  健康接口   房间 {0} / 连接 {1} / 进行中 {2} / 运行 {3}s" -f `
            $h.rooms, $h.conns, $h.playing, $h.uptime) -ForegroundColor Green
        if ($h.record) {
            Write-Host ("  记录器     写入 {0} 条，丢弃 {1}，错误 {2}" -f `
                $h.record.written, $h.record.dropped, $h.record.errors)
        }
    } catch {
        Write-Host "  健康接口   取不到（服务没起来？）" -ForegroundColor Yellow
    }
    Write-Host ("  隧道进程   " + $(if ($tun) { "运行中 PID $($tun.Id)" } else { "未运行" }))
    # 公网可达性（走域名，要经过 nginx + frps + frpc）
    try {
        $r = Invoke-WebRequest -Uri 'https://maj.xjtuwhy.com/api/health' -TimeoutSec 12 `
            -UseBasicParsing
        $j = $r.Content | ConvertFrom-Json
        Write-Host ("  公网 https://maj.xjtuwhy.com  ✅ ({0} 字节, 房间 {1})" -f `
            $r.RawContentLength, $j.rooms) -ForegroundColor Green
    } catch {
        Write-Host "  公网 https://maj.xjtuwhy.com  ❌ $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Watch-App {
    Write-Host "看门狗启动：崩了 5 秒后自动重启（最多 $MaxRestarts 次）" -ForegroundColor Cyan
    $n = 0
    while ($true) {
        $p = Get-Alive $PidFile
        if (-not $p) {
            if ($n -ge $MaxRestarts) {
                Write-Host "重启次数到上限 $MaxRestarts，退出" -ForegroundColor Red
                return
            }
            $n++
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 服务不在了，第 $n 次重启" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            Start-App | Out-Null
        }
        Start-Sleep -Seconds 5
    }
}

switch ($Action) {
    'start' {
        Write-Host "启动土豆麻将平台…" -ForegroundColor Cyan
        $app = Get-Alive $PidFile
        if ($app) { Write-Host "  服务已经在跑 (PID $($app.Id))，先停掉" -ForegroundColor Yellow; Stop-All }
        if (Test-Port $Port) {
            Write-Host "  ⚠ 端口 $Port 被别的进程占着，先清一下" -ForegroundColor Yellow
            Get-Process python -ErrorAction SilentlyContinue |
                Where-Object { $_.Path -like "$Root*" } |
                Stop-Process -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
        if (Start-App) { Start-Tunnel }
        Show-Status
    }
    'stop' {
        Write-Host "停止土豆麻将平台…" -ForegroundColor Cyan
        Stop-All
        Write-Host "  已停止" -ForegroundColor Green
    }
    'restart' {
        Write-Host "重启土豆麻将平台…" -ForegroundColor Cyan
        Stop-All
        Start-Sleep -Seconds 2
        if (Start-App) { Start-Tunnel }
        Show-Status
    }
    'status' { Show-Status }
    'logs' {
        Write-Host "跟着看 maj-server.log（Ctrl-C 退出）" -ForegroundColor Cyan
        Get-Content (Join-Path $LogDir 'maj-server.log') -Wait -Tail 40
    }
    'watch' { Watch-App }
}
