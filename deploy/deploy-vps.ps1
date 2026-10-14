# 土豆麻将 · 网页对战平台 —— VPS 侧一键部署（nginx + TLS + 证书续期）
#
# 用法（在**本机**跑，通过 ssh 推到 VPS）：
#
#     pwsh deploy/deploy-vps.ps1            # 部署/更新 nginx 配置 + 申请证书
#     pwsh deploy/deploy-vps.ps1 -DryRun    # 只打印要做什么，不动服务器
#
# 前提：
#   * DNS 里 maj.xjtuwhy.com 已经指向 VPS（47.82.118.193）
#   * 本机能免密 ssh root@47.82.118.193（用 C:\Windows\System32\OpenSSH\ssh.exe）
#   * VPS 上 frps 在跑，vhostHTTPPort = 8080
#
# 这个脚本**不动 frps 的配置**（那台机器上还有别的服务在用），
# 只加一个 nginx server 块 + 一张证书。

[CmdletBinding()]
param(
    [string]$Vps = 'root@47.82.118.193',
    [string]$Domain = 'maj.xjtuwhy.com',
    [string]$Email = 'whyderivation@gmail.com',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$Scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Conf = Join-Path $Root 'deploy\nginx-maj.conf'
$Service = Join-Path $Root 'deploy\maj.service'
$Logrotate = Join-Path $Root 'deploy\maj.logrotate'

function Invoke-Ssh([string]$Cmd) {
    if ($DryRun) { Write-Host "  [dry-run] ssh $Vps `"$Cmd`"" -ForegroundColor DarkGray; return }
    & $Ssh -o BatchMode=yes -o ConnectTimeout=20 $Vps $Cmd
    if ($LASTEXITCODE -ne 0) { throw "ssh 失败（exit $LASTEXITCODE）：$Cmd" }
}

function Send-File([string]$Local, [string]$Remote) {
    if ($DryRun) { Write-Host "  [dry-run] scp $Local -> $Remote" -ForegroundColor DarkGray; return }
    & $Scp -o BatchMode=yes $Local "${Vps}:$Remote"
    if ($LASTEXITCODE -ne 0) { throw "scp 失败：$Local" }
}

Write-Host "=== 土豆麻将 · VPS 部署（$Domain -> $Vps）===" -ForegroundColor Cyan

Write-Host "`n[1/6] 检查连通性与现状" -ForegroundColor Yellow
Invoke-Ssh "set -e; hostname; nproc; systemctl is-active nginx; systemctl is-active frps; grep -E 'vhostHTTPPort' /etc/frp/frps.toml; which certbot"

Write-Host "`n[2/6] 上传 nginx 配置 / systemd 单元 / 日志轮转" -ForegroundColor Yellow
Send-File $Conf "/etc/nginx/sites-available/maj"
Send-File $Service "/etc/systemd/system/maj-relay.service"
Send-File $Logrotate "/etc/logrotate.d/maj"
Invoke-Ssh "mkdir -p /var/log/maj /var/www/html; mkdir -p /etc/nginx/sites-available"

Write-Host "`n[3/6] 先放一个 80 端口的临时块，好让 certbot 能校验" -ForegroundColor Yellow
Invoke-Ssh @"
set -e
if [ ! -f /etc/letsencrypt/live/$Domain/fullchain.pem ]; then
  cat > /etc/nginx/sites-available/maj-bootstrap <<'EOF'
server {
    listen 80;
    server_name $Domain;
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 200 'maj bootstrap'; }
}
EOF
  ln -sf /etc/nginx/sites-available/maj-bootstrap /etc/nginx/sites-enabled/maj
  nginx -t && systemctl reload nginx
  echo 'BOOTSTRAP_OK'
else
  echo 'CERT_ALREADY_EXISTS'
fi
"@

Write-Host "`n[4/6] 申请 / 续期证书" -ForegroundColor Yellow
Invoke-Ssh @"
set -e
if [ ! -f /etc/letsencrypt/live/$Domain/fullchain.pem ]; then
  certbot certonly --webroot -w /var/www/html -d $Domain \
    --non-interactive --agree-tos -m $Email --keep-until-expiring
fi
ls -l /etc/letsencrypt/live/$Domain/
"@

Write-Host "`n[5/6] 启用正式配置并 reload" -ForegroundColor Yellow
Invoke-Ssh @"
set -e
rm -f /etc/nginx/sites-enabled/maj-bootstrap
ln -sf /etc/nginx/sites-available/maj /etc/nginx/sites-enabled/maj
nginx -t
systemctl reload nginx
systemctl daemon-reload
systemctl enable --now maj-relay 2>/dev/null || true
echo '--- systemd 定时续期 ---'
systemctl list-timers 'certbot*' --no-pager | head -3 || true
"@

Write-Host "`n[6/6] 从公网验证（走完整链路：nginx -> frps -> 隧道 -> 本机服务）" -ForegroundColor Yellow
if ($DryRun) {
    Write-Host "  [dry-run] 跳过公网验证" -ForegroundColor DarkGray
} else {
    foreach ($path in @('/api/health', '/')) {
        try {
            $r = Invoke-WebRequest -Uri "https://$Domain$path" -TimeoutSec 20 -UseBasicParsing
            Write-Host ("  https://{0}{1}  ->  HTTP {2}（{3} 字节）" -f `
                $Domain, $path, $r.StatusCode, $r.RawContentLength) -ForegroundColor Green
        } catch {
            Write-Host ("  https://{0}{1}  ->  ❌ {2}" -f $Domain, $path, `
                $_.Exception.Message) -ForegroundColor Red
        }
    }
}

Write-Host "`n完成。接下来在本机跑：pwsh deploy/maj.ps1 start" -ForegroundColor Cyan
Write-Host "（它会拉起 run.py 和 frpc，然后 status 里会打印公网可达性）" -ForegroundColor DarkGray
