#!/bin/bash
# 双击这个文件即可启动（macOS 只有 .command 才支持双击运行）
#
# 启动 **Electron 版**：自带浏览器开雀魂，把「打这张的收益」直接画在牌面上。
#
# 牌全部来自**协议**（解析雀魂的 WebSocket 消息），不截图、不识别画面 ——
# 所以不会因为界面变暗、被窗口挡住、牌被压住而认错。
#
# 想加参数就在终端里跑 ./run-electron.sh，常用的有：
#   --badge     额外开一个独立的状态窗口
#   --no-dump   不记录牌局帧（默认记录，只存牌局消息，登录/账号一律不存）
#   --verbose   逐帧打印收到的消息（排查「没反应」时用）
#
# 换模型：菜单栏 「模型」→ 三麻/四麻 → 选一个（写进 build/models.json）。
#
# 启动日志同时写到 build/app.log，出问题把那个文件发我。

cd "$(dirname "$0")" || exit 1

if [ ! -f ./run-electron.sh ]; then
    echo "❌ 这个文件得和 run-electron.sh 放在同一个目录里"
    read -r -p "按回车关闭…"
    exit 1
fi
if [ ! -x ./run-electron.sh ]; then
    echo "▸ run-electron.sh 没有执行权限，正在补上…"
    chmod +x ./run-electron.sh || {
        echo "❌ 加执行权限失败，请在终端里执行：chmod +x run-electron.sh"
        read -r -p "按回车关闭…"
        exit 1
    }
fi

exec ./run-electron.sh "$@"
