#!/bin/bash
# 双击这个文件即可开一桌（macOS 只有 .command 支持双击运行）
#
# Java 版牌桌：Swing 窗口，人类点牌 / 选模型 / AI 对战 / 回放 / 换主视角。
# 跟根目录那个「土豆麻将.command」（Electron 版雀魂悬浮提示）是**两套东西**，
# 互不影响，各用各的。
#
# 想加参数就在终端里跑 ./aimodel/ui/java/run.sh --help，常用的有：
#   --port 8765      换端口
#   --seats human,maj-normalv1.1,maj-flashv1.1   指定座位
#   --delay 1.0       AI 每手停顿秒数
#   --no-server       复用已经在跑的后端
cd "$(dirname "$0")" || exit 1

if [ ! -x ./aimodel/ui/java/run.sh ]; then
    echo "❌ 找不到 aimodel/ui/java/run.sh"
    echo "   如果你把它移到别处了，改这个脚本里的路径；"
    echo "   或者先编译一次：bash aimodel/ui/java/build.sh"
    read -r -p "按回车关闭…"
    exit 1
fi

exec ./aimodel/ui/java/run.sh "$@"
