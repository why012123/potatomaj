#!/bin/bash
# 双击这个文件即可开一桌三麻（macOS 只有 .command 才支持双击运行）
# 你 / AI 助手 / 脚本 AI 同桌，浏览器里点牌打
cd "$(dirname "$0")" || exit 1
exec ./run-battle.sh "$@"
